"""Heads-up poker: train a base policy, adapt per player, and evaluate.

Entry point only. Poker rules live in poker_env.py; neural-network losses and
the differentiable inner update live in poker_learning.py. See POKER_README.md.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
try:
    import numpy as np
    import torch
    # A leftover namespace named "torch" can import successfully without being
    # a complete PyTorch installation. Validate the APIs used by this program.
    from torch import nn as _torch_nn
    from torch.func import functional_call as _torch_functional_call
except (ImportError, OSError) as exc:
    loaded_torch = sys.modules.get("torch")
    torch_file = getattr(loaded_torch, "__file__", None)
    torch_paths = list(getattr(loaded_torch, "__path__", []))
    raise SystemExit(
        "PyTorch could not load its required nn/func modules.\n"
        f"Python: {sys.executable}\n"
        f"Python version: {sys.version.split()[0]}\n"
        f"torch file: {torch_file or '(no package file / not loaded)'}\n"
        f"torch search paths: {torch_paths}\n"
        f"Project environment: {ROOT / '.venv'}\n"
        f"Original error: {exc}\n"
        "Run .\\run_poker.ps1 to check and repair the project environment."
    ) from exc

from poker_env import ACTION_NAMES, Holdem, cards_text
from poker_learning import (FEATURE_DIM, Bot, PokerNet, Rollout, choose_action,
                            collect_rollouts, make_decision, meta_train_step,
                            train_step)
from poker_offline import offline_train_step
from poker_replay import ReplayStore

SCHEMA = 1
STYLES = ("balanced", "tight", "loose", "aggressive", "calling", "random")


def configure(seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def atomic_save(path: Path, payload: dict) -> None:
    """Commit a complete checkpoint; interrupted writes keep the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_payload(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Checkpoint not found: {path}\nTrain first: .\\run_poker.ps1 train")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Unsupported checkpoint format.")
    if value["config"].get("feature_dim") != FEATURE_DIM:
        raise ValueError("Checkpoint observation format does not match this program.")
    return value


def model_from(payload: dict, state_key: str = "model") -> PokerNet:
    model = PokerNet(hidden=payload["config"]["hidden"])
    model.load_state_dict(payload[state_key])
    return model


def new_config(hidden: int, stack_bb: int) -> dict:
    return {"hidden": hidden, "stack_bb": stack_bb, "feature_dim": FEATURE_DIM}


def state_digest(model: PokerNet) -> str:
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def base_path(args) -> Path:
    return Path(args.data_dir).resolve() / "base.pt"


def player_path(args, name: str) -> Path:
    # User names are labels, never filesystem paths (also supports Chinese IDs).
    key = hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
    return Path(args.data_dir).resolve() / "players" / f"{key}.pt"


def replay_path(args) -> Path:
    return Path(args.data_dir).resolve() / "shared_replay.sqlite3"


def archive_player(store, path: Path, saved: dict) -> int:
    """Recover committed outbox events and the only surviving legacy hands."""
    added = 0
    for event in saved.get("replay_outbox", []):
        added += store.add_hand(**event)
    if not saved.get("replay_enabled"):
        batch = Rollout.from_dict(saved["pending"]) if saved.get("pending") else Rollout()
        for hand, reward in enumerate(batch.rewards_bb):
            number = saved["hands"] - batch.hands + hand + 1
            added += store.add_hand(
                event_id=f"legacy:{path.stem}:{saved.get('base_id', 'unknown')}:{number}",
                source_group=path.stem, source_type="legacy_unknown", hand_no=number,
                decisions=[d for d in batch.decisions if d["hand"] == hand],
                reward_bb=reward, stack_bb=saved["config"]["stack_bb"])
    return added


def sync_replay(args, store) -> int:
    added = 0
    for path in sorted((Path(args.data_dir).resolve() / "players").glob("*.pt")):
        added += archive_player(store, path, load_payload(path))
    return added


def command_replay_info(args) -> None:
    with ReplayStore(replay_path(args)) as store:
        added = sync_replay(args, store)
        print(f"Replay: {replay_path(args)} | recovered hands={added}")
        for source in ("human", "bot", "legacy_unknown"):
            stats = store.stats(source_types=(source,))
            print(f"{source}: hands={stats['hands']} decisions={stats['decisions']} groups={stats['groups']}")
    print("Shared training uses human data; --include-legacy also accepts old pending hands of unknown origin.")


def command_train_shared(args) -> None:
    if not 1.0 <= args.max_weight <= 1.0e6:
        raise ValueError("--max-weight must be between 1 and 1000000.")
    configure(args.seed, args.threads)
    path = base_path(args)
    saved = load_payload(path)
    model = model_from(saved)
    anchor = model_from(saved, "shared_anchor" if saved.get("shared_anchor") else "model")
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if saved.get("offline_optimizer"):
        optimizer.load_state_dict(saved["offline_optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    completed = saved.get("shared_epochs", 0)
    sources = ("human", "legacy_unknown") if args.include_legacy else ("human",)
    with ReplayStore(replay_path(args)) as store:
        sync_replay(args, store)
        stats = store.stats(source_types=sources)
        if not stats["decisions"]:
            raise ValueError("No eligible AI decisions in replay. Collect human hands with play first, or use --include-legacy for old pending experience; replay-info shows available data.")
        backup = path.with_name("base-before-shared.pt")
        atomic_save(backup, saved)
        print(f"Shared offline training: {stats['trainable_hands']} usable / {stats['hands']} hands from "
              f"{stats['trainable_groups']} groups; sources={','.join(sources)}")
        print("Balanced sampling across players; identities are never network inputs. Offline loss is not an evaluation of playing strength.")
        print(f"Previous base saved to: {backup}")
        try:
            for epoch in range(completed, completed + args.epochs):
                started = time.perf_counter()
                rng = random.Random(args.seed + epoch)
                metrics = []
                for _ in range(args.steps_per_epoch):
                    hands = store.sample_hands(args.batch_hands, rng, source_types=sources)
                    metrics.append(offline_train_step(model, hands, optimizer, anchor=anchor,
                        temperature=args.temperature, max_weight=args.max_weight,
                        anchor_coef=args.anchor_coef))
                payload = dict(saved)
                payload.update(model=model.state_dict(), optimizer=None,
                    offline_optimizer=optimizer.state_dict(), shared_anchor=anchor.state_dict(),
                    shared_epochs=epoch + 1, base_id=state_digest(model),
                    method="shared-offline", algorithm="AWR-inspired offline regression on pooled human AI trajectories",
                    shared_sources=list(sources), shared_replay_hands=stats["hands"])
                atomic_save(path, payload)
                print(f"shared_epoch={epoch + 1} offline_loss={statistics.mean(m['loss'] for m in metrics):.5f} "
                      f"sampled_hands={args.steps_per_epoch * args.batch_hands} seconds={time.perf_counter()-started:.1f}", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. The last completed shared epoch remains saved.")
    print(f"Base checkpoint: {path}")


def command_train(args) -> None:
    configure(args.seed, args.threads)
    path = base_path(args)
    existing = path.exists()
    if existing:
        saved = load_payload(path)
        model = model_from(saved)
        config = saved["config"]
        if args.stack_bb != config["stack_bb"] or args.hidden != config["hidden"]:
            raise ValueError("Resume requires the saved --stack-bb and --hidden settings; use another --data-dir for a new model.")
        completed = saved.get("epochs", 0)
        snapshots = saved.get("opponent_pool", [])
    else:
        config = new_config(args.hidden, args.stack_bb)
        model = PokerNet(hidden=args.hidden)
        completed, snapshots = 0, []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if existing and saved.get("optimizer"):
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    print(f"{'Resuming' if existing else 'Starting'} base policy: {path}")
    print("Method: ordinary on-policy actor-critic." if args.method == "ordinary" else
          "Method: actor-critic warmup + sampled-surrogate MAML (L2E-inspired); no GTO claim.")
    try:
        for epoch in range(completed, completed + args.epochs):
            started = time.perf_counter()
            configure(args.seed + epoch, args.threads)
            task_rng = random.Random(args.seed + epoch)
            task_metrics = []
            rewards = []
            phase = "warmup" if args.method == "ordinary" or epoch < args.warmup_epochs else "meta"
            for task in range(args.tasks):
                if snapshots and task_rng.random() < 0.25:
                    opponent = PokerNet(hidden=config["hidden"])
                    opponent.load_state_dict(task_rng.choice(snapshots))
                    opponent.eval()
                else:
                    opponent = Bot(task_rng.choice(STYLES))
                seed = args.seed + epoch * 100000 + task * 1000
                if phase == "warmup":
                    batch = collect_rollouts(model, opponent, args.support_hands + args.query_hands,
                                             seed, config["stack_bb"])
                    metrics = train_step(model, batch, optimizer)
                    rewards.extend(batch.rewards_bb)
                else:
                    metrics = meta_train_step(model, optimizer, opponent,
                        support_hands=args.support_hands, query_hands=args.query_hands,
                        seed=seed, stack_bb=config["stack_bb"], inner_lr=args.inner_lr,
                        first_order=args.first_order)
                    # Keep reporting robust to optional diagnostic keys.
                    rewards.extend(metrics.get("query_rewards_bb", []))
                task_metrics.append(metrics)
            if (epoch + 1) % args.snapshot_every == 0:
                snapshots.append(copy.deepcopy(model.state_dict()))
                snapshots = snapshots[-6:]
            payload = {"schema": SCHEMA, "config": config, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "epochs": epoch + 1,
                       "opponent_pool": snapshots, "method": args.method,
                       "algorithm": "sampled-surrogate MAML; scripted + historical opponents",
                       "base_id": state_digest(model), "seed": args.seed}
            atomic_save(path, payload)
            loss_values = [float(m.get("loss", m.get("query_loss", 0.0))) for m in task_metrics]
            reward_text = f" sample_bb/100={100 * statistics.mean(rewards):+.1f}" if rewards else ""
            print(f"epoch={epoch + 1} phase={phase} loss={statistics.mean(loss_values):.5f}"
                  f"{reward_text} seconds={time.perf_counter()-started:.1f}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. The last completed epoch remains saved; run train again to resume.")
    print(f"Base checkpoint: {path}" if path.exists() else "No complete epoch yet; no checkpoint was created.")


def load_player(args, name: str) -> tuple[dict, PokerNet, PokerNet, torch.optim.Optimizer, Rollout]:
    path = player_path(args, name)
    if path.exists():
        saved = load_payload(path)
        if saved.get("player") != name:
            raise ValueError("Player checkpoint name mismatch.")
    else:
        base = load_payload(base_path(args))
        saved = {"schema": SCHEMA, "config": base["config"], "player": name,
                 "base_id": base.get("base_id", "unknown"), "model": base["model"],
                 "anchor": copy.deepcopy(base["model"]), "hands": 0, "updates": 0,
                 "net_bb": 0.0, "action_counts": [0] * 6, "pending": None}
    model = model_from(saved)
    anchor = model_from(saved, "anchor")
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    # Match the clipped SGD adaptation that the meta-training inner loop learns.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if saved.get("optimizer"):
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    pending = Rollout.from_dict(saved["pending"]) if saved.get("pending") else Rollout()
    return saved, model, anchor, optimizer, pending


def save_player(args, saved, model, optimizer, pending) -> None:
    saved["model"] = model.state_dict()
    saved["optimizer"] = optimizer.state_dict()
    saved["pending"] = pending.to_dict()
    atomic_save(player_path(args, saved["player"]), saved)
    # The checkpoint commits the hand first. If archiving is interrupted, the
    # persisted outbox is replayed at startup; SQLite deduplicates its event ID.
    if saved.get("replay_outbox"):
        with ReplayStore(replay_path(args)) as store:
            for event in saved["replay_outbox"]:
                store.add_hand(**event)
        saved["replay_outbox"] = []


def update_player(args, saved, model, anchor, optimizer, pending) -> Rollout:
    metrics = train_step(model, pending, optimizer, anchor=anchor, anchor_coef=args.anchor_coef)
    if not metrics.get("skipped", False):
        saved["updates"] += 1
    pending = Rollout()
    save_player(args, saved, model, optimizer, pending)
    if metrics.get("skipped", False):
        print("Batch had no AI decisions; cleared without changing the model.", flush=True)
    else:
        print(f"Personal update={saved['updates']} | hands={saved['hands']} | cumulative AI net={saved['net_bb']:+.1f} bb", flush=True)
    return pending


def action_label(env: Holdem, action: int) -> str:
    target = env.legal_actions()[action]
    if action == 0:
        return "fold"
    if action == 1:
        cost = target - env.street_bets[env.player]
        return "check" if cost == 0 else f"call {cost / 2:g} bb"
    return f"{ACTION_NAMES[action]} to {target / 2:g} bb (this street total)"


def prepare_match(saved: dict, *, new_match: bool = False) -> dict:
    """Keep chips attached to identities [AI, opponent], even as seats alternate."""
    match = saved.get("match")
    initial = saved["config"]["stack_bb"] * Holdem.big_blind
    if match is None or new_match:
        match = {"number": 1 if match is None else match["number"] + 1,
                 "stacks": [initial, initial], "hands": 0}
        saved["match"] = match
    stacks = match.get("stacks", [])
    if (len(stacks) != 2 or any(type(v) is not int or v < 0 for v in stacks)
            or sum(stacks) != 2 * initial):
        raise ValueError("Invalid saved match stacks; use --new-match to start a new match.")
    return match


def show_match(saved: dict) -> None:
    match = saved["match"]
    initial = saved["config"]["stack_bb"]
    ai, other = (chips / Holdem.big_blind for chips in match["stacks"])
    print(f"Match {match['number']} | hands={match['hands']} | "
          f"you={other:g} bb, AI={ai:g} bb | "
          f"match net: you={other-initial:+g} bb, AI={ai-initial:+g} bb | "
          f"lifetime net: you={-saved['net_bb']:+g} bb, AI={saved['net_bb']:+g} bb",
          flush=True)


def play_hand(model, opponent, *, seed: int, stack_bb: int, hero: int,
              human: bool = False, counts=None, starting_stacks=None):
    env = Holdem(seed=seed, stack_bb=stack_bb, button=0,
                 starting_stacks=starting_stacks)
    decisions = []
    streets = ("preflop", "flop", "turn", "river")
    announced_street = None
    # Independent streams prevent one player's decision count changing the other's RNG.
    hero_rng = np.random.default_rng(seed + 19 + hero)
    other_rng = np.random.default_rng(seed + 7919 + hero)
    while not env.done:
        seat = env.player
        obs = env.observe()
        # Show newly dealt cards before either player acts on that street.
        # The AI may act first, so announcing only at the human prompt makes
        # its new-street bet appear to have happened before those cards.
        if human and env.street != announced_street:
            print(f"\n{streets[env.street]} | your cards: {cards_text(env.hole[1-hero])} | board: {cards_text(env.board) or '-'}")
            announced_street = env.street
        if seat == hero:
            action = choose_action(model, obs, rng=hero_rng)
            decisions.append(make_decision(obs, action))
            if human:
                print(f"AI [{streets[env.street]}]: {action_label(env, action)}")
        elif human:
            print(f"pot={env.pot / 2:g} bb | remaining stacks: you={env.stacks[seat] / 2:g} bb, AI={env.stacks[hero] / 2:g} bb")
            print(f"Street bets [{streets[env.street]}]: you={env.street_bets[seat] / 2:g} bb, "
                  f"AI={env.street_bets[hero] / 2:g} bb | "
                  f"to call={min(env.to_call, env.stacks[seat]) / 2:g} bb")
            legal = env.legal_actions()
            print(" | ".join(f"{a}: {action_label(env, a)}" for a in legal))
            while True:
                try:
                    answer = input("Your action (q cancels current hand and saves previous hands): ").strip().lower()
                except EOFError:
                    return None
                if answer in ("q", "quit", "exit"):
                    return None
                if answer.isdigit() and int(answer) in legal:
                    action = int(answer)
                    break
                print("Choose one of the displayed action numbers.")
            if counts is not None:
                counts[action] += 1
            print(f"You [{streets[env.street]}]: {action_label(env, action)}")
        else:
            action = opponent.act(obs, rng=other_rng) if isinstance(opponent, Bot) else choose_action(opponent, obs, rng=other_rng)
            if counts is not None:
                counts[action] += 1
        env.step(action)
    if human:
        if env.folded is None:
            print(f"Showdown | board: {cards_text(env.board)} | your cards: {cards_text(env.hole[1-hero])} | AI: {cards_text(env.hole[hero])}")
        else:
            print("Hand ended by fold; unrevealed opponent cards stay hidden.")
        print(f"Your net: {env.rewards[1-hero]:+.1f} bb | AI net: {env.rewards[hero]:+.1f} bb")
    return decisions, float(env.rewards[hero]), list(env.stacks)


def command_play(args) -> None:
    configure(args.seed, args.threads)
    name = args.player.strip()
    if not name:
        raise ValueError("Player name must not be empty.")
    saved, model, anchor, optimizer, pending = load_player(args, name)
    personal_model = model
    if args.base_only:
        base = load_payload(base_path(args))
        if base["config"] != saved["config"]:
            raise ValueError("This player's configuration differs from the shared base; use a new player name for --base-only.")
        model = model_from(base)
    # Import surviving legacy hands before any personal update can clear them.
    # No player name is stored in replay, only an opaque balancing group.
    with ReplayStore(replay_path(args)) as store:
        archive_player(store, player_path(args, name), saved)
    saved["replay_enabled"] = True
    saved.setdefault("replay_stream_id", uuid.uuid4().hex)
    saved["replay_outbox"] = []
    save_player(args, saved, personal_model, optimizer, pending)
    legacy_match = saved.get("match") is None and saved["hands"] > 0
    previous_match = saved.get("match")
    ended_before = previous_match is not None and 0 in previous_match["stacks"]
    # Recover a hand that was committed just before its learning update was
    # interrupted. Do this before --new-match replaces the terminal stacks.
    try:
        if not args.base_only and pending.hands and (pending.hands >= args.update_every or ended_before):
            pending = update_player(args, saved, personal_model, anchor, optimizer, pending)
    except KeyboardInterrupt:
        print("\nLearning interrupted; saved hands and chips are retained. Run play again to retry.")
        return
    match = prepare_match(saved, new_match=args.new_match)
    save_player(args, saved, personal_model, optimizer, pending)
    opponent = Bot(args.bot) if args.bot else None
    print(f"Player: {name} | completed hands={saved['hands']} | updates={saved['updates']} | pending={pending.hands}")
    print("Chips carry between hands; blinds stay 0.5/1 bb and seats alternate.")
    print("Policy: shared base, fixed for this session; hands are archived without personal learning." if args.base_only else
          f"Policy: personal; AI learns every {args.update_every} completed hands, or when a match ends.")
    print("Use the SAME player name to resume chips and learning. Use --new-match for fresh chips.")
    if legacy_match:
        print("Previous save has no match stacks: starting a fresh match; learning and lifetime results are retained.")
    show_match(saved)
    try:
        for _ in range(args.hands):
            if 0 in match["stacks"]:
                break
            counts = [0] * 6
            number = saved["hands"]
            hero = number % 2
            seat_stacks = [0, 0]
            seat_stacks[hero], seat_stacks[1 - hero] = match["stacks"]
            result = play_hand(model, opponent, seed=args.seed + number * 2,
                               stack_bb=saved["config"]["stack_bb"], hero=hero,
                               human=opponent is None, counts=counts, starting_stacks=seat_stacks)
            if result is None:
                break
            decisions, reward, ending_stacks = result
            match["stacks"] = [ending_stacks[hero], ending_stacks[1 - hero]]
            match["hands"] += 1
            if not args.base_only:
                pending.add_hand(decisions, reward, saved["config"]["stack_bb"])
            saved["hands"] += 1
            saved["net_bb"] += reward
            saved["action_counts"] = [a+b for a,b in zip(saved["action_counts"], counts)]
            saved["replay_outbox"].append({
                "event_id": f"{saved['replay_stream_id']}:{saved['hands']}",
                "source_group": player_path(args, name).stem,
                "source_type": "bot" if opponent is not None else "human",
                "hand_no": saved["hands"], "decisions": decisions,
                "reward_bb": reward, "stack_bb": saved["config"]["stack_bb"]})
            # Save the complete experience before training. A crash never loses the hand.
            save_player(args, saved, personal_model, optimizer, pending)
            if not args.base_only and (pending.hands >= args.update_every or 0 in match["stacks"]):
                pending = update_player(args, saved, personal_model, anchor, optimizer, pending)
            show_match(saved)
            if not opponent:
                print(f"Saved and archived. Total hands={saved['hands']}; shared base unchanged during play." if args.base_only else
                      f"Saved and archived. Total hands={saved['hands']}; learning batch={pending.hands}/{args.update_every}.")
    except KeyboardInterrupt:
        print("\nStopped; incomplete hand discarded. Completed hands are already saved.")
    if 0 in match["stacks"]:
        winner = "You win" if match["stacks"][0] == 0 else "AI wins"
        print(f"Match over: {winner}. Use play --player {name!r} --new-match to play again; learning is retained.")
    print(f"Player checkpoint: {player_path(args, name)}")


def command_evaluate(args) -> None:
    configure(args.seed, args.threads)
    saved = load_payload(Path(args.checkpoint).resolve() if args.checkpoint else base_path(args))
    base = model_from(saved)
    stack = saved["config"]["stack_bb"]
    pairs = args.hands // 2
    print(f"Frozen evaluation: {pairs} independent deals, each played from both seats ({2*pairs} hands per policy).")
    print("bb/100 and paired difference with approximate 95% confidence interval; no updates during evaluation.")
    for index, style in enumerate(STYLES if args.opponent == "all" else [args.opponent]):
        adapted = copy.deepcopy(base)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=args.lr)
        opponent = Bot(style)
        remaining = args.adapt_hands
        batch_index = 0
        while remaining:
            hands = min(args.update_every, remaining)
            batch = collect_rollouts(adapted, opponent, hands,
                args.seed + 10000000 + index * 100000 + batch_index * 1000, stack)
            train_step(adapted, batch, optimizer, anchor=base, anchor_coef=args.anchor_coef)
            remaining -= hands
            batch_index += 1
        base_pairs, adapted_pairs = [], []
        for pair in range(pairs):
            seed = args.seed + 20000000 + index * 100000 + pair
            br, ar = [], []
            for hero in (0, 1):
                br.append(play_hand(base, opponent, seed=seed, stack_bb=stack, hero=hero)[1])
                ar.append(play_hand(adapted, opponent, seed=seed, stack_bb=stack, hero=hero)[1])
            base_pairs.append(statistics.mean(br))
            adapted_pairs.append(statistics.mean(ar))
        differences = [a-b for a,b in zip(adapted_pairs, base_pairs)]
        diff = 100 * statistics.mean(differences)
        half_width = 196 * statistics.stdev(differences) / math.sqrt(pairs) if pairs > 1 else float("nan")
        print(f"{style:10s} base={100*statistics.mean(base_pairs):+9.1f} adapted={100*statistics.mean(adapted_pairs):+9.1f}"
              f" delta={diff:+8.1f} CI95=[{diff-half_width:+.1f},{diff+half_width:+.1f}]", flush=True)


def command_players(args) -> None:
    paths = sorted((Path(args.data_dir).resolve() / "players").glob("*.pt"))
    if not paths:
        print("No saved players yet.")
    for path in paths:
        saved = load_payload(path)
        pending = Rollout.from_dict(saved["pending"]) if saved.get("pending") else Rollout()
        match = saved.get("match")
        chips = (f" match={match['number']} AI_stack={match['stacks'][0]/Holdem.big_blind:g} bb"
                 f" your_stack={match['stacks'][1]/Holdem.big_blind:g} bb" if match else "")
        print(f"{saved['player']} | hands={saved['hands']} updates={saved['updates']} pending={pending.hands}"
              f" AI_net={saved['net_bb']:+.1f} bb{chips} | {path}")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive.")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative.")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive.")
    return value


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=str(ROOT / "poker_data"))
    common.add_argument("--seed", type=nonnegative_int, default=42)
    common.add_argument("--threads", type=positive_int, default=2)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", parents=[common], help="Train/resume the shared base model")
    train.add_argument("--epochs", type=positive_int, default=200, help="Additional epochs on resume")
    train.add_argument("--method", choices=("meta", "ordinary"), default="meta")
    train.add_argument("--warmup-epochs", type=nonnegative_int, default=20)
    train.add_argument("--tasks", type=positive_int, default=4)
    train.add_argument("--support-hands", type=positive_int, default=16)
    train.add_argument("--query-hands", type=positive_int, default=16)
    train.add_argument("--hidden", type=positive_int, default=128)
    train.add_argument("--stack-bb", type=positive_int, default=100)
    train.add_argument("--lr", type=positive_float, default=0.0003)
    train.add_argument("--inner-lr", type=positive_float, default=0.05)
    train.add_argument("--snapshot-every", type=positive_int, default=10)
    train.add_argument("--first-order", action="store_true", help="Ignore inner-update Hessian terms")
    train.set_defaults(func=command_train)
    shared = sub.add_parser("train-shared", parents=[common], help="Improve the shared base from pooled human hands, without player identity inputs")
    shared.add_argument("--epochs", type=positive_int, default=20)
    shared.add_argument("--steps-per-epoch", type=positive_int, default=20)
    shared.add_argument("--batch-hands", type=positive_int, default=64)
    shared.add_argument("--lr", type=positive_float, default=0.0001)
    shared.add_argument("--temperature", type=positive_float, default=0.1)
    shared.add_argument("--max-weight", type=positive_float, default=20.0)
    shared.add_argument("--anchor-coef", type=positive_float, default=0.1)
    shared.add_argument("--include-legacy", action="store_true", help="Also use surviving old pending hands whose human/bot origin was not recorded")
    shared.set_defaults(func=command_train_shared)
    replay = sub.add_parser("replay-info", parents=[common], help="Recover saved experiences and show pooled hand counts")
    replay.set_defaults(func=command_replay_info)
    play = sub.add_parser("play", parents=[common], help="Play and save a separate adaptive model per person")
    play.add_argument("--player", required=True)
    play.add_argument("--hands", type=positive_int, default=1000000)
    play.add_argument("--new-match", action="store_true", help="Start with fresh chips; retain this player's learned model and lifetime results")
    play.add_argument("--base-only", action="store_true", help="Use the shared base without personal adaptation; archive completed hands")
    play.add_argument("--bot", choices=STYLES, help="Simulation/demo instead of human keyboard input")
    play.add_argument("--update-every", type=positive_int, default=32)
    play.add_argument("--lr", type=positive_float, default=0.05)
    play.add_argument("--anchor-coef", type=positive_float, default=0.1)
    play.set_defaults(func=command_play)
    evaluate = sub.add_parser("evaluate", parents=[common], help="Compare base/adapted policies on fresh paired deals")
    evaluate.add_argument("--opponent", choices=("all",)+STYLES, default="all")
    evaluate.add_argument("--hands", type=positive_int, default=1000)
    evaluate.add_argument("--adapt-hands", type=nonnegative_int, default=128)
    evaluate.add_argument("--update-every", type=positive_int, default=32)
    evaluate.add_argument("--lr", type=positive_float, default=0.05)
    evaluate.add_argument("--anchor-coef", type=positive_float, default=0.1)
    evaluate.add_argument("--checkpoint", help="Evaluate a specific checkpoint, e.g. poker_data/base-before-shared.pt")
    evaluate.set_defaults(func=command_evaluate)
    players = sub.add_parser("players", parents=[common], help="List saved player models")
    players.set_defaults(func=command_players)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "evaluate" and (args.hands < 4 or args.hands % 2):
        raise SystemExit("--hands must be an even number >= 4 for paired evaluation.")
    try:
        args.func(args)
    except (ValueError, OSError, FloatingPointError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
