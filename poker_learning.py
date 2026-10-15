"""Neural policies and sampled-surrogate meta-learning for heads-up Hold'em.

This is an educational L2E-inspired implementation, not a reproduction of L2E
or a GTO solver. ``meta_train_step`` differentiates through one inner update
and evaluates on *fresh* on-policy query hands. It treats sampled support/query
trajectories as constants and omits the support-trajectory likelihood correction
in the exact stochastic meta-gradient. Hard-OSG and MMD opponents are not used.

Only ``Holdem.observe()`` is passed to decision makers. Training rewards are
terminal NET big blinds / starting stack in big blinds; reported rewards stay
in big blinds. A whole hand's return is assigned to its decisions (gamma=1).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from poker_env import ACTION_NAMES, Holdem, rank_cards

HISTORY_LENGTH = 12
HISTORY_WIDTH = 11
FEATURE_DIM = 104 + 4 + 20 + HISTORY_LENGTH * HISTORY_WIDTH
BOT_STYLES = ("balanced", "tight", "loose", "aggressive", "calling", "random")


def _action_id(action: Any) -> int:
    if isinstance(action, str):
        try:
            return ACTION_NAMES.index(action)
        except (ValueError, AttributeError):
            return -1
    return int(action)


def visible_strength(obs: Mapping[str, Any]) -> float:
    """A cheap *heuristic*, not equity; uses only own cards and public board."""
    hole, board = list(obs["hole"]), list(obs["board"])
    ranks = sorted((int(c) % 13 + 2 for c in hole), reverse=True)
    high, low = ranks
    pre = 0.15 + 0.45 * (high - 2) / 12 + 0.16 * (low - 2) / 12
    pre += 0.23 if high == low else -0.02 * max(0, high - low - 1)
    pre += 0.04 if hole[0] // 13 == hole[1] // 13 else 0.0
    if len(board) < 3:
        return float(np.clip(pre, 0.04, 0.97))
    rank = rank_cards(hole + board)
    category = int(rank[0])
    top_rank = int(rank[1]) if len(rank) > 1 else high
    bases = (0.18, 0.43, 0.62, 0.72, 0.81, 0.87, 0.93, 0.97, 0.995)
    strength = bases[category] + (top_rank - 2) / 12 * 0.12
    # Board-only made hands are shared; modestly reduce the apparent strength.
    if len(board) == 5 and rank_cards(board) == rank:
        strength *= 0.72
    return float(np.clip(strength, 0.04, 0.995))


def encode(obs: Mapping[str, Any]) -> np.ndarray:
    """260 visible features: 52 hole + 52 board indicators, 4 streets,
    20 scalars and the last 12 public actions (11 features each).

    Scalar order: button, own/other remaining stacks, pot, call cost, own/other
    street bets, own/other contributions, last raise, BB/stack, board count,
    heuristic strength, paired-board flag, largest board suit count, own/other
    observed aggression fractions, other's total paid, history count and last
    aggressor (+1 self, -1 other, 0 absent). Monetary features use starting
    stack. History: self flag, street/3, 6 action indicators, paid, total bet,
    presence. No opponent hole cards or future cards are used.
    """
    result = np.zeros(FEATURE_DIM, dtype=np.float32)
    result[np.asarray(obs["hole"], dtype=np.int64)] = 1.0
    if obs["board"]:
        result[52 + np.asarray(obs["board"], dtype=np.int64)] = 1.0
    street = int(obs["street"])
    result[104 + min(3, street)] = 1.0
    player = int(obs["player"])
    other = 1 - player
    scale = max(float(obs["starting_stack"]), 1.0)
    history = list(obs.get("history", []))
    my_actions = [h for h in history if int(h["player"]) == player]
    their_actions = [h for h in history if int(h["player"]) == other]
    aggressors = [h for h in history if _action_id(h["action"]) >= 2]
    last_aggressor = (1.0 if int(aggressors[-1]["player"]) == player else -1.0) if aggressors else 0.0
    board_ranks = [int(c) % 13 for c in obs["board"]]
    suit_counts = [sum(int(c) // 13 == s for c in obs["board"]) for s in range(4)]
    aggression = lambda actions: sum(_action_id(h["action"]) >= 2 for h in actions) / max(1, len(actions))
    scalars = [
        float(player == int(obs["button"])),
        float(obs["stacks"][player]) / scale,
        float(obs["stacks"][other]) / scale,
        float(obs["pot"]) / scale,
        float(obs["to_call"]) / scale,
        float(obs["street_bets"][player]) / scale,
        float(obs["street_bets"][other]) / scale,
        float(obs["contributions"][player]) / scale,
        float(obs["contributions"][other]) / scale,
        float(obs["last_raise"]) / scale,
        float(obs["big_blind"]) / scale,
        len(obs["board"]) / 5.0,
        visible_strength(obs),
        float(len(set(board_ranks)) < len(board_ranks)),
        max(suit_counts) / 5.0,
        aggression(my_actions),
        aggression(their_actions),
        sum(float(h["paid"]) for h in their_actions) / scale,
        min(len(history), 64) / 64.0,
        last_aggressor,
    ]
    result[108:128] = scalars
    for index, item in enumerate(history[-HISTORY_LENGTH:]):
        offset = 128 + index * HISTORY_WIDTH
        action = _action_id(item["action"])
        result[offset] = float(int(item["player"]) == player)
        result[offset + 1] = int(item["street"]) / 3.0
        if 0 <= action < 6:
            result[offset + 2 + action] = 1.0
        result[offset + 8] = float(item["paid"]) / scale
        result[offset + 9] = float(item["to_total"]) / scale
        result[offset + 10] = 1.0
    if not np.isfinite(result).all():
        raise ValueError("Observation contains nonfinite features")
    return result


class PokerNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = int(hidden)
        self.body = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, 6)
        self.value = nn.Linear(hidden, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        # A mild initial check/call prior avoids a uniform 1/6 all-in policy.
        with torch.no_grad():
            self.policy.bias.copy_(torch.tensor([-0.4, 1.0, 0.0, -0.2, -0.5, -1.8]))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observations)
        return self.policy(features), self.value(features).squeeze(-1)


def _forward(model: PokerNet, states: torch.Tensor, params=None):
    return model(states) if params is None else functional_call(model, params, (states,))


def _masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if not bool(masks.any(dim=-1).all()):
        raise ValueError("Every decision must have at least one legal action")
    return logits.masked_fill(~masks, -1.0e9)


def choose_action(model: PokerNet, obs, params=None, rng=None, deterministic=False) -> int:
    rng = np.random.default_rng() if rng is None else rng
    device = next(model.parameters()).device
    state = torch.as_tensor(encode(obs), device=device).unsqueeze(0)
    mask = torch.as_tensor(obs["legal_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = _forward(model, state, params)
        probabilities = torch.softmax(_masked_logits(logits, mask), dim=-1)[0].cpu().numpy().astype(np.float64)
    probabilities /= probabilities.sum()
    if deterministic:
        return int(np.argmax(probabilities))
    return int(rng.choice(6, p=probabilities))


def make_decision(obs, action: int) -> dict:
    action = int(action)
    mask = [bool(v) for v in obs["legal_mask"]]
    if not 0 <= action < 6 or not mask[action]:
        raise ValueError("Cannot record an illegal action")
    return {"state": encode(obs).tolist(), "mask": mask, "action": action}


class Rollout:
    """JSON-safe fresh on-policy hands; consume once, then discard after update."""
    def __init__(self):
        self.decisions: list[dict] = []
        self.rewards_bb: list[float] = []
        self.hand_seeds: list[int] = []
        self.hero_seats: list[int] = []

    @property
    def hands(self) -> int:
        return len(self.rewards_bb)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    def add_hand(self, decisions, reward_bb: float, stack_bb: float) -> None:
        if stack_bb <= 0 or not math.isfinite(float(reward_bb)):
            raise ValueError("Invalid stack or terminal reward")
        hand_id = self.hands
        normalized_return = float(reward_bb) / float(stack_bb)
        for decision in decisions:
            self.decisions.append({"state": list(decision["state"]), "mask": list(decision["mask"]),
                                   "action": int(decision["action"]), "return": normalized_return, "hand": hand_id})
        self.rewards_bb.append(float(reward_bb))

    def to_dict(self) -> dict:
        return {"decisions": self.decisions, "rewards_bb": self.rewards_bb,
                "hand_seeds": self.hand_seeds, "hero_seats": self.hero_seats}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rollout":
        result = cls()
        result.rewards_bb = [float(v) for v in data.get("rewards_bb", [])]
        if not all(math.isfinite(v) for v in result.rewards_bb):
            raise ValueError("Nonfinite saved rewards")
        for d in data.get("decisions", []):
            state = [float(v) for v in d["state"]]
            mask = [bool(v) for v in d["mask"]]
            action, hand, ret = int(d["action"]), int(d["hand"]), float(d["return"])
            if (len(state) != FEATURE_DIM or len(mask) != 6 or not 0 <= action < 6
                    or not mask[action] or not 0 <= hand < result.hands
                    or not all(math.isfinite(v) for v in state) or not math.isfinite(ret)):
                raise ValueError("Invalid saved rollout")
            result.decisions.append({"state": state, "mask": mask, "action": action, "return": ret, "hand": hand})
        result.hand_seeds = [int(v) for v in data.get("hand_seeds", [])]
        result.hero_seats = [int(v) for v in data.get("hero_seats", [])]
        return result

    def tensors(self, model: PokerNet):
        if not self.decisions:
            raise ValueError("No hero decisions to train on")
        device = next(model.parameters()).device
        states = torch.tensor([d["state"] for d in self.decisions], dtype=torch.float32, device=device)
        masks = torch.tensor([d["mask"] for d in self.decisions], dtype=torch.bool, device=device)
        actions = torch.tensor([d["action"] for d in self.decisions], dtype=torch.long, device=device)
        returns = torch.tensor([d["return"] for d in self.decisions], dtype=torch.float32, device=device)
        return states, masks, actions, returns


class Bot:
    """Rule-based training opponents; no environment/private-deck access."""
    def __init__(self, style: str = "balanced"):
        if style not in BOT_STYLES:
            raise ValueError(f"Unknown bot style {style!r}; choose {BOT_STYLES}")
        self.style = style

    def act(self, obs, rng=None) -> int:
        rng = np.random.default_rng() if rng is None else rng
        mask = np.asarray(obs["legal_mask"], dtype=bool)
        legal = np.flatnonzero(mask)
        if self.style == "random":
            return int(rng.choice(legal))
        strength = visible_strength(obs)
        call = float(obs["to_call"])
        pot = float(obs["pot"])
        odds = call / max(1.0, pot + call)
        profiles = {
            "balanced": (0.34, 0.66, 0.12, 0.50),
            "tight": (0.52, 0.77, 0.025, 0.40),
            "loose": (0.19, 0.59, 0.20, 0.50),
            "aggressive": (0.26, 0.46, 0.34, 0.85),
            "calling": (0.07, 0.92, 0.015, 0.12),
        }
        fold_threshold, raise_threshold, bluff_rate, pressure = profiles[self.style]
        if call > 0 and mask[0] and strength < fold_threshold + odds * 0.45:
            if rng.random() < (0.90 if self.style == "tight" else 0.78):
                return 0
        raises = [a for a in (2, 3, 4, 5) if mask[a]]
        if raises and (strength > raise_threshold or rng.random() < bluff_rate):
            if rng.random() < pressure:
                preferred = 5 if strength > 0.94 else (4 if self.style == "aggressive" else 3)
                return preferred if mask[preferred] else raises[0]
            return raises[0]
        return 1 if mask[1] else int(legal[0])


def collect_rollouts(model, opponent, hands: int, seed: int, stack_bb: int = 100, params=None) -> Rollout:
    """Identical seed/count pairs reproduce cards and seats for paired evaluation.

    Hand h: deck seed ``seed + 104729*h``, hero seat ``h % 2``, button seat 0.
    Action RNG starts at ``seed + 37``. Call with an even hand count to balance
    positions. Different action paths can consume this RNG differently.
    """
    if hands < 1:
        raise ValueError("hands must be positive")
    rng = np.random.default_rng(int(seed) + 37)
    batch = Rollout()
    for hand in range(hands):
        hand_seed, hero = int(seed) + 104729 * hand, hand % 2
        env = Holdem(seed=hand_seed, stack_bb=stack_bb, button=0)
        records = []
        while not env.done:
            obs = env.observe()
            if env.player == hero:
                action = choose_action(model, obs, params=params, rng=rng)
                records.append(make_decision(obs, action))
            elif isinstance(opponent, PokerNet):
                action = choose_action(opponent, obs, rng=rng)
            elif hasattr(opponent, "act"):
                action = opponent.act(obs, rng=rng)
            else:
                action = opponent(obs, rng)
            env.step(action)
        batch.add_hand(records, float(env.rewards[hero]), stack_bb)
        batch.hand_seeds.append(hand_seed)
        batch.hero_seats.append(hero)
    return batch


def policy_loss(model, batch: Rollout, params=None, entropy_coef=0.01, value_coef=0.5,
                anchor=None, anchor_coef=0.0):
    """REINFORCE with a learned baseline plus value/entropy/optional anchor loss.

    Actor terms sum per episode, then average over hands (not decisions), so
    longer trajectories do not silently change the episodic objective. The
    value and entropy terms average decisions. Advantage is stop-gradient.
    """
    states, masks, actions, returns = batch.tensors(model)
    logits, values = _forward(model, states, params)
    log_probs = torch.log_softmax(_masked_logits(logits, masks), dim=-1)
    probs = log_probs.exp()
    advantage = (returns - values).detach()
    selected = log_probs.gather(1, actions[:, None]).squeeze(1)
    actor = -(advantage * selected).sum() / batch.hands
    value = 0.5 * (values - returns).square().mean()
    entropy = -(probs * log_probs).sum(-1).mean()
    anchor_kl = logits.new_zeros(())
    if anchor is not None and anchor_coef:
        with torch.no_grad():
            ref_logits, _ = anchor(states)
            ref_logs = torch.log_softmax(_masked_logits(ref_logits, masks), dim=-1)
            ref_probs = ref_logs.exp()
        anchor_kl = (ref_probs * (ref_logs - log_probs)).sum(-1).mean()
    loss = actor + value_coef * value - entropy_coef * entropy + anchor_coef * anchor_kl
    metrics = {"loss": float(loss.detach()), "policy_loss": float(actor.detach()),
               "value_loss": float(value.detach()), "entropy": float(entropy.detach()),
               "anchor_kl": float(anchor_kl.detach()), "mean_reward_bb": float(np.mean(batch.rewards_bb)),
               "hands": batch.hands, "decisions": batch.decision_count}
    return loss, metrics


def _apply_update(model, optimizer, loss: torch.Tensor) -> float:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite policy loss; update cancelled")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(grad_norm)


def train_step(model, batch: Rollout, optimizer, anchor=None, anchor_coef=0.02) -> dict:
    """Exactly one on-policy update. Caller must discard the consumed batch."""
    if not batch.decision_count:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "anchor_kl": 0.0, "grad_norm": 0.0, "mean_reward_bb": float(np.mean(batch.rewards_bb)) if batch.hands else 0.0,
                "hands": batch.hands, "decisions": 0, "skipped": True}
    loss, metrics = policy_loss(model, batch, anchor=anchor, anchor_coef=anchor_coef)
    metrics["grad_norm"] = _apply_update(model, optimizer, loss)
    return metrics


def adapted_parameters(model, support: Rollout, inner_lr: float = 0.05, first_order: bool = False):
    """One differentiable inner SGD step without mutating the base model."""
    params = OrderedDict(model.named_parameters())
    loss, _ = policy_loss(model, support)
    gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=not first_order)
    if any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Nonfinite inner gradient")
    # Keep the differentiable update bounded without severing its dependence on theta.
    norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    factor = torch.clamp(1.0 / (norm + 1.0e-8), max=1.0)
    return OrderedDict((name, parameter - inner_lr * gradient * factor)
                       for (name, parameter), gradient in zip(params.items(), gradients))


def meta_train_step(model, optimizer, opponent, support_hands=8, query_hands=8,
                    seed=0, stack_bb=100, inner_lr=0.05, first_order=False) -> dict:
    """Sample support -> inner update -> new query -> base meta-update.

    Full differentiation through the inner update is the default. Even in that
    mode this remains the sampled-surrogate approximation documented above.
    ``first_order=True`` further drops the inner Hessian contribution.
    """
    if inner_lr <= 0:
        raise ValueError("inner_lr must be positive")
    support_seed, query_seed = int(seed), int(seed) + 1_000_003
    support = collect_rollouts(model, opponent, support_hands, support_seed, stack_bb)
    # With normal paired hands there are decisions; raise rather than conceal an
    # unsupported empty-support update for unusually short caller configurations.
    params = adapted_parameters(model, support, inner_lr, first_order)
    query = collect_rollouts(model, opponent, query_hands, query_seed, stack_bb, params=params)
    loss, metrics = policy_loss(model, query, params=params)
    metrics["grad_norm"] = _apply_update(model, optimizer, loss)
    metrics.update(support_mean_reward_bb=float(np.mean(support.rewards_bb)),
                   query_mean_reward_bb=float(np.mean(query.rewards_bb)),
                   query_rewards_bb=list(query.rewards_bb),
                   support_seed=support_seed, query_seed=query_seed,
                   support_hands=support.hands, query_hands=query.hands,
                   first_order=bool(first_order))
    return metrics
