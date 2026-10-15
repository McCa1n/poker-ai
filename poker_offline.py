"""Conservative shared-policy regression from completed historical AI hands.

This is an AWR-inspired offline update, not on-policy REINFORCE/PPO and not a
full reproduction of AWR (https://arxiv.org/abs/1910.00177). Old actions need
no recorded behavior probabilities: they are supervised regression targets,
weighted by exp((Monte Carlo return - detached value baseline) / temperature).
The baseline fits the dataset's mixture of past policies/opponents, not an
optimal value function or an unbiased value of the current policy. Poker's
single-hand outcomes are noisy; this update cannot guarantee stronger play.

Differences from the paper include joint actor/value optimization with a
detached pre-update baseline, equal weighting of hands, and a frozen initial
base-policy KL penalty. Return targets use the existing normalized per-hand
NET reward. No counterfactual outcomes or unseen opponent cards are inferred.
Only state, legal mask, action and return become tensors; names and all other
metadata are ignored. Source-balanced hand sampling belongs to the caller.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import torch

from poker_learning import FEATURE_DIM, PokerNet, _masked_logits


def _hand_tensors(hands: Sequence[Mapping], model: PokerNet):
    """Average decisions within each nonempty hand, then average those hands."""
    states, masks, actions, returns, hand_weights = [], [], [], [], []
    trained_hands = 0
    for hand in hands:
        if not isinstance(hand, Mapping) or not isinstance(hand.get("decisions"), (list, tuple)):
            raise ValueError("Each offline hand must contain a decisions list")
        decisions = hand["decisions"]
        if not decisions:
            continue  # An automatic blind all-in may give the AI no decision.
        trained_hands += 1
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("Offline decisions must be mappings")
            try:
                state = [float(v) for v in decision["state"]]
                mask = list(decision["mask"])
                action = decision["action"]
                reward = float(decision["return"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid offline decision fields") from exc
            if (len(state) != FEATURE_DIM or not all(math.isfinite(v) for v in state)
                    or len(mask) != 6 or any(v not in (False, True, 0, 1) for v in mask)
                    or isinstance(action, bool) or not isinstance(action, Integral)
                    or not 0 <= action < 6 or not mask[action] or not math.isfinite(reward)):
                raise ValueError("Offline decision has invalid features, legality or return")
            states.append(state)
            masks.append(mask)
            actions.append(int(action))
            returns.append(reward)
            hand_weights.append(1.0 / len(decisions))
    if not actions:
        return None
    device = next(model.parameters()).device
    tensors = (
        torch.tensor(states, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(returns, dtype=torch.float32, device=device),
        torch.tensor(hand_weights, dtype=torch.float32, device=device) / trained_hands,
    )
    if not all(bool(torch.isfinite(t).all()) for t in (tensors[0], tensors[3])):
        raise ValueError("Offline values exceed the model's finite float32 range")
    return (*tensors, trained_hands)


def _advantage_weights(returns, values, hand_weights, temperature, max_weight):
    """Detached positive weights; never amplify a uniformly poor batch."""
    advantage = (returns - values).detach()
    if not bool(torch.isfinite(advantage).all()):
        raise FloatingPointError("Nonfinite offline advantage; update cancelled")
    weights = torch.exp(torch.clamp(advantage / temperature, min=-20.0, max=math.log(max_weight)))
    # Scale down only. Dividing by a tiny mean would turn very poor samples
    # back into strong imitation targets and overwhelm the anchor penalty.
    weights = weights / torch.clamp((hand_weights * weights).sum(), min=1.0)
    return weights, advantage


def offline_train_step(model: PokerNet, hands: list[dict], optimizer, *,
                       anchor: PokerNet, temperature: float = 0.1,
                       max_weight: float = 20.0, anchor_coef: float = 0.1,
                       value_coef: float = 0.5) -> dict:
    """One bounded weighted-regression update on reusable historical hands.

    ``hands`` contains AI decisions with ``state``, ``mask``, ``action`` and
    normalized ``return`` fields. Other fields never enter the network.
    ``anchor`` must be a separate copy of the original base, reused unchanged
    for the entire fine-tune run. It is frozen here and cannot be in optimizer.
    Passing older-policy decisions is intentional and requires no importance
    ratios. It remains biased regression on observed actions, not evaluation
    of unchosen actions or exact policy-gradient optimization.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(max_weight) or not 1.0 <= max_weight <= 1.0e6:
        raise ValueError("max_weight must be between 1 and 1000000")
    if any(not math.isfinite(c) or c < 0 for c in (anchor_coef, value_coef)):
        raise ValueError("Loss coefficients must be finite and nonnegative")
    if anchor is None or anchor is model:
        raise ValueError("Offline training requires a separate frozen base anchor")
    parameters = list(model.parameters())
    parameter_ids = {id(p) for p in parameters}
    if any(id(p) not in parameter_ids for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("Optimizer may contain only the trainable model's parameters")
    storage = {p.untyped_storage().data_ptr() for p in parameters}
    if any(p.untyped_storage().data_ptr() in storage for p in anchor.parameters()):
        raise ValueError("Anchor cannot share parameter storage with the trainable model")
    if next(anchor.parameters()).device != parameters[0].device:
        raise ValueError("Model and anchor must be on the same device")
    batch = _hand_tensors(hands, model)
    if batch is None:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "anchor_kl": 0.0,
                "mean_weight": 0.0, "max_weight": 0.0, "mean_advantage": 0.0,
                "grad_norm": 0.0, "hands": len(hands), "trainable_hands": 0,
                "decisions": 0, "skipped": True}
    states, masks, actions, returns, hand_weights, trained_hands = batch
    model.train()
    anchor.eval()
    anchor.requires_grad_(False)
    anchor.zero_grad(set_to_none=True)
    logits, values = model(states)
    if not all(bool(torch.isfinite(t).all()) for t in (logits, values)):
        raise FloatingPointError("Nonfinite offline model output; update cancelled")
    log_probs = torch.log_softmax(_masked_logits(logits, masks), dim=-1)
    weights, advantages = _advantage_weights(returns, values, hand_weights, temperature, max_weight)
    selected = log_probs.gather(1, actions[:, None]).squeeze(1)
    policy_loss = -(hand_weights * weights * selected).sum()
    value_loss = (hand_weights * 0.5 * (values - returns).square()).sum()
    with torch.no_grad():
        reference_logits, _ = anchor(states)
        if not bool(torch.isfinite(reference_logits).all()):
            raise FloatingPointError("Nonfinite anchor output; update cancelled")
        reference_logs = torch.log_softmax(_masked_logits(reference_logits, masks), dim=-1)
        reference_probs = reference_logs.exp()
    anchor_kl = (hand_weights * (reference_probs * (reference_logs - log_probs)).sum(-1)).sum()
    loss = policy_loss + value_coef * value_loss + anchor_coef * anchor_kl
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite offline loss; update cancelled")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError("Optimizer produced nonfinite parameters; do not save this model")
    return {
        "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()), "anchor_kl": float(anchor_kl.detach()),
        "mean_weight": float((hand_weights * weights).sum()), "max_weight": float(weights.max()),
        "mean_advantage": float((hand_weights * advantages).sum()), "grad_norm": float(grad_norm),
        "hands": len(hands), "trainable_hands": trained_hands, "decisions": len(actions), "skipped": False,
    }
