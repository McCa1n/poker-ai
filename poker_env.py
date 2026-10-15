"""Small, dependency-free heads-up no-limit Texas Hold'em environment.

Chips are integers: small blind = 1, big blind = 2.  A hand can start with
unequal carried stacks.  The button posts the small blind and acts first
before the flop; the other seat acts first on later streets.  Raises are capped at the effective
stack, so no uncallable chips or side pots are introduced.

Cards are ``suit * 13 + rank - 2`` with suits clubs, diamonds, hearts, spades.
A supplied deck must be a permutation of range(52), consumed from the front.
Deal one card to the big blind, one to the button, repeat, then burn before
each board street.  ``observe`` never exposes the deck or an opponent's cards.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random
from typing import Iterable


ACTION_NAMES = ("fold", "check/call", "min-raise", "half-pot", "pot", "all-in")


def card_text(card: int) -> str:
    if not isinstance(card, int) or not 0 <= card < 52:
        raise ValueError("card must be an integer from 0 to 51")
    return "23456789TJQKA"[card % 13] + "cdhs"[card // 13]


def cards_text(cards: Iterable[int]) -> str:
    return " ".join(card_text(card) for card in cards)


def _rank_five(cards: tuple[int, ...]) -> tuple[int, ...]:
    ranks = sorted((c % 13 + 2 for c in cards), reverse=True)
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({c // 13 for c in cards}) == 1
    unique = sorted(counts, reverse=True)
    if 14 in counts:
        unique.append(1)
    straight = 0
    for index in range(len(unique) - 4):
        if unique[index] - unique[index + 4] == 4:
            straight = unique[index]
            break
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if counts[r] == 1), reverse=True))
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        return (2, *pairs, next(r for r in ranks if counts[r] == 1))
    if pairs:
        return (1, pairs[0], *sorted((r for r in ranks if counts[r] == 1), reverse=True))
    return (0, *ranks)


def rank_cards(cards: Iterable[int]) -> tuple[int, ...]:
    """Rank the best five-card hand among 5--7 cards; larger tuples win."""
    hand = tuple(cards)
    if not 5 <= len(hand) <= 7:
        raise ValueError("rank_cards needs five to seven cards")
    if len(set(hand)) != len(hand) or any(not isinstance(c, int) or not 0 <= c < 52 for c in hand):
        raise ValueError("cards must be distinct integers from 0 to 51")
    return max(_rank_five(five) for five in combinations(hand, 5))


class Holdem:
    """One heads-up hand. Rewards are final net winnings measured in big blinds.

    ``starting_stacks`` optionally supplies each seat's chips before blinds.
    ``stack_bb`` remains the nominal feature scale and the equal-stack default.
    A seat with no chips cannot start another hand.
    """

    big_blind = 2

    def __init__(self, seed: int = 0, stack_bb: int = 100, button: int = 0,
                 deck: list[int] | None = None, *,
                 starting_stacks: list[int] | None = None):
        if isinstance(stack_bb, bool) or not isinstance(stack_bb, int) or stack_bb < 1:
            raise ValueError("stack_bb must be a positive integer")
        if button not in (0, 1):
            raise ValueError("button must be seat 0 or 1")
        # Keep this nominal scale stable for existing observation features.
        self.starting_stack = stack_bb * self.big_blind
        if starting_stacks is None:
            self.starting_stacks = [self.starting_stack, self.starting_stack]
        else:
            if (not isinstance(starting_stacks, list) or len(starting_stacks) != 2
                    or any(isinstance(chips, bool) or not isinstance(chips, int)
                           or chips < 1 for chips in starting_stacks)):
                raise ValueError("starting_stacks must contain two positive integer chip counts")
            self.starting_stacks = list(starting_stacks)
        self.button = button
        self._deck = list(range(52)) if deck is None else list(deck)
        if len(self._deck) != 52 or set(self._deck) != set(range(52)):
            raise ValueError("deck must be a permutation of all 52 cards")
        if deck is None:
            random.Random(seed).shuffle(self._deck)
        self._next_card = 0
        self.hole: list[list[int]] = [[], []]
        for _ in range(2):
            for seat in (1 - button, button):
                self.hole[seat].append(self._draw())
        self.board: list[int] = []
        self.stacks = list(self.starting_stacks)
        self.street_bets = [0, 0]
        self.contributions = [0, 0]
        self.pot = 0
        self.street = 0
        self.player = button
        self.done = False
        self.folded: int | None = None
        self.rewards = [0.0, 0.0]
        self.history: list[dict] = []
        self.last_raise = self.big_blind
        self._acted_since_full_raise: set[int] = set()
        self._pay(button, min(self.stacks[button], 1))
        self._pay(1 - button, min(self.stacks[1 - button], self.big_blind))
        self._pending = {seat for seat in (0, 1) if self.stacks[seat] > 0}
        self._advance_if_ready()

    @property
    def to_call(self) -> int:
        return max(self.street_bets) - self.street_bets[self.player]

    def _draw(self) -> int:
        card = self._deck[self._next_card]
        self._next_card += 1
        return card

    def _pay(self, seat: int, amount: int) -> None:
        if not 0 <= amount <= self.stacks[seat]:
            raise ValueError("invalid chip payment")
        self.stacks[seat] -= amount
        self.street_bets[seat] += amount
        self.contributions[seat] += amount
        self.pot += amount

    def legal_actions(self) -> dict[int, int]:
        """Action ID -> target contribution on this street (fold uses -1).

        Fractional pot raises use the pot after a call.  Duplicate raise sizes
        are removed, preserving all-in first, then the smaller action ID.
        """
        if self.done:
            return {}
        seat = self.player
        other = 1 - seat
        current = max(self.street_bets)
        owed = current - self.street_bets[seat]
        legal: dict[int, int] = {1: min(current, self.street_bets[seat] + self.stacks[seat])}
        if owed:
            legal[0] = -1
        cap = min(self.street_bets[seat] + self.stacks[seat],
                  self.street_bets[other] + self.stacks[other])
        if (cap > current and self.stacks[other] > 0
                and seat not in self._acted_since_full_raise):
            minimum = current + self.last_raise
            candidates = {
                2: minimum,
                3: current + max(self.last_raise, (self.pot + owed) // 2),
                4: current + max(self.last_raise, self.pot + owed),
                5: cap,
            }
            legal[5] = cap
            seen = {cap}
            for action in (2, 3, 4):
                target = candidates[action]
                if minimum <= target <= cap and target not in seen:
                    legal[action] = target
                    seen.add(target)
        return dict(sorted(legal.items()))

    def observe(self, player: int | None = None) -> dict:
        """Return an independent snapshot, with only the requested seat's cards.

        ``player`` is the current actor and ``seat`` is the observing player.
        An observer who is not acting receives no legal actions.
        """
        seat = self.player if player is None else player
        if seat not in (0, 1):
            raise ValueError("player must be seat 0 or 1")
        legal = self.legal_actions() if seat == self.player else {}
        return {
            "player": self.player, "seat": seat, "button": self.button,
            "street": self.street, "hole": list(self.hole[seat]),
            "board": list(self.board), "stacks": list(self.stacks),
            "street_bets": list(self.street_bets),
            "contributions": list(self.contributions), "pot": self.pot,
            "history": [dict(item) for item in self.history],
            "rewards": list(self.rewards), "done": self.done, "folded": self.folded,
            "starting_stack": self.starting_stack, "big_blind": self.big_blind,
            "starting_stacks": list(self.starting_stacks),
            "legal_mask": [action in legal for action in range(6)],
            "legal_targets": legal,
            "to_call": max(self.street_bets) - self.street_bets[seat],
            "last_raise": self.last_raise,
        }

    def step(self, action: int) -> None:
        if self.done:
            raise ValueError("hand is already finished")
        legal = self.legal_actions()
        if isinstance(action, bool) or not isinstance(action, int) or action not in legal:
            raise ValueError(f"illegal action {action!r}; allowed: {legal}")
        seat = self.player
        target = legal[action]
        current = max(self.street_bets)
        paid = 0 if action == 0 else target - self.street_bets[seat]
        self.history.append({"player": seat, "street": self.street,
                             "action": action, "paid": paid, "to_total": target})
        if action == 0:
            self.folded = seat
            self._finish(1 - seat)
            return
        self._pay(seat, paid)
        self._pending.discard(seat)
        if target > current:
            increment = target - current
            if increment >= self.last_raise:
                self.last_raise = increment
                self._acted_since_full_raise = {seat}
            else:
                self._acted_since_full_raise.add(seat)
            self._pending = {1 - seat} if self.stacks[1 - seat] > 0 else set()
        else:
            self._acted_since_full_raise.add(seat)
        self.player = 1 - seat
        self._advance_if_ready()

    def _advance_if_ready(self) -> None:
        self._pending = {seat for seat in self._pending if self.stacks[seat] > 0}
        if 0 in self.stacks:
            # A live player still gets the decision whether to call an all-in.
            needs_response = {seat for seat in self._pending
                              if self.street_bets[seat] < max(self.street_bets)}
            if needs_response:
                self._pending = needs_response
                self.player = next(iter(needs_response))
                return
            self._refund_unmatched()
            while self.street < 3:
                self._deal_next_street()
            self._showdown()
            return
        if self._pending:
            if self.player not in self._pending:
                self.player = next(iter(self._pending))
            return
        if self.street == 3:
            self._showdown()
            return
        self._deal_next_street()
        self._pending = {0, 1}
        self.player = 1 - self.button

    def _deal_next_street(self) -> None:
        self.street += 1
        self.street_bets = [0, 0]
        self.last_raise = self.big_blind
        self._acted_since_full_raise = set()
        self._draw()  # Burn card.
        self.board.extend(self._draw() for _ in range(3 if self.street == 1 else 1))

    def _refund_unmatched(self) -> None:
        difference = self.contributions[0] - self.contributions[1]
        if difference:
            seat = 0 if difference > 0 else 1
            refund = abs(difference)
            self.contributions[seat] -= refund
            self.street_bets[seat] -= refund
            self.stacks[seat] += refund
            self.pot -= refund

    def _showdown(self) -> None:
        first = rank_cards(self.hole[0] + self.board)
        second = rank_cards(self.hole[1] + self.board)
        self._finish(None if first == second else (0 if first > second else 1))

    def _finish(self, winner: int | None) -> None:
        self._refund_unmatched()
        if winner is None:
            share, odd = divmod(self.pot, 2)
            self.stacks[0] += share
            self.stacks[1] += share
            # First seat to the left of the button receives an odd chip.
            self.stacks[1 - self.button] += odd
        else:
            self.stacks[winner] += self.pot
        self.pot = 0
        self.rewards = [(stack - start) / self.big_blind
                        for stack, start in zip(self.stacks, self.starting_stacks)]
        self.done = True
        self._pending.clear()
