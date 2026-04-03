"""RL environment wrapper for 31 with a win/loss-focused interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from computer import ComputerStrategy
from game import Game
from rules import Card, score_hand

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["hearts", "diamonds", "clubs", "spades"]
CARD_TO_INDEX = {
    (rank, suit): suit_index * len(RANKS) + rank_index
    for suit_index, suit in enumerate(SUITS)
    for rank_index, rank in enumerate(RANKS)
}

# Action mapping:
# 0: knock
# 1-4: draw_discard + discard index [0..3]
# 5-8: draw_deck + discard index [0..3]
NUM_ACTIONS = 9


class RLControlledStrategy(ComputerStrategy):
    """Placeholder strategy object used as the learning agent player."""

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        raise RuntimeError(
            "RLControlledStrategy is controlled externally by ThirtyOneEnv."
        )

    def choose_discard_index(self, hand: List[Card]) -> int:
        raise RuntimeError(
            "RLControlledStrategy is controlled externally by ThirtyOneEnv."
        )


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


class ThirtyOneEnv:
    """Two-player environment: RL agent versus one baseline strategy."""

    def __init__(
        self,
        opponent_factory: Callable[[], ComputerStrategy],
        seed: Optional[int] = None,
        max_turns_per_game: int = 200,
    ) -> None:
        self.opponent_factory = opponent_factory
        self.rng = random.Random(seed)
        self.max_turns_per_game = max_turns_per_game

        self.game: Optional[Game] = None
        self.agent: Optional[RLControlledStrategy] = None
        self.opponent: Optional[ComputerStrategy] = None
        self.current_index: int = 0
        self.done: bool = False
        self.turn_count: int = 0
        self.truncated_by_limit: bool = False
        self.seen_cards: set[Card] = set()

    @property
    def obs_dim(self) -> int:
        # hand one-hot (52) + top discard one-hot (52) + seen cards binary (52) + scalars (9)
        return 165

    def reset(
        self, start_player: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Reset the environment and return initial observation/info."""
        self.agent = RLControlledStrategy()
        self.opponent = self.opponent_factory()
        self.game = Game(strategies=[self.agent, self.opponent], debug=False, rng=self.rng)
        self.game._deal_initial()

        self.seen_cards.clear()
        # Track all initial cards as seen
        for strategy in self.game.strategies:
            for card in self.game.hands[strategy]:
                self.seen_cards.add(card)
        if self.game.discard_pile:
            self.seen_cards.add(self.game.discard_pile[0])

        self.current_index = (
            self.rng.randint(0, 1) if start_player is None else int(start_player)
        )
        self.done = False
        self.turn_count = 0
        self.truncated_by_limit = False

        if self.current_index != 0:
            self._play_until_agent_turn_or_done()

        obs = self._build_observation()
        info = {"action_mask": self._action_mask()}
        return obs, info

    def step(self, action: int) -> StepResult:
        """Advance one RL decision step."""
        if self.done:
            raise RuntimeError(
                "Cannot call step() after termination. Call reset() first."
            )
        if self.current_index != 0:
            raise RuntimeError("step() called when it is not the RL agent turn.")

        reward = 0.0
        invalid_penalty = 0.0

        mask = self._action_mask()
        if action < 0 or action >= NUM_ACTIONS or not mask[action]:
            invalid_penalty = -0.05
            legal_actions = np.flatnonzero(mask)
            action = int(legal_actions[0]) if len(legal_actions) > 0 else 0

        acting_strategy = self.game.strategies[self.current_index]
        self._execute_agent_action(acting_strategy, action)
        self._post_turn_update(acting_strategy)

        if not self.done:
            self._play_until_agent_turn_or_done()

        if self.done:
            reward += self._terminal_reward()

        reward += invalid_penalty
        obs = self._build_observation()
        info = {
            "action_mask": self._action_mask(),
            "invalid_action": invalid_penalty < 0,
        }
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=self.done,
            truncated=self.truncated_by_limit,
            info=info,
        )

    def _play_until_agent_turn_or_done(self) -> None:
        while not self.done and self.current_index != 0:
            strategy = self.game.strategies[self.current_index]
            self._execute_strategy_turn(strategy)
            self._post_turn_update(strategy)

    def _post_turn_update(self, strategy: ComputerStrategy) -> None:
        self.turn_count += 1

        if self.game.knock_remaining is not None and strategy != self.game.knocked_by:
            self.game.knock_remaining -= 1

        if self.game.knock_remaining is not None and self.game.knock_remaining <= 0:
            self.done = True
            return

        if self.turn_count >= self.max_turns_per_game:
            self.done = True
            self.truncated_by_limit = True
            return

        self.current_index = (self.current_index + 1) % len(self.game.strategies)

    def _execute_strategy_turn(self, strategy: ComputerStrategy) -> None:
        hand = self.game.hands[strategy]
        top_discard = self.game.discard_pile[-1] if self.game.discard_pile else None

        action = strategy.choose_action(
            hand,
            top_discard,
            knock_allowed=self.game.knocked_by is None,
        )

        if action == "knock":
            self.game.knocked_by = strategy
            self.game.knock_remaining = len(self.game.strategies) - 1
            return

        card = None
        if action == "draw_discard" and top_discard:
            card = self.game.discard_pile.pop()
        else:
            card = self.game.deck.draw_card()
            if card:
                strategy.on_card_drawn(card)
                self.seen_cards.add(card)
            if not card and top_discard:
                card = self.game.discard_pile.pop()

        if not card:
            return

        hand.append(card)
        discard_index = strategy.choose_discard_index(hand)
        discarded = self.game._discard_card(strategy, discard_index)
        self.seen_cards.add(discarded)

    def _execute_agent_action(self, strategy: ComputerStrategy, action: int) -> None:
        hand = self.game.hands[strategy]
        top_discard = self.game.discard_pile[-1] if self.game.discard_pile else None

        if action == 0:
            self.game.knocked_by = strategy
            self.game.knock_remaining = len(self.game.strategies) - 1
            return

        draw_discard = 1 <= action <= 4
        discard_index = (action - 1) if draw_discard else (action - 5)

        card = None
        if draw_discard and top_discard:
            card = self.game.discard_pile.pop()
        else:
            card = self.game.deck.draw_card()
            if card:
                strategy.on_card_drawn(card)
                self.seen_cards.add(card)
            if not card and top_discard:
                card = self.game.discard_pile.pop()

        if not card:
            return

        hand.append(card)
        discard_index = max(0, min(discard_index, len(hand) - 1))
        discarded = self.game._discard_card(strategy, discard_index)
        self.seen_cards.add(discarded)

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self.done or self.current_index != 0:
            return mask

        hand = self.game.hands[self.agent]
        top_discard_exists = bool(self.game.discard_pile)
        knock_allowed = self.game.knocked_by is None

        if knock_allowed:
            mask[0] = True

        # draw_discard + discard_idx
        if top_discard_exists:
            mask[1:5] = True

        # draw_deck + discard_idx (deck draw may fallback to discard if deck is empty)
        if len(hand) == 3 and (len(self.game.deck) > 0 or top_discard_exists):
            mask[5:9] = True

        return mask

    def _build_observation(self) -> np.ndarray:
        if self.game is None or self.agent is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        hand = self.game.hands[self.agent]
        top_discard = self.game.discard_pile[-1] if self.game.discard_pile else None

        hand_vec = np.zeros(52, dtype=np.float32)
        for card in hand:
            hand_vec[CARD_TO_INDEX[card]] = 1.0

        top_vec = np.zeros(52, dtype=np.float32)
        if top_discard is not None:
            top_vec[CARD_TO_INDEX[top_discard]] = 1.0

        # Seen cards: binary indicator for each card we've observed
        seen_vec = np.zeros(52, dtype=np.float32)
        for card in self.seen_cards:
            seen_vec[CARD_TO_INDEX[card]] = 1.0

        max_knock_count = max(1, len(self.game.strategies) - 1)
        knock_remaining = (
            self.game.knock_remaining if self.game.knock_remaining is not None else 0
        )

        # Game state scalars
        context = np.array(
            [
                float(score_hand(hand) / 31.0),
                float(len(self.game.deck) / 52.0),
                float(len(self.game.discard_pile) / 52.0),
                float(self.current_index == 0),
                float(self.game.knocked_by is not None),
                float(self.game.knocked_by == self.agent),
                float(knock_remaining / max_knock_count),
                float(len(self.seen_cards) / 52.0),  # fraction of cards we've seen
                float(len(self.game.discard_pile) > 0),  # discard pile exists
            ],
            dtype=np.float32,
        )

        return np.concatenate([hand_vec, top_vec, seen_vec, context], dtype=np.float32)

    def _terminal_reward(self) -> float:
        scores: Dict[ComputerStrategy, float] = {}
        for strategy in self.game.strategies:
            scores[strategy] = score_hand(self.game.hands[strategy])

        best_score = max(scores.values())
        winners = [s for s, val in scores.items() if val == best_score]

        if len(winners) > 1:
            return 0.0
        return 1.0 if winners[0] == self.agent else -1.0
