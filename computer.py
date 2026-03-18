from typing import Dict, List, Optional, Set, Tuple
import sys
from secrets import SystemRandom

from rules import Card, score_hand, card_label


class ComputerStrategy:
    """Interface for computer turn decisions."""

    ALL_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    ALL_SUITS = ["hearts", "diamonds", "clubs", "spades"]

    @property
    def name(self) -> str:
        """Return the strategy name (defaults to class name)."""
        return self.__class__.__name__

    def observe_card(self, card: Card) -> None:
        """Called when a card is revealed (from initial hand, discard pile, etc.)."""
        pass

    def on_initial_cards(self, hand: List[Card], starter: Card) -> None:
        """Called when initial cards are dealt."""
        for card in hand:
            self.observe_card(card)
        self.observe_card(starter)

    def on_card_drawn(self, card: Card) -> None:
        """Called when a card is drawn from deck."""
        self.observe_card(card)

    def on_card_discarded(self, card: Card) -> None:
        """Called when a card is discarded."""
        self.observe_card(card)

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Decide action for this turn.
        Return one of: "knock", "draw_discard", "draw_deck".
        """
        raise NotImplementedError

    def choose_discard_index(self, hand: List[Card]) -> int:
        """Return the index to discard to maximize the resulting hand score."""
        raise NotImplementedError

    def _best_score_with_added_card(self, hand: List[Card], card: Card) -> float:
        """Return the best score achievable by adding card and discarding optimally."""
        temp_hand = hand + [card]
        best_score = -1.0
        for idx in range(len(temp_hand)):
            candidate = temp_hand[:idx] + temp_hand[idx + 1 :]
            best_score = max(best_score, score_hand(candidate))
        return best_score

    def _get_unseen_cards(self, seen_cards: Set[Card]) -> List[Card]:
        """Return all cards that haven't been seen yet."""
        all_cards = [(rank, suit) for suit in self.ALL_SUITS for rank in self.ALL_RANKS]
        return [card for card in all_cards if card not in seen_cards]

    def _calculate_expected_value(self, hand: List[Card]) -> float:
        """Calculate average best score across all unseen cards."""
        unseen_cards = self._get_unseen_cards(self.seen_cards)
        if not unseen_cards:
            sys.exit("No unseen cards to calculate expected value.")
        total = sum(self._best_score_with_added_card(hand, card) for card in unseen_cards)
        return total / len(unseen_cards)

class RandomStrategy(ComputerStrategy):
    """Strategy that randomly chooses between knock, draw deck, and draw discard equally."""

    def __init__(self) -> None:
        self.rng = SystemRandom()

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        # Equally random between all three actions
        actions = ["knock", "draw_deck", "draw_discard"]
        return self.rng.choice(actions)

    def choose_discard_index(self, hand: List[Card]) -> int:
        # Randomly pick a card to discard
        return self.rng.randint(0, len(hand) - 1)


class RandomStrategyWithKnockScore(ComputerStrategy):
    """Strategy that makes random decisions (with a knock threshold)."""

    KNOCK_SCORE = 27

    def __init__(self) -> None:
        self.rng = SystemRandom()

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        if knock_allowed and score_hand(hand) >= self.KNOCK_SCORE:
            return "knock"

        # Randomly choose between drawing from deck or picking up discard
        if self.rng.choice([True, False]):
            return "draw_discard"
        return "draw_deck"

    def choose_discard_index(self, hand: List[Card]) -> int:
        # Randomly pick a card to discard
        return self.rng.randint(0, len(hand) - 1)

class DiscardIncreaseStrategy(ComputerStrategy):
    """If drawing the discard improves the hand, take it; otherwise draw from deck."""

    KNOCK_SCORE = 27

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        if knock_allowed and score_hand(hand) >= self.KNOCK_SCORE:
            return "knock"

        best_with_discard = self._best_score_with_added_card(hand, top_discard)
        if best_with_discard > score_hand(hand):
            return "draw_discard"

        return "draw_deck"

    def choose_discard_index(self, hand: List[Card]) -> int:
        best_score = -1.0
        best_idx = 0
        for idx in range(len(hand)):
            candidate = hand[:idx] + hand[idx + 1 :]
            score = score_hand(candidate)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

class ConservativeExpectedValueStrategy(ComputerStrategy):
    """
    Risk-averse EV strategy:
    - Prefers guaranteed discard improvements unless the deck's risk-adjusted EV
      beats it by a margin.
    - Risk-adjusted EV = mean(best score with each unseen card) - risk_aversion * stddev.
    """

    KNOCK_SCORE = 27
    ALL_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    ALL_SUITS = ["hearts", "diamonds", "clubs", "spades"]

    def __init__(self) -> None:
        self.seen_cards: Set[Card] = set()
        # Hardcoded margin: deck must beat discard by this amount
        self.margin = 1.5

    def observe_card(self, card: Card) -> None:
        self.seen_cards.add(card)

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        current_score = score_hand(hand)
        if knock_allowed and current_score >= self.KNOCK_SCORE:
            return "knock"

        # Always check if discard improves hand vs deck expected value
        discard_value = self._best_score_with_added_card(hand, top_discard)
        discard_improvement = discard_value - current_score
        
        # If discard improves hand, check against deck EV with margin
        deck_ev = self._calculate_expected_value(hand)
        deck_improvement = deck_ev - current_score
        
        if debug:
            print(f"Observed Cards: {", ".join(card_label(card) for card in sorted(self.seen_cards))}")
            print(f"Discard: +{discard_improvement:.2f}")
            print(f"Deck: +{deck_improvement:.2f} (margin={self.margin})")

        # If discard adds 0 (no improvement), always draw from deck
        if discard_improvement <= 0:
            return "draw_deck"
        
        if deck_ev >= discard_value + self.margin:
            return "draw_deck"
        return "draw_discard"

    def choose_discard_index(self, hand: List[Card]) -> int:
        best_score = -1.0
        best_idx = 0
        for idx in range(len(hand)):
            candidate = hand[:idx] + hand[idx + 1 :]
            score = score_hand(candidate)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

class CurrentTurnExpectedValueStrategy(ComputerStrategy):
    """
    Strategy that tracks seen cards and compares expected value of drawing 
    from deck vs picking up the known top discard.
    """

    KNOCK_SCORE = 27
    ALL_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    ALL_SUITS = ["hearts", "diamonds", "clubs", "spades"]

    def __init__(self) -> None:
        self.seen_cards: Set[Card] = set()

    def observe_card(self, card: Card) -> None:
        """Mark a card as seen."""
        self.seen_cards.add(card)

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        if knock_allowed and score_hand(hand) >= self.KNOCK_SCORE:
            return "knock"

        # Calculate expected value of drawing from deck
        unseen_cards = self._get_unseen_cards(self.seen_cards)
        if not unseen_cards:
            # If no unseen cards, must take discard if available
            return "draw_discard" if top_discard else "draw_deck"

        expected_deck_value = self._calculate_expected_value(hand)

        # Calculate value of taking the known top discard
        discard_value = self._best_score_with_added_card(hand, top_discard)
        # Choose whichever gives better expected outcome
        if discard_value > expected_deck_value:
            return "draw_discard"

        return "draw_deck"

    def choose_discard_index(self, hand: List[Card]) -> int:
        best_score = -1.0
        best_idx = 0
        for idx in range(len(hand)):
            candidate = hand[:idx] + hand[idx + 1 :]
            score = score_hand(candidate)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx
