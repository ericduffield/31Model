"""
Computer strategies for the card game 31 (Scat).

Provides various AI strategies for deciding actions and discards in 31.
Strategies range from purely random to sophisticated expected-value-based
decision-making. All strategies inherit from ComputerStrategy base class.
"""

from typing import List, Optional, Set
import sys
import random

from rules import Card, score_hand, card_label


class ComputerStrategy:
    """
    Abstract base class for computer player strategies in 31.

    Subclasses must implement choose_action() and choose_discard_index()
    to define their playing strategy. Strategies can observe card reveals
    to inform future decisions (e.g., for tracking unseen cards).

    Attributes:
        ALL_RANKS (List[str]): Standard poker ranks.
        ALL_SUITS (List[str]): Standard card suits.
        name (str): Strategy name (defaults to class name).
    """

    ALL_RANKS = [
        "A",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "J",
        "Q",
        "K",
    ]
    ALL_SUITS = ["hearts", "diamonds", "clubs", "spades"]

    @property
    def name(self) -> str:
        """
        Return the strategy name.

        Returns:
            str: The name of this strategy (defaults to the class name).

        Example:
            >>> RandomStrategy().name
            'RandomStrategy'
        """
        return self.__class__.__name__

    def observe_card(self, card: Card) -> None:
        """
        Record that a card has been revealed during play.

        Called when any card becomes visible (initial hand, discard pile, revealed).
        Subclasses can override to track seen cards for probability calculations.

        Args:
            card (Card): A card (rank, suit) that has been revealed.

        Example:
            >>> strategy = ConservativeExpectedValueStrategy()
            >>> strategy.observe_card(("A", "hearts"))
        """
        pass

    def on_initial_cards(self, hand: List[Card], starter: Card) -> None:
        """
        Record the initial cards dealt at the start of a round.

        Notifies the strategy of the initial hand and the starting card in
        the discard pile. Default implementation calls observe_card for each.

        Args:
            hand (List[Card]): The initial 3 cards dealt to this player.
            starter (Card): The first card placed in the discard pile.

        Example:
            >>> strategy = RandomStrategy()
            >>> strategy.on_initial_cards([("A", "hearts"), ("K", "spades"), ("7", "clubs")], ("3", "diamonds"))
        """
        for card in hand:
            self.observe_card(card)
        self.observe_card(starter)

    def on_card_drawn(self, card: Card) -> None:
        """
        Record that a card has been drawn from the deck.

        Args:
            card (Card): The card drawn from the deck.

        Example:
            >>> strategy = ConservativeExpectedValueStrategy()
            >>> strategy.on_card_drawn(("10", "hearts"))
        """
        self.observe_card(card)

    def on_card_discarded(self, card: Card) -> None:
        """
        Record that a card has been discarded.

        Args:
            card (Card): The card that was discarded.

        Example:
            >>> strategy = ConservativeExpectedValueStrategy()
            >>> strategy.on_card_discarded(("2", "diamonds"))
        """
        self.observe_card(card)

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Decide what action to take on this turn.

        Must be implemented by subclasses. Decision is based on the current hand,
        the visible discard card, and whether knocking is available.

        Args:
            hand (List[Card]): The player's current 3-card hand.
            top_discard (Optional[Card]): The card on top of the discard pile,
                or None if the pile is empty.
            knock_allowed (bool): True if the player can knock (no one has knocked yet).
            debug (bool): If True, print debug information. Default False.

        Returns:
            str: One of "knock", "draw_discard", or "draw_deck".
                - "knock": End the round (hand must score >= KNOCK_SCORE).
                - "draw_discard": Take the top discard card.
                - "draw_deck": Draw a card from the deck.

        Raises:
            NotImplementedError: If not overridden by a subclass.

        Example:
            >>> # (Implemented in subclasses)
            >>> strategy = RandomStrategy()
            >>> action = strategy.choose_action([("7", "h"), ("8", "h"), ("9", "h")], ("5", "d"), True)
            >>> action in ["knock", "draw_discard", "draw_deck"]
            True
        """
        raise NotImplementedError

    def choose_discard_index(self, hand: List[Card]) -> int:
        """
        Select which card to discard after drawing.

        Must be implemented by subclasses. The player has 4 cards after drawing,
        and must discard one to return to 3 cards.

        Args:
            hand (List[Card]): The current 4-card hand (after a draw).

        Returns:
            int: The index (0-3) of the card in hand to discard.

        Raises:
            NotImplementedError: If not overridden by a subclass.

        Example:
            >>> # (Implemented in subclasses)
            >>> strategy = RandomStrategy()
            >>> idx = strategy.choose_discard_index([("2", "h"), ("3", "d"), ("4", "c"), ("5", "s")])
            >>> 0 <= idx < 4
            True
        """
        raise NotImplementedError

    def _best_score_with_added_card(self, hand: List[Card], card: Card) -> float:
        """
        Compute the best hand score if we add a card and discard optimally.

        Helper method for strategies that evaluate potential card additions.
        Tries removing each card from the 4-card temporary hand and returns
        the maximum score achievable.

        Args:
            hand (List[Card]): The current 3-card hand.
            card (Card): A card being considered for addition.

        Returns:
            float: The best score achievable after adding card and discarding one.

        Example:
            >>> strategy = DiscardIncreaseStrategy()
            >>> best = strategy._best_score_with_added_card([("A", "h"), ("K", "h"), ("Q", "h")], ("J", "h"))
            >>> best  # One of the 4-card hands will score 31
            31.0
        """
        temp_hand = hand + [card]
        best_score = -1.0
        for idx in range(len(temp_hand)):
            candidate = temp_hand[:idx] + temp_hand[idx + 1 :]
            best_score = max(best_score, score_hand(candidate))
        return best_score

    def _get_unseen_cards(self, seen_cards: Set[Card]) -> List[Card]:
        """
        Return all cards that haven't been observed yet.

        Helper for expected-value strategies. Computes the deck of unseen cards
        by removing all seen cards from the full 52-card deck.

        Args:
            seen_cards (Set[Card]): Set of cards that have been observed.

        Returns:
            List[Card]: All 52-card deck cards that are not in seen_cards.

        Example:
            >>> strategy = CurrentTurnExpectedValueStrategy()
            >>> strategy.see cards = {("A", "h"), ("K", "s")}
            >>> unseen = strategy._get_unseen_cards(strategy.seen_cards)
            >>> len(unseen)
            50
        """
        all_cards = [(rank, suit) for suit in self.ALL_SUITS for rank in self.ALL_RANKS]
        return [card for card in all_cards if card not in seen_cards]

    def _calculate_expected_value(self, hand: List[Card]) -> float:
        """
        Compute expected value of drawing a random unseen card.

        Averages the best hand score achievable across all possible unseen cards.
        Used by expected-value strategies to compare drawing from deck vs. discard.

        Args:
            hand (List[Card]): The current 3-card hand.

        Returns:
            float: The average of best-achievable scores for each unseen card.

        Raises:
            SystemExit: If no unseen cards remain (should not happen in practice).

        Example:
            >>> strategy = ConservativeExpectedValueStrategy()
            >>> expected = strategy._calculate_expected_value([("A", "h"), ("K", "h"), ("Q", "h")])
            >>> 20.0 < expected < 31.0  # Rough range
            True
        """
        unseen_cards = self._get_unseen_cards(self.seen_cards)
        if not unseen_cards:
            sys.exit("No unseen cards to calculate expected value.")
        total = sum(
            self._best_score_with_added_card(hand, card) for card in unseen_cards
        )
        return total / len(unseen_cards)


class RandomStrategy(ComputerStrategy):
    """
    Baseline strategy that makes purely random decisions.

    Equally likely to knock (if allowed), draw from the discard pile, or draw
    from the deck. Dismisses the card and discard cards are also random. Serves
    as a weak baseline for evaluating more sophisticated strategies.
    """

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Choose an action uniformly at random from all possible actions.

        Ignores the current hand, discard pile, and game state.

        Args:
            hand (List[Card]): Ignored.
            top_discard (Optional[Card]): Ignored.
            knock_allowed (bool): Ignored.
            debug (bool): Ignored.

        Returns:
            str: One of "knock", "draw_deck", "draw_discard" with equal probability.
        """
        actions = ["knock", "draw_deck", "draw_discard"]
        return random.choice(actions)

    def choose_discard_index(self, hand: List[Card]) -> int:
        """
        Choose a card index to discard uniformly at random.

        Args:
            hand (List[Card]): The 4-card hand (ignored).

        Returns:
            int: A random index from 0 to len(hand)-1.
        """
        return random.randint(0, len(hand) - 1)


class RandomStrategyWithKnockScore(ComputerStrategy):
    """
    Simple strategy with a knock threshold but random other decisions.

    Knocks if the hand scores >= 27 points. Otherwise, randomly chooses
    between drawing from the discard pile or the deck. Card discards remain
    random. Serves as a mild baseline that incorporates one heuristic.

    Attributes:
        KNOCK_SCORE (int): Minimum score required to knock (27).
    """

    KNOCK_SCORE = 27

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Knock if hand score >= 27, otherwise randomly choose draw source.

        Args:
            hand (List[Card]): The current 3-card hand.
            top_discard (Optional[Card]): The visible discard (unused).
            knock_allowed (bool): If True and score >= KNOCK_SCORE, may knock.
            debug (bool): Ignored.

        Returns:
            str: "knock" if score >= 27, else "draw_discard" or "draw_deck" with 50% prob.
        """
        if knock_allowed and score_hand(hand) >= self.KNOCK_SCORE:
            return "knock"

        # Randomly choose between drawing from deck or picking up discard
        if random.choice([True, False]):
            return "draw_discard"
        return "draw_deck"

    def choose_discard_index(self, hand: List[Card]) -> int:
        """
        Choose a card to discard uniformly at random.

        Args:
            hand (List[Card]): The 4-card hand (ignored).

        Returns:
            int: A random index from 0 to len(hand)-1.
        """
        return random.randint(0, len(hand) - 1)


class DiscardIncreaseStrategy(ComputerStrategy):
    """
    Greedy strategy based on immediate hand improvement.

    Knocks if score >= 27. Otherwise, draws from the discard pile if and only if
    taking it and discarding optimally improves the current hand score. Otherwise
    draws from the deck. Discards chosen to maximize remaining hand score.

    This strategy does not track seen cards or use probabilities, making it
    computationally cheap and intuitive.

    Attributes:
        KNOCK_SCORE (int): Minimum score required to knock (27).
    """

    KNOCK_SCORE = 27

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Knock if eligible, else take discard if it improves, otherwise draw from deck.

        Args:
            hand (List[Card]): The current 3-card hand.
            top_discard (Optional[Card]): The card on top of the discard pile.
            knock_allowed (bool): If True and score >= 27, may knock.
            debug (bool): Ignored.

        Returns:
            str: "knock" if score >= 27, "draw_discard" if discard improves hand,
                else "draw_deck".

        Example:
            >>> strategy = DiscardIncreaseStrategy()
            >>> # If hand = (A♥, K♥, Q♥) with score 31, and discard = 2♣:
            >>> action = strategy.choose_action([("A","h"), ("K","h"), ("Q","h")], ("2","c"), True)
            >>> action  # Will be "knock" since score is 31 >= 27
            'knock'
        """
        if knock_allowed and score_hand(hand) >= self.KNOCK_SCORE:
            return "knock"

        best_with_discard = self._best_score_with_added_card(hand, top_discard)
        if best_with_discard > score_hand(hand):
            return "draw_discard"

        return "draw_deck"

    def choose_discard_index(self, hand: List[Card]) -> int:
        """
        Choose the card to discard that maximizes the remaining hand score.

        Tries removing each of the 4 cards and picks the one that leaves
        the highest-scoring 3-card hand.

        Args:
            hand (List[Card]): The 4-card hand after drawing.

        Returns:
            int: The index in hand of the card that, when removed, maximizes score.

        Example:
            >>> strategy = DiscardIncreaseStrategy()
            >>> # Hand: 2♣ 3♦ A♥ K♠
            >>> idx = strategy.choose_discard_index([("2","c"), ("3","d"), ("A","h"), ("K","s")])
            >>> idx  # Will be 0, removing the 2♣
            0
        """
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
    Risk-averse expected-value strategy using seen-card tracking.

    Tracks all observed cards and computes expected value of drawing from deck.
    Knocks if eligible. Takes discard if it improves the hand, but only if the
    deck's expected value does not beat the discard by a hard margin (1.5 points).
    This mixing of deterministic improvement and probabilistic betting makes it
    conservative—it prefers sure gains unless the unseen deck is very attractive.

    Attributes:
        KNOCK_SCORE (int): Minimum score to knock (27).
        margin (float): Margin (1.5 points) by which deck EV must beat discard.
        seen_cards (Set[Card]): Tracks all observed cards for EV calculation.
    """

    KNOCK_SCORE = 27
    ALL_RANKS = [
        "A",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "J",
        "Q",
        "K",
    ]
    ALL_SUITS = ["hearts", "diamonds", "clubs", "spades"]

    def __init__(self) -> None:
        """Initialize with empty seen_cards set and margin of 1.5."""
        self.seen_cards: Set[Card] = set()
        self.margin = 1.5

    def observe_card(self, card: Card) -> None:
        """
        Track a newly observed card.

        Args:
            card (Card): The (rank, suit) card to record as seen.
        """
        self.seen_cards.add(card)

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Knock if eligible; else compare discard vs. risk-adjusted deck EV.

        Takes discard only if it improves the hand AND the deck's expected value
        does not exceed discard value + margin. Prefers guaranteed improvements.

        Args:
            hand (List[Card]): The current 3-card hand.
            top_discard (Optional[Card]): The visible discard card.
            knock_allowed (bool): If True and score >= 27, may knock.
            debug (bool): If True, print decision details.

        Returns:
            str: "knock", "draw_discard", or "draw_deck".

        Example:
            >>> strategy = ConservativeExpectedValueStrategy()
            >>> action = strategy.choose_action([("7","h"), ("8","h"), ("9","h")], ("2","c"), False)
            >>> action in ["draw_discard", "draw_deck"]
            True
        """
        current_score = score_hand(hand)
        if knock_allowed and current_score >= self.KNOCK_SCORE:
            return "knock"

        discard_value = self._best_score_with_added_card(hand, top_discard)
        discard_improvement = discard_value - current_score

        deck_ev = self._calculate_expected_value(hand)
        deck_improvement = deck_ev - current_score

        if debug:
            print(f"Observed Cards: {
                    ', '.join(
                        card_label(card) for card in sorted(
                            self.seen_cards))}")
            print(f"Discard: +{discard_improvement:.2f}")
            print(f"Deck: +{deck_improvement:.2f} (margin={self.margin})")

        if discard_improvement <= 0:
            return "draw_deck"

        if deck_ev >= discard_value + self.margin:
            return "draw_deck"
        return "draw_discard"

    def choose_discard_index(self, hand: List[Card]) -> int:
        """
        Choose the card to discard that maximizes the remaining 3-card score.

        Args:
            hand (List[Card]): The 4-card hand.

        Returns:
            int: Index of the card to remove for maximum remaining score.
        """
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
    Pure expected-value strategy using full card-tracking.

    Tracks all observed cards and computes the expected value of each possible
    action. Directly compares the known score of the discard pile card against
    the expected value of drawing from the (partially known) deck. Chooses the
    action with higher expected payoff. More aggressive than ConservativeEV
    because it uses true expected values without safety margins.

    Attributes:
        KNOCK_SCORE (int): Minimum score to knock (27).
        seen_cards (Set[Card]): Tracks all observed cards.
    """

    KNOCK_SCORE = 27
    ALL_RANKS = [
        "A",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "J",
        "Q",
        "K",
    ]
    ALL_SUITS = ["hearts", "diamonds", "clubs", "spades"]

    def __init__(self) -> None:
        """Initialize with empty seen_cards set."""
        self.seen_cards: Set[Card] = set()

    def observe_card(self, card: Card) -> None:
        """
        Track a newly observed card.

        Args:
            card (Card): The (rank, suit) card to record as seen.
        """
        self.seen_cards.add(card)

    def choose_action(
        self,
        hand: List[Card],
        top_discard: Optional[Card],
        knock_allowed: bool,
        debug: bool = False,
    ) -> str:
        """
        Knock if eligible; else choose action with highest expected value.

        Directly compares the deterministic discard value against the expected
        value of drawing from the remaining deck. Picks the action with the
        highest expected final hand score.

        Args:
            hand (List[Card]): The current 3-card hand.
            top_discard (Optional[Card]): The visible discard card.
            knock_allowed (bool): If True and score >= 27, may knock.
            debug (bool): Ignored.

        Returns:
            str: "knock", "draw_discard", or "draw_deck" (whichever maximizes EV).

        Example:
            >>> strategy = CurrentTurnExpectedValueStrategy()
            >>> action = strategy.choose_action([("A","h"), ("K","h"), ("Q","h")], ("2","c"), False)
            >>> action  # Will prefer draw_discard if (2,c) pair improves hand, else draw_deck
            'draw_deck'  # (depends on actual EV calculation)
        """
        if knock_allowed and score_hand(hand) >= self.KNOCK_SCORE:
            return "knock"

        unseen_cards = self._get_unseen_cards(self.seen_cards)
        if not unseen_cards:
            # If all cards are seen, take discard if available
            return "draw_discard" if top_discard else "draw_deck"

        expected_deck_value = self._calculate_expected_value(hand)

        # Calculate value of taking the known top discard
        discard_value = self._best_score_with_added_card(hand, top_discard)

        # Choose whichever gives better expected outcome
        if discard_value > expected_deck_value:
            return "draw_discard"

        return "draw_deck"

    def choose_discard_index(self, hand: List[Card]) -> int:
        """
        Choose the card to discard that maximizes the remaining 3-card score.

        Args:
            hand (List[Card]): The 4-card hand.

        Returns:
            int: Index of the card to remove for maximum remaining score.

        Example:
            >>> strategy = CurrentTurnExpectedValueStrategy()
            >>> idx = strategy.choose_discard_index([("A","h"), ("K","s"), ("Q","c"), ("2","d")])
            >>> idx  # Will be 3, removing the 2♦
            3
        """
        best_score = -1.0
        best_idx = 0
        for idx in range(len(hand)):
            candidate = hand[:idx] + hand[idx + 1 :]
            score = score_hand(candidate)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx
