"""
Standard playing deck implementation for 31 (Scat).

Provides a shuffled 52-card deck that can be drawn from one card at a time.
"""

import random
from typing import List, Optional, Tuple


class Deck:
    """
    Represents a standard 52-card playing deck (shuffled on creation).

    Cards can be drawn sequentially from the deck using draw_card().
    Once a card is drawn, it is removed and cannot be drawn again.

    Attributes:
        _RANKS (List[str]): Standard poker ranks: A, 2-10, J, Q, K.
        _SUITS (List[str]): Standard suits: hearts, diamonds, clubs, spades.
        _cards (List[Tuple[str, str]]): Internal list of remaining cards in the deck.
    """

    _RANKS: List[str] = [
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
    _SUITS: List[str] = ["hearts", "diamonds", "clubs", "spades"]

    def __init__(self) -> None:
        """
        Initialize a standard 52-card deck and shuffle it.

        Creates an internal list of all 52 cards (13 ranks × 4 suits) and
        shuffles them using random.shuffle(). The deck is ready to draw from
        immediately after construction.

        Example:
            >>> deck = Deck()
            >>> len(deck)
            52
        """
        self._cards: List[Tuple[str, str]] = [
            (rank, suit) for suit in self._SUITS for rank in self._RANKS
        ]
        random.shuffle(self._cards)

    def draw_card(self) -> Optional[Tuple[str, str]]:
        """
        Remove and return the top card from the deck.

        Cards are drawn in the order determined by the initial shuffle.
        Once drawn, a card is no longer in the deck.

        Returns:
            Optional[Tuple[str, str]]: A card (rank, suit) if the deck is not empty,
                or None if all cards have been drawn.

        Example:
            >>> deck = Deck()
            >>> card = deck.draw_card()
            >>> isinstance(card, tuple) and len(card) == 2
            True
            >>> len(deck)
            51
        """
        if not self._cards:
            return None
        return self._cards.pop()

    def __len__(self) -> int:
        """
        Return the number of cards remaining in the deck.

        Returns:
            int: The count of cards not yet drawn. Ranges from 52 (full deck)
                to 0 (empty deck).

        Example:
            >>> deck = Deck()
            >>> len(deck)
            52
            >>> deck.draw_card()
            >>> len(deck)
            51
        """
        return len(self._cards)
