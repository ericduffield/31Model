import random
from typing import List, Optional, Tuple


class Deck:
    """Represents a standard 52-card deck that is shuffled on creation."""

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
        # Build the full deck and shuffle once per game.
        self._cards: List[Tuple[str, str]] = [
            (rank, suit) for suit in self._SUITS for rank in self._RANKS
        ]
        random.shuffle(self._cards)

    def draw_card(self) -> Optional[Tuple[str, str]]:
        """Remove and return the top card; returns None if the deck is empty."""
        if not self._cards:
            return None
        return self._cards.pop()

    def __len__(self) -> int:
        return len(self._cards)
