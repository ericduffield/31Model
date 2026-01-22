from typing import Dict, List, Tuple

Card = Tuple[str, str]
HAND_SIZE = 3

CARD_VALUES: Dict[str, int] = {
    "A": 11,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
}


def card_label(card: Card) -> str:
    rank, suit = card
    suit_symbols = {
        "hearts": "♥",
        "diamonds": "♦",
        "clubs": "♣",
        "spades": "♠",
    }
    symbol = suit_symbols.get(suit, suit)
    return f"{rank}{symbol}"


def score_hand(hand: List[Card]) -> float:
    """Return the standard 31/Scat score for a 3-card hand."""
    if len(hand) == HAND_SIZE:
        ranks = {rank for rank, _ in hand}
        if len(ranks) == 1:
            return 30.5  # Three-of-a-kind special case.

    per_suit: Dict[str, int] = {}
    for rank, suit in hand:
        per_suit[suit] = per_suit.get(suit, 0) + CARD_VALUES.get(rank, 0)

    return float(max(per_suit.values(), default=0))
