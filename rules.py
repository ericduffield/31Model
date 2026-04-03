"""
Rules and constants for the card game 31 (Scat).

Defines card representation, scoring logic, and game constants.
"""

from typing import Dict, List, Tuple

# Type alias: a card is represented as a tuple of (rank, suit).
Card = Tuple[str, str]

# Standard hand size for 31: each player holds 3 cards.
HAND_SIZE = 3

# Point values for each rank. Face cards (10, J, Q, K) are worth 10 points.
# Aces are worth 11. Number cards are worth face value.
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
    """
    Convert a card to a human-readable string with Unicode suit symbols.

    Args:
        card (Card): A card represented as (rank, suit).
            rank is a string like "A", "2", ..., "K".
            suit is one of "hearts", "diamonds", "clubs", "spades".

    Returns:
        str: A formatted string like "A♥" or "10♣".

    Example:
        >>> card_label(("A", "hearts"))
        'A♥'
        >>> card_label(("K", "spades"))
        'K♠'
    """
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
    """
    Compute the standard 31/Scat game score for a 3-card hand.

    The score is the sum of card values in the suit with the highest total.
    Special case: three cards of the same rank (e.g., three 7s) scores 30.5.

    Args:
        hand (List[Card]): A list of cards. Typically contains 3 cards.
            Each card is a tuple (rank, suit).

    Returns:
        float: The score of the hand. Ranges from 0 to 31 (or 30.5 for three-of-a-kind).

    Example:
        >>> score_hand([("A", "hearts"), ("K", "hearts"), ("Q", "hearts")])
        31.0  # 11 + 10 + 10
        >>> score_hand([("7", "hearts"), ("7", "diamonds"), ("7", "clubs")])
        30.5  # Three-of-a-kind special case
        >>> score_hand([("A", "hearts"), ("2", "diamonds"), ("3", "clubs")])
        11.0  # Best suit is hearts with 11
    """
    if len(hand) == HAND_SIZE:
        ranks = {rank for rank, _ in hand}
        if len(ranks) == 1:
            return 30.5  # Three-of-a-kind special case.

    per_suit: Dict[str, int] = {}
    for rank, suit in hand:
        per_suit[suit] = per_suit.get(suit, 0) + CARD_VALUES.get(rank, 0)

    return float(max(per_suit.values(), default=0))
