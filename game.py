from typing import Dict, List, Optional

from deck import Deck
from computer import ComputerStrategy, ConservativeExpectedValueStrategy
from rules import HAND_SIZE, Card, card_label, score_hand


class Game:
    """Interactive command-line game of 31 (Scat)."""

    def __init__(
        self,
        strategies: List[ComputerStrategy],
        debug: bool = False,
    ) -> None:
        self.strategies: List[ComputerStrategy] = strategies
        self.deck = Deck()
        self.hands: Dict[ComputerStrategy, List[Card]] = {
            strategy: [] for strategy in self.strategies
        }
        self.discard_pile: List[Card] = []
        self.knocked_by: Optional[ComputerStrategy] = None
        self.knock_remaining: Optional[int] = None
        self.debug: bool = debug

    def play(self) -> None:
        self._deal_initial()
        current = 0

        while True:
            strategy = self.strategies[current]

            if self.debug:
                self._display_state()
                input("Press Enter to continue...")

            self._strategy_turn(strategy)

            if self.knock_remaining is not None and strategy != self.knocked_by:
                self.knock_remaining -= 1

            if self.knock_remaining is not None and self.knock_remaining <= 0:
                break

            current = (current + 1) % len(self.strategies)

        self._end_round()

    def _deal_initial(self) -> None:
        for _ in range(HAND_SIZE):
            for strategy in self.strategies:
                card = self.deck.draw_card()
                if card:
                    self.hands[strategy].append(card)
        starter = self.deck.draw_card()
        if starter:
            self.discard_pile.append(starter)
        # Notify all strategies of initial cards
        for strategy in self.strategies:
            hand = self.hands[strategy]
            strategy.on_initial_cards(hand, starter)

    def _strategy_turn(self, strategy: ComputerStrategy) -> None:
        """Execute a turn using a computer strategy."""
        hand = self.hands[strategy]
        top_discard = self.discard_pile[-1]
        current_score = score_hand(hand)

        if self.debug:
            print(f"\n>>> {strategy.name}'s turn")
            print(
                f"{strategy.name}'s hand: {', '.join(card_label(c) for c in hand)} (score {current_score})"
            )

        action = strategy.choose_action(
            hand,
            top_discard,
            knock_allowed=self.knocked_by is None,
            debug=self.debug,
        )

        if self.debug:
            print(f"Action: {action}")

        if action == "knock":
            self.knocked_by = strategy
            self.knock_remaining = len(self.strategies) - 1
            if self.debug:
                print(f"{strategy.name} knocks!")
            return

        card_source = "deck"
        card: Optional[Card]

        if action == "draw_discard":
            card = self.discard_pile.pop()
            card_source = "discard"
        else:
            card = self.deck.draw_card()
            if card:
                strategy.on_card_drawn(card)
            if not card and top_discard:
                card = self.discard_pile.pop()
                card_source = "discard"

        if not card:
            if self.debug:
                print("No cards available to draw.")
            return

        if self.debug:
            print(f"Drew {card_label(card)} from {card_source}")

        hand.append(card)
        if self.debug:
            # Show hand with drawn card before discard
            # Find best 3-card combo from the 4 cards
            best_score = -1.0
            best_combo = []
            for idx in range(len(hand)):
                candidate = hand[:idx] + hand[idx + 1 :]
                candidate_score = score_hand(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_combo = candidate
            combo_str = ", ".join(card_label(c) for c in best_combo)
            print(
                f"Hand with draw: {', '.join(card_label(c) for c in hand)} (best: {combo_str} = {best_score})"
            )

        discard_index = strategy.choose_discard_index(hand)
        discarded = self._discard_card(strategy, discard_index)
        if self.debug:
            print(f"Discarded {card_label(discarded)}")
            # Show updated hand
            new_score = score_hand(hand)
            print(
                f"Hand after turn: {', '.join(card_label(c) for c in hand)} (score {new_score})"
            )

    def _discard_card(self, strategy: ComputerStrategy, index: int) -> Card:
        hand = self.hands[strategy]
        removed = hand.pop(index)
        self.discard_pile.append(removed)
        # Notify all strategies
        for strat in self.strategies:
            strat.on_card_discarded(removed)
        return removed

    def _display_state(self) -> None:
        """Display full game state."""
        print("\n" + "=" * 60)
        print("GAME STATE")
        print("=" * 60)
        for strategy in self.strategies:
            hand = self.hands[strategy]
            score = score_hand(hand)
            labels = ", ".join(card_label(c) for c in hand)
            print(f"{strategy.name}: {labels} (score {score})")

        if self.discard_pile:
            print(f"Top discard: {card_label(self.discard_pile[-1])}")
        else:
            print("Discard pile: empty")

        if self.knocked_by:
            print(
                f"Knocked by: {self.knocked_by.name} ({self.knock_remaining} turns left)"
            )
        print("=" * 60)

    def _end_round(self) -> None:
        print("\nRound complete. Final hands:")
        scores: Dict[str, float] = {}
        for strategy in self.strategies:
            hand = self.hands[strategy]
            score = score_hand(hand)
            scores[strategy] = score
            labels = ", ".join(card_label(c) for c in hand)
            print(f"  {strategy.name}: {labels} => {score}")

        best = max(scores.values())
        winners = [s for s, val in scores.items() if val == best]
        if len(winners) == 1:
            print(f"{winners[0]} wins with {best}!")
        else:
            joined = ", ".join(winners)
            print(f"Tie between {joined} at {best}.")


def main() -> None:
    game = Game(
        strategies=[
            ConservativeExpectedValueStrategy(),
            ConservativeExpectedValueStrategy(),
        ],
        debug=True,
    )
    game.play()


if __name__ == "__main__":
    main()
