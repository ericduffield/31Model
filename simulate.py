"""Simulation framework for running matchups between different baseline strategies.

Runs head-to-head games between strategies, collects win/loss statistics,
knock frequencies, and generates comparison tables. Useful for empirically
evaluating baseline strength before training RL agents.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from game import Game
from computer import (
    ComputerStrategy,
    DiscardIncreaseStrategy,
    CurrentTurnExpectedValueStrategy,
    ConservativeExpectedValueStrategy,
    ScoreAdaptiveExpectedValueStrategy,
)


class Simulation:
    """Run multiple games between strategies and track detailed statistics.

    Supports all-vs-all matchups with head-to-head win percentages, knock counts,
    total scores, and knock-win rates. Tracks per-opponent results for each
    strategy including aggregate statistics.
    """

    def __init__(self, debug: bool = False):
        """Initialize an empty simulation runner.

        Args:
            debug (bool): If True, print game-level debugging info.
                Default False (silent game runs).

        Example:
            >>> sim = Simulation(debug=False)
            >>> sim.results
            {}
        """
        self.results: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "total_score": 0,
                "games": 0,
                "total_knock_score": 0,
                "knock_count": 0,
                "knock_losses": 0,
            }
        )
        # Track head-to-head results: matchup_results[strategy1][strategy2] =
        # {wins, losses, draws}
        self.matchup_results: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {"wins": 0, "losses": 0, "draws": 0, "knocks": 0}
            )
        )
        self.debug = debug

    def run_matchup(
        self,
        strategy1: ComputerStrategy,
        strategy2: ComputerStrategy,
        num_games: int,
    ) -> None:
        """Run a series of games between two strategies and record results.

        Creates fresh instances of each strategy for each game, alternates
        who goes first, and accumulates win/loss/draw tallies plus knock stats.
        Prints progress every 10 games and final summary.

        Args:
            strategy1 (ComputerStrategy): First strategy (not special significance).
            strategy2 (ComputerStrategy): Second strategy.
            num_games (int): Number of games to play.

        Example:
            >>> sim = Simulation()
            >>> sim.run_matchup(
            ...     RandomStrategy(), DiscardIncreaseStrategy(), num_games=100
            ... )
            >>> sim.results['Random']['wins']  # How many times Random beat Discard
            >>> sim.matchup_results['Random']['Discard Increase']['wins']
        """

        for game_num in range(num_games):
            # Create fresh instances for each game
            strat1 = type(strategy1)()
            strat2 = type(strategy2)()

            # Alternate who goes first
            if game_num % 2 == 0:
                strategies = [strat1, strat2]
            else:
                strategies = [strat2, strat1]

            game = Game(
                strategies=strategies,
                debug=self.debug,
            )
            # Deal initial cards and notify strategies
            game._deal_initial()

            # Run game silently and get results
            # Map strategies back to names for result tracking
            name_map = {strat1: strat1.name, strat2: strat2.name}
            (
                winner_strategies,
                strategy_scores,
                knocked_by_strat,
                knock_score,
            ) = self._run_game(game)

            # Convert to names for tracking
            winner = [name_map[s] for s in winner_strategies]
            scores = {name_map[s]: score for s, score in strategy_scores.items()}
            knocked_by = name_map[knocked_by_strat] if knocked_by_strat else None

            # Record results
            for player, score in scores.items():
                self.results[player]["total_score"] += score
                self.results[player]["games"] += 1

            # Record knock score
            if knocked_by:
                self.results[knocked_by]["total_knock_score"] += knock_score
                self.results[knocked_by]["knock_count"] += 1
                # Record head-to-head knock count
                opponent = strat1.name if knocked_by == strat2.name else strat2.name
                self.matchup_results[knocked_by][opponent]["knocks"] += 1

            if len(winner) == 1:
                self.results[winner[0]]["wins"] += 1
                loser = strat1.name if winner[0] == strat2.name else strat2.name
                self.results[loser]["losses"] += 1
                # Record head-to-head results
                self.matchup_results[winner[0]][loser]["wins"] += 1
                self.matchup_results[loser][winner[0]]["losses"] += 1
            else:
                # Draw
                for player in winner:
                    self.results[player]["draws"] += 1
                # Record head-to-head draws
                self.matchup_results[strat1.name][strat2.name]["draws"] += 1
                self.matchup_results[strat2.name][strat1.name]["draws"] += 1

            # If a strategy knocked and lost, track it
            if knocked_by and len(winner) == 1 and winner[0] != knocked_by:
                self.results[knocked_by]["knock_losses"] += 1

            # Progress indicator
            if (game_num + 1) % 10 == 0:
                print(f"  Completed {game_num + 1}/{num_games} games")

        print(f"Completed matchup: {strat1.name} vs {strat2.name}")

    def _run_game(self, game: Game) -> Tuple[
        List[ComputerStrategy],
        Dict[ComputerStrategy, float],
        Optional[ComputerStrategy],
        float,
    ]:
        """Run a single game and return winner(s), scores, who knocked, and knock score.

        Args:
            game (Game): Initialized Game object (pre-dealt).

        Returns:
            Tuple of:
                - winners (List[ComputerStrategy]): Strategies that tied for best score
                - scores (Dict[ComputerStrategy, float]): Final hand score for each player
                - knocked_by (Optional[ComputerStrategy]): Who knocked, or None
                - knock_score (float): Score of knocking player (0 if no knock)
        """
        from rules import score_hand

        current = 0
        knock_score = 0.0

        while True:
            strategy = game.strategies[current]

            # Check if knock happened this turn
            knocked_before = game.knocked_by
            self._execute_strategy_turn(game, strategy)
            if game.knocked_by and not knocked_before:
                # Record knock score
                knock_score = score_hand(game.hands[strategy])

            if game.knock_remaining is not None and strategy != game.knocked_by:
                game.knock_remaining -= 1

            if game.knock_remaining is not None and game.knock_remaining <= 0:
                break

            current = (current + 1) % len(game.strategies)

        # Calculate final scores
        scores = {}
        for strategy in game.strategies:
            scores[strategy] = score_hand(game.hands[strategy])

        # Determine winner(s)
        best_score = max(scores.values())
        winners = [p for p, s in scores.items() if s == best_score]

        return winners, scores, game.knocked_by, knock_score

    def _execute_strategy_turn(self, game: Game, strategy: ComputerStrategy) -> None:
        """Execute one turn for a strategy without printing.

        Args:
            game (Game): The game object managing state.
            strategy (ComputerStrategy): The player to move.
        """
        hand = game.hands[strategy]
        top_discard = game.discard_pile[-1] if game.discard_pile else None

        action = strategy.choose_action(
            hand, top_discard, knock_allowed=game.knocked_by is None
        )

        if action == "knock":
            game.knocked_by = strategy
            game.knock_remaining = len(game.strategies) - 1
            return

        card = None
        if action == "draw_discard" and top_discard:
            card = game.discard_pile.pop()
        else:
            card = game.deck.draw_card()
            if card:
                strategy.on_card_drawn(card)
            if not card and top_discard:
                card = game.discard_pile.pop()

        if not card:
            return

        hand.append(card)
        discard_index = strategy.choose_discard_index(hand)
        discarded = hand.pop(discard_index)
        game.discard_pile.append(discarded)
        # Notify all strategies of the discard
        for strat in game.strategies:
            strat.on_card_discarded(discarded)

    def print_head_to_head(self, only_strategies: Optional[List[str]] = None) -> None:
        """Print head-to-head win percentages in a readable table format.

        Shows vs-all win% (excluding draws), draw%, and overall knock%.
        Filters to a subset of strategies if only_strategies is provided.

        Args:
            only_strategies (Optional[List[str]]): If provided, only show these
                strategy names. If None, show all strategies in results.

        Example:
            >>> sim = Simulation()
            >>> sim.run_matchup(...)
            >>> sim.print_head_to_head(only_strategies=['Random', 'DiscardIncrease'])
        """
        print("HEAD-TO-HEAD WIN PERCENTAGES")
        print("=" * 70)

        all_strategies = sorted(self.matchup_results.keys())
        if only_strategies:
            focus_set = set(only_strategies)
            all_strategies = [s for s in all_strategies if s in focus_set]

        for strategy in all_strategies:
            print(f"\n{strategy}:")
            # Overall knock percentage for this strategy
            overall_stats = self.results.get(strategy, {})
            games_total = overall_stats.get("games", 0)
            knocks_total = overall_stats.get("knock_count", 0)
            overall_knock_pct = (
                (knocks_total / games_total * 100) if games_total > 0 else 0
            )
            print(f"Overall knock %: {overall_knock_pct:>.1f}%")
            print("-" * 50)

            opponents = sorted(self.matchup_results[strategy].keys())
            for opponent in opponents:
                stats = self.matchup_results[strategy][opponent]
                wins = stats["wins"]
                losses = stats["losses"]
                draws = stats["draws"]
                knocks = stats.get("knocks", 0)
                total_games = wins + losses + draws
                non_draw_games = wins + losses

                if total_games > 0:
                    win_pct_excl_draws = (
                        (wins / non_draw_games * 100) if non_draw_games > 0 else 0
                    )
                    draw_pct = (draws / total_games) * 100
                    knock_pct = (knocks / total_games) * 100
                    print(f"  vs {
                            opponent:<35} Win(ex Draws) {
                            win_pct_excl_draws:>5.1f}% | Draw {
                            draw_pct:>5.1f}% | Knock {
                            knock_pct:>5.1f}%")

        print("\n" + "=" * 70)

    def reset(self) -> None:
        """Clear all results and ready for new simulations."""
        self.matchup_results.clear()


def main():
    """Run demo matchups between baseline strategies and print results.

    Runs all-vs-all matchups for a set of baseline strategies (configurable
    via NUM_GAMES, strategies, and SHOW_MATCHUP_TABLE). Prints average
    win%, draws%, avg score, knockout counts, etc.
    """

    simulation = Simulation(debug=False)

    # Define strategies to test
    strategies = [
        # RandomStrategy,
        # RandomStrategyWithKnockScore,
        ConservativeExpectedValueStrategy,  # 2
        CurrentTurnExpectedValueStrategy,  # 1
        DiscardIncreaseStrategy,  # 3
        ScoreAdaptiveExpectedValueStrategy,  # Base: no negative, slope=0.08, cap=1.0
    ]

    # Module constants
    NUM_GAMES = 20000  # 10,000 seems to give most consistent results
    SHOW_MATCHUP_TABLE = False  # Set to False to hide matchup table

    # All-vs-all matchups (no duplicates, no self-play)
    num_strategies = len(strategies)
    for i in range(num_strategies):
        for j in range(i + 1, num_strategies):
            simulation.run_matchup(
                strategies[i](), strategies[j](), num_games=NUM_GAMES
            )

    # Print aggregate results per strategy
    print("\n" + "=" * 120)
    print("AVERAGE RESULTS")
    print("=" * 120)
    print(f"{
            'Strategy':<35} {
            'Win% ex Draws':>15} {
                'Draw %':>10} {
                    'Avg Score':>12} {
                        'Avg Knock':>12} {
                            'Knock Loss %':>14}")
    print("-" * 120)
    for strategy_name in sorted(simulation.results.keys()):
        stats = simulation.results[strategy_name]
        wins = stats["wins"]
        losses = stats["losses"]
        draws = stats["draws"]
        games = stats["games"]
        non_draw_games = wins + losses
        avg_win_pct = (wins / non_draw_games * 100) if non_draw_games > 0 else 0
        draw_pct = (draws / games * 100) if games > 0 else 0
        avg_score = (stats["total_score"] / games) if games > 0 else 0
        knock_count = stats["knock_count"]
        avg_knock = (stats["total_knock_score"] / knock_count) if knock_count > 0 else 0
        knock_loss_pct = (
            (stats["knock_losses"] / knock_count * 100) if knock_count > 0 else 0
        )
        print(f"{
                strategy_name:<35} {
                avg_win_pct:>15.2f} {
                draw_pct:>10.2f} {
                    avg_score:>12.2f} {
                        avg_knock:>12.2f} {
                            knock_loss_pct:>14.2f}")
    print("=" * 120)

    # Optionally print head-to-head matchup table
    if SHOW_MATCHUP_TABLE:
        simulation.print_head_to_head()


if __name__ == "__main__":
    main()
