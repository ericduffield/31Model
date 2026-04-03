"""Evaluate a trained DQN agent against all baseline opponent strategies.

Loads a saved checkpoint and runs offline evaluation: plays multiple games
against each of 5 baseline strategies and reports win rates, average scores,
knock frequencies, and knock-win percentages.
"""

import argparse
import random

import numpy as np
import torch
from typing import Dict, List, Sequence, Type

from computer import (
    ComputerStrategy,
    ConservativeExpectedValueStrategy,
    CurrentTurnExpectedValueStrategy,
    DiscardIncreaseStrategy,
    RandomStrategy,
    RandomStrategyWithKnockScore,
)
from dqn_agent import DQNAgent
from rl_env import NUM_ACTIONS, ThirtyOneEnv
from rules import score_hand

OpponentClass = Type[ComputerStrategy]
"""Type alias for ComputerStrategy subclasses."""


def evaluate(
    agent: DQNAgent,
    opponents: Sequence[OpponentClass],
    games_per_opponent: int,
    seed: int,
    include_extra_stats: bool,
) -> List[Dict[str, float]]:
    """Run agent against multiple opponent strategies and collect stats.

    Args:
        agent (DQNAgent): Trained agent to evaluate (epsilon=0, greedy).
        opponents (Sequence[OpponentClass]): List of opponent classes to test.
        games_per_opponent (int): Number of games per opponent.
        seed (int): Reproducibility seed.
        include_extra_stats (bool): If True, also compute avg scores, knock%,
            knock-win%, etc. If False, only win/loss/draw.

    Returns:
        List[Dict[str, object]]: List of result dicts (one per opponent) with keys:
            - "opponent": str, class name
            - "wins", "losses", "draws": int counts
            - "win_rate": float percentage
            - "avg_score", "knock_rate", "avg_knock_score", "knock_win_rate": float
              (only if include_extra_stats=True)

    Example:
        >>> agent = DQNAgent(obs_dim=165, num_actions=9)
        >>> agent.load('checkpoints/dqn.pt')
        >>> results = evaluate(
        ...     agent,
        ...     [RandomStrategy, DiscardIncreaseStrategy],
        ...     games_per_opponent=100,
        ...     seed=42,
        ...     include_extra_stats=True
        ... )
        >>> results[0]['opponent']
        'RandomStrategy'
        >>> results[0]['win_rate']
        75.5
    """
    rows: List[Dict[str, float]] = []

    for idx, opp_cls in enumerate(opponents):
        env = ThirtyOneEnv(opponent_factory=opp_cls, seed=seed + idx)
        wins = 0
        losses = 0
        draws = 0
        total_score = 0.0
        knock_count = 0
        knock_win_count = 0
        total_knock_score = 0.0

        for game_idx in range(games_per_opponent):
            obs, info = env.reset(start_player=game_idx % 2)
            done = False

            while not done:
                action = agent.select_action(obs, info["action_mask"], epsilon=0.0)
                result = env.step(action)
                obs = result.observation
                info = result.info
                done = result.terminated or result.truncated

            if result.reward > 0:
                wins += 1
            elif result.reward < 0:
                losses += 1
            else:
                draws += 1

            if include_extra_stats:
                agent_score = score_hand(env.game.hands[env.agent])
                total_score += agent_score

                if env.game.knocked_by == env.agent:
                    knock_count += 1
                    total_knock_score += agent_score
                    if result.reward > 0:
                        knock_win_count += 1

        total = wins + losses + draws
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        avg_score = (total_score / total) if include_extra_stats and total > 0 else 0.0
        knock_rate = (
            (knock_count / total * 100.0) if include_extra_stats and total > 0 else 0.0
        )
        avg_knock_score = (
            (total_knock_score / knock_count)
            if include_extra_stats and knock_count > 0
            else 0.0
        )
        knock_win_rate = (
            (knock_win_count / knock_count * 100.0)
            if include_extra_stats and knock_count > 0
            else 0.0
        )

        rows.append(
            {
                "opponent": opp_cls.__name__,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate,
                "avg_score": avg_score,
                "knock_rate": knock_rate,
                "avg_knock_score": avg_knock_score,
                "knock_win_rate": knock_win_rate,
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation script.

    Returns:
        argparse.Namespace: Parsed args with attributes:
            - model_path (str): Path to saved agent checkpoint
            - games (int): Games per opponent
            - seed (int): Random seed
            - include_extra_stats (bool): Compute extra metrics

    Example:
        >>> args = parse_args()  # Reads sys.argv
        >>> args.model_path
        'checkpoints/dqn_best.pt'
    """
    parser = argparse.ArgumentParser(description="Evaluate DQN for 31.")
    parser.add_argument("--model-path", type=str, default="checkpoints/dqn_best.pt")
    parser.add_argument("--games", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--include-extra-stats",
        action="store_true",
        help="Include avg score and knock-related metrics in the output table.",
    )
    return parser.parse_args()


def main() -> None:
    """Load agent, run evaluation against all baselines, and print results table.

    Reads command-line args (--model-path, --games, --seed, --include-extra-stats),
    loads the DQN agent from checkpoint, and runs evaluate() against all 5 baseline
    opponent classes. Prints results in a formatted ASCII table.

    Example:
        >>> # From command line:
        >>> # $ python evaluate_dqn.py --model-path checkpoints/dqn_best.pt \
        >>> #     --games 5000 --include-extra-stats
        >>> # Prints: | RandomStrategy | 1234 | 765 | ... | 61.72 | ...
    """
    args = parse_args()

    # Set global seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    opponents: List[OpponentClass] = [
        RandomStrategy,
        RandomStrategyWithKnockScore,
        DiscardIncreaseStrategy,
        CurrentTurnExpectedValueStrategy,
        ConservativeExpectedValueStrategy,
    ]

    probe_env = ThirtyOneEnv(opponent_factory=RandomStrategy, seed=args.seed)
    agent = DQNAgent(obs_dim=probe_env.obs_dim, num_actions=NUM_ACTIONS)
    agent.load(args.model_path)

    rows = evaluate(
        agent,
        opponents,
        games_per_opponent=args.games,
        seed=args.seed,
        include_extra_stats=args.include_extra_stats,
    )

    if args.include_extra_stats:
        header = f"| {
                'Opponent':<34} | {
                'Wins':>6} | {
                'Losses':>7} | {
                    'Draws':>5} | " f"{
                        'Win Rate %':>10} | {
                            'Avg Score':>9} | {
                                'Knock %':>7} | {
                                    'Avg Knock Score':>15} | {
                                        'Knock Win %':>10} |"
    else:
        header = f"| {
                'Opponent':<34} | {
                'Wins':>6} | {
                'Losses':>7} | {
                    'Draws':>5} | {
                        'Win Rate %':>10} |"

    rule = "-" * len(header)
    border = "=" * len(header)
    print(border)
    print(header)
    print(rule)

    total_wr = 0.0
    total_avg_score = 0.0
    total_knock_rate = 0.0
    total_avg_knock_score = 0.0
    total_knock_win_rate = 0.0

    for row in rows:
        total_wr += row["win_rate"]
        if args.include_extra_stats:
            total_avg_score += row["avg_score"]
            total_knock_rate += row["knock_rate"]
            total_avg_knock_score += row["avg_knock_score"]
            total_knock_win_rate += row["knock_win_rate"]

            print(
                f"| {row['opponent']:<34} | {int(row['wins']):>6} | {int(row['losses']):>7} | {int(row['draws']):>5} | "
                f"{row['win_rate']:>10.2f} | {row['avg_score']:>9.2f} | {row['knock_rate']:>7.2f} | "
                f"{row['avg_knock_score']:>15.2f} | {row['knock_win_rate']:>10.2f} |"
            )
        else:
            print(
                f"| {row['opponent']:<34} | {int(row['wins']):>6} | {int(row['losses']):>7} | {int(row['draws']):>5} | "
                f"{row['win_rate']:>10.2f} |"
            )

    count = max(1, len(rows))
    average_win_pct = total_wr / count
    if args.include_extra_stats:
        average_score = total_avg_score / count
        average_knock_pct = total_knock_rate / count
        average_knock_score = total_avg_knock_score / count
        average_knock_win_pct = total_knock_win_rate / count

        print(rule)
        print(f"| {
                'Average Win %':<34} | {
                '':>6} | {
                '':>7} | {
                    '':>5} | " f"{
                        average_win_pct:>10.2f} | {
                            average_score:>9.2f} | {
                                average_knock_pct:>7.2f} | " f"{
                                    average_knock_score:>15.2f} | {
                                        average_knock_win_pct:>10.2f} |")
    else:
        print(rule)
        print(f"| {
                'Average Win %':<34} | {
                '':>6} | {
                '':>7} | {
                    '':>5} | {
                        average_win_pct:>10.2f} |")

    print(border)


if __name__ == "__main__":
    main()
