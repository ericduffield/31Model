"""Evaluate a trained DQN agent against baseline opponent strategies.

Prints outcome metrics (win/loss/draw, decisive win rate, non-loss rate) and
quality metrics (scores, margins, knock behavior).
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Sequence, Type

import numpy as np
import torch

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

OPPONENT_LABELS = {
    "RandomStrategy": "Random",
    "RandomStrategyWithKnockScore": "Random+Knock27",
    "DiscardIncreaseStrategy": "DiscardIncrease",
    "CurrentTurnExpectedValueStrategy": "CurrentTurnEV",
    "ConservativeExpectedValueStrategy": "ConservativeEV",
    "Average": "Average",
}


def _format_table(
    title: str, columns: List[tuple[str, str]], rows: List[Dict[str, Any]]
) -> None:
    """Print a markdown-compatible table from column specs and rows."""
    header_cells = [label for label, _ in columns]
    body_rows: List[List[str]] = []

    for row in rows:
        formatted: List[str] = []
        for _, key in columns:
            value = row[key]
            if isinstance(value, float):
                formatted.append(f"{value:.2f}")
            else:
                formatted.append(str(value))
        body_rows.append(formatted)

    widths = [len(cell) for cell in header_cells]
    for row in body_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_line(cells: List[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print(f"\n{title}")
    print(render_line(header_cells))
    print(separator)
    for row in body_rows:
        print(render_line(row))


def evaluate(
    agent: DQNAgent,
    opponents: Sequence[OpponentClass],
    games_per_opponent: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run greedy evaluation against multiple opponents and collect metrics."""
    rows: List[Dict[str, Any]] = []

    for idx, opp_cls in enumerate(opponents):
        env = ThirtyOneEnv(opponent_factory=opp_cls, seed=seed + idx)
        wins = 0
        losses = 0
        draws = 0

        total_agent_score = 0.0
        total_opp_score = 0.0
        total_score_margin = 0.0
        total_win_score = 0.0
        total_loss_score = 0.0

        knock_count = 0
        knock_win_count = 0
        knock_loss_count = 0
        total_knock_score = 0.0
        action_knock_count = 0
        action_draw_discard_count = 0
        action_draw_deck_count = 0
        total_agent_turns = 0

        for game_idx in range(games_per_opponent):
            obs, info = env.reset(start_player=game_idx % 2)
            done = False

            while not done:
                action = agent.select_action(obs, info["action_mask"], epsilon=0.0)
                total_agent_turns += 1
                if action == 0:
                    action_knock_count += 1
                elif 1 <= action <= 4:
                    action_draw_discard_count += 1
                elif 5 <= action <= 8:
                    action_draw_deck_count += 1
                result = env.step(action)
                obs = result.observation
                info = result.info
                done = result.terminated or result.truncated

            agent_score = score_hand(env.game.hands[env.agent])
            opp_score = score_hand(env.game.hands[env.opponent])
            total_agent_score += agent_score
            total_opp_score += opp_score
            total_score_margin += agent_score - opp_score

            if result.reward > 0:
                wins += 1
                total_win_score += agent_score
            elif result.reward < 0:
                losses += 1
                total_loss_score += agent_score
            else:
                draws += 1

            if env.game.knocked_by == env.agent:
                knock_count += 1
                total_knock_score += agent_score
                if result.reward > 0:
                    knock_win_count += 1
                elif result.reward < 0:
                    knock_loss_count += 1

        total = wins + losses + draws
        decisive = wins + losses

        rows.append(
            {
                "opponent": opp_cls.__name__,
                "opponent_short": OPPONENT_LABELS.get(
                    opp_cls.__name__, opp_cls.__name__
                ),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": (wins / total * 100.0) if total > 0 else 0.0,
                "draw_rate": (draws / total * 100.0) if total > 0 else 0.0,
                "non_loss_rate": ((wins + draws) / total * 100.0) if total > 0 else 0.0,
                "decisive_win_rate": (wins / decisive * 100.0) if decisive > 0 else 0.0,
                "avg_agent_score": (total_agent_score / total) if total > 0 else 0.0,
                "avg_opp_score": (total_opp_score / total) if total > 0 else 0.0,
                "avg_margin": (total_score_margin / total) if total > 0 else 0.0,
                "avg_win_score": (total_win_score / wins) if wins > 0 else 0.0,
                "avg_loss_score": (total_loss_score / losses) if losses > 0 else 0.0,
                "knock_rate": (knock_count / total * 100.0) if total > 0 else 0.0,
                "avg_knock_score": (
                    (total_knock_score / knock_count) if knock_count > 0 else 0.0
                ),
                "knock_win_rate": (
                    (knock_win_count / knock_count * 100.0) if knock_count > 0 else 0.0
                ),
                "knock_loss_rate": (
                    (knock_loss_count / knock_count * 100.0) if knock_count > 0 else 0.0
                ),
                "action_knock_rate": (
                    action_knock_count / total_agent_turns * 100.0
                    if total_agent_turns > 0
                    else 0.0
                ),
                "action_draw_discard_rate": (
                    action_draw_discard_count / total_agent_turns * 100.0
                    if total_agent_turns > 0
                    else 0.0
                ),
                "action_draw_deck_rate": (
                    action_draw_deck_count / total_agent_turns * 100.0
                    if total_agent_turns > 0
                    else 0.0
                ),
            }
        )

    return rows


def _add_average_row(rows: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Append a macro-average row across opponents for numeric fields."""
    if not rows:
        return rows

    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key != "opponent" and isinstance(value, (int, float))
    ]
    avg_row: Dict[str, Any] = {
        "opponent": label,
        "opponent_short": OPPONENT_LABELS.get(label, label),
    }
    for key in numeric_keys:
        avg_row[key] = sum(float(row[key]) for row in rows) / len(rows)

    return [*rows, avg_row]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate DQN for 31.")
    parser.add_argument("--model-path", type=str, default="checkpoints/dqn_best.pt")
    parser.add_argument("--games", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint, evaluate against baseline opponents, and print tables."""
    args = parse_args()

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
        agent=agent,
        opponents=opponents,
        games_per_opponent=args.games,
        seed=args.seed,
    )
    rows_with_avg = _add_average_row(rows, label="Average")

    _format_table(
        title="Outcome Counts",
        columns=[
            ("Opponent", "opponent_short"),
            ("Wins", "wins"),
            ("Losses", "losses"),
            ("Draws", "draws"),
            ("Win %", "win_rate"),
            ("Decisive Win %", "decisive_win_rate"),
        ],
        rows=rows_with_avg,
    )

    _format_table(
        title="Score Metrics",
        columns=[
            ("Opponent", "opponent_short"),
            ("Avg Agent", "avg_agent_score"),
            ("Avg Opp", "avg_opp_score"),
            ("Avg Margin", "avg_margin"),
            ("Avg Win", "avg_win_score"),
            ("Avg Loss", "avg_loss_score"),
        ],
        rows=rows_with_avg,
    )

    _format_table(
        title="Knock Metrics",
        columns=[
            ("Opponent", "opponent_short"),
            ("Knock %", "knock_rate"),
            ("Avg Knock", "avg_knock_score"),
            ("Knock Win %", "knock_win_rate"),
            ("Knock Loss %", "knock_loss_rate"),
        ],
        rows=rows_with_avg,
    )

    _format_table(
        title="Action Distribution",
        columns=[
            ("Opponent", "opponent_short"),
            ("Action Knock %", "action_knock_rate"),
            ("Action DrawDiscard %", "action_draw_discard_rate"),
            ("Action DrawDeck %", "action_draw_deck_rate"),
        ],
        rows=rows_with_avg,
    )


if __name__ == "__main__":
    main()
