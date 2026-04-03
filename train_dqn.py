"""
Training script for a DQN agent on the card game 31.

Orchestrates the training loop: episode generation, replay buffer management,
network updates, and periodic evaluation against baseline strategies.
Supports run naming, checkpointing, and detailed logging for monitoring.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from typing import List, Sequence, Tuple, Type

import numpy as np

from computer import (
    ComputerStrategy,
    DiscardIncreaseStrategy,
    RandomStrategy,
)
from dqn_agent import DQNAgent
from rl_env import NUM_ACTIONS, ThirtyOneEnv

OpponentClass = Type[ComputerStrategy]
CHECKPOINT_DIR = "checkpoints"


def set_global_seed(seed: int) -> None:
    """Set the seed for random number generators across libraries."""
    random.seed(seed)
    np.random.seed(seed)


def evaluate_agent(
    agent: DQNAgent,
    opponents: Sequence[OpponentClass],
    games_per_opponent: int,
    seed: int,
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Evaluate the agent against multiple baseline strategies.

    Runs the agent in greedy mode (epsilon=0) against each opponent and
    computes per-opponent and aggregate win rates.

    Args:
        agent (DQNAgent): The trained agent to evaluate.
        opponents (Sequence[OpponentClass]): Strategy classes to play against.
        games_per_opponent (int): Number of test games per opponent.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[float, List[Tuple[str, float]]]: Aggregate win rate (%) and
            list of (opponent_name, win_rate_%) tuples.
    """
    per_opponent: List[Tuple[str, float]] = []

    for idx, opp_cls in enumerate(opponents):
        print(
            f"  Eval progress: opponent {idx + 1}/{len(opponents)} -> {opp_cls.__name__}",
            flush=True,
        )
        env = ThirtyOneEnv(opponent_factory=opp_cls, seed=seed + idx)
        wins = 0
        losses = 0
        draws = 0

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

        total = wins + losses + draws
        win_rate = (wins / total * 100.0) if total > 0 else 0.0
        per_opponent.append((opp_cls.__name__, win_rate))

    aggregate = sum(wr for _, wr in per_opponent) / len(per_opponent)
    return aggregate, per_opponent


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments with all hyperparameters
            and configuration options.
    """
    parser = argparse.ArgumentParser(description="Train DQN for 31.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Name for this training run; checkpoints are saved under checkpoints/<run-name>/.",
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--eval-games", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--buffer-capacity", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--epsilon-start", type=float, required=True)
    parser.add_argument("--epsilon-end", type=float, required=True)
    parser.add_argument("--epsilon-decay-episodes", type=int, required=True)
    parser.add_argument("--target-update-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def epsilon_by_episode(
    episode: int,
    eps_start: float,
    eps_end: float,
    eps_decay_episodes: int,
) -> float:
    """
    Compute the epsilon (exploration rate) for a given training episode.

    Linear decay from eps_start to eps_end over eps_decay_episodes.
    After decay period, epsilon remains at eps_end.

    Args:
        episode (int): Current episode number (1-indexed).
        eps_start (float): Initial exploration rate.
        eps_end (float): Final exploration rate.
        eps_decay_episodes (int): Number of episodes over which to decay.

    Returns:
        float: Epsilon value for this episode.
    """


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.run_name)
    # Start this run from a clean run-specific checkpoint directory.
    if os.path.exists(run_checkpoint_dir):
        shutil.rmtree(run_checkpoint_dir)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Run name: {args.run_name}", flush=True)
    print(f"Checkpoint directory: {run_checkpoint_dir}", flush=True)

    train_opponents: List[OpponentClass] = [
        # RandomStrategy,
        # RandomStrategyWithKnockScore,
        DiscardIncreaseStrategy,
        # CurrentTurnExpectedValueStrategy,
        # ConservativeExpectedValueStrategy,
    ]

    probe_env = ThirtyOneEnv(opponent_factory=RandomStrategy, seed=args.seed)
    obs_dim = probe_env.obs_dim

    agent = DQNAgent(
        obs_dim=obs_dim,
        num_actions=NUM_ACTIONS,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        buffer_capacity=args.buffer_capacity,
        seed=args.seed,
    )

    global_step = 0
    best_eval_wr = -1.0
    best_eval_episode = 0
    best_eval_by_opp: List[Tuple[str, float]] = []
    moving_rewards: List[float] = []
    moving_losses: List[float] = []

    for episode in range(1, args.episodes + 1):
        opp_cls = random.choice(train_opponents)
        env = ThirtyOneEnv(opponent_factory=opp_cls, seed=args.seed + episode)

        obs, info = env.reset(start_player=episode % 2)
        done = False
        episode_reward = 0.0
        epsilon = epsilon_by_episode(
            episode=episode,
            eps_start=args.epsilon_start,
            eps_end=args.epsilon_end,
            eps_decay_episodes=args.epsilon_decay_episodes,
        )

        while not done:
            action_mask = info["action_mask"]
            action = agent.select_action(obs, action_mask, epsilon=epsilon)

            result = env.step(action)
            next_obs = result.observation
            next_info = result.info
            done = result.terminated or result.truncated

            agent.add_transition(
                obs=obs,
                action=action,
                reward=result.reward,
                next_obs=next_obs,
                done=done,
                action_mask=action_mask,
                next_action_mask=next_info["action_mask"],
            )

            if global_step >= args.warmup_steps:
                loss = agent.train_step(args.batch_size)
                if loss is not None:
                    moving_losses.append(loss)
                    if len(moving_losses) > 500:
                        moving_losses.pop(0)

            if global_step % args.target_update_steps == 0:
                agent.update_target()

            obs = next_obs
            info = next_info
            episode_reward += result.reward
            global_step += 1

        moving_rewards.append(episode_reward)
        if len(moving_rewards) > 200:
            moving_rewards.pop(0)

        if episode % args.log_every == 0:
            avg_reward = sum(moving_rewards) / len(moving_rewards)
            avg_loss = (
                (sum(moving_losses) / len(moving_losses))
                if moving_losses
                else float("nan")
            )
            q_diag = agent.q_diagnostics(sample_size=128)
            if q_diag is None:
                mean_max_q = float("nan")
                mean_abs_q = float("nan")
            else:
                mean_max_q, mean_abs_q = q_diag
            print(
                f"Episode {episode:>6}/{args.episodes} | epsilon={epsilon:.3f} | "
                f"avg_reward(200)={avg_reward:.3f} | avg_loss(500)={avg_loss:.4f} | "
                f"mean_max_q={mean_max_q:.3f} | mean_abs_q={mean_abs_q:.3f} | "
                f"replay={len(agent.replay)}",
                flush=True,
            )

        if episode % args.eval_every == 0:
            print(
                f"Starting evaluation at episode {episode} "
                f"({len(train_opponents)} opponents x {args.eval_games} games)",
                flush=True,
            )
            avg_wr, by_opp = evaluate_agent(
                agent=agent,
                opponents=train_opponents,
                games_per_opponent=args.eval_games,
                seed=args.seed + 100_000 + episode,
            )
            avg_reward = sum(moving_rewards) / len(moving_rewards)
            print(
                f"Episode {
                    episode:>6} | epsilon={
                    epsilon:.3f} | avg_reward(200)={
                    avg_reward:.3f} " f"| eval_win_rate={
                    avg_wr:.2f}%",
                flush=True,
            )
            for name, wr in by_opp:
                print(f"  vs {name:<35} {wr:>6.2f}%", flush=True)

            latest_path = os.path.join(run_checkpoint_dir, "dqn_latest.pt")
            agent.save(latest_path)

            if avg_wr > best_eval_wr:
                best_eval_wr = avg_wr
                best_eval_episode = episode
                best_eval_by_opp = by_opp.copy()
                best_path = os.path.join(run_checkpoint_dir, "dqn_best.pt")
                agent.save(best_path)
                print(f"  New best aggregate win rate: {
                        best_eval_wr:.2f}% -> saved {best_path}", flush=True)

    print("Training complete.")
    if best_eval_episode > 0:
        print(
            f"Best eval average win rate: {
                best_eval_wr:.2f}% at episode {best_eval_episode}",
            flush=True,
        )
        for name, wr in best_eval_by_opp:
            print(f"  best vs {name:<35} {wr:>6.2f}%", flush=True)
    else:
        print(
            "No evaluation was run, so no best eval win rate is available.",
            flush=True,
        )


if __name__ == "__main__":
    main()
