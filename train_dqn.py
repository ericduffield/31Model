"""
Training script for a DQN agent on the card game 31.

Orchestrates the training loop: episode generation, replay buffer management,
network updates, and periodic evaluation against baseline strategies.
Supports run naming, checkpointing, and detailed logging for monitoring.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple, Type

import numpy as np
import torch

from computer import (
    ComputerStrategy,
    ConservativeExpectedValueStrategy,
    DiscardIncreaseStrategy,
    RandomStrategy,
    RandomStrategyWithKnockScore,
    CurrentTurnExpectedValueStrategy
)
from dqn_agent import DQNAgent
from rl_env import NUM_ACTIONS, ThirtyOneEnv
from rules import score_hand

OpponentClass = Type[ComputerStrategy]
CHECKPOINT_DIR = "checkpoints"
TOP_CHECKPOINT_FILENAMES = ["dqn_best.pt", "dqn_second_best.pt", "dqn_third_best.pt"]


def set_global_seed(seed: int) -> None:
    """Set the seed for random number generators across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Force deterministic kernels where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)



def write_logs_to_file(logs: List[Dict], run_checkpoint_dir: str) -> None:
    """Write current logs to JSON file for inspection during training."""
    log_path = os.path.join(run_checkpoint_dir, "training_logs.json")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write logs to {log_path}: {e}", flush=True)


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
        required=True,
        help="Name for this training run; checkpoints are saved under checkpoints/<run-name>/.",
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument(
        "--eval-games",
        type=int,
        default=2000,
        help="Evaluation games per opponent (default: 2000).",
    )
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
    if eps_decay_episodes <= 0:
        return eps_end

    progress = min(1.0, episode / float(eps_decay_episodes))
    return eps_start + progress * (eps_end - eps_start)


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

    # Save run command + hyperparameters for reproducibility.
    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "command": " ".join([sys.executable, *sys.argv]),
        "argv": sys.argv,
        "hyperparameters": vars(args),
        "determinism": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
    }
    run_meta_path = os.path.join(run_checkpoint_dir, "run_metadata.json")
    with open(run_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Run metadata saved to {run_meta_path}", flush=True)

    train_opponents: List[OpponentClass] = [
        # RandomStrategy,
        # RandomStrategyWithKnockScore,
        # ConservativeExpectedValueStrategy,
        # CurrentTurnExpectedValueStrategy,
        DiscardIncreaseStrategy,
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
    best_logged_avg_reward = float("-inf")
    top_checkpoints: List[Dict[str, object]] = []
    moving_rewards: List[float] = []
    moving_outcomes: List[float] = []
    moving_losses: List[float] = []
    moving_knock_flags: List[float] = []
    moving_knock_scores: List[float] = []
    moving_end_scores: List[float] = []
    logs: List[Dict] = []  # Accumulate logs for JSON export

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
        moving_outcomes.append(episode_reward)
        if len(moving_outcomes) > 200:
            moving_outcomes.pop(0)

        agent_end_score = score_hand(env.game.hands[env.agent])
        agent_knocked = (env.game.knocked_by == env.agent)
        moving_end_scores.append(agent_end_score)
        if len(moving_end_scores) > 200:
            moving_end_scores.pop(0)
        moving_knock_flags.append(1.0 if agent_knocked else 0.0)
        if len(moving_knock_flags) > 200:
            moving_knock_flags.pop(0)
        if agent_knocked:
            moving_knock_scores.append(agent_end_score)
            if len(moving_knock_scores) > 200:
                moving_knock_scores.pop(0)

        if episode % args.log_every == 0:
            avg_reward = sum(moving_rewards) / len(moving_rewards)
            wins_200 = sum(1 for outcome in moving_outcomes if outcome > 0)
            losses_200 = sum(1 for outcome in moving_outcomes if outcome < 0)
            draws_200 = sum(1 for outcome in moving_outcomes if outcome == 0)
            total_200 = len(moving_outcomes)
            avg_win_rate_200 = (wins_200 / total_200 * 100.0) if total_200 > 0 else 0.0
            avg_draw_rate_200 = (draws_200 / total_200 * 100.0) if total_200 > 0 else 0.0
            decisive_games_200 = wins_200 + losses_200
            avg_decisive_win_rate_200 = (
                wins_200 / decisive_games_200 * 100.0 if decisive_games_200 > 0 else 0.0
            )
            knock_rate_200 = (
                sum(moving_knock_flags) / len(moving_knock_flags) * 100.0
                if moving_knock_flags
                else 0.0
            )
            avg_knock_score_200 = (
                sum(moving_knock_scores) / len(moving_knock_scores)
                if moving_knock_scores
                else float("nan")
            )
            avg_end_score_200 = (
                sum(moving_end_scores) / len(moving_end_scores)
                if moving_end_scores
                else float("nan")
            )
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
            
            log_entry = {
                "episode": episode,
                "epsilon": epsilon,
                "avg_reward": avg_reward,
                "avg_win_rate_200": avg_win_rate_200,
                "avg_draw_rate_200": avg_draw_rate_200,
                "avg_decisive_win_rate_200": avg_decisive_win_rate_200,
                "knock_rate_200": knock_rate_200,
                "avg_knock_score_200": avg_knock_score_200,
                "avg_end_hand_score_200": avg_end_score_200,
                "avg_loss": avg_loss,
                "mean_max_q": mean_max_q,
                "mean_abs_q": mean_abs_q,
                "replay_size": len(agent.replay),
            }
            logs.append(log_entry)
            write_logs_to_file(logs, run_checkpoint_dir)
            
            print(
                f"Episode {episode:>6}/{args.episodes} | epsilon={epsilon:.3f} | "
                f"avg_reward(200)={avg_reward:.3f} | "
                f"win%(200)={avg_win_rate_200:.2f} | draw%(200)={avg_draw_rate_200:.2f} | "
                f"decisive_win%(200)={avg_decisive_win_rate_200:.2f} | "
                f"knock%(200)={knock_rate_200:.2f} | "
                f"avg_knock_score(200)={avg_knock_score_200:.2f} | "
                f"avg_end_score(200)={avg_end_score_200:.2f} | "
                f"avg_loss(500)={avg_loss:.4f} | "
                f"mean_max_q={mean_max_q:.3f} | mean_abs_q={mean_abs_q:.3f} | "
                f"replay={len(agent.replay)}",
                flush=True,
            )

        reward_peak_eval = False
        if logs and logs[-1]["episode"] == episode:
            current_avg_reward = float(logs[-1]["avg_reward"])
            if current_avg_reward > best_logged_avg_reward:
                best_logged_avg_reward = current_avg_reward
                reward_peak_eval = True

        periodic_eval = (episode % args.eval_every == 0)
        if periodic_eval or reward_peak_eval:
            eval_reason: List[str] = []
            if periodic_eval:
                eval_reason.append("periodic")
            if reward_peak_eval:
                eval_reason.append("new best avg_reward")
            print(
                f"Starting evaluation at episode {episode} "
                f"[{', '.join(eval_reason)}] "
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

            # Log evaluation results
            eval_log = {
                "episode": episode,
                "eval_win_rate": avg_wr,
                "eval_by_opponent": [{"opponent": name, "win_rate": wr} for name, wr in by_opp],
            }
            if logs and logs[-1]["episode"] == episode:
                logs[-1].update(eval_log)
            else:
                logs.append(eval_log)
            write_logs_to_file(logs, run_checkpoint_dir)

            latest_path = os.path.join(run_checkpoint_dir, "dqn_latest.pt")
            agent.save(latest_path)

            should_track = (len(top_checkpoints) < 3) or (
                avg_wr > float(top_checkpoints[-1]["win_rate"])
            )
            if should_track:
                insert_idx = len(top_checkpoints)
                for i, ckpt in enumerate(top_checkpoints):
                    if avg_wr > float(ckpt["win_rate"]):
                        insert_idx = i
                        break

                if insert_idx < 3:
                    temp_path = os.path.join(run_checkpoint_dir, "_dqn_candidate.pt")
                    agent.save(temp_path)

                    for rank in range(min(len(top_checkpoints), 2), insert_idx - 1, -1):
                        if rank >= 2:
                            continue
                        src_name = TOP_CHECKPOINT_FILENAMES[rank]
                        dst_name = TOP_CHECKPOINT_FILENAMES[rank + 1]
                        src_path = os.path.join(run_checkpoint_dir, src_name)
                        dst_path = os.path.join(run_checkpoint_dir, dst_name)
                        if os.path.exists(src_path):
                            shutil.copy2(src_path, dst_path)

                    target_name = TOP_CHECKPOINT_FILENAMES[insert_idx]
                    target_path = os.path.join(run_checkpoint_dir, target_name)
                    shutil.copy2(temp_path, target_path)
                    os.remove(temp_path)

                    top_checkpoints.insert(
                        insert_idx,
                        {
                            "episode": episode,
                            "win_rate": avg_wr,
                            "by_opp": by_opp.copy(),
                            "path": target_path,
                        },
                    )
                    if len(top_checkpoints) > 3:
                        top_checkpoints.pop()

                    previous_best_episode = best_eval_episode
                    best_eval_wr = float(top_checkpoints[0]["win_rate"])
                    best_eval_episode = int(top_checkpoints[0]["episode"])
                    best_eval_by_opp = list(top_checkpoints[0]["by_opp"])
                    if best_eval_episode != previous_best_episode:
                        best_path = os.path.join(run_checkpoint_dir, TOP_CHECKPOINT_FILENAMES[0])
                        print(
                            f"  New best aggregate win rate: {best_eval_wr:.2f}% -> saved {best_path}",
                            flush=True,
                        )

    print("Training complete.")
    if best_eval_episode > 0:
        print(
            f"Best eval average win rate: {
                best_eval_wr:.2f}% at episode {best_eval_episode}",
            flush=True,
        )
        for name, wr in best_eval_by_opp:
            print(f"  best vs {name:<35} {wr:>6.2f}%", flush=True)
        print("Top checkpoints:", flush=True)
        for rank, ckpt in enumerate(top_checkpoints, start=1):
            print(
                f"  #{rank}: episode {int(ckpt['episode'])} | win_rate={float(ckpt['win_rate']):.2f}% | {ckpt['path']}",
                flush=True,
            )
    else:
        print(
            "No evaluation was run, so no best eval win rate is available.",
            flush=True,
        )
    
    # Save logs to JSON file (already written periodically, but ensure final save)
    write_logs_to_file(logs, run_checkpoint_dir)
    print("Training logs saved periodically to training_logs.json", flush=True)


if __name__ == "__main__":
    main()
