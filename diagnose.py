"""Quick diagnostic to identify training instability root cause."""

import random
import numpy as np
import torch

from computer import RandomStrategy, DiscardIncreaseStrategy
from dqn_agent import DQNAgent
from rl_env import NUM_ACTIONS, ThirtyOneEnv


def diagnose(num_episodes: int = 10, seed: int = 42) -> None:
    """Run short training and log detailed diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = DQNAgent(obs_dim=168, num_actions=NUM_ACTIONS, seed=seed)
    opponent_cls = RandomStrategy
    
    invalid_count = 0
    no_mask_count = 0
    total_steps = 0
    
    for ep in range(1, num_episodes + 1):
        env = ThirtyOneEnv(opponent_factory=opponent_cls, seed=seed + ep)
        obs, info = env.reset(start_player=ep % 2)
        done = False
        episode_reward = 0.0
        steps_in_episode = 0
        
        while not done:
            total_steps += 1
            steps_in_episode += 1
            
            action_mask = info["action_mask"]
            if not action_mask.any():
                print(f"[Ep {ep}, Step {steps_in_episode}] ERROR: No legal actions in mask!")
                no_mask_count += 1
            
            action = agent.select_action(obs, action_mask, epsilon=0.1)
            result = env.step(action)
            
            was_invalid = result.info.get("invalid_action", False)
            if was_invalid:
                invalid_count += 1
                print(f"[Ep {ep}, Step {steps_in_episode}] INVALID ACTION: {action}, reward={result.reward:.3f}")
            
            obs = result.observation
            info = result.info
            done = result.terminated or result.truncated
            episode_reward += result.reward
        
        print(f"Episode {ep:3d} | steps={steps_in_episode:3d} | total_reward={episode_reward:.2f}")
    
    print("\n" + "="*60)
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Invalid actions: {invalid_count} ({100*invalid_count/max(1, total_steps):.1f}%)")
    print(f"No-legal-action errors: {no_mask_count}")
    print("="*60)


if __name__ == "__main__":
    diagnose(num_episodes=20, seed=42)
