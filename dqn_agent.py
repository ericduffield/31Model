"""DQN agent implementation for 31."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class Transition:
    """Single replay transition."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    """Simple FIFO replay buffer."""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return self.rng.sample(self.storage, batch_size)

    def __len__(self) -> int:
        return len(self.storage)


class QNetwork(nn.Module):
    """Small MLP for Q-value estimation."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN with target network and action masking."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 128,
        buffer_capacity: int = 100_000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_q = QNetwork(obs_dim, num_actions, hidden_size=hidden_size).to(self.device)
        self.target_q = QNetwork(obs_dim, num_actions, hidden_size=hidden_size).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.target_q.eval()

        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, epsilon: float, debug: bool = False) -> int:
        """Epsilon-greedy action selection with legal-action masking."""
        legal_actions = np.flatnonzero(action_mask)
        if len(legal_actions) == 0:
            if debug:
                print("[WARNING] No legal actions available!")
            return 0

        if random.random() < epsilon:
            return int(random.choice(legal_actions.tolist()))

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_q(obs_tensor).squeeze(0).cpu().numpy()

        masked_q = np.full_like(q_values, -1e9)
        masked_q[action_mask] = q_values[action_mask]
        best_action = int(np.argmax(masked_q))
        
        if debug:
            print(f"Q-values (masked): {masked_q}")
            print(f"Selected action: {best_action}")
        
        return best_action

    def add_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.replay.add(
            Transition(
                obs=obs.copy(),
                action=action,
                reward=reward,
                next_obs=next_obs.copy(),
                done=done,
                action_mask=action_mask.copy(),
                next_action_mask=next_action_mask.copy(),
            )
        )

    def train_step(self, batch_size: int) -> Optional[float]:
        """Run one gradient step if replay has enough samples."""
        if len(self.replay) < batch_size:
            return None

        batch = self.replay.sample(batch_size)

        obs = torch.as_tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        next_masks = torch.as_tensor(np.stack([t.next_action_mask for t in batch]), dtype=torch.bool, device=self.device)

        q_values = self.online_q(obs).gather(1, actions).squeeze(1)

        with torch.no_grad():
            has_legal = next_masks.any(dim=1)

            # Standard DQN target.
            next_target_q = self.target_q(next_obs)
            masked_next_q = torch.where(
                next_masks, next_target_q, torch.full_like(next_target_q, -1e9)
            )
            max_raw = masked_next_q.max(dim=1).values
            max_next_q = torch.where(has_legal, max_raw, torch.zeros_like(max_raw))

            target = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_q.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_q.load_state_dict(self.online_q.state_dict())

    def q_diagnostics(self, sample_size: int = 256) -> Optional[Tuple[float, float]]:
        """Return (mean_max_q, mean_abs_q) over a replay sample for monitoring."""
        if len(self.replay) == 0:
            return None

        size = min(sample_size, len(self.replay))
        batch = self.replay.sample(size)
        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            q_values = self.online_q(obs)
            mean_max_q = q_values.max(dim=1).values.mean().item()
            mean_abs_q = q_values.abs().mean().item()

        return float(mean_max_q), float(mean_abs_q)

    def save(self, path: str) -> None:
        payload = {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "online_state_dict": self.online_q.state_dict(),
            "target_state_dict": self.target_q.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.online_q.load_state_dict(payload["online_state_dict"])
        self.target_q.load_state_dict(payload["target_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
