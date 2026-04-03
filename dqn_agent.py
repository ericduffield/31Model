"""
Deep Q-Network (DQN) agent implementation for the card game 31.

Includes the neural network (QNetwork), experience replay buffer (ReplayBuffer),
and the full DQN agent (DQNAgent) with target networks, epsilon-greedy exploration,
and action masking for safe play.
"""

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
    """
    A single transition tuple for experience replay.

    Stores a state-action-reward-nextstate tuple along with action masks
    to enable safe action selection during learning.

    Attributes:
        obs (np.ndarray): The observation (state) before action.
        action (int): The action taken (0 to num_actions-1).
        reward (float): The immediate reward received.
        next_obs (np.ndarray): The observation (state) after action.
        done (bool): Whether the episode terminated.
        action_mask (np.ndarray): Boolean mask for legal actions in current state.
        next_action_mask (np.ndarray): Boolean mask for legal actions in next state.
    """

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray


class ReplayBuffer:
    """
    Fixed-size FIFO buffer for storing and sampling experience transitions.

    Manages a circular queue of transitions used for batch learning in DQN.
    When capacity is exceeded, old transitions are discarded. This enables
    training on diverse past experiences rather than just recent ones.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        storage (Deque[Transition]): Circular buffer of transitions.
        rng (random.Random): Random number generator for reproducible sampling.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to keep.
            seed (Optional[int]): Seed for the random sampler (for reproducibility).

        Example:
            >>> buffer = ReplayBuffer(capacity=10000, seed=42)
            >>> len(buffer)
            0
        """
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        """
        Add a transition to the buffer.

        If the buffer is at capacity, the oldest transition is removed.

        Args:
            transition (Transition): The transition to store.

        Example:
            >>> buffer = ReplayBuffer(capacity=100)
            >>> obs = np.zeros(10)
            >>> trans = Transition(obs, 1, 1.0, obs, False, np.array([1,0]), np.array([1,1]))
            >>> buffer.add(trans)
            >>> len(buffer)
            1
        """
        self.storage.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            List[Transition]: A list of randomly sampled transitions.

        Raises:
            ValueError: If batch_size > len(buffer).

        Example:
            >>> buffer = ReplayBuffer(capacity=100, seed=42)
            >>> [buffer.add(Transition(...)) for _ in range(50)]  # Add 50 transitions
            >>> batch = buffer.sample(10)
            >>> len(batch)
            10
        """
        return self.rng.sample(self.storage, batch_size)

    def __len__(self) -> int:
        """
        Return the current number of transitions in the buffer.

        Returns:
            int: Number of stored transitions (0 to capacity).
        """
        return len(self.storage)


class QNetwork(nn.Module):
    """
    Multi-layer perceptron (MLP) for estimating Q-values.

    Input: observation vector -> Hidden layers with ReLU -> Output: Q-values for each action.
    Used by DQN to approximate the optimal action-value function.

    Architecture:
        obs_dim -> hidden_size -> hidden_size -> num_actions
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 128) -> None:
        """
        Initialize the Q-value network.

        Args:
            obs_dim (int): Dimension of the observation/state vector.
            num_actions (int): Number of possible actions.
            hidden_size (int): Size of hidden layers (default 128).

        Example:
            >>> net = QNetwork(obs_dim=165, num_actions=9, hidden_size=128)
            >>> obs = torch.randn(1, 165)
            >>> q_values = net(obs)
            >>> q_values.shape
            torch.Size([1, 9])
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for the given observations.

        Args:
            x (torch.Tensor): Batch of observations, shape (batch_size, obs_dim).

        Returns:
            torch.Tensor: Q-values, shape (batch_size, num_actions).

        Example:
            >>> net = QNetwork(obs_dim=10, num_actions=4)
            >>> x = torch.randn(32, 10)  # 32-sample batch
            >>> q = net(x)
            >>> q.shape
            torch.Size([32, 4])
        """
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target networks.

    Learns an optimal policy for 31 by maintaining two networks:
    - online_q: Updated on every training step.
    - target_q: Updated periodically; used for stable value targets.

    Supports action masking to enforce legal moves and epsilon-greedy exploration.

    Attributes:
        obs_dim (int): Observation space dimensionality.
        num_actions (int): Number of possible actions.
        gamma (float): Discount factor for future rewards.
        online_q (QNetwork): The primary (learning) Q-network.
        target_q (QNetwork): The target Q-network (for stability).
        optimizer (torch.optim.Adam): Optimizer for the online network.
        replay (ReplayBuffer): Experience replay buffer.
        device (torch.device): Device (CPU or GPU) for computation.
    """

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
        """
        Initialize the DQN agent.

        Args:
            obs_dim (int): Observation space dimension.
            num_actions (int): Number of discrete actions.
            learning_rate (float): Adam optimizer learning rate (default 1e-3).
            gamma (float): Discount factor (default 0.99).
            hidden_size (int): Hidden layer size for Q-networks (default 128).
            buffer_capacity (int): Replay buffer capacity (default 100,000).
            seed (Optional[int]): Random seed for reproducibility.
            device (Optional[str]): Torch device ("cpu", "cuda", or None for auto).

        Example:
            >>> agent = DQNAgent(
            ...     obs_dim=165,
            ...     num_actions=9,
            ...     learning_rate=1e-4,
            ...     gamma=0.95,
            ...     buffer_capacity=100000,
            ...     seed=42
            ... )
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online_q = QNetwork(obs_dim, num_actions, hidden_size=hidden_size).to(
            self.device
        )
        self.target_q = QNetwork(obs_dim, num_actions, hidden_size=hidden_size).to(
            self.device
        )
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.target_q.eval()

        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(capacity=buffer_capacity, seed=seed)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
        debug: bool = False,
    ) -> int:
        """
        Select an action using epsilon-greedy exploration with action masking.

        With probability epsilon, selects a uniformly random legal action.
        Otherwise, selects the legal action with the highest Q-value.

        Args:
            obs (np.ndarray): Current observation, shape (obs_dim,).
            action_mask (np.ndarray): Boolean mask of legal actions, shape (num_actions,).
            epsilon (float): Exploration probability (0 to 1).
            debug (bool): If True, print debug information (default False).

        Returns:
            int: Index of the selected action (0 to num_actions-1).

        Raises:
            IndexError: If action_mask has no True values (no legal actions).

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4, seed=42)
            >>> obs = np.random.randn(10)
            >>> mask = np.array([True, True, False, True])
            >>> action = agent.select_action(obs, mask, epsilon=0.1)
            >>> action in [0, 1, 3]  # Legal actions only
            True
        """
        legal_actions = np.flatnonzero(action_mask)
        if len(legal_actions) == 0:
            if debug:
                print("[WARNING] No legal actions available!")
            return 0

        if random.random() < epsilon:
            return int(random.choice(legal_actions.tolist()))

        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
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
        """
        Store a transition in the replay buffer.

        Args:
            obs (np.ndarray): The observation before the action.
            action (int): The action taken.
            reward (float): The immediate reward.
            next_obs (np.ndarray): The observation after the action.
            done (bool): Whether the episode ended.
            action_mask (np.ndarray): Legal action mask in current state.
            next_action_mask (np.ndarray): Legal action mask in next state.

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4)
            >>> obs = np.random.randn(10)
            >>> agent.add_transition(obs, 2, 1.0, obs, False, np.ones(4, dtype=bool), np.ones(4, dtype=bool))
            >>> len(agent.replay)
            1
        """
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
        """
        Perform one optimization step using a batch of transitions.

        Samples transitions from replay buffer, computes DQN loss using the
        Bellman target equation, and updates the online network weights.
        Action masking is applied to prevent learning on illegal actions.

        Args:
            batch_size (int): Number of transitions to sample per step.

        Returns:
            Optional[float]: The loss value if training occurred, None if buffer
                has fewer than batch_size transitions.

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4)
            >>> # ... add transitions to replay buffer ...
            >>> loss = agent.train_step(batch_size=32)
            >>> if loss is not None:
            ...     print(f"Loss: {loss:.4f}")
        """
        if len(self.replay) < batch_size:
            return None

        batch = self.replay.sample(batch_size)

        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_obs = torch.as_tensor(
            np.stack([t.next_obs for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        next_masks = torch.as_tensor(
            np.stack([t.next_action_mask for t in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        # Current Q-values
        q_values = self.online_q(obs).gather(1, actions).squeeze(1)

        with torch.no_grad():
            has_legal = next_masks.any(dim=1)

            # Masked target Q-values: set illegal actions to very negative
            # value
            next_target_q = self.target_q(next_obs)
            masked_next_q = torch.where(
                next_masks, next_target_q, torch.full_like(next_target_q, -1e9)
            )
            max_raw = masked_next_q.max(dim=1).values
            # If no legal next actions, max_next_q = 0; else use max Q
            max_next_q = torch.where(has_legal, max_raw, torch.zeros_like(max_raw))

            # Bellman target: r + γ * (1 - done) * max Q'(s', a')
            target = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_q.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        """
        Update the target network to match the online network.

        Called periodically (e.g., every 500 steps) to stabilize learning.
        Copies all weights from online_q to target_q.

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4)
            >>> # ... training updates online_q ...
            >>> agent.update_target()  # Synchronize target network
        """
        self.target_q.load_state_dict(self.online_q.state_dict())

    def q_diagnostics(self, sample_size: int = 256) -> Optional[Tuple[float, float]]:
        """
        Compute diagnostic statistics over Q-values.

        Samples transitions from replay buffer and computes the average
        of the maximum Q-values (exploration potential) and the average
        absolute Q-value (magnitude). Useful for monitoring training health.

        Args:
            sample_size (int): Number of samples to use for diagnostics (default 256).

        Returns:
            Optional[Tuple[float, float]]: A tuple (mean_max_q, mean_abs_q), or
                None if the replay buffer is empty.
                - mean_max_q: Average of max Q-values across sampled states.
                - mean_abs_q: Average of absolute Q-values.

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4)
            >>> # ... add transitions and train ...
            >>> diag = agent.q_diagnostics(sample_size=128)
            >>> if diag:
            ...     mean_max_q, mean_abs_q = diag
            ...     print(f"Mean max Q: {mean_max_q:.3f}, Mean abs Q: {mean_abs_q:.3f}")
        """
        if len(self.replay) == 0:
            return None

        size = min(sample_size, len(self.replay))
        batch = self.replay.sample(size)
        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            q_values = self.online_q(obs)
            mean_max_q = q_values.max(dim=1).values.mean().item()
            mean_abs_q = q_values.abs().mean().item()

        return float(mean_max_q), float(mean_abs_q)

    def save(self, path: str) -> None:
        """
        Save the agent's networks and optimizer state to disk.

        Stores the online and target Q-networks and the optimizer state,
        enabling training continuation or deployment.

        Args:
            path (str): File path to save the checkpoint (typically a .pt file).

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4)
            >>> # ... train ...
            >>> agent.save("checkpoints/agent_best.pt")
        """
        payload = {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "online_state_dict": self.online_q.state_dict(),
            "target_state_dict": self.target_q.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """
        Load the agent's networks and optimizer state from disk.

        Restores an agent from a saved checkpoint.

        Args:
            path (str): File path of the checkpoint to load.

        Example:
            >>> agent = DQNAgent(obs_dim=10, num_actions=4)
            >>> agent.load("checkpoints/agent_best.pt")
        """
        payload = torch.load(path, map_location=self.device)
        self.online_q.load_state_dict(payload["online_state_dict"])
        self.target_q.load_state_dict(payload["target_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
