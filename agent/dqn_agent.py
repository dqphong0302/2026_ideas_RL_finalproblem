"""
DQN Agent Implementation
Agent Deep Q-Network cho Microgrid Optimization

Mô tả thuật toán DQN:
1. Sử dụng neural network để approximate Q(s, a)
2. Experience Replay để decorrelate samples
3. Target Network để stable training
4. Epsilon-greedy exploration strategy

Công thức cập nhật:
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q_target(s', a') - Q(s, a)]
    
Loss function:
    L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))²]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import copy

from .networks import QNetwork, DuelingQNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent
    
    Triển khai đầy đủ DQN với:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network
    - Gradient clipping
    
    Args:
        state_dim: Số chiều của state
        action_dim: Số actions có thể
        hidden_sizes: Kích thước hidden layers
        learning_rate: Tốc độ học
        gamma: Discount factor
        epsilon_start: Epsilon ban đầu (exploration)
        epsilon_end: Epsilon tối thiểu
        epsilon_decay: Tốc độ giảm epsilon
        buffer_size: Kích thước replay buffer
        batch_size: Batch size cho training
        target_update_freq: Tần suất update target network
        device: CPU hoặc CUDA
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_dueling: bool = False,
        device: str = "auto",
    ):
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        
        self.q_network = NetworkClass(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network = NetworkClass(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network không cần gradient
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training stats
        self.training_step = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Chọn action sử dụng epsilon-greedy policy
        
        Epsilon-greedy:
        - Với xác suất epsilon: chọn action ngẫu nhiên (explore)
        - Với xác suất (1 - epsilon): chọn action có Q cao nhất (exploit)
        
        Args:
            state: Current state observation
            training: Nếu True, sử dụng epsilon-greedy. Nếu False, chỉ exploit
        
        Returns:
            action: Integer action index
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_dim)
        
        # Exploitation: best action theo Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Lưu transition vào replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Thực hiện một bước training
        
        DQN Training Step:
        1. Sample mini-batch từ replay buffer
        2. Tính Q(s, a) từ q_network
        3. Tính target: r + γ * max_a' Q_target(s', a')
        4. Tính loss và backpropagate
        5. Update target network nếu đến lúc
        
        Returns:
            loss: Training loss (None nếu buffer chưa đủ samples)
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss (Huber loss / Smooth L1 - more robust than MSE)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def _update_target_network(self):
        """
        Soft update target network
        
        Hard update: copy toàn bộ weights
        θ_target = θ_q
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """
        Giảm epsilon sau mỗi episode
        
        Epsilon decay cho phép agent:
        - Explore nhiều ở đầu (high epsilon)
        - Exploit nhiều hơn về sau (low epsilon)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "losses": self.losses[-1000:],  # Keep last 1000 losses
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.losses = checkpoint.get("losses", [])
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions at a state (for visualization)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent
    
    Cải tiến so với DQN thường:
    - Sử dụng q_network để CHỌN action
    - Sử dụng target_network để ĐÁNH GIÁ action đó
    
    Công thức:
        target = r + γ * Q_target(s', argmax_a' Q(s', a'))
    
    Ưu điểm: Giảm overestimation của Q-values
    """
    
    def update(self) -> Optional[float]:
        """
        Double DQN update với action selection từ q_network
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Select action using q_network, evaluate using target_network
        with torch.no_grad():
            # Select best action with online network
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Evaluate with target network
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
