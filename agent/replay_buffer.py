"""
Experience Replay Buffer for DQN
Bộ nhớ trải nghiệm cho DQN

Mô tả:
Experience Replay là kỹ thuật quan trọng trong DQN:
1. Lưu trữ các transitions (s, a, r, s', done) vào buffer
2. Sample ngẫu nhiên mini-batches để training
3. Phá vỡ correlation giữa các consecutive samples
4. Tăng sample efficiency - reuse data nhiều lần
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, List, NamedTuple
import torch


class Transition(NamedTuple):
    """
    Một transition trong RL
    
    Attributes:
        state: Trạng thái hiện tại s
        action: Hành động được chọn a
        reward: Phần thưởng nhận được r
        next_state: Trạng thái tiếp theo s'
        done: Cờ kết thúc episode
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Replay Buffer sử dụng deque để efficient memory management
    
    Khi buffer đầy, các transitions cũ nhất sẽ bị xóa tự động
    
    Args:
        capacity: Số transitions tối đa lưu trữ
    
    Giải thích tại sao cần Replay Buffer:
    1. **Decorrelation**: Các samples liên tiếp có correlation cao
       -> Training không ổn định. Random sampling giải quyết vấn đề này.
    2. **Sample Efficiency**: Mỗi experience có thể được sử dụng nhiều lần
    3. **Stable Learning**: Gradients ổn định hơn với diverse batches
    """
    
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Thêm một transition vào buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample random mini-batch từ buffer
        
        Args:
            batch_size: Số transitions cần sample
        
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first for efficiency
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        
        # Convert to PyTorch tensors
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER)
    
    Ưu tiên sample những transitions có TD-error cao
    -> Học từ những experiences "surprising" hơn
    
    Công thức priority: p_i = |delta_i| + epsilon
    Xác suất sample: P(i) = p_i^alpha / sum(p_j^alpha)
    
    Note: Đây là implementation đơn giản, không tối ưu cho production
    Recommended: SumTree data structure cho O(log n) sampling
    
    Không bắt buộc cho assignment, nhưng là extension tốt
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Importance sampling weight
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,  # Small constant to avoid zero priority
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = None
    ):
        """Add transition with priority"""
        transition = Transition(state, action, reward, next_state, done)
        
        # New transitions get max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        priority = td_error if td_error else max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with prioritization"""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in buffer")
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states = torch.FloatTensor([t.state for t in transitions])
        actions = torch.LongTensor([t.action for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        next_states = torch.FloatTensor([t.next_state for t in transitions])
        dones = torch.FloatTensor([t.done for t in transitions])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
