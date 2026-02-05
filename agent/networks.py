"""
Neural Network Architectures for DQN
Kiến trúc mạng neural cho Deep Q-Network

Mô tả:
- QNetwork: Mạng chính để ước lượng Q-values
- Sử dụng fully connected layers với ReLU activation
- Output layer không có activation (Q-values có thể âm hoặc dương)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QNetwork(nn.Module):
    """
    Deep Q-Network để ước lượng Q(s, a)
    
    Architecture:
        Input (state_dim) -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU -> Output (action_dim)
    
    Giải thích:
    - Dùng 3 hidden layers để capture complex patterns
    - ReLU activation cho non-linearity
    - Không dùng activation ở output vì Q-values có thể có giá trị bất kỳ
    
    Args:
        state_dim: Số chiều của state space
        action_dim: Số actions có thể (discrete)
        hidden_sizes: Tuple các hidden layer sizes
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build layers dynamically
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights sử dụng Xavier/Glorot initialization
        Giúp training ổn định hơn
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass để tính Q-values
        
        Args:
            state: Tensor shape (batch_size, state_dim)
        
        Returns:
            Q-values: Tensor shape (batch_size, action_dim)
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get best action for a single state (greedy policy)
        
        Args:
            state: Tensor shape (state_dim,) hoặc (1, state_dim)
        
        Returns:
            action: Integer action với Q-value cao nhất
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Architecture (optional, more advanced)
    
    Tách Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    
    Ưu điểm:
    - Học Value function và Advantage function riêng biệt
    - Hiệu quả hơn cho states mà actions không quan trọng
    
    Không bắt buộc cho assignment này, nhưng là extension tốt
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        
        # Value stream - V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
        )
        
        # Advantage stream - A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], action_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass với Dueling architecture
        
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        
        # Combine using mean subtraction for stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: torch.Tensor) -> int:
        """Get best action for a single state"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
