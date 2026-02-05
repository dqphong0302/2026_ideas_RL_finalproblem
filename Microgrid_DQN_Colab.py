# -*- coding: utf-8 -*-
"""
================================================================================
🔋 MICROGRID ENERGY OPTIMIZATION USING DEEP REINFORCEMENT LEARNING
================================================================================
Google Colab Standalone Script - Run All Cells to Train & Evaluate DQN Agent

This notebook contains:
1. Environment setup and installations
2. Microgrid Environment (Gymnasium-compatible)
3. DQN Agent with Experience Replay and Target Network
4. Training Loop with Logging
5. Evaluation and Visualization
6. Full Execution Pipeline

Author: Deep RL Assignment
Date: February 2026

================================================================================
📝 HƯỚNG DẪN CHO SINH VIÊN (STUDENT CUSTOMIZATION GUIDE)
================================================================================
File này là BÀI MẪU cho 5-7 sinh viên tham khảo.
Để BÀI LÀM của mỗi người KHÁC NHAU, hãy thay đổi các giá trị được đánh dấu:
    🔧 [CUSTOMIZABLE] - Có thể thay đổi số liệu
    ⚠️ [REQUIRED CHANGE] - BẮT BUỘC thay đổi để tránh trùng lặp

Các gợi ý thay đổi:
- Thay đổi hyperparameters (learning rate, batch size, hidden layers)
- Thay đổi cấu trúc mạng neural (số layers, neurons)
- Thay đổi reward weights
- Thay đổi seed để có kết quả khác
- Thay đổi số episodes training
================================================================================
"""

#@title 1️⃣ Install Dependencies & Imports
!pip install gymnasium torch numpy matplotlib -q

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

#@title 2️⃣ Configuration Parameters
"""
================================================================================
⚙️ HYPERPARAMETERS - CẤU HÌNH THAM SỐ
================================================================================
🔧 [CUSTOMIZABLE] - Thay đổi các giá trị này để tạo bài làm khác biệt!

GỢI Ý THAY ĐỔI CHO MỖI SINH VIÊN:
- Student 1: seed=42, lr=1e-4, hidden=[256,256,128], episodes=500
- Student 2: seed=123, lr=5e-4, hidden=[512,256], episodes=600
- Student 3: seed=2024, lr=2e-4, hidden=[128,128,64], episodes=400
- Student 4: seed=999, lr=1e-3, hidden=[256,128,64], episodes=550
- Student 5: seed=7777, lr=3e-4, hidden=[384,192,96], episodes=450
================================================================================
"""

CONFIG = {
    # =========================================================================
    # 🔋 ENVIRONMENT PARAMETERS - Tham số môi trường Microgrid
    # =========================================================================
    
    # 🔧 [CUSTOMIZABLE] Battery capacity in kWh
    # Gợi ý: 80-150 kWh (thay đổi ±20% so với 100)
    "battery_capacity": 100.0,
    
    # 🔧 [CUSTOMIZABLE] Battery round-trip efficiency
    # Gợi ý: 0.90-0.98 (pin lithium thường 0.92-0.96)
    "battery_efficiency": 0.95,
    
    # 🔧 [CUSTOMIZABLE] Maximum charge/discharge rate in kW
    # Gợi ý: 15-30 kW
    "max_charge_rate": 20.0,
    "max_discharge_rate": 20.0,
    
    # 🔧 [CUSTOMIZABLE] Renewable energy capacity
    # Gợi ý: Solar 40-70 kW, Wind 20-50 kW
    "max_solar": 50.0,
    "max_wind": 30.0,
    
    # 🔧 [CUSTOMIZABLE] Consumer demand parameters
    # Gợi ý: base 30-60 kW, std 5-15 kW
    "base_demand": 40.0,
    "demand_std": 10.0,
    
    # 🔧 [CUSTOMIZABLE] Grid electricity price ($/kWh)
    # Gợi ý: min 0.03-0.08, max 0.20-0.35
    "grid_price_min": 0.05,
    "grid_price_max": 0.25,
    
    # Hours per episode (1 day = 24 hours)
    "hours_per_episode": 24,
    
    # =========================================================================
    # 🧠 DQN PARAMETERS - Tham số thuật toán DQN
    # =========================================================================
    
    # State and Action dimensions (KHÔNG THAY ĐỔI)
    "state_dim": 8,
    "action_dim": 5,
    
    # ⚠️ [REQUIRED CHANGE] Neural network architecture
    # Mỗi sinh viên PHẢI thay đổi cấu trúc này!
    # Gợi ý: [128,128,64], [256,128], [512,256,128], [384,192,96]
    "hidden_dims": [256, 256, 128],
    
    # ⚠️ [REQUIRED CHANGE] Learning rate
    # Gợi ý: 1e-4, 2e-4, 5e-4, 1e-3, 3e-4
    "learning_rate": 1e-4,
    
    # 🔧 [CUSTOMIZABLE] Discount factor (gamma)
    # Gợi ý: 0.95-0.99
    "gamma": 0.99,
    
    # 🔧 [CUSTOMIZABLE] Epsilon-greedy exploration
    # Gợi ý: start 0.9-1.0, end 0.01-0.05, decay 0.990-0.998
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    
    # 🔧 [CUSTOMIZABLE] Batch size for training
    # Gợi ý: 32, 64, 128, 256
    "batch_size": 64,
    
    # 🔧 [CUSTOMIZABLE] Replay buffer size
    # Gợi ý: 50000-200000
    "buffer_size": 100000,
    
    # 🔧 [CUSTOMIZABLE] Target network update frequency
    # Gợi ý: 500-2000 steps
    "target_update_freq": 1000,
    
    # =========================================================================
    # 📊 TRAINING PARAMETERS - Tham số huấn luyện
    # =========================================================================
    
    # ⚠️ [REQUIRED CHANGE] Number of training episodes
    # Mỗi sinh viên nên dùng số khác nhau: 400-700
    "num_episodes": 500,
    
    # Steps per episode (= hours per day)
    "max_steps_per_episode": 24,
    
    # Checkpoint save frequency
    "save_freq": 100,
    
    # Logging frequency
    "log_freq": 10,
    
    # ⚠️ [REQUIRED CHANGE] Random seed - MỖI SINH VIÊN PHẢI KHÁC!
    # Gợi ý: 42, 123, 456, 789, 999, 2024, 7777
    "seed": 42,
    
    # =========================================================================
    # 🎯 REWARD SHAPING - Trọng số phần thưởng
    # =========================================================================
    # 🔧 [CUSTOMIZABLE] Reward weights - Thay đổi để ưu tiên mục tiêu khác nhau
    
    # Positive reward for using renewable energy
    # Gợi ý: 0.5-2.0
    "reward_renewable": 1.0,
    
    # Negative penalty for grid electricity purchase
    # Gợi ý: -1.0 đến -3.0
    "reward_grid_penalty": -2.0,
    
    # Heavy penalty for unmet demand
    # Gợi ý: -3.0 đến -10.0
    "reward_unmet_penalty": -5.0,
    
    # Small penalty for battery wear (cycling)
    # Gợi ý: -0.05 đến -0.2
    "reward_battery_wear": -0.1,
    
    # Bonus for avoiding grid during peak hours
    # Gợi ý: 0.3-1.0
    "reward_peak_bonus": 0.5,
    
    # =========================================================================
    # 🛑 EPISODE TERMINATION CONDITIONS - Điều kiện kết thúc episode
    # =========================================================================
    # 🔧 [CUSTOMIZABLE] Thêm điều kiện kết thúc sớm
    
    # Critical low battery threshold (fraction of capacity)
    # Nếu pin < threshold, episode kết thúc. Gợi ý: 0.05-0.15
    "battery_critical_low": 0.05,
    
    # Critical high battery threshold (fraction of capacity)
    # Nếu pin > threshold, kết thúc. Gợi ý: 0.95-1.0
    "battery_critical_high": 1.0,
    
    # Maximum cumulative unmet demand ratio before termination
    # Gợi ý: 0.15-0.30
    "max_unmet_ratio": 0.20,
}

print("✅ Configuration loaded!")
print(f"   Episodes: {CONFIG['num_episodes']}")
print(f"   Learning Rate: {CONFIG['learning_rate']}")
print(f"   Gamma: {CONFIG['gamma']}")
print(f"   Hidden Layers: {CONFIG['hidden_dims']}")
print(f"   Seed: {CONFIG['seed']}")

#@title 3️⃣ Microgrid Environment
"""
================================================================================
🔋 MICROGRID ENVIRONMENT - Môi trường mô phỏng lưới điện siêu nhỏ
================================================================================

Gymnasium-compatible environment simulating:
- Solar and wind renewable generation (phát điện tái tạo)
- Battery storage with efficiency losses (lưu trữ pin với tổn hao)
- Grid connection with time-varying prices (kết nối lưới với giá biến đổi)
- Consumer demand with stochastic variations (nhu cầu ngẫu nhiên)

State Space (8D continuous):
    [battery_level, demand, solar, wind, grid_price, hour_sin, hour_cos, prev_action]

Action Space (5 discrete):
    0: Discharge battery (Xả pin)
    1: Charge from renewable (Sạc từ năng lượng tái tạo)
    2: Buy from grid (Mua từ lưới điện)
    3: Renewable + Discharge (Kết hợp tái tạo + xả pin)
    4: Renewable + Grid (Kết hợp tái tạo + lưới)
================================================================================
"""

class MicrogridEnv:
    """
    Microgrid Energy Management Environment
    
    Môi trường quản lý năng lượng lưới điện siêu nhỏ, mô phỏng:
    - Phát điện mặt trời (solar) theo thời gian trong ngày
    - Phát điện gió (wind) với biến động ngẫu nhiên
    - Pin lưu trữ với hiệu suất round-trip
    - Nhu cầu tiêu thụ với peak sáng và tối
    - Giá điện lưới biến đổi theo giờ
    """
    
    def __init__(self, config: Dict):
        """
        Khởi tạo môi trường với cấu hình.
        
        Args:
            config: Dictionary chứa tất cả tham số cấu hình
        """
        self.config = config
        
        # 🔋 Battery parameters - Tham số pin
        self.battery_capacity = config["battery_capacity"]  # kWh
        self.battery_efficiency = config["battery_efficiency"]  # Hiệu suất
        self.max_charge = config["max_charge_rate"]  # kW - tốc độ sạc tối đa
        self.max_discharge = config["max_discharge_rate"]  # kW - tốc độ xả tối đa
        
        # ☀️ Renewable energy parameters - Tham số năng lượng tái tạo
        self.max_solar = config["max_solar"]  # kW peak solar
        self.max_wind = config["max_wind"]  # kW peak wind
        
        # 🏠 Demand parameters - Tham số nhu cầu tiêu thụ
        self.base_demand = config["base_demand"]  # kW trung bình
        self.demand_std = config["demand_std"]  # Độ lệch chuẩn
        
        # 💰 Grid price parameters - Tham số giá điện lưới
        self.grid_price_min = config["grid_price_min"]  # $/kWh off-peak
        self.grid_price_max = config["grid_price_max"]  # $/kWh peak
        
        # ⏰ Time parameters
        self.hours_per_episode = config["hours_per_episode"]
        
        # 🎯 Reward weights - Trọng số phần thưởng
        self.r_renewable = config["reward_renewable"]
        self.r_grid = config["reward_grid_penalty"]
        self.r_unmet = config["reward_unmet_penalty"]
        self.r_wear = config["reward_battery_wear"]
        self.r_bonus = config["reward_peak_bonus"]
        
        # 🛑 Termination conditions - Điều kiện kết thúc
        self.battery_critical_low = config.get("battery_critical_low", 0.05)
        self.battery_critical_high = config.get("battery_critical_high", 1.0)
        self.max_unmet_ratio = config.get("max_unmet_ratio", 0.20)
        
        # State tracking
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        Đặt lại môi trường về trạng thái ban đầu.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation vector (8D)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 🔧 [CUSTOMIZABLE] Initial battery level
        # Gợi ý: Thay đổi mức pin ban đầu (0.3-0.7 của capacity)
        self.battery_level = self.battery_capacity * 0.5  # Start at 50%
        self.current_hour = 0
        self.prev_action = 0
        
        # Episode tracking - Theo dõi episode
        self.total_demand = 0.0
        self.total_renewable_used = 0.0
        self.total_grid_cost = 0.0
        self.total_unmet = 0.0
        self.episode_history = []
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """
        Generate normalized observation vector.
        Tạo vector quan sát đã chuẩn hóa.
        
        Observation vector gồm 8 thành phần:
        - battery_level: Mức pin hiện tại [0,1]
        - demand: Nhu cầu tiêu thụ [0,1]
        - solar: Sản lượng điện mặt trời [0,1]
        - wind: Sản lượng điện gió [0,1]
        - grid_price: Giá điện lưới [0,1]
        - hour_sin: Sin của giờ (cyclic encoding)
        - hour_cos: Cos của giờ (cyclic encoding)
        - prev_action: Hành động trước đó [0,1]
        
        Returns:
            Normalized observation array (8D)
        """
        demand = self._get_demand(self.current_hour)
        solar = self._get_solar(self.current_hour)
        wind = self._get_wind(self.current_hour)
        price = self._get_price(self.current_hour)
        
        # Normalize all features to [0, 1]
        obs = np.array([
            self.battery_level / self.battery_capacity,
            demand / (self.base_demand * 2),
            solar / self.max_solar,
            wind / self.max_wind,
            (price - self.grid_price_min) / (self.grid_price_max - self.grid_price_min),
            (np.sin(2 * np.pi * self.current_hour / 24) + 1) / 2,
            (np.cos(2 * np.pi * self.current_hour / 24) + 1) / 2,
            self.prev_action / 4.0,
        ], dtype=np.float32)
        
        return np.clip(obs, 0.0, 1.0)
    
    def _get_demand(self, hour: int) -> float:
        """
        Generate stochastic demand with morning and evening peaks.
        Tạo nhu cầu ngẫu nhiên với đỉnh sáng và tối.
        
        🔧 [CUSTOMIZABLE] Có thể thay đổi:
        - Giờ peak sáng (mặc định: 8h)
        - Giờ peak tối (mặc định: 19h)
        - Cường độ peak (0.3 và 0.4)
        
        Args:
            hour: Current hour (0-23)
            
        Returns:
            Demand in kW
        """
        # 🔧 [CUSTOMIZABLE] Peak hours - Thay đổi giờ cao điểm
        morning_peak_hour = 8  # Gợi ý: 7-9
        evening_peak_hour = 19  # Gợi ý: 18-21
        
        morning_peak = np.exp(-((hour - morning_peak_hour) ** 2) / 8)
        evening_peak = np.exp(-((hour - evening_peak_hour) ** 2) / 8)
        
        # 🔧 [CUSTOMIZABLE] Peak intensity - Thay đổi cường độ peak
        base = self.base_demand * (0.5 + 0.3 * morning_peak + 0.4 * evening_peak)
        noise = np.random.normal(0, self.demand_std * 0.3)
        
        return max(0, base + noise)
    
    def _get_solar(self, hour: int) -> float:
        """
        Generate solar power based on time of day.
        Tạo sản lượng điện mặt trời theo thời gian trong ngày.
        
        🔧 [CUSTOMIZABLE] Có thể thay đổi:
        - Giờ mặt trời mọc/lặn (mặc định: 6-18h)
        - Biến động thời tiết (0.8-1.2)
        
        Args:
            hour: Current hour (0-23)
            
        Returns:
            Solar power in kW
        """
        # 🔧 [CUSTOMIZABLE] Sunrise/sunset hours
        sunrise = 6  # Gợi ý: 5-7
        sunset = 18  # Gợi ý: 17-19
        
        if sunrise <= hour <= sunset:
            base = self.max_solar * np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))
            # 🔧 [CUSTOMIZABLE] Weather variability
            noise = np.random.uniform(0.8, 1.2)  # Gợi ý: 0.7-1.3
            return max(0, base * noise)
        return 0.0
    
    def _get_wind(self, hour: int) -> float:
        """
        Generate stochastic wind power.
        Tạo sản lượng điện gió ngẫu nhiên.
        
        🔧 [CUSTOMIZABLE] Có thể thay đổi:
        - Base wind level (0.5)
        - Variation amplitude
        - Noise range (0.5-1.5)
        
        Args:
            hour: Current hour (0-23)
            
        Returns:
            Wind power in kW
        """
        # 🔧 [CUSTOMIZABLE] Wind pattern
        base_level = 0.5  # Gợi ý: 0.3-0.6
        base = self.max_wind * base_level
        variation = self.max_wind * (1 - base_level) * np.sin(np.pi * hour / 12)
        
        # 🔧 [CUSTOMIZABLE] Wind variability
        noise = np.random.uniform(0.5, 1.5)  # Gợi ý: 0.4-1.6
        
        return max(0, (base + variation) * noise)
    
    def _get_price(self, hour: int) -> float:
        """
        Generate time-varying grid electricity price.
        Tạo giá điện lưới biến đổi theo thời gian.
        
        🔧 [CUSTOMIZABLE] Có thể thay đổi:
        - Peak hours (7-9, 18-21)
        - Off-peak hours (22-6)
        - Price variation (0.9-1.1)
        
        Args:
            hour: Current hour (0-23)
            
        Returns:
            Price in $/kWh
        """
        # 🔧 [CUSTOMIZABLE] Peak/off-peak hours
        if 7 <= hour <= 9 or 18 <= hour <= 21:  # Peak hours
            base = self.grid_price_max
        elif 22 <= hour or hour <= 6:  # Off-peak
            base = self.grid_price_min
        else:  # Mid-peak
            base = (self.grid_price_min + self.grid_price_max) / 2
        
        # 🔧 [CUSTOMIZABLE] Price variability
        noise = np.random.uniform(0.9, 1.1)  # Gợi ý: 0.85-1.15
        
        return base * noise
    
    def _check_termination(self) -> Tuple[bool, str]:
        """
        Check if episode should terminate early.
        Kiểm tra điều kiện kết thúc sớm episode.
        
        Returns:
            (should_terminate, reason)
        """
        # Check end of day
        if self.current_hour >= self.hours_per_episode:
            return True, "end_of_day"
        
        # Check critical battery level
        battery_ratio = self.battery_level / self.battery_capacity
        if battery_ratio < self.battery_critical_low:
            return True, "battery_critical_low"
        if battery_ratio > self.battery_critical_high:
            return True, "battery_critical_high"
        
        # Check cumulative unmet demand ratio
        if self.total_demand > 0:
            unmet_ratio = self.total_unmet / self.total_demand
            if unmet_ratio > self.max_unmet_ratio:
                return True, "max_unmet_exceeded"
        
        return False, ""
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step.
        Thực hiện một bước thời gian (1 giờ).
        
        Args:
            action: Integer action (0-4)
            
        Returns:
            (next_state, reward, done, info)
        """
        demand = self._get_demand(self.current_hour)
        solar = self._get_solar(self.current_hour)
        wind = self._get_wind(self.current_hour)
        price = self._get_price(self.current_hour)
        renewable = solar + wind
        
        # Initialize energy flows
        renewable_used = 0.0
        grid_purchased = 0.0
        battery_charge = 0.0
        battery_discharge = 0.0
        unmet_demand = 0.0
        
        # Process action - Xử lý hành động
        if action == 0:  # Discharge battery - Xả pin
            discharge = min(self.battery_level, self.max_discharge, demand)
            battery_discharge = discharge
            self.battery_level -= discharge
            remaining = demand - discharge * self.battery_efficiency
            if remaining > 0:
                unmet_demand = remaining
        
        elif action == 1:  # Charge from renewable - Sạc từ tái tạo
            supply = min(renewable, demand)
            renewable_used = supply
            remaining = demand - supply
            if remaining > 0:
                unmet_demand = remaining
            excess = renewable - supply
            if excess > 0:
                charge = min(excess, self.max_charge, 
                           self.battery_capacity - self.battery_level)
                battery_charge = charge
                self.battery_level += charge * self.battery_efficiency
        
        elif action == 2:  # Buy from grid - Mua từ lưới
            grid_purchased = demand
        
        elif action == 3:  # Renewable + Discharge - Tái tạo + Xả pin
            renewable_used = min(renewable, demand)
            remaining = demand - renewable_used
            if remaining > 0:
                discharge = min(self.battery_level, self.max_discharge, remaining)
                battery_discharge = discharge
                self.battery_level -= discharge
                remaining -= discharge * self.battery_efficiency
            if remaining > 0:
                unmet_demand = remaining
        
        elif action == 4:  # Renewable + Grid - Tái tạo + Lưới
            renewable_used = min(renewable, demand)
            remaining = demand - renewable_used
            if remaining > 0:
                grid_purchased = remaining
        
        # Calculate reward - Tính phần thưởng
        normalized_price = (price - self.grid_price_min) / (self.grid_price_max - self.grid_price_min)
        is_peak = 18 <= self.current_hour <= 21
        
        reward = (
            self.r_renewable * (renewable_used / self.base_demand) +
            self.r_grid * (grid_purchased / self.base_demand) * normalized_price +
            self.r_unmet * (unmet_demand / self.base_demand) +
            self.r_wear * ((battery_charge + battery_discharge) / self.max_charge)
        )
        
        # Bonus for avoiding grid during peak
        if is_peak and grid_purchased == 0:
            reward += self.r_bonus
        
        # Update tracking
        self.total_demand += demand
        self.total_renewable_used += renewable_used
        self.total_grid_cost += grid_purchased * price
        self.total_unmet += unmet_demand
        
        # Store history
        self.episode_history.append({
            "hour": self.current_hour,
            "demand": demand,
            "solar": solar,
            "wind": wind,
            "price": price,
            "action": action,
            "renewable_used": renewable_used,
            "grid_purchased": grid_purchased,
            "battery_level": self.battery_level,
            "reward": reward,
        })
        
        # Advance time
        self.current_hour += 1
        self.prev_action = action
        
        # Check termination
        done, termination_reason = self._check_termination()
        
        info = {
            "total_cost": self.total_grid_cost,
            "renewable_ratio": self.total_renewable_used / max(1, self.total_demand),
            "unmet_ratio": self.total_unmet / max(1, self.total_demand),
            "termination_reason": termination_reason,
        }
        
        return self._get_obs(), reward, done, info

print("✅ MicrogridEnv class defined!")

#@title 4️⃣ Neural Network & Replay Buffer
"""
================================================================================
🧠 DQN COMPONENTS - Các thành phần DQN
================================================================================
- Q-Network: Multi-layer perceptron với ReLU activations
- Replay Buffer: Uniform sampling for experience replay
================================================================================
"""

class QNetwork(nn.Module):
    """
    Deep Q-Network for value function approximation.
    Mạng Q sâu để xấp xỉ hàm giá trị.
    
    🔧 [CUSTOMIZABLE] Có thể thay đổi:
    - Số layers và neurons (trong CONFIG['hidden_dims'])
    - Dropout rate
    - Activation functions
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        Khởi tạo mạng Q.
        
        Args:
            state_dim: Dimension of state space (8)
            action_dim: Number of actions (5)
            hidden_dims: List of hidden layer sizes, e.g., [256, 256, 128]
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # 🔧 [CUSTOMIZABLE] Dropout rate
            # Gợi ý: 0.0-0.3
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """
        Xavier initialization for stable training.
        Khởi tạo Xavier để huấn luyện ổn định.
        
        🔧 [CUSTOMIZABLE] Có thể thay đổi phương pháp init:
        - xavier_uniform_
        - kaiming_uniform_
        - orthogonal_
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass qua mạng."""
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Bộ đệm phát lại kinh nghiệm cho huấn luyện DQN.
    
    Giúp phá vỡ correlation giữa các mẫu liên tiếp.
    """
    
    def __init__(self, capacity: int):
        """
        Khởi tạo buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Thêm transition vào buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample random batch from buffer.
        Lấy mẫu ngẫu nhiên từ buffer.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)

print("✅ QNetwork and ReplayBuffer defined!")

#@title 5️⃣ DQN Agent (with Double DQN)
"""
================================================================================
🤖 DOUBLE DQN AGENT - Agent sử dụng thuật toán Double DQN
================================================================================

Cải tiến so với vanilla DQN:
- Separate target network: Mạng target riêng để ổn định Q-values
- Double DQN: Giảm overestimation bias
- Epsilon-greedy exploration: Khám phá với epsilon decay

Reference paper:
- DQN: Mnih et al., 2015 "Human-level control through deep RL"
- Double DQN: Van Hasselt et al., 2016 "Deep RL with Double Q-learning"
================================================================================
"""

class DQNAgent:
    """
    Double DQN Agent for Microgrid Optimization.
    Agent Double DQN cho tối ưu hóa Microgrid.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Khởi tạo agent.
        
        Args:
            config: Configuration dictionary
            device: torch.device (cuda/cpu)
        """
        self.config = config
        self.device = device
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        
        # Epsilon parameters - Tham số epsilon cho exploration
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        
        # Networks - Mạng neural
        self.q_network = QNetwork(
            self.state_dim, 
            self.action_dim, 
            config["hidden_dims"]
        ).to(device)
        
        self.target_network = QNetwork(
            self.state_dim, 
            self.action_dim, 
            config["hidden_dims"]
        ).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 🔧 [CUSTOMIZABLE] Optimizer
        # Gợi ý thay đổi: SGD, RMSprop, AdamW
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config["learning_rate"]
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config["buffer_size"])
        
        # Training tracking
        self.training_step = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        Chọn hành động theo chiến lược epsilon-greedy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action (0-4)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.
        Lưu transition vào replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, float(done))
    
    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.
        Thực hiện một bước cập nhật gradient.
        
        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        # Double DQN: dùng mạng online chọn action, mạng target đánh giá
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 🔧 [CUSTOMIZABLE] Loss function
        # Gợi ý thay đổi: MSELoss, SmoothL1Loss (Huber)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 🔧 [CUSTOMIZABLE] Gradient clipping
        # Gợi ý: 1.0-20.0
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        return loss_value
    
    def decay_epsilon(self):
        """
        Decay exploration rate.
        Giảm tỷ lệ khám phá.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """
        Save model weights.
        Lưu trọng số model.
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
        }, path)
    
    def load(self, path: str):
        """
        Load model weights.
        Tải trọng số model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]

print("✅ DQNAgent class defined!")

#@title 6️⃣ Training Function
"""
================================================================================
🚀 TRAINING LOOP - Vòng lặp huấn luyện
================================================================================

Implements the main training procedure with:
- Episode-based learning
- Periodic logging
- Best model checkpointing
================================================================================
"""

def train(config: Dict, device: torch.device) -> Dict[str, List]:
    """
    Train the DQN agent.
    Huấn luyện agent DQN.
    
    Args:
        config: Configuration dictionary
        device: torch.device
        
    Returns:
        Training history dictionary
    """
    
    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment and agent
    env = MicrogridEnv(config)
    agent = DQNAgent(config, device)
    
    # Training history
    history = {
        "rewards": [],
        "costs": [],
        "renewable_ratios": [],
        "epsilons": [],
        "losses": [],
    }
    
    best_reward = float("-inf")
    
    # Training loop
    for episode in range(config["num_episodes"]):
        state = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_losses = []
        
        for step in range(config["max_steps_per_episode"]):
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store and learn
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Store history
        history["rewards"].append(episode_reward)
        history["costs"].append(info["total_cost"])
        history["renewable_ratios"].append(info["renewable_ratio"])
        history["epsilons"].append(agent.epsilon)
        if episode_losses:
            history["losses"].append(np.mean(episode_losses))
        
        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("best_model.pt")
        
        # Logging
        if (episode + 1) % config["log_freq"] == 0:
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Cost: ${info['total_cost']:6.2f} | "
                  f"Renewable: {info['renewable_ratio']*100:5.1f}% | "
                  f"ε: {agent.epsilon:.3f}")
    
    # Save final model
    agent.save("final_model.pt")
    
    print("=" * 60)
    print(f"✅ Training complete! Best reward: {best_reward:.2f}")
    print("=" * 60)
    
    return history, agent, env

#@title 7️⃣ Evaluation Functions
"""
================================================================================
📊 EVALUATION - Đánh giá model
================================================================================

Functions for:
- Testing trained agent
- Comparing with random baseline
- Generating visualizations
================================================================================
"""

def evaluate_agent(agent: DQNAgent, env: MicrogridEnv, 
                   num_episodes: int = 20, seed: int = 42) -> Dict:
    """
    Evaluate trained agent.
    Đánh giá agent đã huấn luyện.
    """
    
    rewards = []
    costs = []
    renewable_ratios = []
    unmet_ratios = []
    
    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        rewards.append(episode_reward)
        costs.append(info["total_cost"])
        renewable_ratios.append(info["renewable_ratio"])
        unmet_ratios.append(info["unmet_ratio"])
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_cost": np.mean(costs),
        "mean_renewable": np.mean(renewable_ratios),
        "mean_unmet": np.mean(unmet_ratios),
    }


def evaluate_random(env: MicrogridEnv, num_episodes: int = 20, seed: int = 42) -> Dict:
    """
    Evaluate random baseline.
    Đánh giá baseline ngẫu nhiên.
    """
    
    rewards = []
    costs = []
    renewable_ratios = []
    unmet_ratios = []
    
    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        episode_reward = 0
        
        while True:
            action = random.randint(0, 4)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        rewards.append(episode_reward)
        costs.append(info["total_cost"])
        renewable_ratios.append(info["renewable_ratio"])
        unmet_ratios.append(info["unmet_ratio"])
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_cost": np.mean(costs),
        "mean_renewable": np.mean(renewable_ratios),
        "mean_unmet": np.mean(unmet_ratios),
    }


def plot_results(history: Dict):
    """
    Generate training visualization plots.
    Tạo biểu đồ trực quan hóa training.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("🔋 DQN Training Results for Microgrid Optimization", fontsize=14, fontweight='bold')
    
    # Reward curve
    ax = axes[0, 0]
    rewards = history["rewards"]
    ax.plot(rewards, alpha=0.4, label="Raw")
    if len(rewards) >= 20:
        smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
        ax.plot(range(19, len(rewards)), smoothed, linewidth=2, label="Smoothed (MA20)")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cost curve
    ax = axes[0, 1]
    costs = history["costs"]
    ax.plot(costs, alpha=0.4, color='orange', label="Raw")
    if len(costs) >= 20:
        smoothed = np.convolve(costs, np.ones(20)/20, mode='valid')
        ax.plot(range(19, len(costs)), smoothed, linewidth=2, color='red', label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Daily Grid Cost ($)")
    ax.set_title("Grid Electricity Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Renewable ratio
    ax = axes[1, 0]
    ratios = [r * 100 for r in history["renewable_ratios"]]
    ax.plot(ratios, alpha=0.4, color='green', label="Raw")
    if len(ratios) >= 20:
        smoothed = np.convolve(ratios, np.ones(20)/20, mode='valid')
        ax.plot(range(19, len(ratios)), smoothed, linewidth=2, color='darkgreen', label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Renewable Usage (%)")
    ax.set_title("Renewable Energy Utilization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax = axes[1, 1]
    ax.plot(history["epsilons"], color='purple', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Exploration Rate Decay")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved: training_curves.png")


def plot_episode_analysis(env: MicrogridEnv):
    """
    Analyze single episode behavior.
    Phân tích hành vi trong một episode.
    """
    
    if not env.episode_history:
        print("No episode history to plot!")
        return
    
    history = env.episode_history
    hours = [h["hour"] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("🔋 24-Hour Episode Analysis", fontsize=14, fontweight='bold')
    
    # Energy balance
    ax = axes[0, 0]
    ax.bar(hours, [h["demand"] for h in history], alpha=0.5, label="Demand", color='red')
    ax.bar(hours, [h["solar"] for h in history], alpha=0.7, label="Solar", color='gold')
    ax.bar(hours, [h["wind"] for h in history], alpha=0.7, bottom=[h["solar"] for h in history], 
           label="Wind", color='skyblue')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Energy Supply & Demand")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Battery level
    ax = axes[0, 1]
    ax.plot(hours, [h["battery_level"] for h in history], 'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label="Max Capacity")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Battery Level (kWh)")
    ax.set_title("Battery State of Charge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    
    # Actions taken
    ax = axes[1, 0]
    action_names = ["Discharge", "Charge", "Grid", "Renew+Disch", "Renew+Grid"]
    actions = [h["action"] for h in history]
    colors = ['red', 'green', 'orange', 'blue', 'purple']
    for i, h in enumerate(history):
        ax.bar(h["hour"], 1, bottom=0, color=colors[h["action"]], alpha=0.7)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Action")
    ax.set_title("Actions Taken by Agent")
    ax.set_yticks([])
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.7) for c in colors]
    ax.legend(handles, action_names, loc='upper right', fontsize=8)
    
    # Cumulative reward
    ax = axes[1, 1]
    cumulative = np.cumsum([h["reward"] for h in history])
    ax.plot(hours, cumulative, 'g-', linewidth=2, marker='o', markersize=4)
    ax.fill_between(hours, cumulative, alpha=0.3, color='green')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Accumulation")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("episode_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved: episode_analysis.png")

print("✅ Evaluation functions defined!")

#@title 8️⃣ 🚀 RUN TRAINING & EVALUATION
"""
================================================================================
🎯 MAIN EXECUTION - Thực thi chính
================================================================================

Run this cell to train and evaluate the DQN agent!
Chạy cell này để huấn luyện và đánh giá agent DQN!
================================================================================
"""

# Train the agent
history, agent, env = train(CONFIG, device)

# Plot training results
plot_results(history)

# Evaluate trained agent
print("\n" + "=" * 60)
print("📊 EVALUATION RESULTS")
print("=" * 60)

agent_metrics = evaluate_agent(agent, env, num_episodes=20)
random_metrics = evaluate_random(env, num_episodes=20)

print(f"\n{'Metric':<25} {'Trained Agent':>15} {'Random Baseline':>15} {'Improvement':>12}")
print("-" * 70)
print(f"{'Mean Reward':<25} {agent_metrics['mean_reward']:>15.2f} {random_metrics['mean_reward']:>15.2f} "
      f"{(agent_metrics['mean_reward'] - random_metrics['mean_reward'])/abs(random_metrics['mean_reward'])*100:>+10.1f}%")
print(f"{'Daily Cost ($)':<25} {agent_metrics['mean_cost']:>15.2f} {random_metrics['mean_cost']:>15.2f} "
      f"{(random_metrics['mean_cost'] - agent_metrics['mean_cost'])/random_metrics['mean_cost']*100:>+10.1f}%")
print(f"{'Renewable Usage (%)':<25} {agent_metrics['mean_renewable']*100:>14.1f}% {random_metrics['mean_renewable']*100:>14.1f}% "
      f"{(agent_metrics['mean_renewable'] - random_metrics['mean_renewable'])*100:>+10.1f}pp")
print(f"{'Unmet Demand (%)':<25} {agent_metrics['mean_unmet']*100:>14.1f}% {random_metrics['mean_unmet']*100:>14.1f}% "
      f"{(random_metrics['mean_unmet'] - agent_metrics['mean_unmet'])*100:>+10.1f}pp")

# Run and analyze a demo episode
print("\n" + "=" * 60)
print("🎮 DEMO EPISODE")
print("=" * 60)

state = env.reset(seed=999)
total_reward = 0

print(f"\n{'Hour':>5} | {'Action':<15} | {'Demand':>8} | {'Solar':>8} | {'Wind':>8} | {'Battery':>8} | {'Reward':>8}")
print("-" * 80)

action_names = ["Discharge", "Charge", "Grid", "Renew+Disch", "Renew+Grid"]
for step in range(24):
    action = agent.select_action(state, training=False)
    next_state, reward, done, info = env.step(action)
    
    h = env.episode_history[-1]
    print(f"{h['hour']:>5} | {action_names[action]:<15} | {h['demand']:>8.1f} | {h['solar']:>8.1f} | "
          f"{h['wind']:>8.1f} | {h['battery_level']:>8.1f} | {reward:>+8.2f}")
    
    state = next_state
    total_reward += reward
    if done:
        break

print("-" * 80)
print(f"{'TOTAL':>5} | {'':15} | {'':>8} | {'':>8} | {'':>8} | {'':>8} | {total_reward:>+8.2f}")
print(f"\n📈 Total Reward: {total_reward:.2f}")
print(f"💰 Grid Cost: ${info['total_cost']:.2f}")
print(f"🌱 Renewable Usage: {info['renewable_ratio']*100:.1f}%")

# Plot episode analysis
plot_episode_analysis(env)

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
print("\nFiles created:")
print("  📁 best_model.pt - Best trained model weights")
print("  📁 final_model.pt - Final trained model weights")
print("  📁 training_curves.png - Training visualization")
print("  📁 episode_analysis.png - 24-hour episode analysis")
