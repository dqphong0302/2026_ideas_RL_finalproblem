# Configuration for Microgrid RL Project
# Cấu hình cho dự án Reinforcement Learning tối ưu hóa microgrid

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvironmentConfig:
    """Cấu hình môi trường microgrid"""
    # Battery parameters
    battery_capacity: float = 100.0  # kWh - Dung lượng pin tối đa
    battery_efficiency: float = 0.95  # Hiệu suất sạc/xả (5% loss)
    max_charge_rate: float = 20.0  # kW - Tốc độ sạc tối đa
    max_discharge_rate: float = 20.0  # kW - Tốc độ xả tối đa
    
    # Renewable generation parameters
    max_solar_generation: float = 50.0  # kW peak
    max_wind_generation: float = 30.0  # kW peak
    
    # Demand parameters
    base_demand: float = 40.0  # kW - Nhu cầu cơ sở
    demand_std: float = 10.0  # Standard deviation của demand
    
    # Grid parameters
    grid_price_min: float = 0.05  # $/kWh - Giá thấp nhất
    grid_price_max: float = 0.25  # $/kWh - Giá cao nhất
    
    # Episode parameters
    hours_per_episode: int = 24  # Số giờ mỗi episode (1 ngày)
    time_step_hours: float = 1.0  # Mỗi step = 1 giờ


@dataclass
class DQNConfig:
    """Cấu hình DQN Agent
    
    Giải thích hyperparameters:
    - learning_rate: Tốc độ học, quá cao gây unstable, quá thấp học chậm
    - gamma: Discount factor, gần 1 = quan tâm tương lai nhiều hơn
    - epsilon: Xác suất explore ngẫu nhiên thay vì exploit
    - batch_size: Số samples mỗi lần update
    - buffer_size: Kích thước replay buffer
    """
    # Network architecture
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    
    # Learning parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    
    # Exploration parameters (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Replay buffer
    buffer_size: int = 100_000
    batch_size: int = 64
    
    # Target network
    target_update_freq: int = 1000  # Steps giữa mỗi lần update target network
    
    # Training
    num_episodes: int = 1000
    max_steps_per_episode: int = 24  # 24 giờ
    
    # Saving
    save_freq: int = 100  # Save model mỗi 100 episodes


@dataclass 
class RewardConfig:
    """Cấu hình hàm reward
    
    Reward shaping để khuyến khích:
    1. Sử dụng năng lượng tái tạo (positive)
    2. Giảm mua điện từ grid (negative)
    3. Đáp ứng đủ nhu cầu (penalty nếu không đủ)
    4. Bảo vệ pin (penalty nếu charge/discharge quá nhiều)
    """
    renewable_usage_reward: float = 1.0  # Thưởng cho mỗi kWh renewable sử dụng
    grid_purchase_penalty: float = -2.0  # Phạt cho mỗi kWh mua từ grid
    unmet_demand_penalty: float = -5.0  # Phạt nặng cho mỗi kWh không đáp ứng
    battery_wear_penalty: float = -0.1  # Phạt nhẹ cho wear
    cost_efficiency_bonus: float = 0.5  # Bonus khi vận hành hiệu quả


# Default configurations
ENV_CONFIG = EnvironmentConfig()
DQN_CONFIG = DQNConfig()
REWARD_CONFIG = RewardConfig()
