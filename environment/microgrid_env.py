"""
Microgrid Environment for Reinforcement Learning
Môi trường mô phỏng lưới điện nhỏ (microgrid) cho RL

Mô tả:
- Microgrid bao gồm: pin lưu trữ, solar panels, wind turbines, kết nối grid
- Agent quyết định cách phân phối năng lượng mỗi giờ
- Mục tiêu: Tối đa hóa renewable usage, tối thiểu hóa chi phí và unmet demand
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional


class MicrogridEnv(gym.Env):
    """
    Gymnasium Environment cho Microgrid Energy Optimization
    
    State Space (8 dimensions):
        - battery_level: Mức pin [0, 1]
        - demand: Nhu cầu năng lượng [0, 1]
        - solar_generation: Sản lượng solar [0, 1]
        - wind_generation: Sản lượng wind [0, 1]
        - grid_price: Giá điện grid [0, 1]
        - hour_sin: Sin của giờ (cyclic encoding)
        - hour_cos: Cos của giờ (cyclic encoding)
        - prev_action: Hành động trước đó (one-hot)
    
    Action Space (5 discrete actions):
        0: Discharge battery - Xả pin để đáp ứng nhu cầu
        1: Charge from renewable - Sạc pin từ năng lượng tái tạo dư
        2: Buy from grid - Mua điện từ lưới chính
        3: Mixed: renewable + discharge - Ưu tiên renewable, xả pin nếu thiếu
        4: Mixed: renewable + grid - Ưu tiên renewable, mua grid nếu thiếu
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        battery_capacity: float = 100.0,
        battery_efficiency: float = 0.95,
        max_charge_rate: float = 20.0,
        max_discharge_rate: float = 20.0,
        max_solar: float = 50.0,
        max_wind: float = 30.0,
        base_demand: float = 40.0,
        demand_std: float = 10.0,
        grid_price_min: float = 0.05,
        grid_price_max: float = 0.25,
        hours_per_episode: int = 24,
        reward_config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # Environment parameters
        self.battery_capacity = battery_capacity
        self.battery_efficiency = battery_efficiency
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.max_solar = max_solar
        self.max_wind = max_wind
        self.base_demand = base_demand
        self.demand_std = demand_std
        self.grid_price_min = grid_price_min
        self.grid_price_max = grid_price_max
        self.hours_per_episode = hours_per_episode
        self.render_mode = render_mode
        
        # Reward configuration
        self.reward_config = reward_config or {
            "renewable_usage": 1.0,
            "grid_purchase": -2.0,
            "unmet_demand": -5.0,
            "battery_wear": -0.1,
            "cost_efficiency": 0.5,
        }
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        
        # State: [battery, demand, solar, wind, price, hour_sin, hour_cos, prev_action]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Initialize state
        self._reset_state()
    
    def _reset_state(self):
        """Reset internal state variables"""
        self.battery_level = self.battery_capacity * 0.5  # Start at 50%
        self.current_hour = 0
        self.prev_action = 0
        self.total_cost = 0.0
        self.total_renewable_used = 0.0
        self.total_demand = 0.0
        self.total_unmet = 0.0
        self.episode_history = []
    
    def _get_solar_generation(self, hour: int) -> float:
        """
        Tính sản lượng điện mặt trời dựa trên giờ
        Peak vào giữa trưa (12h), không có vào ban đêm
        """
        if 6 <= hour <= 18:
            # Solar curve: peak at noon
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
            noise = np.random.uniform(0.8, 1.2)  # Weather variability
            return self.max_solar * solar_factor * noise
        return 0.0
    
    def _get_wind_generation(self, hour: int) -> float:
        """
        Tính sản lượng điện gió - ít predictable hơn solar
        Có xu hướng mạnh hơn vào đêm
        """
        # Wind tends to be stronger at night
        base_factor = 0.6 if 6 <= hour <= 18 else 0.8
        noise = np.random.uniform(0.3, 1.5)  # High variability
        return self.max_wind * base_factor * noise
    
    def _get_demand(self, hour: int) -> float:
        """
        Tính nhu cầu năng lượng theo pattern hàng ngày
        Peak vào sáng sớm và tối
        """
        # Daily demand pattern: peaks at 8am and 7pm
        morning_peak = np.exp(-((hour - 8) ** 2) / 8)
        evening_peak = np.exp(-((hour - 19) ** 2) / 8)
        base_pattern = 0.5 + 0.3 * morning_peak + 0.4 * evening_peak
        
        demand = self.base_demand * base_pattern
        noise = np.random.normal(0, self.demand_std)
        return max(0, demand + noise)
    
    def _get_grid_price(self, hour: int) -> float:
        """
        Tính giá điện grid theo giờ
        Cao vào peak hours (7-9am, 5-9pm)
        """
        # Time-of-use pricing
        if 7 <= hour <= 9 or 17 <= hour <= 21:
            # Peak hours - expensive
            price_factor = 0.8 + np.random.uniform(0, 0.2)
        elif 22 <= hour or hour <= 6:
            # Off-peak - cheap
            price_factor = 0.2 + np.random.uniform(0, 0.2)
        else:
            # Mid-peak
            price_factor = 0.5 + np.random.uniform(0, 0.2)
        
        price_range = self.grid_price_max - self.grid_price_min
        return self.grid_price_min + price_range * price_factor
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation from current state"""
        # Normalize values to [0, 1]
        battery_norm = self.battery_level / self.battery_capacity
        demand = self._get_demand(self.current_hour)
        demand_norm = min(1.0, demand / (self.base_demand * 2))
        
        solar = self._get_solar_generation(self.current_hour)
        solar_norm = solar / self.max_solar if self.max_solar > 0 else 0
        
        wind = self._get_wind_generation(self.current_hour)
        wind_norm = wind / self.max_wind if self.max_wind > 0 else 0
        
        price = self._get_grid_price(self.current_hour)
        price_norm = (price - self.grid_price_min) / (self.grid_price_max - self.grid_price_min)
        
        # Cyclic encoding for hour
        hour_sin = np.sin(2 * np.pi * self.current_hour / 24)
        hour_cos = np.cos(2 * np.pi * self.current_hour / 24)
        
        # Normalize to [0, 1] range
        hour_sin_norm = (hour_sin + 1) / 2
        hour_cos_norm = (hour_cos + 1) / 2
        
        # Previous action normalized
        prev_action_norm = self.prev_action / 4
        
        return np.array([
            battery_norm,
            demand_norm,
            solar_norm,
            wind_norm,
            price_norm,
            hour_sin_norm,
            hour_cos_norm,
            prev_action_norm,
        ], dtype=np.float32)
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self._reset_state()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary"""
        return {
            "hour": self.current_hour,
            "battery_level": self.battery_level,
            "total_cost": self.total_cost,
            "total_renewable_used": self.total_renewable_used,
            "total_demand": self.total_demand,
            "total_unmet": self.total_unmet,
            "renewable_ratio": (
                self.total_renewable_used / self.total_demand 
                if self.total_demand > 0 else 0
            ),
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step
        
        Args:
            action: Integer action (0-4)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current environment state
        demand = self._get_demand(self.current_hour)
        solar = self._get_solar_generation(self.current_hour)
        wind = self._get_wind_generation(self.current_hour)
        grid_price = self._get_grid_price(self.current_hour)
        
        renewable_available = solar + wind
        self.total_demand += demand
        
        # Initialize energy flows
        renewable_used = 0.0
        battery_discharge = 0.0
        battery_charge = 0.0
        grid_purchased = 0.0
        unmet_demand = 0.0
        
        # Process action
        if action == 0:  # Discharge battery
            # Use battery to meet demand
            discharge_needed = min(demand, self.battery_level, self.max_discharge_rate)
            battery_discharge = discharge_needed * self.battery_efficiency
            self.battery_level -= discharge_needed
            remaining_demand = demand - battery_discharge
            unmet_demand = remaining_demand
            
        elif action == 1:  # Charge from renewable
            # Use renewable for demand first, then charge battery
            renewable_for_demand = min(renewable_available, demand)
            renewable_used = renewable_for_demand
            remaining_demand = demand - renewable_for_demand
            unmet_demand = remaining_demand
            
            # Charge battery with excess
            excess_renewable = renewable_available - renewable_for_demand
            charge_amount = min(
                excess_renewable,
                self.battery_capacity - self.battery_level,
                self.max_charge_rate
            )
            battery_charge = charge_amount
            self.battery_level += charge_amount * self.battery_efficiency
            
        elif action == 2:  # Buy from grid
            # Buy all needed energy from grid
            grid_purchased = demand
            
        elif action == 3:  # Renewable + discharge
            # Prioritize renewable, use battery if not enough
            renewable_used = min(renewable_available, demand)
            remaining = demand - renewable_used
            
            if remaining > 0:
                discharge = min(remaining, self.battery_level, self.max_discharge_rate)
                battery_discharge = discharge * self.battery_efficiency
                self.battery_level -= discharge
                remaining -= battery_discharge
            
            unmet_demand = max(0, remaining)
            
        elif action == 4:  # Renewable + grid
            # Prioritize renewable, buy from grid if not enough
            renewable_used = min(renewable_available, demand)
            remaining = demand - renewable_used
            grid_purchased = remaining
        
        # Calculate reward
        reward = self._calculate_reward(
            renewable_used=renewable_used,
            grid_purchased=grid_purchased,
            unmet_demand=unmet_demand,
            battery_charge=battery_charge,
            battery_discharge=battery_discharge,
            grid_price=grid_price,
        )
        
        # Update tracking variables
        self.total_renewable_used += renewable_used
        self.total_cost += grid_purchased * grid_price
        self.total_unmet += unmet_demand
        self.prev_action = action
        
        # Store history for visualization
        self.episode_history.append({
            "hour": self.current_hour,
            "demand": demand,
            "solar": solar,
            "wind": wind,
            "renewable_used": renewable_used,
            "battery_level": self.battery_level,
            "grid_purchased": grid_purchased,
            "unmet_demand": unmet_demand,
            "reward": reward,
            "action": action,
            "grid_price": grid_price,
        })
        
        # Advance time
        self.current_hour += 1
        
        # Check termination
        terminated = self.current_hour >= self.hours_per_episode
        truncated = False
        
        # Critical failure: battery depleted and high unmet demand
        if self.battery_level <= 0 and unmet_demand > demand * 0.5:
            terminated = True
        
        obs = self._get_obs()
        info = self._get_info()
        info["step_details"] = self.episode_history[-1]
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(
        self,
        renewable_used: float,
        grid_purchased: float,
        unmet_demand: float,
        battery_charge: float,
        battery_discharge: float,
        grid_price: float,
    ) -> float:
        """
        Calculate reward based on current step
        
        Reward shaping:
        - Positive: Sử dụng renewable, tiết kiệm chi phí
        - Negative: Mua grid, unmet demand, battery wear
        """
        reward = 0.0
        
        # Reward for using renewable energy
        reward += self.reward_config["renewable_usage"] * (renewable_used / self.base_demand)
        
        # Penalty for grid purchase (weighted by price)
        normalized_price = grid_price / self.grid_price_max
        reward += self.reward_config["grid_purchase"] * (grid_purchased / self.base_demand) * normalized_price
        
        # Heavy penalty for unmet demand
        reward += self.reward_config["unmet_demand"] * (unmet_demand / self.base_demand)
        
        # Small penalty for battery wear (charge + discharge cycles)
        battery_activity = (battery_charge + battery_discharge) / self.battery_capacity
        reward += self.reward_config["battery_wear"] * battery_activity
        
        # Bonus for cost efficiency (minimal grid use during peak prices)
        if grid_purchased == 0 and normalized_price > 0.7:
            reward += self.reward_config["cost_efficiency"]
        
        return reward
    
    def render(self):
        """Render current state"""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"\n=== Hour {self.current_hour} ===")
            print(f"Battery: {self.battery_level:.1f}/{self.battery_capacity} kWh")
            print(f"Total Cost: ${self.total_cost:.2f}")
            print(f"Renewable Ratio: {info['renewable_ratio']*100:.1f}%")
            print(f"Unmet Demand: {self.total_unmet:.1f} kWh")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the episode for analysis"""
        return {
            "total_cost": self.total_cost,
            "total_renewable_used": self.total_renewable_used,
            "total_demand": self.total_demand,
            "total_unmet": self.total_unmet,
            "renewable_ratio": (
                self.total_renewable_used / self.total_demand
                if self.total_demand > 0 else 0
            ),
            "unmet_ratio": (
                self.total_unmet / self.total_demand
                if self.total_demand > 0 else 0
            ),
            "history": self.episode_history,
        }
