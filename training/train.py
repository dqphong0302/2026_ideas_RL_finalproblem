"""
Training Loop for DQN Agent
Vòng lặp huấn luyện cho DQN Agent

Mô tả:
- Chạy nhiều episodes training
- Tracking metrics: reward, loss, epsilon
- Logging và checkpointing
- Early stopping nếu đạt performance target
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.microgrid_env import MicrogridEnv
from agent.dqn_agent import DQNAgent, DoubleDQNAgent
from config import ENV_CONFIG, DQN_CONFIG, REWARD_CONFIG


def train(
    num_episodes: int = 1000,
    max_steps: int = 24,
    save_dir: str = "checkpoints",
    save_freq: int = 100,
    log_freq: int = 10,
    use_double_dqn: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, List]:
    """
    Training loop chính cho DQN agent
    
    Quy trình training:
    1. Khởi tạo environment và agent
    2. Với mỗi episode:
        a. Reset environment
        b. Với mỗi step:
            - Agent chọn action (epsilon-greedy)
            - Environment execute action
            - Lưu transition vào buffer
            - Agent update (nếu đủ samples)
        c. Decay epsilon
        d. Log metrics
    3. Save checkpoint định kỳ
    
    Args:
        num_episodes: Số episodes training
        max_steps: Số steps tối đa mỗi episode
        save_dir: Thư mục lưu checkpoints
        save_freq: Tần suất save model
        log_freq: Tần suất log metrics
        use_double_dqn: Sử dụng Double DQN thay vì DQN thường
        seed: Random seed để reproducibility
    
    Returns:
        training_history: Dictionary chứa metrics qua từng episode
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize environment
    env = MicrogridEnv(
        battery_capacity=ENV_CONFIG.battery_capacity,
        battery_efficiency=ENV_CONFIG.battery_efficiency,
        max_charge_rate=ENV_CONFIG.max_charge_rate,
        max_discharge_rate=ENV_CONFIG.max_discharge_rate,
        max_solar=ENV_CONFIG.max_solar_generation,
        max_wind=ENV_CONFIG.max_wind_generation,
        base_demand=ENV_CONFIG.base_demand,
        demand_std=ENV_CONFIG.demand_std,
        grid_price_min=ENV_CONFIG.grid_price_min,
        grid_price_max=ENV_CONFIG.grid_price_max,
        hours_per_episode=ENV_CONFIG.hours_per_episode,
        reward_config={
            "renewable_usage": REWARD_CONFIG.renewable_usage_reward,
            "grid_purchase": REWARD_CONFIG.grid_purchase_penalty,
            "unmet_demand": REWARD_CONFIG.unmet_demand_penalty,
            "battery_wear": REWARD_CONFIG.battery_wear_penalty,
            "cost_efficiency": REWARD_CONFIG.cost_efficiency_bonus,
        },
    )
    
    # Initialize agent
    AgentClass = DoubleDQNAgent if use_double_dqn else DQNAgent
    agent = AgentClass(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=DQN_CONFIG.hidden_sizes,
        learning_rate=DQN_CONFIG.learning_rate,
        gamma=DQN_CONFIG.gamma,
        epsilon_start=DQN_CONFIG.epsilon_start,
        epsilon_end=DQN_CONFIG.epsilon_end,
        epsilon_decay=DQN_CONFIG.epsilon_decay,
        buffer_size=DQN_CONFIG.buffer_size,
        batch_size=DQN_CONFIG.batch_size,
        target_update_freq=DQN_CONFIG.target_update_freq,
    )
    
    # Training history
    history = {
        "episode_rewards": [],
        "episode_costs": [],
        "episode_renewable_ratios": [],
        "episode_unmet_ratios": [],
        "episode_losses": [],
        "epsilons": [],
    }
    
    # Best model tracking
    best_reward = float("-inf")
    
    print("=" * 60)
    print(f"Starting training with {'Double ' if use_double_dqn else ''}DQN")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset(seed=seed + episode if seed else None)
        episode_reward = 0
        episode_losses = []
        
        # Episode loop
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Episode finished
        agent.decay_epsilon()
        
        # Get episode summary
        summary = env.get_episode_summary()
        
        # Record history
        history["episode_rewards"].append(episode_reward)
        history["episode_costs"].append(summary["total_cost"])
        history["episode_renewable_ratios"].append(summary["renewable_ratio"])
        history["episode_unmet_ratios"].append(summary["unmet_ratio"])
        history["episode_losses"].append(
            np.mean(episode_losses) if episode_losses else 0
        )
        history["epsilons"].append(agent.epsilon)
        
        # Logging
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(history["episode_rewards"][-log_freq:])
            avg_cost = np.mean(history["episode_costs"][-log_freq:])
            avg_renewable = np.mean(history["episode_renewable_ratios"][-log_freq:])
            
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {avg_reward:7.2f} | "
                f"Cost: ${avg_cost:6.2f} | "
                f"Renewable: {avg_renewable*100:5.1f}% | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Speed: {eps_per_sec:.1f} ep/s"
            )
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode + 1}.pt")
            agent.save(checkpoint_path)
            
            # Save history
            history_path = os.path.join(save_dir, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(history, f)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(save_dir, "best_model.pt")
            agent.save(best_path)
    
    # Training completed
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Best episode reward: {best_reward:.2f}")
    print("=" * 60)
    
    # Save final model and history
    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    return history


if __name__ == "__main__":
    # Run training with default parameters
    history = train(
        num_episodes=500,
        max_steps=24,
        save_dir="checkpoints",
        save_freq=50,
        log_freq=10,
        use_double_dqn=True,
        seed=42,
    )
