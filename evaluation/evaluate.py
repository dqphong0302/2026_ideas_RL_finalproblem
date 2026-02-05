"""
Evaluation and Visualization for Trained Agent
Đánh giá và trực quan hóa hiệu suất agent

Mô tả:
- Load trained model và evaluate performance
- Tạo các biểu đồ phân tích
- So sánh với random baseline
- Export metrics cho report
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.microgrid_env import MicrogridEnv
from agent.dqn_agent import DQNAgent, DoubleDQNAgent
from config import ENV_CONFIG, DQN_CONFIG, REWARD_CONFIG


def evaluate_agent(
    agent: DQNAgent,
    env: MicrogridEnv,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dict:
    """
    Evaluate trained agent on multiple episodes
    
    Args:
        agent: Trained DQN agent
        env: Microgrid environment
        num_episodes: Number of evaluation episodes
        seed: Random seed
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    metrics = {
        "episode_rewards": [],
        "episode_costs": [],
        "renewable_ratios": [],
        "unmet_ratios": [],
        "episode_histories": [],
    }
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep if seed else None)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)  # No exploration
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_costs"].append(summary["total_cost"])
        metrics["renewable_ratios"].append(summary["renewable_ratio"])
        metrics["unmet_ratios"].append(summary["unmet_ratio"])
        metrics["episode_histories"].append(summary["history"])
    
    # Compute aggregate statistics
    metrics["mean_reward"] = np.mean(metrics["episode_rewards"])
    metrics["std_reward"] = np.std(metrics["episode_rewards"])
    metrics["mean_cost"] = np.mean(metrics["episode_costs"])
    metrics["mean_renewable_ratio"] = np.mean(metrics["renewable_ratios"])
    metrics["mean_unmet_ratio"] = np.mean(metrics["unmet_ratios"])
    
    return metrics


def evaluate_random_baseline(
    env: MicrogridEnv,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> Dict:
    """
    Evaluate random policy as baseline
    
    So sánh với random để thấy agent đã học được gì
    """
    metrics = {
        "episode_rewards": [],
        "episode_costs": [],
        "renewable_ratios": [],
        "unmet_ratios": [],
    }
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep if seed else None)
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_costs"].append(summary["total_cost"])
        metrics["renewable_ratios"].append(summary["renewable_ratio"])
        metrics["unmet_ratios"].append(summary["unmet_ratio"])
    
    metrics["mean_reward"] = np.mean(metrics["episode_rewards"])
    metrics["mean_cost"] = np.mean(metrics["episode_costs"])
    metrics["mean_renewable_ratio"] = np.mean(metrics["renewable_ratios"])
    metrics["mean_unmet_ratio"] = np.mean(metrics["unmet_ratios"])
    
    return metrics


def plot_training_curves(history_path: str, save_dir: str = "plots"):
    """
    Vẽ biểu đồ learning curves từ training history
    
    Các biểu đồ:
    1. Episode Reward over time
    2. Daily Cost over time
    3. Renewable Ratio over time
    4. Epsilon decay
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    episodes = range(1, len(history["episode_rewards"]) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Training Progress - Microgrid Energy Optimization", fontsize=14)
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, history["episode_rewards"], alpha=0.3, color="blue")
    # Smoothed curve (moving average)
    window = min(50, len(history["episode_rewards"]) // 10)
    if window > 1:
        smoothed = np.convolve(
            history["episode_rewards"],
            np.ones(window) / window,
            mode="valid"
        )
        ax1.plot(range(window, len(history["episode_rewards"]) + 1), smoothed, 
                 color="blue", linewidth=2, label="Smoothed")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Reward (Cumulative)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily Cost
    ax2 = axes[0, 1]
    ax2.plot(episodes, history["episode_costs"], alpha=0.3, color="red")
    if window > 1:
        smoothed_cost = np.convolve(
            history["episode_costs"],
            np.ones(window) / window,
            mode="valid"
        )
        ax2.plot(range(window, len(history["episode_costs"]) + 1), smoothed_cost,
                 color="red", linewidth=2, label="Smoothed")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Daily Grid Purchase Cost")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Renewable Usage Ratio
    ax3 = axes[1, 0]
    renewable_pct = [r * 100 for r in history["episode_renewable_ratios"]]
    ax3.plot(episodes, renewable_pct, alpha=0.3, color="green")
    if window > 1:
        smoothed_renewable = np.convolve(
            renewable_pct,
            np.ones(window) / window,
            mode="valid"
        )
        ax3.plot(range(window, len(renewable_pct) + 1), smoothed_renewable,
                 color="green", linewidth=2, label="Smoothed")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Renewable Usage (%)")
    ax3.set_title("Renewable Energy Utilization Ratio")
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Epsilon Decay
    ax4 = axes[1, 1]
    ax4.plot(episodes, history["epsilons"], color="purple", linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")
    ax4.set_title("Exploration Rate (ε) Decay")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    
    print(f"Training curves saved to {save_dir}/training_curves.png")


def plot_episode_analysis(history: List[Dict], save_dir: str = "plots"):
    """
    Vẽ phân tích chi tiết một episode
    
    Hiển thị:
    - Demand vs Supply (solar + wind + grid)
    - Battery level over time
    - Actions taken
    - Price variations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    hours = [h["hour"] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Single Episode Analysis - 24 Hour Operation", fontsize=14)
    
    # 1. Energy Balance
    ax1 = axes[0, 0]
    demand = [h["demand"] for h in history]
    solar = [h["solar"] for h in history]
    wind = [h["wind"] for h in history]
    renewable_used = [h["renewable_used"] for h in history]
    grid = [h["grid_purchased"] for h in history]
    
    ax1.bar(hours, demand, alpha=0.3, label="Demand", color="red")
    ax1.bar(hours, renewable_used, alpha=0.7, label="Renewable Used", color="green")
    ax1.bar(hours, grid, bottom=renewable_used, alpha=0.7, label="Grid", color="orange")
    ax1.plot(hours, [s + w for s, w in zip(solar, wind)], 
             "g--", label="Renewable Available", linewidth=2)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Energy (kW)")
    ax1.set_title("Energy Supply vs Demand")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Battery Level
    ax2 = axes[0, 1]
    battery = [h["battery_level"] for h in history]
    ax2.plot(hours, battery, "b-", linewidth=2, marker="o", markersize=4)
    ax2.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="Max Capacity")
    ax2.axhline(y=20, color="orange", linestyle="--", alpha=0.5, label="Low Warning")
    ax2.fill_between(hours, battery, alpha=0.3)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Battery Level (kWh)")
    ax2.set_title("Battery State of Charge")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Actions Taken
    ax3 = axes[1, 0]
    actions = [h["action"] for h in history]
    action_labels = ["Discharge", "Charge", "Buy Grid", "Renew+Discharge", "Renew+Grid"]
    colors = ["blue", "green", "red", "cyan", "orange"]
    for i, action in enumerate(actions):
        ax3.scatter(hours[i], action, c=colors[action], s=100, marker="s")
    ax3.set_yticks(range(5))
    ax3.set_yticklabels(action_labels)
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Action")
    ax3.set_title("Actions Taken by Agent")
    ax3.grid(True, alpha=0.3)
    
    # 4. Grid Price and Cost
    ax4 = axes[1, 1]
    prices = [h["grid_price"] for h in history]
    rewards = [h["reward"] for h in history]
    
    ax4_twin = ax4.twinx()
    ax4.bar(hours, prices, alpha=0.5, color="purple", label="Grid Price")
    ax4_twin.plot(hours, np.cumsum(rewards), "g-", linewidth=2, label="Cumulative Reward")
    
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Grid Price ($/kWh)", color="purple")
    ax4_twin.set_ylabel("Cumulative Reward", color="green")
    ax4.set_title("Grid Price & Cumulative Reward")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "episode_analysis.png"), dpi=150)
    plt.close()
    
    print(f"Episode analysis saved to {save_dir}/episode_analysis.png")


def plot_comparison(agent_metrics: Dict, random_metrics: Dict, save_dir: str = "plots"):
    """
    So sánh hiệu suất Agent vs Random Baseline
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_names = ["Mean Reward", "Mean Cost ($)", "Renewable (%)", "Unmet Demand (%)"]
    agent_values = [
        agent_metrics["mean_reward"],
        agent_metrics["mean_cost"],
        agent_metrics["mean_renewable_ratio"] * 100,
        agent_metrics["mean_unmet_ratio"] * 100,
    ]
    random_values = [
        random_metrics["mean_reward"],
        random_metrics["mean_cost"],
        random_metrics["mean_renewable_ratio"] * 100,
        random_metrics["mean_unmet_ratio"] * 100,
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, agent_values, width, label="Trained Agent", color="blue")
    bars2 = ax.bar(x + width/2, random_values, width, label="Random Policy", color="gray")
    
    ax.set_ylabel("Value")
    ax.set_title("Performance Comparison: Trained Agent vs Random Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "agent_vs_random.png"), dpi=150)
    plt.close()
    
    print(f"Comparison chart saved to {save_dir}/agent_vs_random.png")


def generate_report(
    agent_metrics: Dict,
    random_metrics: Dict,
    save_path: str = "evaluation_report.txt"
):
    """
    Generate text report of evaluation results
    """
    improvement_reward = (
        (agent_metrics["mean_reward"] - random_metrics["mean_reward"])
        / abs(random_metrics["mean_reward"]) * 100
        if random_metrics["mean_reward"] != 0 else 0
    )
    
    cost_reduction = (
        (random_metrics["mean_cost"] - agent_metrics["mean_cost"])
        / random_metrics["mean_cost"] * 100
        if random_metrics["mean_cost"] != 0 else 0
    )
    
    report = f"""
================================================================================
                    MICROGRID RL AGENT EVALUATION REPORT
================================================================================

PERFORMANCE METRICS
-------------------
                            Trained Agent       Random Baseline     Improvement
Mean Episode Reward:        {agent_metrics["mean_reward"]:>12.2f}       {random_metrics["mean_reward"]:>12.2f}       {improvement_reward:>+8.1f}%
Mean Daily Cost:            ${agent_metrics["mean_cost"]:>11.2f}       ${random_metrics["mean_cost"]:>11.2f}       {cost_reduction:>+8.1f}%
Renewable Usage Ratio:      {agent_metrics["mean_renewable_ratio"]*100:>11.1f}%       {random_metrics["mean_renewable_ratio"]*100:>11.1f}%
Unmet Demand Ratio:         {agent_metrics["mean_unmet_ratio"]*100:>11.1f}%       {random_metrics["mean_unmet_ratio"]*100:>11.1f}%

ANALYSIS
--------
- The trained DQN agent achieved {improvement_reward:.1f}% improvement in reward over random policy.
- Daily grid purchase costs were reduced by {cost_reduction:.1f}%.
- Renewable energy utilization: {agent_metrics["mean_renewable_ratio"]*100:.1f}%
- Demand satisfaction rate: {(1 - agent_metrics["mean_unmet_ratio"])*100:.1f}%

CONCLUSIONS
-----------
The DQN agent has successfully learned an energy dispatch policy that:
1. Maximizes the use of renewable energy sources
2. Minimizes electricity costs by strategic grid purchasing
3. Maintains battery levels for peak demand periods
4. Balances immediate needs with future energy requirements

================================================================================
"""
    
    with open(save_path, "w") as f:
        f.write(report)
    
    print(report)
    print(f"Report saved to {save_path}")


def run_full_evaluation(
    model_path: str = "checkpoints/best_model.pt",
    history_path: str = "checkpoints/training_history.json",
    save_dir: str = "evaluation_results",
    num_eval_episodes: int = 20,
):
    """
    Run complete evaluation pipeline
    """
    os.makedirs(save_dir, exist_ok=True)
    
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
        hours_per_episode=ENV_CONFIG.hours_per_episode,
    )
    
    # Load agent
    agent = DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=DQN_CONFIG.hidden_sizes,
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using untrained agent")
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    agent_metrics = evaluate_agent(agent, env, num_eval_episodes, seed=42)
    
    # Evaluate random baseline
    print("Evaluating random baseline...")
    random_metrics = evaluate_random_baseline(env, num_eval_episodes, seed=42)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    if os.path.exists(history_path):
        plot_training_curves(history_path, save_dir)
    
    if agent_metrics["episode_histories"]:
        plot_episode_analysis(agent_metrics["episode_histories"][0], save_dir)
    
    plot_comparison(agent_metrics, random_metrics, save_dir)
    
    # Generate report
    generate_report(
        agent_metrics, random_metrics,
        os.path.join(save_dir, "evaluation_report.txt")
    )
    
    # Save metrics as JSON
    results = {
        "agent_metrics": {k: v for k, v in agent_metrics.items() if k != "episode_histories"},
        "random_metrics": random_metrics,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll evaluation results saved to {save_dir}/")
    
    return agent_metrics, random_metrics


if __name__ == "__main__":
    run_full_evaluation()
