"""
Main Entry Point for Microgrid RL Project
Điểm khởi đầu cho dự án Reinforcement Learning tối ưu hóa Microgrid

Sử dụng:
    python main.py train          # Huấn luyện agent
    python main.py evaluate       # Đánh giá agent đã train
    python main.py demo           # Demo một episode   
    python main.py --help         # Xem hướng dẫn
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.microgrid_env import MicrogridEnv
from agent.dqn_agent import DQNAgent, DoubleDQNAgent
from training.train import train
from evaluation.evaluate import run_full_evaluation
from config import ENV_CONFIG, DQN_CONFIG


def demo_episode(model_path: str = None, render: bool = True):
    """
    Demo một episode với agent đã train (hoặc random nếu không có model)
    """
    # Initialize environment
    env = MicrogridEnv(
        battery_capacity=ENV_CONFIG.battery_capacity,
        max_solar=ENV_CONFIG.max_solar_generation,
        max_wind=ENV_CONFIG.max_wind_generation,
        base_demand=ENV_CONFIG.base_demand,
        hours_per_episode=ENV_CONFIG.hours_per_episode,
        render_mode="human" if render else None,
    )
    
    # Initialize agent
    agent = DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=DQN_CONFIG.hidden_sizes,
    )
    
    # Load model if available
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Loaded trained model from {model_path}")
        use_trained = True
    else:
        print("⚠️ No trained model found, using random actions")
        use_trained = False
    
    # Run episode
    state, _ = env.reset(seed=42)
    total_reward = 0
    
    print("\n" + "=" * 60)
    print("🔋 MICROGRID ENERGY OPTIMIZATION DEMO")
    print("=" * 60)
    
    action_names = [
        "Discharge Battery",
        "Charge from Renewable", 
        "Buy from Grid",
        "Renewable + Discharge",
        "Renewable + Grid"
    ]
    
    for step in range(ENV_CONFIG.hours_per_episode):
        # Get action
        if use_trained:
            action = agent.select_action(state, training=False)
        else:
            action = env.action_space.sample()
        
        # Execute
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print step info
        step_info = info.get("step_details", {})
        print(f"\nHour {step:2d}:00 | Action: {action_names[action]}")
        print(f"  Demand: {step_info.get('demand', 0):.1f} kW | "
              f"Solar: {step_info.get('solar', 0):.1f} kW | "
              f"Wind: {step_info.get('wind', 0):.1f} kW")
        print(f"  Battery: {step_info.get('battery_level', 0):.1f} kWh | "
              f"Grid Price: ${step_info.get('grid_price', 0):.3f}/kWh")
        print(f"  Reward: {reward:+.2f} | Cumulative: {total_reward:.2f}")
        
        state = next_state
        
        if terminated or truncated:
            break
    
    # Summary
    summary = env.get_episode_summary()
    print("\n" + "=" * 60)
    print("📊 EPISODE SUMMARY")
    print("=" * 60)
    print(f"Total Reward:        {total_reward:.2f}")
    print(f"Total Grid Cost:     ${summary['total_cost']:.2f}")
    print(f"Renewable Usage:     {summary['renewable_ratio']*100:.1f}%")
    print(f"Unmet Demand Ratio:  {summary['unmet_ratio']*100:.1f}%")
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Microgrid Energy Optimization with Deep Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --episodes 500        # Train for 500 episodes
  python main.py train --double-dqn          # Use Double DQN
  python main.py evaluate                    # Evaluate trained model
  python main.py demo                        # Run demo episode
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the DQN agent")
    train_parser.add_argument("--episodes", type=int, default=500,
                              help="Number of training episodes (default: 500)")
    train_parser.add_argument("--double-dqn", action="store_true",
                              help="Use Double DQN instead of vanilla DQN")
    train_parser.add_argument("--save-dir", type=str, default="checkpoints",
                              help="Directory to save checkpoints")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed for reproducibility")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("--model", type=str, default="checkpoints/best_model.pt",
                             help="Path to trained model")
    eval_parser.add_argument("--episodes", type=int, default=20,
                             help="Number of evaluation episodes")
    eval_parser.add_argument("--save-dir", type=str, default="evaluation_results",
                             help="Directory to save results")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo episode")
    demo_parser.add_argument("--model", type=str, default="checkpoints/best_model.pt",
                             help="Path to trained model")
    demo_parser.add_argument("--no-render", action="store_true",
                             help="Disable rendering")
    
    args = parser.parse_args()
    
    if args.command == "train":
        print("🚀 Starting training...")
        train(
            num_episodes=args.episodes,
            max_steps=ENV_CONFIG.hours_per_episode,
            save_dir=args.save_dir,
            use_double_dqn=args.double_dqn,
            seed=args.seed,
        )
        
    elif args.command == "evaluate":
        print("📊 Starting evaluation...")
        run_full_evaluation(
            model_path=args.model,
            history_path=os.path.join(os.path.dirname(args.model), "training_history.json"),
            save_dir=args.save_dir,
            num_eval_episodes=args.episodes,
        )
        
    elif args.command == "demo":
        demo_episode(
            model_path=args.model,
            render=not args.no_render,
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
