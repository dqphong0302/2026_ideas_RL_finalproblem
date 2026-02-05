#!/usr/bin/env python3
"""
Microgrid DQN Training Script - Standalone Local Version
Quick training and evaluation for demonstration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os

# ═══════════════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
EPISODES = 200
LEARNING_RATE = 0.0001

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# ═══════════════════════════════════════════════════════════════════════════
# MÔI TRƯỜNG MICROGRID
# ═══════════════════════════════════════════════════════════════════════════
class MicrogridEnv:
    def __init__(self):
        self.battery_capacity = 100.0
        self.battery_level = 50.0
        self.current_hour = 0
        
    def reset(self):
        self.battery_level = 50.0
        self.current_hour = 0
        return self._get_state()
    
    def _get_state(self):
        hour = self.current_hour % 24
        solar = max(0, np.sin((hour - 6) * np.pi / 12)) * 50 * (0.8 + 0.4 * np.random.random())
        wind = 30 * np.random.random()
        demand = 40 + 20 * np.sin((hour - 6) * np.pi / 12) + 10 * np.random.random()
        price = 0.15 + 0.1 * (1 if 17 <= hour <= 21 else 0)
        
        return np.array([
            self.battery_level / self.battery_capacity,
            demand / 100, solar / 50, wind / 30, price / 0.25,
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), 0.5
        ], dtype=np.float32)
    
    def step(self, action):
        state = self._get_state()
        demand = state[1] * 100
        solar = state[2] * 50
        wind = state[3] * 30
        price = state[4] * 0.25
        renewable = solar + wind
        
        grid_used = 0
        renewable_used = 0
        battery_change = 0
        
        if action == 0:  # Discharge
            discharge = min(20, self.battery_level, demand)
            battery_change = -discharge
            renewable_used = min(renewable, demand - discharge)
            grid_used = max(0, demand - discharge - renewable_used)
        elif action == 1:  # Charge from renewable
            renewable_used = min(renewable, demand)
            grid_used = max(0, demand - renewable_used)
            excess = renewable - renewable_used
            battery_change = min(20, self.battery_capacity - self.battery_level, excess)
        elif action == 2:  # Grid only
            grid_used = demand
        elif action == 3:  # Renewable + discharge
            renewable_used = min(renewable, demand)
            remaining = demand - renewable_used
            discharge = min(20, self.battery_level, remaining)
            battery_change = -discharge
            grid_used = max(0, remaining - discharge)
        else:  # Renewable + grid
            renewable_used = min(renewable, demand)
            grid_used = max(0, demand - renewable_used)
        
        self.battery_level = np.clip(self.battery_level + battery_change, 0, self.battery_capacity)
        
        reward = renewable_used / 40 - 2 * grid_used / 40 * price - 0.1 * abs(battery_change) / 20
        if grid_used == 0 and 17 <= self.current_hour % 24 <= 21:
            reward += 0.5
        
        self.current_hour += 1
        done = self.current_hour >= 24
        
        return self._get_state(), reward, done, {
            'renewable_used': renewable_used, 'grid_used': grid_used, 'cost': grid_used * price
        }

# ═══════════════════════════════════════════════════════════════════════════
# DQN NETWORK & AGENT
# ═══════════════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.buffer = deque(maxlen=10000)
        self.epsilon = 1.0
        self.gamma = 0.99
        self.update_count = 0
        
    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, 4)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.q_net(state_t).argmax().item()
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.buffer) < 64:
            return
        
        batch = random.sample(self.buffer, 64)
        states = torch.FloatTensor([t[0] for t in batch]).to(device)
        actions = torch.LongTensor([t[1] for t in batch]).to(device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(device)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("🚀 TRAINING DQN AGENT")
print("=" * 60)

env = MicrogridEnv()
agent = DQNAgent()

rewards_history = []
renewable_history = []

for episode in range(1, EPISODES + 1):
    state = env.reset()
    total_reward = 0
    total_renewable = 0
    total_grid = 0
    
    for _ in range(24):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        
        state = next_state
        total_reward += reward
        total_renewable += info['renewable_used']
        total_grid += info['grid_used']
        
        if done:
            break
    
    agent.epsilon = max(0.01, agent.epsilon * 0.995)
    rewards_history.append(total_reward)
    renewable_ratio = total_renewable / (total_renewable + total_grid + 1e-6) * 100
    renewable_history.append(renewable_ratio)
    
    if episode % 20 == 0:
        avg = np.mean(rewards_history[-20:])
        print(f"Episode {episode:3d} | Reward: {avg:+7.2f} | ε: {agent.epsilon:.3f} | Renewable: {renewable_ratio:.1f}%")

print("\n✅ Training complete!")

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 EVALUATION: Trained Agent vs Random Baseline")
print("=" * 60)

def evaluate(agent, use_random=False, num_episodes=20):
    results = {'rewards': [], 'costs': [], 'renewable': []}
    
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward, ep_cost, ep_renewable, ep_total = 0, 0, 0, 0
        
        for _ in range(24):
            if use_random:
                action = random.randint(0, 4)
            else:
                action = agent.select_action(state, eval_mode=True)
            
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_cost += info['cost']
            ep_renewable += info['renewable_used']
            ep_total += info['renewable_used'] + info['grid_used']
            state = next_state
            if done:
                break
        
        results['rewards'].append(ep_reward)
        results['costs'].append(ep_cost)
        results['renewable'].append(ep_renewable / ep_total * 100 if ep_total > 0 else 0)
    
    return {k: np.mean(v) for k, v in results.items()}

trained = evaluate(agent, use_random=False)
random_baseline = evaluate(agent, use_random=True)

reward_imp = ((trained['rewards'] - random_baseline['rewards']) / abs(random_baseline['rewards'])) * 100
cost_imp = ((random_baseline['costs'] - trained['costs']) / random_baseline['costs']) * 100

print(f"\n{'Metric':<25} {'Trained':>12} {'Random':>12} {'Improvement':>12}")
print("-" * 65)
print(f"{'Mean Episode Reward':<25} {trained['rewards']:>+12.2f} {random_baseline['rewards']:>+12.2f} {reward_imp:>+11.1f}%")
print(f"{'Daily Cost ($)':<25} {trained['costs']:>12.2f} {random_baseline['costs']:>12.2f} {cost_imp:>+11.1f}%")
print(f"{'Renewable Usage (%)':<25} {trained['renewable']:>11.1f}% {random_baseline['renewable']:>11.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE CHARTS
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs('evaluation_results', exist_ok=True)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(rewards_history, alpha=0.3, color='blue')
window = 10
smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
axes[0].plot(range(window-1, len(rewards_history)), smoothed, color='blue', linewidth=2)
axes[0].set_title('Training Reward', fontsize=12)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].grid(True, alpha=0.3)

axes[1].plot(renewable_history, color='green', linewidth=2)
axes[1].set_title('Renewable Usage %', fontsize=12)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('%')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results/training_curves.png', dpi=150, bbox_inches='tight')
print("\n📁 Saved: evaluation_results/training_curves.png")

# Comparison chart
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = ['Reward', 'Cost ($)', 'Renewable %']
trained_vals = [trained['rewards'], trained['costs'], trained['renewable']]
random_vals = [random_baseline['rewards'], random_baseline['costs'], random_baseline['renewable']]

for i, (m, t, r) in enumerate(zip(metrics, trained_vals, random_vals)):
    bars = axes[i].bar(['Trained', 'Random'], [t, r], color=['#2ecc71', '#e74c3c'])
    axes[i].set_title(m, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results/agent_vs_random.png', dpi=150, bbox_inches='tight')
print("📁 Saved: evaluation_results/agent_vs_random.png")

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
