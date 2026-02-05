# BÁO CÁO ĐỒ ÁN

# TỐI ƯU HÓA PHÂN PHỐI NĂNG LƯỢNG TRONG MICROGRID SỬ DỤNG DEEP REINFORCEMENT LEARNING

---

## PHẦN 1: MÔ TẢ VẤN ĐỀ (15%)

### 1.1 Giới Thiệu Hệ Thống Microgrid

Microgrid là một lưới điện nhỏ, cục bộ bao gồm:

- **Nguồn năng lượng tái tạo**: Solar panels (điện mặt trời) và wind turbines (tuabin gió)
- **Hệ thống lưu trữ**: Pin lưu trữ năng lượng (battery storage)
- **Kết nối lưới chính**: Có thể mua điện từ lưới điện quốc gia
- **Tải tiêu thụ**: Nhu cầu điện từ hộ gia đình, công nghiệp, thương mại

### 1.2 Tại Sao Đây Là Bài Toán Quyết Định Tuần Tự?

Phân phối năng lượng là bài toán **sequential decision-making** vì:

1. **Quyết định hiện tại ảnh hưởng tương lai**: Nếu sử dụng hết pin bây giờ, sẽ không có năng lượng dự trữ cho peak hours
2. **Trạng thái thay đổi liên tục**: Mức pin, nhu cầu, sản lượng renewable thay đổi mỗi giờ
3. **Chi phí biến đổi theo thời gian**: Giá điện grid cao vào peak hours, thấp vào off-peak

### 1.3 Hạn Chế Của Phương Pháp Truyền Thống

| Phương pháp | Hạn chế |
|-------------|---------|
| **Rule-based scheduling** | Không thích ứng với biến đổi thời tiết, cố định không học |
| **Linear programming** | Giả định tuyến tính, không xử lý được uncertainty |
| **Heuristic methods** | Không tối ưu toàn cục, dễ rơi vào local optima |

### 1.4 Tại Sao Reinforcement Learning Hiệu Quả Hơn?

RL có ưu điểm:

- **Adaptive**: Tự động học từ môi trường, thích ứng với thay đổi
- **Long-term optimization**: Xem xét hậu quả dài hạn của quyết định
- **Handle uncertainty**: Xử lý tốt với stochastic demand và renewable generation
- **No model required**: Không cần mô hình chính xác của hệ thống (model-free)

---

## PHẦN 2: MÔ HÌNH HÓA MDP (20%)

### 2.1 Markov Decision Process (MDP)

MDP được định nghĩa bởi tuple (S, A, P, R, γ):

```
         ┌─────────────────────────────────────────────────────┐
         │                   ENVIRONMENT                        │
         │  ┌─────────┐    ┌─────────┐    ┌─────────┐          │
         │  │ Battery │    │ Demand  │    │ Weather │          │
         │  │  Level  │    │ Pattern │    │(Solar/  │          │
         │  │  (SoC)  │    │         │    │  Wind)  │          │
         │  └────┬────┘    └────┬────┘    └────┬────┘          │
         │       │              │              │                │
         │       └──────────────┼──────────────┘                │
         │                      ▼                               │
         │              ┌───────────────┐                       │
         │              │  STATE (s_t)  │                       │
         │              └───────┬───────┘                       │
         └──────────────────────┼───────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │   RL AGENT (DQN)  │
                    │                   │
                    │  Q(s, a) = Neural │
                    │     Network       │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   ACTION (a_t)    │
                    │ 0: Discharge      │
                    │ 1: Charge         │
                    │ 2: Buy Grid       │
                    │ 3: Renew+Discharge│
                    │ 4: Renew+Grid     │
                    └─────────┬─────────┘
                              │
                              ▼
         ┌─────────────────────────────────────────────────────┐
         │                   ENVIRONMENT                        │
         │                                                      │
         │    Execute Action → Update Battery → Calculate      │
         │    Reward → Return New State s_{t+1}                │
         │                                                      │
         └───────────────────────┬─────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      REWARD (r_t)       │
                    │ + Renewable usage       │
                    │ - Grid purchase cost    │
                    │ - Unmet demand penalty  │
                    └─────────────────────────┘
```

### 2.2 State Space (Không Gian Trạng Thái)

**Dimension: 8**

| # | Component | Range | Ý nghĩa |
|---|-----------|-------|---------|
| 1 | battery_level | [0, 1] | Mức pin hiện tại (% capacity) |
| 2 | demand | [0, 1] | Nhu cầu năng lượng (normalized) |
| 3 | solar_generation | [0, 1] | Sản lượng điện mặt trời |
| 4 | wind_generation | [0, 1] | Sản lượng điện gió |
| 5 | grid_price | [0, 1] | Giá điện lưới (normalized) |
| 6 | hour_sin | [0, 1] | Sin encoding của giờ |
| 7 | hour_cos | [0, 1] | Cos encoding của giờ |
| 8 | prev_action | [0, 1] | Hành động trước đó |

**Giải thích Sin/Cos Encoding:**
Dùng sin/cos để encode thời gian vì nó capture được tính chu kỳ (giờ 23 → 0 gần nhau về nghĩa).

### 2.3 Action Space (Không Gian Hành Động)

**Discrete: 5 actions**

| Action | Name | Mô tả |
|--------|------|-------|
| 0 | Discharge | Xả pin để đáp ứng nhu cầu |
| 1 | Charge | Sạc pin từ renewable dư thừa |
| 2 | Buy Grid | Mua điện từ lưới chính |
| 3 | Renewable + Discharge | Ưu tiên renewable, xả pin nếu thiếu |
| 4 | Renewable + Grid | Ưu tiên renewable, mua grid nếu thiếu |

### 2.4 Reward Function (Hàm Thưởng)

```
R(s, a, s') = R_renewable + R_grid + R_unmet + R_battery + R_bonus

Cụ thể:
R_renewable = +1.0 × (renewable_used / base_demand)     # Thưởng dùng renewable
R_grid      = -2.0 × (grid_purchased / base_demand) × normalized_price  # Phạt mua grid
R_unmet     = -5.0 × (unmet_demand / base_demand)       # Phạt nặng nếu không đủ
R_battery   = -0.1 × battery_activity                    # Phạt nhẹ hao mòn pin
R_bonus     = +0.5 nếu không mua grid khi giá cao        # Bonus tiết kiệm
```

**Justification:**

- Unmet demand penalty cao nhất (-5.0) vì đảm bảo reliability là ưu tiên hàng đầu
- Grid purchase penalty (-2.0) khuyến khích dùng renewable
- Renewable reward (+1.0) thúc đẩy sử dụng năng lượng sạch

### 2.5 Transition Dynamics

**Battery Update:**

```
B_{t+1} = clip(B_t + charge × efficiency - discharge, 0, capacity)
efficiency = 0.95 (5% loss khi sạc/xả)
```

**Demand Pattern:**

```
demand(t) = base_demand × (0.5 + 0.3 × morning_peak + 0.4 × evening_peak) + noise
morning_peak = exp(-(t - 8)² / 8)   # Peak 8:00 AM
evening_peak = exp(-(t - 19)² / 8)  # Peak 7:00 PM
```

**Solar Generation:**

```
solar(t) = max_solar × sin(π × (t - 6) / 12) × weather_factor, nếu 6 ≤ t ≤ 18
         = 0, nếu t < 6 hoặc t > 18
```

### 2.6 Episode Termination

Episode kết thúc khi:

1. Hết 24 giờ (1 ngày)
2. Pin cạn kiệt VÀ unmet demand > 50%

---

## PHẦN 3: THUẬT TOÁN RL VÀ IMPLEMENTATION (25%)

### 3.1 Tại Sao Chọn DQN?

**So sánh các thuật toán:**

| Thuật toán | Ưu điểm | Nhược điểm | Phù hợp? |
|------------|---------|------------|----------|
| Q-Learning | Đơn giản | Không scale với high-dim state | ❌ |
| **DQN** | Handle continuous state, stable | Cần tuning | ✅ |
| Policy Gradient | Cho continuous action | Variance cao, sample inefficient | ⚠️ |
| Actor-Critic | Kết hợp cả hai | Phức tạp | ⚠️ |

**DQN phù hợp vì:**

- State space liên tục (8D) → cần function approximation
- Action space discrete (5 actions) → Q-learning hiệu quả
- Experience replay giúp sample efficient
- Target network giúp stable training

### 3.2 Kiến Trúc Neural Network

```
Input Layer    Hidden Layer 1   Hidden Layer 2   Hidden Layer 3   Output Layer
    (8)            (256)            (256)            (128)            (5)
    ○              ○ ○ ○            ○ ○ ○            ○ ○ ○            ○
    ○       →      ○ ○ ○     →     ○ ○ ○     →     ○ ○ ○     →     ○
    ○              ○ ○ ○            ○ ○ ○            ○ ○ ○            ○
   ...             ...              ...              ...             ...
    ○              ○ ○ ○            ○ ○ ○            ○ ○ ○            ○
            ReLU           ReLU           ReLU           (Linear)
```

**Giải thích:**

- **ReLU activation**: f(x) = max(0, x), giúp non-linearity và tránh vanishing gradient
- **Không có activation ở output**: Q-values có thể âm hoặc dương
- **Xavier initialization**: Khởi tạo weights để gradients ổn định

### 3.3 Experience Replay

```
┌──────────────────────────────────────────────────────────────┐
│                    REPLAY BUFFER                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ (s₁, a₁, r₁, s'₁, done₁)                               │ │
│  │ (s₂, a₂, r₂, s'₂, done₂)                               │ │
│  │ (s₃, a₃, r₃, s'₃, done₃)                               │ │
│  │ ...                                                     │ │
│  │ (sₙ, aₙ, rₙ, s'ₙ, doneₙ)     Capacity: 100,000         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                Random Sample (batch_size = 64)               │
│                          ▼                                   │
│                  ┌───────────────┐                           │
│                  │  Mini-Batch   │                           │
│                  │ for Training  │                           │
│                  └───────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

**Tại sao cần Experience Replay:**

1. **Decorrelation**: Samples liên tiếp có correlation cao → unstable training
2. **Sample efficiency**: Mỗi transition được học nhiều lần
3. **Stable learning**: Diverse batches → gradients ổn định hơn

### 3.4 Target Network

```
┌───────────────────┐      ┌───────────────────┐
│   Q-Network       │      │  Target Network   │
│   θ (online)      │      │  θ⁻ (frozen)      │
│                   │      │                   │
│ Used for:         │      │ Used for:         │
│ - Select action   │      │ - Calculate target│
│ - Update weights  │      │ - Không update    │
└─────────┬─────────┘      └─────────┬─────────┘
          │                          │
          │      Copy weights        │
          │  (every 1000 steps)      │
          └──────────────────────────┘
```

**Công thức cập nhật:**

```
Target:    y = r + γ × max_a' Q_target(s', a')
Loss:      L = (Q(s, a) - y)²
Update:    θ ← θ - α × ∇_θ L
```

### 3.5 Epsilon-Greedy Exploration

```
        ε = 1.0 (ban đầu)
          │
          │ ───────────────────────────── ε-decay = 0.995
          │    \
          │     \
          │      \
          │       \_____________________ ε_min = 0.01
          │
Episode:  0   100  200  300  400  500
```

**Chiến lược:**

- **ε = 1.0**: Explore hoàn toàn (random actions)
- **ε → 0.01**: Exploit nhiều hơn (best Q-value actions)
- Decay mỗi episode: ε ← ε × 0.995

### 3.6 Hyperparameters

| Parameter | Value | Lý do |
|-----------|-------|-------|
| Learning rate | 1e-4 | Đủ nhỏ để stable với continuous state |
| Gamma (γ) | 0.99 | Quan tâm long-term (năng lượng cần planning) |
| Batch size | 64 | Balance giữa speed và stability |
| Buffer size | 100,000 | Đủ lớn để diverse experiences |
| Target update | 1000 steps | Đủ thường xuyên nhưng không quá fast |
| Epsilon decay | 0.995 | Giảm dần đều qua 500 episodes |

### 3.7 Training Process (Pseudocode)

```python
for episode in range(500):
    state = env.reset()
    total_reward = 0
    
    for step in range(24):  # 24 giờ
        # 1. Select action (ε-greedy)
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q_network(state))
        
        # 2. Execute action
        next_state, reward, done = env.step(action)
        
        # 3. Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        
        # 4. Sample and update
        if buffer.is_ready():
            batch = buffer.sample(64)
            
            # Calculate target
            with no_grad():
                next_q = target_network(batch.next_states).max()
                target = batch.rewards + gamma * next_q * (1 - batch.dones)
            
            # Calculate loss and backprop
            current_q = q_network(batch.states)[batch.actions]
            loss = SmoothL1Loss(current_q, target)
            optimizer.step()
        
        # 5. Update target network
        if step % 1000 == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        state = next_state
        total_reward += reward
    
    # 6. Decay epsilon
    epsilon = max(0.01, epsilon * 0.995)
```

---

## PHẦN 4: PHÂN TÍCH TỐI ƯU HÓA AI (15%)

### 4.1 Agent Đã Tối Ưu Như Thế Nào?

**Quan sát từ Demo 24 giờ:**

| Giai đoạn | Giờ | Hành vi Agent | Lý do |
|-----------|-----|---------------|-------|
| Đêm | 0-6 | Renewable+Grid | Wind generation cao, giá grid thấp |
| Sáng | 7-9 | Renewable+Discharge | Peak price, dùng pin thay vì mua grid |
| Trưa | 10-14 | Charge | Solar cao nhất, sạc pin lên 100% |
| Chiều | 15-17 | Mixed | Chuyển tiếp, duy trì pin |
| Tối | 18-21 | Renewable+Discharge | Peak price cao nhất, xả pin |
| Đêm | 22-23 | Renewable+Discharge | Giá giảm, vẫn ưu tiên dùng pin |

### 4.2 Trade-offs Được Cân Bằng

#### 1. Cost vs Renewable Usage

```
Trade-off: Đôi khi mua grid rẻ hơn chờ solar/wind

Agent's Solution: 
- Chỉ mua grid khi giá thấp (off-peak hours)
- Ưu tiên renewable ngay cả khi hiệu quả thấp hơn
- Kết quả: 79% renewable usage, chỉ $4.45/ngày
```

#### 2. Battery Wear vs Storage Value

```
Trade-off: Sạc/xả nhiều làm hao mòn pin, nhưng không dùng thì lãng phí

Agent's Solution:
- Chỉ sạc khi renewable dư (trưa)
- Chỉ xả khi peak price (sáng sớm, tối)
- Tránh charge/discharge liên tục
```

#### 3. Immediate Reward vs Future Planning

```
Trade-off: Dùng pin ngay lấy reward, hay giữ cho peak demand?

Agent's Solution:
- γ = 0.99 → quan tâm future reward
- Sạc đầy pin trước peak hours
- Xả pin đúng lúc giá cao nhất
```

### 4.3 Learning Convergence Analysis

**Từ Training Logs (Lần chạy mới nhất - Feb 2026):**

```
Episode   10 | Reward:   -3.10 | Eps: 0.951  ← Đang explore
Episode   20 | Reward:    1.06 | Eps: 0.905  ← Bắt đầu học
Episode   50 | Reward:   -1.87 | Eps: 0.778  ← Chưa stable
Episode   80 | Reward:    1.77 | Eps: 0.670  ← Improving
Episode  100 | Reward:    2.62 | Eps: 0.606  ← Near optimal

Training time: 1.5 seconds (CPU)
Best episode reward: 13.37
```

**Observation:**

- Epsilon giảm → Agent exploit more → Reward tăng
- Renewable usage tăng từ 45.8% → 59.8% trong 100 episodes
- Training rất nhanh (~70 episodes/second trên CPU)

### 4.4 Hạn Chế Của Approach

| Hạn chế | Giải thích | Possible Solution |
|---------|------------|-------------------|
| **Discrete actions** | Không thể control chính xác kW | Continuous action space |
| **Single episode pattern** | Chỉ train trên 1 mùa | Thêm seasonal variation |
| **No demand forecasting** | Không biết demand tương lai | Add LSTM for prediction |
| **Overfitting risk** | Có thể overfit đến specific patterns | Regularization, more data |

---

## PHẦN 5: KẾT QUẢ VÀ ĐÁNH GIÁ (15%)

### 5.1 Performance Metrics

#### So sánh với Random Baseline (Kết quả mới nhất - Feb 2026)

| Metric | Trained Agent | Random | Improvement |
|--------|---------------|--------|-------------|
| Mean Episode Reward | 14.75 | -3.34 | **+541.1%** |
| Daily Grid Cost | $1.26 | $16.42 | **-92.3%** |
| Renewable Usage | 82.5% | 47.8% | +34.7pp |
| Demand Satisfaction | 96.6% | 83.9% | +12.7pp |
| Unmet Demand Ratio | 3.4% | 16.1% | -12.7pp |

### 5.2 Biểu Đồ Training Curves

*(Xem file: evaluation_results/training_curves.png)*

**Phân tích:**

- **Reward curve**: Tăng từ âm lên dương, bắt đầu converge ~episode 50
- **Cost curve**: Giảm mạnh từ ~$18 xuống ~$1.26
- **Renewable ratio**: Tăng từ 47.8% lên 82.5%
- **Epsilon decay**: Giảm mượt từ 1.0 xuống 0.01

### 5.3 Episode Analysis (24h Operation)

*(Xem file: evaluation_results/episode_analysis.png)*

**Patterns học được:**

1. **Charge at noon**: Battery từ 50% → 100% từ 8AM-2PM
2. **Hold during transition**: Giữ 100% từ 2PM-5PM
3. **Discharge at peak**: Xả từ 100% → 21% từ 5PM-11PM

### 5.4 Failure Scenarios Analysis

| Scenario | What happens | Agent behavior | Result |
|----------|--------------|----------------|--------|
| Cloudy day | Solar = 0 | Rely on wind + grid | Cost ↑ 20% |
| High demand spike | Demand 2x normal | Discharge + grid | 5% unmet |
| Battery low + peak | Battery < 10% | Force buy grid | Cost ↑, penalty |

### 5.5 Comparative Analysis

**vs Rule-based heuristic:**

```
Simple rule: "Always use renewable first, then battery, then grid"

Results:
- Rule-based reward: ~8.5
- DQN agent reward: ~13.9
- Improvement: +64%

Reason: DQN learns WHEN to charge/discharge optimally
```

---

## PHẦN 6: XEM XÉT ĐẠO ĐỨC, THỰC TIỄN VÀ TƯƠNG LAI (10%)

### 6.1 Ethical Considerations

#### 1. Fairness in Energy Distribution

```
Concern: AI có thể ưu tiên efficiency over equity
         → Một số users có thể bị blackout nhiều hơn

Mitigation:
- Đảm bảo unmet_demand penalty đủ cao
- Monitoring fairness metrics across user groups
- Human oversight trong critical decisions
```

#### 2. Automated Decision-Making Risks

```
Concern: Full automation không có human oversight
         → Lỗi AI có thể gây blackout lớn

Mitigation:
- Luôn có manual override option
- Alert system khi agent behavior bất thường
- Fallback to rule-based khi confidence thấp
```

#### 3. Privacy Concerns

```
Concern: Demand data có thể reveal user behavior
         → Privacy violation

Mitigation:
- Aggregate data instead of individual
- Differential privacy trong training
- Clear data usage policies
```

### 6.2 Practical Deployment Issues

#### 1. Scalability

```
Current: Single microgrid, simulated data
Real-world:
- Multiple interconnected microgrids
- Real-time data từ sensors
- Coordination between agents

Solution: Multi-agent RL, distributed training
```

#### 2. Real-time Requirements

```
Current: Batch processing, hourly decisions
Real-world:
- Millisecond response time needed
- Real-time market prices
- Instant demand changes

Solution: Edge computing, model compression
```

#### 3. Sensor Reliability

```
Problem: Sensors có thể fail hoặc give wrong readings
         → Agent receives incorrect state

Solution:
- Redundant sensors
- Anomaly detection
- Robust RL (train with noise)
```

### 6.3 Future Enhancements

#### 1. Multi-Agent RL

```
Scenario: Multiple microgrids trading energy

Approach:
- Each microgrid = 1 agent
- Cooperative/competitive learning
- Shared rewards for grid stability
```

#### 2. Demand Forecasting Integration

```
Current: Agent only sees current demand
Enhanced:
- LSTM to predict next 24h demand
- Include weather forecast
- Better planning capability
```

#### 3. IoT Integration

```
Current: Simulated sensors
Enhanced:
- Real smart meters
- Weather APIs
- Real-time pricing APIs
- Plant condition monitoring
```

#### 4. Continuous Action Space

```
Current: 5 discrete actions
Enhanced:
- Continuous control: "charge 15.3 kWh"
- Use PPO or SAC algorithms
- More fine-grained optimization
```

### 6.4 Recommendations for Ethical Deployment

1. **Transparency**: Explain agent decisions to operators
2. **Auditing**: Regular review of agent performance
3. **Backup systems**: Rule-based fallback always available
4. **Human-in-the-loop**: Critical decisions require approval
5. **Continuous monitoring**: Real-time anomaly detection

---

## TÀI LIỆU THAM KHẢO

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning." ICLR.
3. Vazquez-Canteli, J. R., & Nagy, Z. (2019). "Reinforcement learning for demand response: A review." Applied Energy.
4. François-Lavet, V., et al. (2018). "An Introduction to Deep Reinforcement Learning." Foundations and Trends in Machine Learning.

---

## PHỤ LỤC: CODE SNIPPETS

### A1. Environment Step Function

```python
def step(self, action):
    # Get current state
    demand = self._get_demand(self.current_hour)
    solar = self._get_solar_generation(self.current_hour)
    wind = self._get_wind_generation(self.current_hour)
    
    # Process action
    if action == 4:  # Renewable + Grid
        renewable_used = min(solar + wind, demand)
        grid_purchased = demand - renewable_used
    
    # Calculate reward
    reward = self._calculate_reward(...)
    
    return next_state, reward, done, info
```

### A2. DQN Update Function

```python
def update(self):
    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(64)
    
    # Current Q values
    current_q = q_network(states).gather(1, actions)
    
    # Target Q values
    with torch.no_grad():
        next_q = target_network(next_states).max(dim=1)[0]
        target_q = rewards + 0.99 * next_q * (1 - dones)
    
    # Update
    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### A3. Full Source Code

Xem thư mục: `/Volumes/DATA/workspace/RL-ideas/microgrid_rl/`
