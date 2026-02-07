# BÁO CÁO: PHƯƠNG PHÁP DQN (Deep Q-Network)

# Tối Ưu Hóa Phân Phối Năng Lượng Trong Microgrid

---

## 1. GIỚI THIỆU THUẬT TOÁN DQN

### 1.1 DQN Là Gì?

**Deep Q-Network (DQN)** là thuật toán kết hợp Q-Learning truyền thống với Deep Neural Network, được đề xuất bởi Mnih et al. (2015) trong paper "Human-level control through deep reinforcement learning".

**Ý tưởng cốt lõi**: Dùng neural network để xấp xỉ hàm Q-value:

```
Q(s, a) ≈ Q_θ(s, a)    (θ = weights của neural network)
```

Q-value cho biết **tổng reward kỳ vọng** khi thực hiện action `a` tại state `s` và follow optimal policy sau đó.

### 1.2 Tại Sao Chọn DQN Cho Microgrid?

| Tiêu chí | Lý do |
|-----------|-------|
| **State space liên tục (8D)** | Neural network xử lý tốt continuous input |
| **Action space discrete (5)** | DQN được thiết kế cho discrete actions |
| **Sample efficiency** | Experience replay giúp tận dụng mỗi transition nhiều lần |
| **Stability** | Target network ngăn oscillation trong training |

### 1.3 So Sánh Với Thuật Toán Khác

| Thuật toán | Ưu điểm | Nhược điểm | Phù hợp? |
|------------|---------|------------|----------|
| Q-Learning | Đơn giản | Không scale với high-dim state | ❌ |
| **DQN** | Stable, sample efficient | Chỉ discrete actions | ✅ |
| Policy Gradient | Continuous actions | High variance | ⚠️ |
| PPO | Linh hoạt, robust | Phức tạp hơn, on-policy | ⚠️ |

---

## 2. KIẾN TRÚC THUẬT TOÁN

### 2.1 Các Thành Phần Chính

```
┌──────────────────────────────────────────────────────────────┐
│                     DQN ARCHITECTURE                          │
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐  │
│  │  Q-Network  │    │   Target    │    │  Replay Buffer   │  │
│  │  (online)   │    │  Network    │    │  (100K samples)  │  │
│  │  θ → update │    │  θ⁻→ frozen │    │  random sample   │  │
│  └─────────────┘    └─────────────┘    └──────────────────┘  │
│         │                  │                    │             │
│         │    copy every    │                    │             │
│         │   1000 steps     │                    │             │
│         └──────────────────┘                    │             │
│                                                  │             │
│  ┌───────────────────────────────────────────────┘            │
│  │  ε-greedy: random(ε) or argmax Q(s,a)(1-ε)               │
│  └────────────────────────────────────────────────────────── │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Neural Network Architecture

```
Input (8)  →  Linear(8, 256) → ReLU → Dropout(0.1)
           →  Linear(256, 256) → ReLU → Dropout(0.1)
           →  Linear(256, 128) → ReLU → Dropout(0.1)
           →  Linear(128, 5) → Q-values (no activation)

Output: Q(s, a₀), Q(s, a₁), Q(s, a₂), Q(s, a₃), Q(s, a₄)
```

**Giải thích:**

- **ReLU**: `f(x) = max(0, x)` — non-linearity, tránh vanishing gradient
- **Dropout(0.1)**: Regularization, tránh overfitting
- **No activation ở output**: Q-values có thể âm hoặc dương
- **Xavier initialization**: Weights khởi tạo cân bằng

### 2.3 Experience Replay Buffer

```
Tại sao cần?
- Samples liên tiếp có correlation cao → unstable training
- Replay buffer phá vỡ correlation bằng random sampling

Hoạt động:
1. Agent tương tác với env → thu (s, a, r, s', done)
2. Lưu vào buffer (size = 100,000)
3. Random sample batch (size = 64) để training
4. Mỗi sample được học nhiều lần (off-policy)
```

### 2.4 Target Network

```
Vấn đề: Q_target = r + γ × max Q(s', a')
         → Q dùng chính nó để tính target → oscillation!

Giải pháp: Dùng 2 mạng riêng biệt
- Q-Network (θ): Update liên tục mỗi step
- Target Network (θ⁻): Copy từ Q-Network mỗi 1000 steps

→ Target ổn định hơn → Training stable hơn
```

### 2.5 Double DQN (Cải tiến)

```
Vanilla DQN:  y = r + γ × max_a' Q_target(s', a')
              → Overestimation bias (Q-values bị inflate)

Double DQN:   a* = argmax_a' Q_online(s', a')    ← chọn action bằng online
              y  = r + γ × Q_target(s', a*)       ← đánh giá bằng target

→ Giảm overestimation → Q-values chính xác hơn
```

---

## 3. CÁCH CHẠY DQN

### 3.1 Files Liên Quan

| File | Mô tả |
|------|--------|
| `Microgrid_DQN_Colab.ipynb` | Notebook đầy đủ cho Google Colab |
| `Microgrid_DQN_Colab.py` | Source code Python tương ứng |
| `Microgrid_DQN_Simple.ipynb` | Notebook đơn giản (3 bước) |
| `run_training.py` | Script chạy local |

### 3.2 Chạy Trên Google Colab (Khuyến nghị)

**Bước 1**: Upload notebook lên Google Colab

```
1. Mở https://colab.research.google.com
2. File → Upload notebook → chọn Microgrid_DQN_Simple.ipynb
3. Runtime → Change runtime type → GPU (T4)
```

**Bước 2**: Cá nhân hóa tham số

```python
# Thay đổi các giá trị này cho bài làm riêng
SEED = 42              # Mỗi SV chọn seed khác: 42, 123, 456, 789, 999
EPISODES = 100         # Số episodes: 50-500
LEARNING_RATE = 0.0001 # Learning rate: 0.0001-0.001
```

**Bước 3**: Chạy 3 ô theo thứ tự

```
Ô 1 (📦 Cài Đặt): Cài thư viện + tạo env + agent    (~10 giây)
Ô 2 (🚀 Huấn Luyện): Training DQN agent               (~30 giây)
Ô 3 (📊 Kết Quả): Xem đồ thị + so sánh               (~5 giây)
```

### 3.3 Chạy Local (Optional)

```bash
# Cài đặt
pip install torch numpy matplotlib

# Chạy training
python run_training.py

# Output: evaluation_results/ (chứa biểu đồ)
```

---

## 4. QUÁ TRÌNH TRAINING

### 4.1 Training Loop (Pseudocode)

```python
for episode in range(500):
    state = env.reset()
    
    for step in range(24):  # 24 giờ mỗi ngày
        # 1. Chọn action (ε-greedy)
        if random() < epsilon:
            action = random_action()        # Explore
        else:
            action = argmax(Q(state))        # Exploit
        
        # 2. Thực hiện action
        next_state, reward, done = env.step(action)
        
        # 3. Lưu vào replay buffer
        buffer.push(state, action, reward, next_state, done)
        
        # 4. Sample batch và update Q-network
        batch = buffer.sample(64)
        target = reward + γ × max Q_target(next_state)
        loss = MSE(Q(state, action), target)
        optimizer.step()
        
        # 5. Copy weights sang target network (mỗi 1000 steps)
        if step_count % 1000 == 0:
            target_network ← q_network
    
    # 6. Giảm epsilon
    epsilon = max(0.01, epsilon × 0.995)
```

### 4.2 Epsilon Decay Schedule

```
ε = 1.0 ─────────\
                    \
                     \          ← ε × 0.995 mỗi episode
                      \
                       \_______ ε_min = 0.01
Episode:  0    100   200   300   400   500

Ý nghĩa:
- ε = 1.0: 100% random → khám phá toàn bộ action space
- ε = 0.5: 50% random, 50% best action → cân bằng explore/exploit
- ε = 0.01: 1% random → chủ yếu exploit policy đã học
```

### 4.3 Công Thức Cập Nhật

```
1. Q-value target:
   y = r + γ × max_a' Q_target(s', a')       (γ = 0.99)

2. Loss function:
   L(θ) = E[(Q_θ(s, a) - y)²]                (MSE Loss)

3. Gradient descent:
   θ ← θ - α × ∇_θ L(θ)                     (α = 0.0001)

4. Target network sync:
   θ⁻ ← θ     (mỗi 1000 steps)
```

---

## 5. HYPERPARAMETERS

| Parameter | Value | Ý nghĩa | Gợi ý thay đổi |
|-----------|-------|---------|----------------|
| Learning rate | 1e-4 | Tốc độ học | 1e-4 ~ 1e-3 |
| Gamma (γ) | 0.99 | Discount factor | 0.95 ~ 0.99 |
| Epsilon start | 1.0 | Exploration ban đầu | 0.9 ~ 1.0 |
| Epsilon end | 0.01 | Exploration tối thiểu | 0.01 ~ 0.05 |
| Epsilon decay | 0.995 | Tốc độ giảm ε | 0.990 ~ 0.998 |
| Batch size | 64 | Kích thước mini-batch | 32, 64, 128 |
| Buffer size | 100,000 | Replay buffer capacity | 50K ~ 200K |
| Target update | 1000 steps | Tần suất sync target | 500 ~ 2000 |
| Hidden layers | [256, 256, 128] | Kiến trúc mạng | Thay đổi kích thước |
| Episodes | 500 | Số lần train | 100 ~ 700 |

---

## 6. KẾT QUẢ ĐÁNH GIÁ

### 6.1 Training Convergence

```
Training Progress:
Episode   10 | Reward:  -3.10 | ε: 0.951  ← Khám phá (explore)
Episode   50 | Reward:  -1.87 | ε: 0.778  ← Bắt đầu học
Episode  100 | Reward:  +2.62 | ε: 0.606  ← Chính sách cải thiện
Episode  200 | Reward:  +8.45 | ε: 0.367  ← Gần optimal
Episode  500 | Reward: +13.37 | ε: 0.010  ← Converged
```

### 6.2 So Sánh Agent vs Random

| Metric | DQN Agent | Random | Improvement |
|--------|-----------|--------|-------------|
| Mean Reward | +14.75 | -3.34 | **+541%** |
| Daily Cost | $1.26 | $16.42 | **-92.3%** |
| Renewable Usage | 82.5% | 47.8% | **+34.7pp** |
| Unmet Demand | 3.4% | 16.1% | **-12.7pp** |

### 6.3 Policy Học Được (24h)

| Giờ | Hành vi | Lý do |
|-----|---------|-------|
| 0-6 (Đêm) | Renewable+Grid | Gió mạnh, giá grid thấp → mua grid rẻ |
| 7-9 (Sáng) | Renewable+Discharge | Peak price → xả pin thay vì mua grid |
| 10-14 (Trưa) | Charge | Solar cao nhất → sạc pin đầy |
| 15-17 (Chiều) | Mixed | Chuyển tiếp, duy trì pin |
| 18-21 (Tối) | Renewable+Discharge | Peak price → xả pin tối đa |
| 22-23 (Khuya) | Discharge | Giá giảm, dùng nốt pin dư |

---

## 7. ƯU ĐIỂM VÀ HẠN CHẾ CỦA DQN

### 7.1 Ưu Điểm

- ✅ **Sample efficient**: Replay buffer cho phép học từ data cũ (off-policy)
- ✅ **Stable training**: Target network + replay buffer giảm oscillation
- ✅ **Proven**: Đã được chứng minh hiệu quả trên nhiều bài toán (Atari games, robotics)
- ✅ **Đơn giản**: Dễ implement và debug

### 7.2 Hạn Chế

- ❌ **Chỉ discrete actions**: Không thể control chính xác kW (phải dùng DDPG/SAC)
- ❌ **Overestimation**: Q-values có thể bị inflate (Double DQN giảm nhưng không loại bỏ hoàn toàn)
- ❌ **Memory intensive**: Replay buffer chiếm nhiều RAM
- ❌ **Không song song**: Khó parallelize training

---

## 8. TÀI LIỆU THAM KHẢO

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
2. Van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." *AAAI*.
3. Lin, L.J. (1992). "Self-improving reactive agents based on RL, planning and teaching." *Machine Learning*, 8(3-4), 293-321.
4. François-Lavet, V., et al. (2018). "An Introduction to Deep Reinforcement Learning." *Foundations and Trends in ML*.
