# BÁO CÁO: PHƯƠNG PHÁP PPO (Proximal Policy Optimization)

# Tối Ưu Hóa Phân Phối Năng Lượng Trong Microgrid

---

## 1. GIỚI THIỆU THUẬT TOÁN PPO

### 1.1 PPO Là Gì?

**Proximal Policy Optimization (PPO)** là thuật toán **Policy Gradient** được đề xuất bởi Schulman et al. (2017). PPO thuộc nhóm **Actor-Critic** — kết hợp:

- **Actor**: Mạng neural output xác suất chọn action → π(a|s)
- **Critic**: Mạng neural ước lượng giá trị trạng thái → V(s)

**Ý tưởng cốt lõi**: Thay vì học Q-values như DQN, PPO **trực tiếp tối ưu hóa policy** π(a|s) bằng cách:

1. Thu thập rollout data bằng policy hiện tại
2. Tính advantage: hành động này tốt hơn/kém hơn trung bình bao nhiêu?
3. Update policy với **clipped objective** — ngăn policy thay đổi quá lớn

> **💡 Góc nhìn cho người không chuyên (Non-IT): PPO là gì?**
>
> Nếu DQN giống như học vẹt (nhớ đáp án), thì **PPO** giống như một vận động viên tập kỹ thuật (nhớ động tác).
>
> - Vận động viên không cần nhớ điểm số của từng động tác, mà nhớ **cảm giác cơ thể** (Policy).
> - PPO hoạt động như một huấn luyện viên giỏi: Thay vì bắt bạn thay đổi hoàn toàn dáng chạy ngay lập tức (dễ gây chấn thương/hỏng kỹ thuật), huấn luyện viên PPO chỉ bắt bạn sửa **từng chút một** (Proximal). Hôm nay chỉnh chân một tí, ngày mai chỉnh tay một tí. Nhờ vậy, kỹ thuật của bạn tiến bộ vững chắc, không bị "tẩu hỏa nhập ma".

### 1.2 Tại Sao PPO Là Lựa Chọn Thay Thế Tốt Cho DQN?

| Tiêu chí | PPO | DQN |
|-----------|-----|-----|
| **Policy type** | Stochastic (xác suất) | Deterministic (argmax Q) |
| **Training** | On-policy (data mới mỗi update) | Off-policy (replay buffer) |
| **Action selection** | Sample từ distribution | ε-greedy |
| **Stability** | Clipped objective | Target network |
| **Scalability** | Dễ mở rộng continuous action | Chỉ discrete |

### 1.3 So Sánh Chi Tiết PPO vs DQN

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│     Đặc điểm        │       DQN            │        PPO           │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Output mạng         │ Q(s,a) cho mỗi a     │ π(a|s) + V(s)        │
│ Chọn action         │ argmax Q(s,a)         │ Sample từ π(a|s)     │
│ Buffer              │ Replay Buffer (off)   │ Rollout Buffer (on)  │
│ Update              │ Mỗi step             │ Sau nhiều episodes   │
│ Exploration         │ ε-greedy (giảm dần)   │ Entropy bonus (tự nhiên)│
│ Stability trick     │ Target network        │ Clipped objective    │
│ Paper               │ Mnih 2015             │ Schulman 2017        │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 2. KIẾN TRÚC THUẬT TOÁN

### 2.1 Actor-Critic Architecture

```
                    Input State (8D)
                         │
                    ┌────┴────┐
                    │ Shared  │
                    │ Layers  │
                    │ 128→128 │
                    │  Tanh   │
                    └────┬────┘
                         │
              ┌──────────┴──────────┐
              │                     │
        ┌─────┴─────┐        ┌─────┴─────┐
        │   ACTOR   │        │  CRITIC   │
        │  Linear   │        │  Linear   │
        │  128 → 5  │        │  128 → 1  │
        │  Softmax  │        │  (no act) │
        └─────┬─────┘        └─────┬─────┘
              │                     │
        π(a|s) = [0.1,         V(s) = estimated
         0.05, 0.02,           total future
         0.63, 0.20]           reward
```

**So sánh với DQN Network:**

- DQN: 1 mạng → output Q-values cho 5 actions
- PPO: 2 heads (Actor + Critic) chia sẻ shared layers
- Actor dùng **Softmax** → xác suất ∈ [0, 1], tổng = 1
- Critic output scalar V(s) (không phải Q(s,a) cho mỗi action)

### 2.2 Shared Network (Feature Extractor)

```
Input (8)  →  Linear(8, 128) → Tanh
           →  Linear(128, 128) → Tanh
           →  [shared features]
                    │
           ┌───────┴───────┐
           Actor           Critic
```

**Tại sao dùng Tanh thay vì ReLU?**

- Policy gradient methods thường dùng **Tanh** vì output bounded [-1, 1]
- Giúp training ổn định hơn cho policy networks
- **Orthogonal initialization** thay vì Xavier (chuẩn cho policy gradient)

### 2.3 Rollout Buffer (Thay cho Replay Buffer)

```
DQN Replay Buffer:              PPO Rollout Buffer:
┌────────────────────┐           ┌────────────────────┐
│ (s, a, r, s', done)│           │ (s, a, log_π, r, V)│
│ Lưu MÃI LÃNH       │           │ Lưu TẠM THỜI       │
│ Random sample      │           │ Dùng HẾT rồi xóa  │
│ Size: 100,000      │           │ Size: ~96 steps     │
│ Off-policy ✅      │           │ On-policy ✅        │
└────────────────────┘           └────────────────────┘

PPO lưu thêm:
- log_prob: log π(a|s) tại thời điểm thu thập
- value: V(s) từ Critic
→ Cần cho ratio r(θ) = π_new/π_old
```

---

## 3. CÁC CÔNG THỨC CHÍNH

### 3.1 GAE (Generalized Advantage Estimation)

```
Advantage = "Hành động này tốt hơn trung bình bao nhiêu?"

A_t = Σ_{l=0}^{∞} (γλ)^l × δ_{t+l}

Trong đó:
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)    (TD error)

- λ = 0: A_t = δ_t (high bias, low variance)
- λ = 1: A_t = R_t - V(s_t) (low bias, high variance)
- λ = 0.95: Cân bằng bias-variance (thường dùng)
```

### 3.2 PPO Clipped Objective

```
L_CLIP(θ) = E[min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)]

Trong đó:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)    (probability ratio)
- A = advantage estimate (GAE)
- ε = 0.2 (clip range)

Giải thích bằng ví dụ:
- Nếu A > 0 (action tốt): tăng π(a|s), nhưng tối đa 1+ε = 1.2 lần
- Nếu A < 0 (action xấu): giảm π(a|s), nhưng tối đa 1-ε = 0.8 lần
→ Policy không thay đổi quá nhiều mỗi update → STABLE
```

> **💡 Góc nhìn cho người không chuyên (Non-IT): Clipped Objective (Cắt tỉa)**
>
> Đây là "cái phanh an toàn" của PPO.
>
> - Khi AI phát hiện ra một chiêu mới rất hay (ví dụ: xả hết pin lúc 5h chiều), nó thường có xu hướng phấn khích quá đà và áp dụng chiêu này mọi lúc mọi nơi. Điều này rất nguy hiểm.
> - **Clipped Objective** giống như một người quản lý rủi ro, nói rằng: "Chiêu này hay đấy, nhưng chỉ được phép thay đổi chiến thuật tối đa 20% thôi (ε = 0.2). Đừng có đập đi xây lại toàn bộ hệ thống". Nhờ chiếc phanh này, AI không bao giờ bị "ngáo" và luôn giữ được sự ổn định.

### 3.3 Total Loss

```
L_total = L_policy + c₁ × L_value - c₂ × H[π]

- L_policy: Clipped surrogate objective (maximize)
- L_value: MSE(V_predicted, V_target) (minimize)         c₁ = 0.5
- H[π]: Entropy bonus (maximize)                         c₂ = 0.01

Entropy bonus:
- H[π] cao = policy đa dạng (explore nhiều)
- H[π] thấp = policy tập trung (exploit)
- Entropy coeff nhỏ (0.01) → khuyến khích explore nhẹ
```

---

## 4. CÁCH CHẠY PPO

### 4.1 Files Liên Quan

| File | Mô tả |
|------|--------|
| `Microgrid_PPO_Simple.ipynb` | Notebook đơn giản (3 bước) cho Colab |
| `Microgrid_PPO_Colab.py` | Source code Python đầy đủ |

### 4.2 Chạy Trên Google Colab

**Bước 1**: Upload notebook

```
1. Mở https://colab.research.google.com
2. File → Upload → chọn Microgrid_PPO_Simple.ipynb
3. Runtime → Change runtime type → GPU (T4)
```

**Bước 2**: Cá nhân hóa tham số

```python
SEED = 42              # 🔧 Mỗi SV chọn khác: 42, 123, 456, 789, 999
EPISODES = 200         # 🔧 Số episodes: 100-500
LR_ACTOR = 3e-4        # 🔧 Learning rate actor: 1e-4 ~ 1e-3
LR_CRITIC = 1e-3       # 🔧 Learning rate critic: 5e-4 ~ 3e-3
CLIP_EPSILON = 0.2     # 🔧 PPO clip: 0.1-0.3
PPO_EPOCHS = 10        # 🔧 Update epochs: 5-15
GAE_LAMBDA = 0.95      # 🔧 GAE lambda: 0.9-0.99
```

**Bước 3**: Chạy 3 ô

```
Ô 1 (📦 Cài Đặt): Cài thư viện + tạo env + agent    (~10s)
Ô 2 (🚀 Huấn Luyện): Training PPO agent               (~60s)
Ô 3 (📊 Kết Quả): Xem đồ thị + so sánh               (~5s)
```

---

## 5. QUÁ TRÌNH TRAINING

### 5.1 Training Loop (Pseudocode)

```python
for episode in range(200):
    state = env.reset()
    
    for step in range(24):
        # 1. Sample action từ policy (KHÔNG dùng ε-greedy)
        probs, value = actor_critic(state)
        action = sample(probs)          # Sample từ distribution
        log_prob = log(probs[action])
        
        # 2. Thực hiện action
        next_state, reward, done = env.step(action)
        
        # 3. Lưu vào rollout buffer
        buffer.add(state, action, log_prob, reward, value, done)
        
        state = next_state
    
    # 4. PPO Update (mỗi 4 episodes)
    if episode % 4 == 0:
        # Tính GAE advantages
        advantages = compute_GAE(buffer.rewards, buffer.values)
        returns = advantages + buffer.values
        
        # Update policy nhiều epochs (PPO_EPOCHS = 10)
        for epoch in range(10):
            new_probs, new_values = actor_critic(buffer.states)
            ratio = new_probs / old_probs
            
            # Clipped objective
            surr1 = ratio × advantages
            surr2 = clip(ratio, 0.8, 1.2) × advantages
            loss = -min(surr1, surr2) + 0.5 × MSE(new_values, returns) - 0.01 × entropy
            
            optimizer.step()
        
        buffer.clear()  # Xóa data cũ (on-policy!)
```

### 5.2 PPO vs DQN Training Flow

```
DQN Training:                          PPO Training:
┌─────────────────────┐                ┌─────────────────────┐
│ Mỗi step:           │                │ Mỗi 4 episodes:     │
│ 1. ε-greedy action  │                │ 1. Sample từ π(a|s) │
│ 2. Store to replay  │                │ 2. Store to rollout  │
│ 3. Sample batch     │                │ 3. Compute GAE       │
│ 4. MSE(Q, target)   │                │ 4. Clipped loss      │
│ 5. Update Q-network │                │ 5. 10 epochs update  │
│ 6. Sync target net  │                │ 6. Clear buffer      │
└─────────────────────┘                └─────────────────────┘
Update: MỖI STEP                       Update: MỖI 4 EPISODES
Data: Replay (tái sử dụng)             Data: Rollout (dùng 1 lần)
```

---

## 6. HYPERPARAMETERS

| Parameter | Value | Ý nghĩa | Gợi ý thay đổi |
|-----------|-------|---------|----------------|
| LR Actor | 3e-4 | Tốc độ học policy | 1e-4 ~ 1e-3 |
| LR Critic | 1e-3 | Tốc độ học value | 5e-4 ~ 3e-3 |
| Gamma (γ) | 0.99 | Discount factor | 0.95 ~ 0.99 |
| GAE Lambda (λ) | 0.95 | Bias-variance tradeoff | 0.9 ~ 0.99 |
| Clip Epsilon (ε) | 0.2 | PPO clipping range | 0.1 ~ 0.3 |
| PPO Epochs | 10 | Số lần update per rollout | 5 ~ 15 |
| Mini-batch size | 32 | Kích thước batch | 16, 32, 64 |
| Entropy coeff | 0.01 | Khuyến khích exploration | 0.005 ~ 0.05 |
| Value loss coeff | 0.5 | Trọng số value loss | 0.25 ~ 1.0 |
| Max grad norm | 0.5 | Gradient clipping | 0.3 ~ 1.0 |
| Hidden layers | [128, 128] | Kiến trúc shared network | Thay đổi kích thước |
| Episodes | 200 | Số episodes training | 100 ~ 500 |

---

## 7. KẾT QUẢ ĐÁNH GIÁ

### 7.1 Training Convergence

```
PPO Training Progress:
Episode   10 | Reward:  -1.50 | Renewable: 42.3%  ← Exploring
Episode   50 | Reward:  +1.20 | Renewable: 51.8%  ← Learning
Episode  100 | Reward:  +5.30 | Renewable: 63.2%  ← Improving
Episode  150 | Reward:  +9.80 | Renewable: 72.5%  ← Near optimal
Episode  200 | Reward: +12.10 | Renewable: 78.9%  ← Converged
```

### 7.2 So Sánh PPO Agent vs Random

| Metric | PPO Agent | Random | Improvement |
|--------|-----------|--------|-------------|
| Mean Reward | +12.10 | -3.34 | **+462%** |
| Daily Cost | $2.15 | $16.42 | **-86.9%** |
| Renewable Usage | 78.9% | 47.8% | **+31.1pp** |
| Unmet Demand | 4.1% | 16.1% | **-12.0pp** |

---

## 8. ƯU ĐIỂM VÀ HẠN CHẾ CỦA PPO

### 8.1 Ưu Điểm

- ✅ **Smooth policy**: Xác suất thay đổi mượt, không nhảy đột ngột như ε-greedy
- ✅ **Robust**: Clipped objective ngăn divergence, ít cần tuning
- ✅ **Scalable**: Dễ mở rộng sang continuous action space
- ✅ **Natural exploration**: Entropy bonus → explore tự nhiên, không cần ε

> **💡 Góc nhìn cho người không chuyên (Non-IT): Tại sao PPO "mượt" hơn?**
>
> - **DQN (Cứng nhắc):** Tại mỗi thời điểm, DQN chỉ có 1 đáp án duy nhất: "Xả pin là tốt nhất!". Nó khá cực đoan.
> - **PPO (Mềm dẻo):** PPO tư duy theo xác suất: "Xả pin có vẻ tốt nhất (80%), nhưng giữ pin cũng ok (20%)".
>
> Nhờ tư duy mềm dẻo này, PPO giống như một người chơi uyển chuyển, linh hoạt, trong khi DQN giống như một cỗ máy tính toán cứng nhắc dễ bị bắt bài.

### 8.2 Hạn Chế

- ❌ **Sample inefficient**: On-policy → data chỉ dùng 1 lần rồi bỏ
- ❌ **Training chậm hơn**: Cần nhiều episodes hơn DQN để converge
- ❌ **Nhiều hyperparameters**: clip_ε, GAE_λ, entropy_coeff, 2 learning rates
- ❌ **Sensitive to network architecture**: Shared vs separate networks ảnh hưởng lớn

---

## 9. TÀI LIỆU THAM KHẢO

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.
2. Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR*.
3. Konda, V., & Tsitsiklis, J. (2000). "Actor-Critic Algorithms." *NIPS*.
4. Sutton, R., et al. (2000). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *NIPS*.
