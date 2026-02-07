# ⚔️ So Sánh Chi Tiết: DQN vs PPO trong Tối Ưu Hóa Năng Lượng Microgrid

Tài liệu này cung cấp một bản đánh giá sâu sắc về hai thuật toán Reinforcement Learning được sử dụng trong dự án: **Deep Q-Network (DQN)** và **Proximal Policy Optimization (PPO)**.

---

## 1. Tổng Quan Về Hai Thuật Toán

| Đặc điểm | 🔵 DQN (Deep Q-Network) | 🟣 PPO (Proximal Policy Optimization) |
| :--- | :--- | :--- |
| **Loại thuật toán** | **Value-based**: Học hàm giá trị $Q(s, a)$ để ước lượng phần thưởng tích lũy. | **Policy-based (Actor-Critic)**: Học trực tiếp chiến thuật $\pi(a\|s)$ và hàm giá trị $V(s)$. |
| **Cơ chế học** | **Off-Policy**: Có thể học từ dữ liệu cũ (Experience Replay). | **On-Policy**: Chỉ học từ dữ liệu mới nhất do chính nó tạo ra. |
| **Không gian hành động** | **Rời rạc (Discrete)**: Chỉ chọn 1 trong $N$ hành động (Vd: Bật/Tắt). | **Cả hai**: Hỗ trợ tốt cả Rời rạc và Liên tục (Continuous - Vd: Chỉnh van 50%). |
| **Độ phức tạp cài đặt** | Trung bình (cần Replay Buffer, Target Network). | Cao (cần GAE, Clipping, 2 mạng Actor-Critic). |
| **Sự ổn định** | Thấp hơn (dễ bị dao động, khó hội tụ nếu hyperparams sai). | Rất cao (nhờ cơ chế Clipping giới hạn update). |

---

## 2. Phân Tích Cơ Chế Hoạt Động (Deep Dive)

### 🔵 DQN: Học Qua Ký Ức (Flashcards Analogy)

**Cách hoạt động:**

1. **Replay Buffer (Bộ nhớ hồi tưởng):** DQN lưu trữ mọi trải nghiệm $(S, A, R, S')$ vào bộ nhớ. Khi học, nó bốc ngẫu nhiên một lô dữ liệu (batch) để training.
    * *Ưu điểm:* **Sample Efficiency** cao. Một trải nghiệm có thể được học đi học lại nhiều lần. Phá vỡ sự tương quan thời gian giữa các mẫu.
2. **Target Network (Mạng mục tiêu):** Để tránh việc "vừa học vừa sửa đáp án", DQN dùng một mạng riêng (Target) để tính nhãn $y$, mạng này chỉ được cập nhật sau mỗi vài nghìn bước.

> **Góc nhìn Non-IT:**  
> DQN giống như **học ôn thi bằng Flashcards**. Bạn trộn lẫn các câu hỏi từ quá khứ để học (Replay), và bạn giữ nguyên đáp án trong một khoảng thời gian để không bị rối (Target Network).

### 🟣 PPO: Học Với Huấn Luyện Viên (Coach Analogy)

**Cách hoạt động:**

1. **Actor-Critic:** Mạng Actor quyết định hành động, mạng Critic đánh giá hành động đó tốt hay xấu ($V(s)$).
2. **Clipped Surrogate Objective:** PPO giới hạn mức độ thay đổi của policy trong mỗi bước update (thường là $\epsilon = 0.2$, tức không đổi quá 20%).
    * *Ưu điểm:* **Độ ổn định cực cao**. Tránh việc policy bị "sập" (collapse) do update quá đà, điều thường thấy ở các thuật toán Policy Gradient cũ.

> **Góc nhìn Non-IT:**  
> PPO giống như **Huấn luyện viên thể thao**. HLV chỉ chỉnh sửa tư thế của bạn từng chút một ("Thấp tay xuống một chút"), không bắt bạn đổi hoàn toàn cách chơi ngay lập tức để tránh chấn thương hoặc mất phong độ.

---

## 3. Hiệu Suất Trên Bài Toán Microgrid

Dựa trên thực nghiệm, dưới đây là so sánh hiệu quả của hai thuật toán đối với bài toán năng lượng:

### 🚀 Tốc độ hội tụ (Convergence Speed)

* **DQN:** Thường hội tụ nhanh hơn ở giai đoạn đầu nhờ tái sử dụng dữ liệu (Replay Buffer). Tuy nhiên, đường cong loss có thể dao động mạnh.
* **PPO:** Hội tụ chậm hơn và mượt mà hơn (monotonic improvement). Cần nhiều dữ liệu (môi trường tương tác) hơn để đạt cùng mức hiệu suất.

### 🎯 Chất lượng chính sách (Policy Quality)

* **DQN:** Có xu hướng tìm ra chiến thuật "cực đoan" (Bang-bang control) do bản chất `argmax` của Q-learning (Vd: Xả hết pin hoặc Sạc đầy pin).
* **PPO:** Có thể học được chiến thuật mềm dẻo hơn (Stochastic policy), đặc biệt nếu chuyển sang action liên tục (Vd: Xả 40% pin).

### 🛠️ Độ nhạy với Siêu tham số (Hyperparameters Sensitivity)

* **DQN:** Rất nhạy cảm. Cần tinh chỉnh kỹ `learning_rate`, `epsilon_decay`, `buffer_size`, `target_update_freq`. Nếu sai, mạng có thể không hội tụ (Q-value phân kỳ).
* **PPO:** Khá "trâu bò" (Robust). Các tham số mặc định (clip=0.2, gamma=0.99) thường hoạt động tốt trên nhiều bài toán khác nhau mà không cần chỉnh sửa nhiều.

---

## 4. Kết Luận & Khuyến Nghị

### Khi nào nên chọn DQN?

1. **Người mới bắt đầu:** DQN dễ hiểu, dễ debug hơn.
2. **Hành động rời rạc:** Bài toán chỉ cần Bật/Tắt thiết bị.
3. **Tài nguyên tính toán thấp:** DQN thường nhẹ hơn PPO một chút.
4. **Muốn tiết kiệm mẫu (Sample efficient):** Khi việc tương tác với môi trường tốn kém thời gian.

### Khi nào nên chọn PPO?

1. **Cần sự ổn định cao:** Không muốn đau đầu chỉnh hyperparams.
2. **Hành động liên tục:** Cần điều khiển công suất mịn (Vd: Điều khiển dòng sạc chính xác từng Ampe).
3. **Muốn kết quả SOTA (State-of-the-Art):** PPO hiện là chuẩn mực cho nhiều bài toán phức tạp (bao gồm cả ChatGPT).
4. **Môi trường ngẫu nhiên (Stochastic):** PPO xử lý nhiễu tốt hơn DQN.

### 🏆 Lựa chọn cho Đồ án Microgrid

Với bài toán này, **DQN là lựa chọn khởi đầu tốt nhất** vì tính trực quan và phù hợp với action space rời rạc (5 hành động). Tuy nhiên, **PPO là bước nâng cao đáng giá** để cải thiện độ ổn định và điểm số trong phần báo cáo.
