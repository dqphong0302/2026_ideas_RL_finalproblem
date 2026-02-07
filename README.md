# 🔋 Microgrid Energy Optimization using Deep Reinforcement Learning

## Tối Ưu Hóa Năng Lượng Microgrid Sử Dụng Deep Reinforcement Learning (DQN & PPO)

> **Phiên bản 3.0 (Update 07/02/2026)**: Cập nhật hướng dẫn chi tiết về tư duy giải quyết bài toán, so sánh DL vs RL, và phân tích sâu thuật toán DQN vs PPO.

### 📁 Files

| File | Mô Tả | Đối Tượng |
|------|-------|-----------|
| **`Microgrid_DQN_Simple.ipynb`** | ⭐ DQN đơn giản, CHỈ 3 BƯỚC | Người mới bắt đầu |
| **`Microgrid_PPO_Simple.ipynb`** | ⭐ PPO đơn giản, CHỈ 3 BƯỚC | Người mới bắt đầu |
| `Microgrid_DQN_Colab.py` | DQN phiên bản đầy đủ, chi tiết | Sinh viên nâng cao |
| `Microgrid_PPO_Colab.py` | PPO phiên bản đầy đủ, chi tiết | Sinh viên nâng cao |
| `REPORT_DQN.md` | Báo cáo chi tiết phương pháp DQN | Tất cả |
| `REPORT_PPO.md` | Báo cáo chi tiết phương pháp PPO | Tất cả |
| `REPORT.md` | Báo cáo tổng hợp đồ án | Tất cả |

---

## 📖 Phần 1: Câu Chuyện (The Story) - Góc Nhìn Non-IT

Để hiểu bài toán này, hãy tưởng tượng AI không phải là những con số vô tri, mà là một **Người Quản Gia Tận Tụy**.

### 🏠 Câu Chuyện: "Người Quản Gia Năng Lượng"

Nhiệm vụ của Quản gia AI là chăm sóc ngôi nhà sử dụng điện mặt trời và pin lưu trữ.
Mục tiêu: **Chủ nhà luôn vui (đủ điện)** và **Ví tiền luôn đầy (tiết kiệm)**.

**Hành trình 1 ngày làm việc của Quản Gia:**

1. **🌅 Sáng (6h-10h):**
    * Trời mới hửng nắng, pin còn ít từ đêm qua.
    * *Quyết định:* Dùng tiết kiệm, hạn chế mua điện giá cao từ lưới.

2. **☀️ Trưa (10h-14h):**
    * Nắng to! Điện mặt trời (Rau trong vườn) dư thừa.
    * *Quyết định:* **Sạc đầy tủ lạnh (Pin)** để dành cho buổi tối.

3. **🌆 Tối (17h-21h) - CAO ĐIỂM:**
    * Điện lưới (Siêu thị) bán giá cắt cổ! 💸
    * *Quyết định:* **Tuyệt đối không mua!** Lấy đồ dự trữ trong tủ lạnh (Pin) ra dùng.

4. **🌙 Đêm (22h-5h):**
    * Điện lưới đại hạ giá.
    * *Quyết định:* Đi mua đầy tủ lạnh (Sạc) để phòng hờ ngày mai mưa bão.

---

## 📚 Phần 2: Kiến Thức Nền (The Foundation)

Tại sao chúng ta dùng **Reinforcement Learning (RL)** mà không phải Deep Learning (DL) truyền thống?

### 1. Supervised Learning: "Học Vẹt" Có Đáp Án 👨‍🎓

* **Sự tương đồng:** Giống như luyện thi đại học có đáp án.
* **Cách học:** Làm bài -> Mở giải xem đáp án -> Sửa lỗi.
* **Tại sao không dùng?** Vì trong Microgrid, **không có đáp án chuẩn** ngay lập tức. Quyết định "xả pin lúc 2h chiều" là đúng hay sai? Chúng ta không biết ngay, phải đợi đến cuối tháng trả tiền điện mới biết!

### 2. Reinforcement Learning: "Tập Xe Đạp" 🚴

* **Sự tương đồng:** Giống như tập đi xe đạp.
* **Cách học:** Tự thử nghiêng trái, nghiêng phải.
  * Ngã -> Đau (Phạt/Negative Reward).
  * Đi được -> Vui (Thưởng/Positive Reward).
* **Tại sao dùng?** AI tự học qua **Thử & Sai (Trial & Error)** để tìm ra cách đi tốt nhất mà không cần ai dạy từng chút một.

### 3. Deep Reinforcement Learning = Mắt Thần + Bộ Não 👁️🧠

Đây là sự kết hợp hoàn hảo:

* **Deep Learning (Mắt thần):** Dùng Neural Network để **cảm nhận** và xử lý thông tin phức tạp (giá điện, thời tiết, lịch sử).
* **Reinforcement Learning (Bộ não):** Dùng Q-Learning/PPO để **ra quyết định** dựa trên những gì mắt nhìn thấy.

---

## 🧮 Phần 3: Mô Hình Hóa MDP (The Math)

Chúng ta chuyển đổi bài toán thực tế thành ngôn ngữ Toán học (Markov Decision Process).

### 1. State Space (Trạng thái - 8 biến)

AI nhìn thấy gì ở mỗi bước?

| # | Tên biến | Phạm vi | Ý nghĩa | Lý do đưa vào |
|---|----------|---------|---------|---------------|
| 1 | `battery_level` | [0, 1] | Mức pin hiện tại | Biết "tủ lạnh" còn bao nhiêu đồ |
| 2 | `demand` | [0, 1] | Nhu cầu tiêu thụ | Biết chủ nhà cần bao nhiêu điện |
| 3 | `solar_generation` | [0, 1] | Điện mặt trời | Biết có bao nhiêu điện miễn phí |
| 4 | `wind_generation` | [0, 1] | Điện gió | Nguồn bổ sung ngẫu nhiên |
| 5 | `grid_price` | [0, 1] | Giá điện lưới | Quyết định mua hay bán |
| 6 | `hour_sin` | [-1, 1] | Giờ (Sin) | Mã hóa thời gian tuần hoàn |
| 7 | `hour_cos` | [-1, 1] | Giờ (Cos) | Để AI hiểu 23h gần với 0h |
| 8 | `prev_action` | [0, 1] | Hành động trước | Giúp hành động mượt mà hơn |

### 2. Action Space (Hành động - 5 lựa chọn)

AI có thể làm gì?

| Action | Tên | Mô tả chi tiết |
|--------|-----|----------------|
| **0** | **Xả pin** | Lấy điện từ pin ra dùng. (Discharge) |
| **1** | **Sạc mặt trời** | Nạp điện dư thừa vào pin. (Solar Charge) |
| **2** | **Mua lưới** | Mua điện từ lưới khi thiếu. (Grid Import) |
| **3** | **Kết hợp 1** | Tái tạo + Xả pin (Ưu tiên xanh). |
| **4** | **Kết hợp 2** | Tái tạo + Mua lưới (Giữ pin). |

### 3. Reward Function (Phần thưởng)

Cách dạy AI ngoan như dạy cún cưng! 🐶

| Hành vi | Điểm thưởng/phạt | Ý nghĩa ("Lời thầy cô phê") |
|---------|------------------|-----------------------------|
| **Dùng điện tái tạo** | **+1.0 điểm** | "Giỏi! Biết tận dụng đồ có sẵn, sạch & free." |
| **Xả pin đúng lúc** | **+0.5 điểm** | "Thông minh! Dùng đồ dự trữ lúc giá cao." |
| **Mua điện giờ cao điểm** | **-2.0 điểm** | "Hoang phí quá! Sao không dùng pin?" |
| **Để nhà mất điện** | **-5.0 điểm** | "Kỷ luật! Việc này không thể chấp nhận được!" 😡 |
| **Xả sạc liên tục** | **-0.1 điểm** | "Cẩn thận, làm thế nhanh hỏng pin (Hao mòn)." |

---

## 🤖 Phần 4: Thuật Toán (The Algorithms - DQN vs PPO)

Dự án này cung cấp 2 giải pháp. Bạn nên chọn cái nào?

### 1. DQN: Học Qua Ký Ức (Flashcards) 🔵

* **Cơ chế:** Lưu lại mọi trải nghiệm vào **Replay Buffer** (Bộ nhớ hồi tưởng). Khi học, bốc ngẫu nhiên các ký ức cũ ra để ôn lại.
* **Analogy:** Giống như ôn thi bằng **Thẻ Flashcards**. Trộn lẫn các câu hỏi lịch sử, toán, văn để học, tránh học tủ.
* **Đặc điểm:**
  * **Off-Policy:** Học được từ quá khứ (Sample Efficient).
  * **Value-based:** Cố gắng đoán giá trị của từng hành động.

### 2. PPO: Học Với Huấn Luyện Viên (Actor-Critic) 🟣

* **Cơ chế:** Có 2 mạng: **Actor** (Diễn viên - Ra quyết định) và **Critic** (Phê bình - Chấm điểm). Sử dụng cơ chế **Clipping** để giới hạn việc thay đổi quá nhanh.
* **Analogy:** Giống như tập thể thao với **Huấn luyện viên**. HLV chỉ sửa tư thế "tay thấp xuống một chút", không bắt đổi toàn bộ cách đánh ngay lập tức (tránh chấn thương/sốc).
* **Đặc điểm:**
  * **On-Policy:** Chỉ học từ trải nghiệm mới nhất.
  * **Policy-based:** Học trực tiếp chiến thuật hành động.

### ⚔️ Bảng So Sánh Chiến Thuật

| Tiêu chí | 🔵 DQN (Deep Q-Network) | 🟣 PPO (Proximal Policy Opt) |
| :--- | :--- | :--- |
| **Cách tiếp cận** | Học giá trị (Value-based) | Học hành vi (Policy-based) |
| **Dữ liệu** | Tái sử dụng (Replay Buffer) | Dùng 1 lần rồi bỏ (On-Policy) |
| **Hành động** | Chỉ rời rạc (Bật/Tắt) | Cả rời rạc & liên tục (Vặn van) |
| **Độ ổn định** | Thấp hơn, khó tune | Rất cao, dễ hội tụ |
| **Lời khuyên** | ✅ Bắt đầu từ đây (Dễ hiểu/Debug) | ✅ Nâng cao (Điểm cộng/SOTA) |

---

## 📊 Phần 5: Kết Quả & Phân Tích (Results)

### Hiệu quả so với Random Baseline

Agent đã chứng minh sự vượt trội so với việc chọn ngẫu nhiên:

| Metric | Random Agent | 🔵 DQN Agent | Cải thiện |
|--------|--------------|--------------|-----------|
| **Reward TB** | +10.72 | **+14.77** | **+37.8%** 🚀 |
| **Chi phí ngày** | $77.02 | **$64.78** | **-15.9%** 💰 |
| **Tỷ lệ điện xanh** | 58.0% | **66.9%** | **+8.9%** 🌱 |

### Phân Tích Hành Vi

Biểu đồ cho thấy AI đã học được chiến thuật "Con người":

1. **Sạc trưa:** Tận dụng điện mặt trời dư thừa.
2. **Xả tối:** Tránh mua điện giờ cao điểm đắt đỏ.
3. **Sạc đêm:** Mua điện rẻ dự phòng cho ngày hôm sau.

![Training Curves](evaluation_results/training_curves.png)
*(Hình ảnh minh họa quá trình loss giảm dần và reward tăng dần)*

---

## ⚖️ Phần 6: Đạo Đức & Tương Lai

### Vấn đề đạo đức (AI Ethics)

* **Công bằng:** Liệu AI có ưu tiên nhà giàu (trả nhiều tiền) hơn nhà nghèo khi thiếu điện?
* **Minh bạch:** Tại sao AI lại cắt điện lúc 2h? Cần có giải thích (Explainable AI).

### Hướng phát triển

1. **Multi-Agent:** Nhiều nhà cùng kết nối, mua bán điện cho nhau (P2P Trading).
2. **Dự báo thời tiết:** Tích hợp AI dự báo nắng/gió để lên kế hoạch tốt hơn.
3. **Thực tế:** Triển khai lên chip nhúng (Raspberry Pi/Jetson) để điều khiển mạch điện thật.

---

### 📦 Hướng Dẫn Nộp Bài (Google Colab)

Cấu trúc file Notebook `.ipynb` chuẩn:

1. **Section 1: Setup:** Cài đặt thư viện (`gymnasium`, `torch`).
2. **Section 2: Environment:** Code class `MicrogridEnv`.
3. **Section 3: Agent:** Code class `DQN` hoặc `PPO`.
4. **Section 4: Training:** Vòng lặp `train()`.
5. **Section 5: Evaluation:** Vẽ biểu đồ và bảng so sánh.

> **Một file duy nhất, chạy từ đầu đến cuối!**

---

*Tài liệu được thiết kế lại theo tiêu chuẩn giáo dục: Dễ hiểu - Trực quan - Chuyên sâu.*
