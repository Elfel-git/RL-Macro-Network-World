# 🤖 DS102: Reinforcement Learning & World Models Project

**Khảo sát khả năng tổng quát hóa của AI trong điều hướng không gian và mô phỏng đánh giá rủi ro tài chính (Financial Precarity)**

---

## 📖 Mục Lục

1. [🚀 Quick Start](#-quick-start) - Bắt đầu nhanh trong 5 phút
2. [📁 Project Structure](#-project-structure) - Cấu trúc dự án
3. [🎮 Algorithm Guide](#-algorithm-guide) - Hướng dẫn chạy các thuật toán RL
4. [📈 Financial Simulation](#-financial-simulation) - Chạy mô phỏng kinh tế
5. [📝 Notebook Files](#-notebook-files) - Hướng dẫn file phân tích
6. [🔗 Liên kết Hữu Ích](#-liên-kết-hữu-ích)
7. [🐛 Troubleshooting](#-troubleshooting) - Xử lý sự cố

---

## 🚀 Quick Start

### Yêu cầu

- Python 3.8+
- Git
- Đủ dung lượng ổ cứng để lưu trữ Replay Buffer (đối với World Models)

### 1️⃣ Clone hoặc tải project

```bash
git clone [https://github.com/your-username/DS102_RL_WorldModels_Project.git](https://github.com/your-username/DS102_RL_WorldModels_Project.git)
cd DS102_RL_WorldModels_Project
2️⃣ Tạo Virtual Environment
Bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Hoặc cmd.exe
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3️⃣ Cài đặt Dependencies
Bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
📁 Project Structure
Plaintext
DS102_RL_WorldModels_Project/
│
├── rl_algorithms/                 # Thuật toán RL cơ bản (DQN, PPO, A2C, SAC)
│   └── test_me_cung.py            # Script đánh giá Agent trong môi trường Maze
│
├── world_models/                  # Mã nguồn & Benchmark cho World Models
│   ├── demo_world_model_visual.py # Trích xuất & render video dự đoán từ WM
│   └── miniworld_wm_benchmark_final.py # Benchmark 4 kiến trúc World Model
│
└── financial_precarity/           # Case study: Mô phỏng rủi ro tài chính
    ├── notebooks/                 # Thư mục chứa Jupyter Notebooks
    │   ├── EDA_Financial_Precarity.ipynb
    │   └── Financial_Precarity_Full_Simulation.ipynb
    └── src/
        ├── vectorized_env.py      # Môi trường Gym vector hóa (chạy 10.000 agents)
        └── full_simulation.py     # Script mô phỏng chính & xuất biểu đồ
🎮 Algorithm Guide
1. Benchmark RL trên môi trường Mê cung (Maze)
Khảo sát khả năng tổng quát hóa (Generalization) của các thuật toán RL khi đối mặt với Overfitting.

Bash
python rl_algorithms/test_me_cung.py
⏱️ Thời gian: Phụ thuộc vào số lượng episodes huấn luyện. Giao diện Pygame sẽ hiện lên để so sánh kết quả (Train WR vs Test WR).

2. Huấn luyện & Đánh giá World Models
Thu thập Replay Buffer, train 4 mô hình (Ha-VAE, PlaNet, DreamerV1, Transformer) trên MiniWorld.

Bash
# Chạy smoke test nhanh để kiểm tra lỗi
python world_models/miniworld_wm_benchmark_final.py --quick

# Chạy full quá trình thu thập data và train
python world_models/miniworld_wm_benchmark_final.py --env_id MiniWorld-OneRoom-v0 --rebuild_buffer
📈 Financial Simulation
Chạy kịch bản mô phỏng hiệu ứng "Compounded Decisions" lên tài sản của Agent (Nokhiz et al., 2024), so sánh giữa Office Worker và Gig Worker.

Bash
python financial_precarity/src/full_simulation.py
🎯 Output: Script sẽ tạo và tự động lưu các biểu đồ so sánh (g1_*.png đến g6_*.png) vào thư mục gốc.

📝 Notebook Files
Các Jupyter notebooks trong project phục vụ mục đích khám phá dữ liệu và trực quan hóa:

financial_precarity/notebooks/EDA_Financial_Precarity.ipynb
Mục đích: Tính toán hệ số Gini, phân tích bất bình đẳng tài chính (Wealth Inequality) và đánh giá tác động của trợ cấp (Subsidies / Tax breaks).

Cách chạy:

Bash
# Khởi động Jupyter Lab
jupyter lab

# Hoặc Jupyter Notebook
jupyter notebook

# Mở file: financial_precarity/notebooks/EDA_Financial_Precarity.ipynb
Cấu hình kernel:

Vào Kernel → Select Kernel

Chọn .venv environment

Chạy các cell theo thứ tự

🔗 Liên kết Hữu Ích
Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/

Gymnasium Documentation: https://gymnasium.farama.org/

Nokhiz et al. Paper: Agent-Based Simulation of Decision-Making Under Uncertainty (2024)

World Models (David Ha): https://worldmodels.github.io/

👤 Author & Maintenance
Đồ án môn học DS102 Thực hiện bởi: Lê Hoài Nam, Võ Phan Kiều My, Nguyễn Bảo Long

Last Updated: Tháng 7, 2026

📋 Checklist - Before Running
[ ] Python 3.8+ installed

[ ] Virtual environment created & activated

[ ] All requirements installed: pip install -r requirements.txt

[ ] Pygame / GUI rendering dependencies cài đặt đủ (cho bài toán maze)

[ ] Đủ dung lượng ổ cứng để lưu model checkpoints

🐛 Troubleshooting
Lỗi ModuleNotFoundError: No module named 'stable_baselines3': Bạn chưa activate môi trường ảo .venv hoặc chưa cài đặt requirements.txt.

Lỗi AttributeError numpy has no attribute 'trapz': Do dự án sử dụng Numpy 2.0+, hàm np.trapz đã bị loại bỏ. Trong code của full_simulation.py đã được sửa thành np.trapezoid. Hãy đảm bảo bạn đang dùng file mới nhất.

Lỗi crash khi render trên server không có màn hình: Nếu chạy trên Kaggle/Colab, hãy tắt cờ render hình ảnh của Pygame.
