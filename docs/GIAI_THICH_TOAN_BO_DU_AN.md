# 📚 GIẢI THÍCH TOÀN BỘ DỰ ÁN ITS - TRÌNH BÀY CHO CÔ GIÁO

**Mục đích:** Tài liệu này giúp bạn hiểu rõ và giải thích toàn bộ dự án từ A-Z, bao gồm: cấu trúc thư mục, chức năng từng file, thuật toán, và cách vận hành hệ thống.

---

## 📋 MỤC LỤC NHANH
1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu trúc thư mục tổng thể](#2-cấu-trúc-thư-mục-tổng-thể)
3. [Giải thích chi tiết từng folder](#3-giải-thích-chi-tiết-từng-folder)
4. [Giải thích chi tiết từng file quan trọng](#4-giải-thích-chi-tiết-từng-file-quan-trọng)
5. [Thuật toán và kỹ thuật sử dụng](#5-thuật-toán-và-kỹ-thuật-sử-dụng)
6. [Quy trình vận hành hệ thống](#6-quy-trình-vận-hành-hệ-thống)
7. [Các câu hỏi thường gặp](#7-các-câu-hỏi-thường-gặp)

---

## 1️⃣ TỔNG QUAN DỰ ÁN

### 🎯 Tên dự án
**ITS (Intelligent Transportation System) - Hệ thống Phát hiện Sự cố Giao thông Tự động**

### 🎓 Đề tài
**"Detecting & Segmenting Abnormal Behavior in Surveillance"**
*(Phát hiện và phân đoạn hành vi bất thường trong camera giám sát)*

### 💡 Mục tiêu
Xây dựng hệ thống AI tự động phát hiện và cảnh báo các sự cố giao thông (tai nạn, xe hỏng, sự kiện bất thường) từ video camera giám sát sử dụng Deep Learning.

### 🔑 Công nghệ cốt lõi
*   **AI/ML:** Convolutional Neural Network (CNN) với Transfer Learning (MobileNetV2).
*   **Backend:** FastAPI (Python) - Async/Await.
*   **Frontend:** Streamlit (Interactive Dashboard).
*   **Database:** PostgreSQL với SQLAlchemy ORM.
*   **Computer Vision:** OpenCV.
*   **MLOps:** MLflow (tracking experiments).

---

## 2️⃣ CẤU TRÚC THƯ MỤC TỔNG THỂ

```plaintext
ITS/
├── 📁 src/                    # Source code chính (CORE)
│   ├── models/               # Các mô hình AI (CNN, ANN, RNN, RBFNN)
│   ├── training/             # Logic huấn luyện mô hình
│   ├── serving/              # API và prediction logic
│   ├── data_processing/      # Xử lý dữ liệu, ảnh
│   ├── database/             # Database models và migrations
│   └── utils/                # Utilities (config, logger)
│
├── 📁 data/                   # Dữ liệu
│   └── images/
│       ├── normal/           # Ảnh giao thông bình thường
│       └── incident/         # Ảnh có sự cố
│
├── 📁 models/                 # Mô hình đã train (saved models)
│   └── CNN_model/
│       └── model.keras       # File model chính
│
├── 📁 docs/                   # Tài liệu
│   ├── README.md             # Index tất cả tài liệu
│   ├── GIAI_THICH_CONG_NGHE.md
│   ├── ARCHITECTURE.md
│   └── ...
│
├── 📁 scripts/                # Scripts tiện ích
│   ├── he_thong.bat          # Menu quản lý hệ thống
│   ├── tao_venv.bat          # Tạo virtual environment
│   └── ...
│
├── 📁 configs/                # File cấu hình
│   └── training_config.yaml
│
├── 📁 tests/                  # Unit tests
│
├── 📁 logs/                   # Log files
│
├── 📄 app.py                  # Ứng dụng Streamlit chính
├── 📄 train_cnn.py            # Script huấn luyện mô hình
├── 📄 start_api.py            # Script khởi động API
├── 📄 requirements.txt        # Dependencies Python
├── 📄 setup.py                # Package setup
└── 📄 README.md               # Documentation chính
```

---

## 3️⃣ GIẢI THÍCH CHI TIẾT TỪNG FOLDER

### 📂 `src/` - Source Code Chính
Đây là thư mục quan trọng nhất, chứa toàn bộ logic của hệ thống.

#### 🧠 `src/models/` - Các Mô hình AI
**Mục đích:** Định nghĩa kiến trúc của các mô hình Machine Learning/Deep Learning.

| File | Mô tả | Kỹ thuật sử dụng |
| :--- | :--- | :--- |
| `base_model.py` | Class cha (abstract) cho tất cả models | OOP, Inheritance |
| `cnn.py` | **MÔ HÌNH CHÍNH** - CNN | Transfer Learning, MobileNetV2 |
| `ann.py` | Artificial Neural Network | Feed-forward, Dense layers |
| `rnn.py` | Recurrent Neural Network | LSTM/GRU cho data tuần tự |
| `rbfnn.py` | Radial Basis Function NN | RBF kernel |

**Giải thích file `cnn.py` (QUAN TRỌNG NHẤT):**
```python
# Đây là file định nghĩa mô hình CNN chính
class CNNModel:
    def __init__(self, base_model='MobileNetV2'):
        # Khởi tạo với base model (MobileNetV2/ResNet50/VGG16)
        # Đây là Transfer Learning - tận dụng model đã train trên ImageNet
        pass
        
    def build_model(self):
        # Xây dựng kiến trúc model:
        # 1. Load base model (đã train sẵn)
        # 2. Freeze các layers đầu (không train lại)
        # 3. Thêm layers mới phía trên (Dense, Dropout)
        # 4. Output layer với 2 neurons (normal/incident)
        pass
```
> **Tại sao dùng CNN?**
> *   CNN được thiết kế đặc biệt cho xử lý ảnh.
> *   Tự động học các đặc trưng (features): cạnh, góc, texture.

#### 🎓 `src/training/` - Logic Huấn Luyện

| File | Mô tả | Chức năng |
| :--- | :--- | :--- |
| `trainer.py` | Core training logic | Quản lý process train: load data, build model, save |
| `evaluator.py` | Đánh giá model | Tính metrics: Accuracy, Precision, Recall, F1 |
| `visualizer.py` | Visualization | Vẽ biểu đồ loss/accuracy |

**Giải thích `trainer.py`:**
```python
class ModelTrainer:
    def prepare_data(self, data_path):
        # 1. Load ảnh từ data/images
        # 2. Resize ảnh về 224x224
        # 3. Normalize pixel (0-255 → 0-1)
        # 4. Split data (Train/Val/Test)
        # 5. Data Augmentation
        pass
        
    def train(self, X_train, y_train, X_val, y_val):
        # 1. Build model
        # 2. Compile (Adam, BinaryCrossentropy)
        # 3. Train với Callbacks
        # 4. Save best model
        # 5. Log metrics MLflow
        pass
```

#### 🚀 `src/serving/` - API và Prediction

| File | Mô tả | Công nghệ |
| :--- | :--- | :--- |
| `api.py` | FastAPI endpoints | RESTful API, Swagger |
| `predictor.py` | Prediction logic | Inference wrapper |
| `temporal_confirmation.py` | **ĐẶC BIỆT** - Thuật toán | Sliding window, K-consecutive |
| `monitoring.py` | System monitoring | Health check |

**Giải thích `temporal_confirmation.py` (THUẬT TOÁN ĐỘC ĐÁO):**
```python
class TemporalConfirmation:
    """
    Vấn đề: Nếu tin ngay 1 frame → nhiều false alarm (bóng cây, ánh sáng).
    Giải pháp: Chỉ cảnh báo khi sự cố xuất hiện LIÊN TỤC trong K frames.
    """
    
    def confirm_incident(self, frame_predictions):
        # Nếu có ít nhất K frames LIÊN TIẾP đều báo incident
        if consecutive_count >= K:
            return True  # CONFIRM
        return False
```
> **Lợi ích:** Giảm 30-50% cảnh báo giả (False Alarms).

#### 🔧 `src/data_processing/` & 🗄️ `src/database/` & ⚙️ `src/utils/`
*   `image_processor.py`: Resize, crop, augmentation.
*   `models.py`: SQLAlchemy ORM models (Lưu lịch sử incident vào DB).
*   `config.py`: Quản lý cấu hình hệ thống.

---

## 4️⃣ GIẢI THÍCH CHI TIẾT TỪNG FILE QUAN TRỌNG

### 📄 `app.py` - Ứng Dụng Streamlit (Dashboard)
**Chức năng:** Giao diện web quản lý và demo.

```python
import streamlit as st
from src.training.trainer import ModelTrainer

# Sidebar navigation
page = st.sidebar.radio("Chọn chức năng:", ["Trang chủ", "Huấn luyện", "Test mô hình"])

if page == "Huấn luyện":
    # Giao diện hiển thị nút bấm train, thanh progress bar
    # Gọi ModelTrainer để train lại model
elif page == "Test mô hình":
    # Cho phép upload ảnh/video và hiển thị kết quả predict
```

### 📄 `train_cnn.py` - Script Huấn Luyện
**Chức năng:** Chạy huấn luyện mô hình từ dòng lệnh.
**Cách chạy:** `python train_cnn.py`
**Luồng:** Load Data -> Init Trainer -> Train -> Evaluate -> Save Model.

### 📄 `start_api.py` - Script Khởi Động API
**Chức năng:** Chạy Backend Server.
```python
if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000)
```
**Endpoints chính:**
*   `POST /predict/image`: Gửi ảnh lên, nhận về kết quả (Normal/Incident).
*   `GET /incidents`: Lấy danh sách lịch sử sự cố.

---

## 5️⃣ THUẬT TOÁN VÀ KỸ THUẬT SỬ DỤNG

### 🧠 1. Transfer Learning với CNN
*   **Ý tưởng:** Không train từ đầu. Dùng **MobileNetV2** (đã học từ ImageNet) làm nền tảng.
*   **Quy trình:**
    1.  Load Pre-trained MobileNetV2.
    2.  **Freeze** các lớp convolution (giữ nguyên kiến thức cũ).
    3.  Thêm lớp **Classification Head** mới (Dense layers) ở cuối.
    4.  Chỉ train các lớp mới này.
*   **Lợi ích:** Cần ít dữ liệu (vài trăm ảnh thay vì hàng triệu), train cực nhanh, độ chính xác cao.

### 📊 2. Data Augmentation
*   **Kỹ thuật:** Xoay ảnh, lật ngang, chỉnh độ sáng.
*   **Mục đích:** Giúp model "thông minh" hơn, nhận diện được xe dù xe đang quay ngang, quay dọc hay trời tối. Chống học vẹt (Overfitting).

### ⏱️ 3. Temporal Confirmation (Xác nhận theo thời gian)
*   **Vấn đề:** Camera bị rung hoặc lá cây bay qua làm model nhận diện nhầm trong tích tắc.
*   **Giải pháp:** Sliding Window (Cửa sổ trượt).
*   **Quy tắc:** "Sự cố phải tồn tại liên tục trong **5 frame** (khoảng 0.5s) thì mới tính là thật."

### 🎯 4. Evaluation Metrics
*   **Precision:** Báo đúng bao nhiêu? (Quan trọng để tránh báo giả).
*   **Recall:** Tìm được bao nhiêu sự cố? (Quan trọng để không bỏ sót).
*   **F1-Score:** Trung bình hài hòa giữa Precision và Recall.

---

## 6️⃣ QUY TRÌNH VẬN HÀNH HỆ THỐNG

### 🚀 Setup Lần Đầu
1.  **Cài đặt:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Chuẩn bị dữ liệu:**
    *   Copy ảnh vào `data/images/normal/` và `data/images/incident/`.

### 🎓 Train Mô Hình
*   **Cách 1 (Dễ):** Mở Streamlit (`app.py`) -> Tab "Huấn luyện" -> Bấm nút.
*   **Cách 2 (Pro):** Chạy lệnh `python train_cnn.py`.

### 🌐 Chạy Hệ Thống Demo
1.  **Backend:** `python start_api.py` (Chạy ở background).
2.  **Frontend:** `python run_streamlit.py`.
3.  Truy cập: `http://localhost:8501`.

---

## 7️⃣ CÁC CÂU HỎI THƯỜNG GẶP (Q&A)

**❓ "Tại sao em chọn MobileNetV2?"**
> "Dạ, vì ưu tiên của hệ thống giao thông là **Tốc độ (Real-time)**. MobileNetV2 nhẹ hơn VGG16 rất nhiều (14MB vs 500MB) nhưng độ chính xác vẫn rất tốt. Nó phù hợp để sau này triển khai lên các thiết bị nhỏ như Jetson Nano ạ."

**❓ "Làm sao giảm báo động giả (False Alarm)?"**
> "Em sử dụng thuật toán **Temporal Confirmation**. Thay vì tin ngay một khung hình đơn lẻ, hệ thống chờ sự nhất quán trong chuỗi 5-10 khung hình liên tiếp rồi mới phát cảnh báo."

**❓ "Em làm dự án này trong bao lâu?"**
> "Giải thích thật: Em vừa research vừa code trong khoảng X tuần. Phần khó nhất là tinh chỉnh model (fine-tuning) và xử lý dữ liệu đầu vào."

**❓ "Em có code từ đầu không?"**
> "Dạ em dùng các thư viện chuẩn như TensorFlow, FastAPI. Phần kiến trúc hệ thống, logic training và thuật toán temporal confirmation là do em tự thiết kế và code ạ."

---
*Tài liệu này được biên soạn để hỗ trợ thuyết trình và bảo vệ đồ án.*
