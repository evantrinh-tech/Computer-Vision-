# 📂 CẤU TRÚC DỰ ÁN ITS

Tài liệu này mô tả chi tiết cấu trúc thư mục và chức năng của các file trong dự án **ITS (Intelligent Transportation System)**.
*(Cập nhật mới nhất sau khi dọn dẹp hệ thống)*

---

## 🌳 Cây Thư Mục Tổng Quát

```plaintext
ITS/
├── 📄 app.py                     # Giao diện Dashboard chính (Streamlit)
├── 📄 train_cnn.py               # Script huấn luyện mô hình CNN
├── 📄 start_api.py               # Script khởi động Backend API
├── 📄 he_thong.bat               # Menu công cụ quản lý toàn bộ hệ thống (Windows)
├── 📄 requirements.txt           # Danh sách thư viện Python cần thiết
├── 📄 setup.py                   # Cấu hình package Python
├── 📄 README.md                  # Tài liệu hướng dẫn chính
│
├── 📁 src/                       # SOURCE CODE CHÍNH (Core Logic)
│   ├── 📁 models/                # Định nghĩa kiến trúc các mô hình AI (CNN,...)
│   ├── 📁 training/              # Logic huấn luyện, evaluation loop
│   ├── 📁 serving/               # API endpoints, logic dự đoán (Inference)
│   ├── 📁 data_processing/       # Xử lý ảnh, data augmentation
│   ├── 📁 database/              # Schema database, ORM models
│   └── 📁 utils/                 # Các tiện ích (Logger, Config loader)
│
├── 📁 configs/                   # File cấu hình (Hyperparameters)
│   └── training_config.yaml
│
├── 📁 data/                      # Dữ liệu (Dataset)
│   └── images/
│       ├── normal/               # Ảnh không có sự cố
│       └── incident/             # Ảnh tai nạn/sự cố
│
├── 📁 docs/                      # Tài liệu dự án
│   ├── GIAI_THICH_TOAN_BO_DU_AN.md  # Tài liệu tổng hợp A-Z
│   ├── KICH_BAN_VA_NOI_DUNG_SLIDE.md # Kịch bản thuyết trình
│   ├── PHAN_CONG_TRINH_BAY.md    # Phân công nhóm
│   └── ...
│
├── 📁 models/                    # Nơi lưu mô hình đã train (.keras)
│   └── CNN_model/
│
├── 📁 logs/                      # Log file của hệ thống
├── 📁 scripts/                   # Các script tiện ích khác (.bat, .ps1)
├── 📁 tests/                     # Unit tests
└── 📁 venv311/                   # Môi trường ảo Python (Recommended)
```

---

## 🔍 Giải Thích Chi Tiết

### 1. Root Directory (Thư mục gốc)
*   **`app.py`**: Entry point cho ứng dụng Web. Chạy bằng `streamlit run app.py`. Cung cấp giao diện để người dùng train model, test model, xem dữ liệu.
*   **`train_cnn.py`**: Entry point để huấn luyện model. Chạy bằng `python train_cnn.py`.
*   **`start_api.py`**: Entry point để chạy API Server. Chạy bằng `python start_api.py`.
*   **`he_thong.bat`**: Công cụ "All-in-one" cho Windows. Click đúp để mở menu chọn chức năng (Train, Run App, Install...).
*   **`requirements.txt`**: Liệt kê các thư viện cần cài đặt (`pip install -r requirements.txt`).

### 2. Source Code (`src/`)
Đây là "bộ não" của hệ thống.
*   **`src/models/`**:
    *   `cnn.py`: Định nghĩa class `CNNModel` (sử dụng MobileNetV2 Transfer Learning).
    *   `base_model.py`: Interface chung cho các model.
*   **`src/training/`**:
    *   `trainer.py`: Class `ModelTrainer` quản lý quy trình train, lưu model.
    *   `evaluator.py`: Tính toán độ chính xác (Accuracy, F1-Score).
*   **`src/serving/`**:
    *   `api.py`: Định nghĩa các API endpoints (FastAPI).
    *   `temporal_confirmation.py`: Thuật toán xác nhận sự cố theo chuỗi thời gian (giảm báo ảo).
*   **`src/data_processing/`**:
    *   `image_processor.py`: Các hàm resize, chuẩn hóa ảnh.
*   **`src/utils/`**:
    *   `config.py`: Đọc cấu hình từ file yaml.
    *   `logger.py`: Cấu hình ghi log.

### 3. Configs & Data
*   **`configs/training_config.yaml`**: Lưu các tham số như `batch_size`, `learning_rate`, `epochs`. Chỉnh sửa file này thay vì sửa code.
*   **`data/images/`**: Chứa dữ liệu huấn luyện. Bắt buộc phải có 2 thư mục con `normal` và `incident`.

### 4. Documentation (`docs/`)
Chứa tài liệu hướng dẫn chi tiết.
*   **`GIAI_THICH_TOAN_BO_DU_AN.md`**: Tài liệu quan trọng nhất để hiểu dự án.
*   **`GIAI_THICH_VAN_HANH_CHI_TIET.md`**: Hướng dẫn chạy code từng bước.

### 5. Scripts (`scripts/`)
Chứa các file kịch bản hỗ trợ.
*   `tao_venv.bat`: Tự động tạo môi trường ảo và cài thư viện.
*   `setup_tensorflow.ps1`: Hỗ trợ cài TensorFlow trên Windows.

---

## 💡 Lưu ý quan trọng
*   Các file dọn dẹp cũ (`cleanup_system.py`) **đã được xóa** để project gọn gàng hơn.
*   Môi trường ảo khuyến nghị là **`venv311`** (Python 3.11).
