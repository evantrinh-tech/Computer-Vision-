<<<<<<< HEAD
# ITS
HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG
=======
#  Hệ thống Phát hiện Sự cố Giao thông Tự động

Hệ thống tự động phát hiện sự cố giao thông (tai nạn, xe hỏng, sự kiện đặc biệt) từ ảnh camera sử dụng Deep Learning và Neural Network. Hệ thống được xây dựng với kiến trúc end-to-end từ xử lý ảnh/video đến dashboard quản lý và API tích hợp.

## 📋 Tổng quan

Hệ thống sử dụng **Convolutional Neural Network (CNN)** với **Transfer Learning** để phân loại ảnh giao thông thành hai loại: **Normal** (bình thường) và **Incident** (có sự cố). Hệ thống tích hợp **Temporal Confirmation** để giảm false alarm rate bằng cách xác nhận sự cố qua nhiều frames liên tiếp.

### Đặc điểm nổi bật

-  **Deep Learning**: CNN với Transfer Learning (MobileNetV2, ResNet50, VGG16)
-  **Real-time Processing**: Xử lý ảnh/video với latency thấp
-  **Temporal Confirmation**: Giảm false alarm bằng xác nhận qua nhiều frames
-  **Web Dashboard**: Giao diện Streamlit trực quan, dễ sử dụng
-  **REST API**: FastAPI với Swagger documentation
-  **Database**: PostgreSQL để lưu trữ incidents và predictions
-  **MLflow Tracking**: Quản lý experiments và model versions
-  **Production Ready**: Kiến trúc mở rộng, hỗ trợ deployment

## 🎯 Tính năng chính

### 1. Phát hiện Sự cố
- **Phát hiện từ ảnh**: Upload ảnh và nhận kết quả ngay lập tức
- **Phát hiện từ video**: Xử lý video frame-by-frame với temporal confirmation
- **RTSP Stream**: Hỗ trợ xử lý stream từ camera (tương lai)
- **Confidence Score**: Hiển thị độ tin cậy của prediction (0-1)

### 2. Huấn luyện Mô hình
- **Transfer Learning**: Sử dụng pre-trained models (ImageNet)
- **Configurable Parameters**: Epochs, batch size, image size, learning rate
- **Data Augmentation**: Tự động augment dữ liệu training
- **Model Selection**: Hỗ trợ MobileNetV2, ResNet50, VGG16
- **Training Monitoring**: Theo dõi loss, accuracy real-time
- **Model Versioning**: Lưu và quản lý nhiều versions

### 3. Giao diện Web (Streamlit)
- **Upload & Predict**: Upload ảnh/video và xem kết quả
- **Training Interface**: Giao diện huấn luyện mô hình trực quan
- **Metrics Visualization**: Biểu đồ loss, accuracy, confusion matrix
- **Incident Management**: Xem và quản lý incidents đã phát hiện
- **Model Management**: Xem danh sách models đã train

### 4. API Service (FastAPI)
- **RESTful API**: Endpoints chuẩn REST
- **Swagger Documentation**: Tự động generate API docs
- **Health Check**: Endpoint kiểm tra trạng thái hệ thống
- **Batch Processing**: Hỗ trợ xử lý nhiều ảnh cùng lúc
- **Async Support**: Xử lý bất đồng bộ cho hiệu suất cao

### 5. Database & Storage
- **PostgreSQL**: Lưu trữ incidents, predictions, model runs
- **SQLAlchemy ORM**: Object-relational mapping
- **Migrations**: Database migration scripts
- **Audit Trail**: Lưu tất cả predictions để audit

### 6. Temporal Confirmation
- **K-frames Confirmation**: Xác nhận qua K frames liên tiếp
- **Moving Average**: Tính toán moving average của probabilities
- **Cooldown Period**: Tránh spam alerts
- **False Alarm Reduction**: Giảm false alarm rate đáng kể

## 🎯 Công nghệ sử dụng

### Core Frameworks
- **Python 3.11**: Ngôn ngữ lập trình chính
- **TensorFlow/Keras**: Deep Learning framework
- **Streamlit**: Giao diện web dashboard
- **FastAPI**: REST API framework (async, high performance)
- **OpenCV**: Xử lý ảnh/video
- **Pillow**: Image manipulation

### Machine Learning
- **TensorFlow/Keras**: CNN models, Transfer Learning
- **scikit-learn**: ML utilities, metrics
- **NumPy**: Numerical computing
- **Pandas**: Data processing

### MLOps & Tools
- **MLflow**: Experiment tracking, model registry
- **SQLAlchemy**: Database ORM
- **PostgreSQL**: Relational database
- **python-dotenv**: Environment variables
- **pyyaml**: Configuration files
- **python-json-logger**: Structured logging

### Development Tools
- **Git**: Version control
- **pytest**: Unit testing
- **Black**: Code formatting (optional)
- **VS Code/PyCharm**: IDE

Xem chi tiết: [requirements.txt](requirements.txt)

##  Cài đặt và Sử dụng

### Yêu cầu Hệ thống

- **Python**: 3.9, 3.10, hoặc 3.11 (khuyến nghị 3.11)
- **OS**: Windows, Linux, macOS
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **GPU**: Không bắt buộc (có GPU sẽ nhanh hơn 5-10 lần)
- **Storage**: Tối thiểu 2GB (cho models và dependencies)

### 1. Clone Repository

```bash
git clone <repository-url>
cd ITS
```

### 2. Tạo Virtual Environment

#### Windows:
```bash
python -m venv venv311
venv311\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv311
source venv311/bin/activate
```

### 3. Cài đặt Dependencies

```bash
# Cập nhật pip
python -m pip install --upgrade pip

# Cài đặt TensorFlow (có thể mất vài phút)
pip install tensorflow

# Cài đặt các dependencies khác
pip install -r requirements.txt
```

**Lưu ý**: Nếu có GPU, cài đặt `tensorflow-gpu` thay vì `tensorflow`:
```bash
pip install tensorflow-gpu
```

### 4. Chuẩn bị Dữ liệu

Đặt ảnh training vào các thư mục:

```
data/images/
├── normal/      # Ảnh giao thông bình thường
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── incident/    # Ảnh có sự cố giao thông
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

**Hỗ trợ định dạng**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`

**Yêu cầu dữ liệu**:
- Tối thiểu: 50 ảnh mỗi class (normal/incident)
- Khuyến nghị: 200+ ảnh mỗi class để có kết quả tốt
- Tỷ lệ: Cân bằng giữa normal và incident (50/50 hoặc 60/40)

### 5. Chạy Hệ thống

#### Cách 1: Menu Windows (Khuyến nghị - Dễ nhất)

```bash
he_thong.bat
```

Menu này cung cấp tất cả chức năng:
- **[1] Giao diện Web (Streamlit)** - Khuyến nghị
- **[2] Chạy API Server**
- **[3] Huấn luyện mô hình** (CNN, ANN, RNN, RBFNN)
- **[4] Test mô hình** (ảnh, video, API, temporal)
- **[5] Kiểm tra trạng thái hệ thống**
- **[6] Tạo Virtual Environment**
- **[7] Setup Database**
- **[8] Dọn dẹp hệ thống**
- **[9] Quick Start** (tự động setup và chạy)

#### Cách 2: Giao diện Web (Streamlit)

```bash
# Windows
python run_streamlit.py

# Hoặc
streamlit run app.py
```

Mở trình duyệt tại: **http://localhost:8501**

**Tính năng trong Streamlit**:
- Upload ảnh/video và predict
- Huấn luyện mô hình với giao diện trực quan
- Xem metrics và training history
- Quản lý incidents

#### Cách 3: API Server

```bash
python start_api.py
```

API sẽ chạy tại: **http://localhost:8000**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

#### Cách 4: Training từ Command Line

```bash
python train_cnn.py
```

Model sẽ được lưu tại: `models/CNN_model/model.keras`

## 📁 Cấu trúc Dự án

```
ITS/
├── app.py                    # Ứng dụng Streamlit chính
├── run_streamlit.py          # Script chạy Streamlit
├── start_api.py              # API server entry point
├── train_cnn.py              # Training script chính
├── test_cnn_image.py         # Test với ảnh
├── test_cnn_video.py         # Test với video
├── test_api.py               # Test API endpoints
├── check_training_status.py  # Kiểm tra trạng thái
├── check_training_detailed.py # Kiểm tra chi tiết
├── cleanup_system.py         # Dọn dẹp hệ thống
│
├── he_thong.bat              # Menu chính hệ thống (Windows)
│
├── src/                      # Source code
│   ├── models/              # Mô hình ML/DL
│   │   ├── base_model.py   # Base class cho models
│   │   ├── cnn.py           # CNN model (MobileNetV2, ResNet50, VGG16)
│   │   ├── ann.py           # ANN model (Feed-forward)
│   │   ├── rnn.py           # RNN model (LSTM/GRU)
│   │   └── rbfnn.py         # RBFNN model
│   ├── training/            # Training pipeline
│   │   ├── trainer.py       # Training logic
│   │   ├── evaluator.py     # Evaluation functions
│   │   └── visualizer.py    # Visualization utilities
│   ├── data_processing/     # Xử lý dữ liệu
│   │   ├── collectors.py    # Data collection
│   │   ├── image_processor.py # Image processing
│   │   ├── preprocessors.py # Preprocessing functions
│   │   ├── feature_engineering.py # Feature engineering
│   │   └── validators.py    # Data validation
│   ├── serving/             # API serving
│   │   ├── api.py            # FastAPI endpoints
│   │   ├── predictor.py      # Prediction logic
│   │   ├── temporal_confirmation.py # Temporal confirmation
│   │   └── monitoring.py    # System monitoring
│   ├── database/            # Database
│   │   ├── models.py        # SQLAlchemy models
│   │   └── migrations/      # Migration scripts
│   │       └── 001_initial_schema.sql
│   └── utils/               # Utilities
│       ├── config.py         # Configuration
│       └── logger.py         # Logging
│
├── pipelines/               # Training pipelines
│   └── training_pipeline.py  # Pipeline cho các models
│
├── docs/                     # Tài liệu
│   ├── ROADMAP.md           # Roadmap 3 phase
│   ├── EVALUATION_PROTOCOL.md # Evaluation protocol
│   ├── BASELINE_COMPARISON.md # Baseline comparison
│   ├── ARCHITECTURE.md       # System architecture
│   ├── BAO_CAO_CUOI.md      # Báo cáo outline
│   ├── HUONG_DAN_SU_DUNG.md # Hướng dẫn sử dụng
│   └── examples/            # Code examples
│
├── data/                     # Dữ liệu
│   ├── images/
│   │   ├── normal/          # Ảnh bình thường
│   │   └── incident/        # Ảnh có sự cố
│   ├── processed/           # Dữ liệu đã xử lý
│   └── raw/                  # Dữ liệu thô
│
├── models/                   # Models đã train
│   └── CNN_model/
│       └── model.keras
│
├── configs/                  # Cấu hình
│   └── training_config.yaml
│
├── tests/                    # Unit tests
│   └── unit/
│       └── test_preprocessors.py
│
├── logs/                     # Log files
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml           # Project metadata
└── README.md                 # File này
```

##  Huấn luyện Mô hình

### Qua Giao diện Web (Streamlit)

1. Mở `http://localhost:8501`
2. Chọn tab **" Huấn luyện mô hình CNN"**
3. Cấu hình parameters:
   - **Epochs**: Số lần train (khuyến nghị: 10-50)
   - **Batch Size**: Kích thước batch (khuyến nghị: 16-32)
   - **Image Size**: Kích thước ảnh (224x224 hoặc 128x128)
   - **Learning Rate**: Tốc độ học (mặc định: 0.001)
   - **Base Model**: MobileNetV2, ResNet50, hoặc VGG16
4. Nhấn **" Bắt đầu huấn luyện"**
5. Theo dõi tiến trình:
   - Loss và Accuracy real-time
   - Training vs Validation metrics
   - ETA (Estimated Time to Arrival)

### Qua Command Line

```bash
python train_cnn.py
```

**Tùy chọn**:
```bash
# Train với parameters tùy chỉnh
python train_cnn.py --epochs 50 --batch_size 32 --image_size 224

# Train với model khác
python pipelines/training_pipeline.py --model ANN --simulate
python pipelines/training_pipeline.py --model RNN --simulate
python pipelines/training_pipeline.py --model RBFNN --simulate
```

### Training Pipeline

1. **Load Dataset**: Đọc ảnh từ `data/images/normal/` và `data/images/incident/`
2. **Split Data**: Train/Validation (80/20)
3. **Data Augmentation**: Rotation, flip, brightness, contrast
4. **Build Model**: Transfer Learning với base model
5. **Compile**: Optimizer (Adam), Loss (binary_crossentropy), Metrics (accuracy)
6. **Train**: Với callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
7. **Evaluate**: Tính metrics trên validation set
8. **Save Model**: Lưu best model vào `models/CNN_model/model.keras`
9. **MLflow Tracking**: Log metrics, parameters, artifacts

### Model được lưu tại

- **Path**: `models/CNN_model/model.keras`
- **Format**: Keras SavedModel format
- **Size**: ~20-50MB (tùy base model)

##  Test Mô hình

### Qua Giao diện Web

1. Chọn tab **" Test mô hình"**
2. Upload ảnh hoặc chọn từ thư mục
3. Xem kết quả:
   - **Prediction**: Normal hoặc Incident
   - **Confidence Score**: 0.0 - 1.0
   - **Visualization**: Ảnh với overlay prediction

### Qua Command Line

#### Test với ảnh:
```bash
python test_cnn_image.py path/to/image.jpg
```

#### Test với video:
```bash
python test_cnn_video.py path/to/video.mp4
```

#### Test API:
```bash
python test_api.py
```

### Qua API

#### Predict từ ảnh:
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "data/images/incident/img1.jpg"}'
```

#### Predict từ video:
```bash
curl -X POST "http://localhost:8000/predict/video" \
  -H "Content-Type: application/json" \
  -d '{"video_path": "path/to/video.mp4"}'
```

#### Lấy danh sách incidents:
```bash
curl -X GET "http://localhost:8000/incidents"
```

##  API Endpoints

### Health Check
```
GET /health
```
Kiểm tra trạng thái hệ thống

### Predict Image
```
POST /predict/image
Body: {
  "image_path": "path/to/image.jpg"
}
Response: {
  "prediction": "incident" | "normal",
  "probability": 0.92,
  "confidence": 0.92
}
```

### Predict Video
```
POST /predict/video
Body: {
  "video_path": "path/to/video.mp4"
}
Response: {
  "predictions": [...],
  "incidents": [...]
}
```

### Get Incidents
```
GET /incidents
Query params:
  - limit: số lượng incidents (default: 100)
  - offset: offset (default: 0)
  - status: filter by status
```

Xem chi tiết tại: **http://localhost:8000/docs** (Swagger UI)

##  Kiểm tra Trạng thái

```bash
# Kiểm tra nhanh
python check_training_status.py

# Kiểm tra chi tiết
python check_training_detailed.py
```

**Thông tin hiển thị**:
- Model đã train
- Dataset size
- System health
- Dependencies status

## 🛠️ Scripts Tiện ích

### Batch Scripts (Windows)
- **`he_thong.bat`**: Menu chính hệ thống
  - Tất cả chức năng trong một menu
  - Tự động setup virtual environment
  - Quick start option

### Python Scripts
- **`check_training_status.py`**: Kiểm tra trạng thái training
- **`check_training_detailed.py`**: Kiểm tra chi tiết hệ thống
- **`cleanup_system.py`**: Dọn dẹp và tổ chức lại dự án
- **`run_streamlit.py`**: Chạy Streamlit app
- **`start_api.py`**: Chạy API server
- **`train_cnn.py`**: Training script
- **`test_cnn_image.py`**: Test với ảnh
- **`test_cnn_video.py`**: Test với video
- **`test_api.py`**: Test API endpoints

## 📚 Cấu hình

### Training Configuration
File: `configs/training_config.yaml`

```yaml
training:
  epochs: 50
  batch_size: 32
  image_size: [224, 224]
  learning_rate: 0.001
  base_model: "MobileNetV2"  # MobileNetV2, ResNet50, VGG16
  validation_split: 0.2
  data_augmentation: true
```

### System Configuration
File: `src/utils/config.py`

- Model paths
- Database connection
- API settings
- Logging configuration

### Streamlit Configuration
File: `.streamlit/config.toml`

- Theme settings
- Server settings
- Browser settings

## 🧪 Testing

### Unit Tests
```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/unit/test_preprocessors.py

# Với coverage
pytest --cov=src tests/
```

### Integration Tests
```bash
# Test API
python test_api.py

# Test CNN với ảnh
python test_cnn_image.py data/images/incident/img1.jpg

# Test CNN với video
python test_cnn_video.py path/to/video.mp4
```

##  Performance Metrics

### Model Performance
- **Accuracy**: >90% (target)
- **Recall**: >85% (target)
- **Precision**: >85% (target)
- **False Alarm Rate**: <10% (target)
- **F1-Score**: >85% (target)

### System Performance
- **Latency (CPU)**: ~200-300ms per frame
- **Latency (GPU)**: ~20-50ms per frame
- **FPS**: >5 FPS (target)
- **Model Size**: <50MB (để deploy edge)

### Temporal Confirmation
- **False Alarm Reduction**: ~30-50%
- **Confirmation Window**: K frames (configurable)
- **Cooldown Period**: Tránh spam alerts

## 🗄️ Database Setup

### PostgreSQL Setup

1. **Cài đặt PostgreSQL** (nếu chưa có)

2. **Tạo database**:
```sql
CREATE DATABASE traffic_incidents;
```

3. **Chạy migration**:
```bash
# Sử dụng SQL script
psql -U postgres -d traffic_incidents -f src/database/migrations/001_initial_schema.sql

# Hoặc sử dụng SQLAlchemy (tự động tạo tables)
python -c "from src.database.models import *; from src.utils.config import get_db_engine; engine = get_db_engine(); Base.metadata.create_all(engine)"
```

### Database Schema

**Tables**:
- `incidents`: Lưu incidents đã phát hiện
- `predictions`: Lưu tất cả predictions (audit trail)
- `model_runs`: Lưu thông tin training runs
- `alerts`: Lưu alert history

Xem chi tiết: `src/database/models.py`

## 📚 Tài liệu

### Tài liệu Kỹ thuật

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Kiến trúc hệ thống chi tiết
  - Pipeline overview
  - Component architecture
  - Data flow diagrams
  - Latency optimization
  - Deployment guide

- **[ROADMAP.md](docs/ROADMAP.md)** - Roadmap 3 phase
  - Phase 1: MVP (Minimum Viable Product)
  - Phase 2: Hybrid (Edge + Cloud)
  - Phase 3: Production (Scalable, Production-ready)

- **[EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)** - Evaluation protocol
  - Dataset split strategy
  - Metrics calculation
  - Threshold selection
  - MTTD (Mean Time To Detection)

- **[BASELINE_COMPARISON.md](docs/BASELINE_COMPARISON.md)** - Baseline & Model Comparison
  - Baseline methods
  - Model comparison (CNN, ANN, RNN, RBFNN)
  - Performance benchmarks

- **[BAO_CAO_CUOI.md](docs/BAO_CAO_CUOI.md)** - Outline báo cáo cuối
  - Cấu trúc báo cáo
  - Nội dung từng chương

- **[HUONG_DAN_SU_DUNG.md](docs/HUONG_DAN_SU_DUNG.md)** - Hướng dẫn sử dụng
  - Module mới
  - Best practices
  - Troubleshooting

- **[PHAN_CONG_BAO_CAO.md](PHAN_CONG_BAO_CAO.md)** - Phân công báo cáo
  - Phân công công việc
  - Khung chi tiết cho từng chương
  - Checklist cho từng thành viên

### Module Mới

- **Temporal Confirmation** (`src/serving/temporal_confirmation.py`)
  - Giảm false alarm bằng cách xác nhận qua nhiều frames
  - K-frames confirmation
  - Moving average window
  - Cooldown period

- **Database Models** (`src/database/models.py`)
  - SQLAlchemy models cho PostgreSQL
  - Incident, Prediction, ModelRun models
  - Relationships và indexes

- **Migration Scripts** (`src/database/migrations/`)
  - Database migration scripts
  - Schema versioning

## 🐛 Troubleshooting

### Lỗi thường gặp

#### 1. Không tìm thấy venv311
```bash
# Giải pháp: Tạo virtual environment
he_thong.bat → [6] Tạo Virtual Environment
```

#### 2. TensorFlow không cài được
```bash
# Kiểm tra Python version (phải 3.9-3.11)
python --version

# Cài đặt lại TensorFlow
pip uninstall tensorflow
pip install tensorflow
```

#### 3. Model không load được
```bash
# Kiểm tra file model có tồn tại
ls models/CNN_model/model.keras

# Nếu không có, cần train model trước
python train_cnn.py
```

#### 4. Database connection error
```bash
# Kiểm tra PostgreSQL đang chạy
# Kiểm tra connection string trong .env hoặc config.py
```

#### 5. Out of memory khi training
```bash
# Giảm batch size
# Giảm image size
# Sử dụng data generator thay vì load all vào memory
```

### Performance Issues

#### Latency cao
- Sử dụng GPU thay vì CPU
- Giảm image size (224x224 → 128x128)
- Batch processing
- Model quantization

#### Memory issues
- Giảm batch size
- Sử dụng data generators
- Clear cache sau mỗi epoch

## 🔒 Security

### Best Practices
- Không commit `.env` files
- Sử dụng environment variables cho secrets
- API authentication (JWT tokens) - tương lai
- Database encryption
- HTTPS cho production

##  Deployment

### Development
```bash
python run_streamlit.py
python start_api.py
```

### Production (Tương lai)
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Cloud**: AWS, GCP, Azure
- **Edge**: Jetson, Coral devices

Xem chi tiết: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết chi tiết

## 👥 Tác giả

**Dự án số 37 - Hệ thống Phát hiện Sự cố Giao thông**

Nhóm phát triển:
- Hùng - Chương 1 & 6
- Phước - Chương 2
- Nhung - Chương 3
- Tài - Chương 4
- Đạt - Chương 5

## 🙏 Acknowledgments

- TensorFlow/Keras team
- Streamlit team
- FastAPI team
- OpenCV community
- Tất cả contributors

---

## 📞 Hỗ trợ & Liên hệ

- **Documentation**: Xem thư mục `docs/`
- **Issues**: Tạo issue trên repository
- **Code Comments**: Đọc docstrings trong source code
- **Examples**: Xem `docs/examples/`

**Chúc bạn sử dụng hệ thống thành công! **

---

*Cập nhật lần cuối: 2024*
>>>>>>> 8b941ce (Initial release: Traffic Incident Detection System with full documentation)
