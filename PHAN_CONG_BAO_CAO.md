# 📋 PHÂN CÔNG BÁO CÁO TIỂU LUẬN - HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG

## 🎯 TỔNG QUAN DỰ ÁN

**Tên đề tài**: Hệ thống Phát hiện Sự cố Giao thông Tự động sử dụng Deep Learning

**Domain**: Traffic Incident Detection (Intersection/Freeway)

**Công nghệ chính**:
- **Computer Vision**: CNN (MobileNetV2, ResNet50, VGG16), Transfer Learning
- **Neural Networks**: ANN, RNN (LSTM/GRU), RBFNN
- **Backend**: FastAPI, PostgreSQL, MLflow
- **Frontend**: Streamlit Dashboard
- **Xử lý**: OpenCV, Temporal Confirmation, Image Processing

**Pipeline**: Camera/Video → Preprocessing → CNN Inference → Temporal Confirmation → Incident Service → Alert/Dashboard

---

## 👥 PHÂN CÔNG CÔNG VIỆC

### **Hùng: Chương 1 & Chương 6**
- **Chương 1**: Tổng quan đề tài (2-3 trang)
- **Chương 6**: Kết luận & Hướng phát triển (2-3 trang)
- **Tổng hợp & Formatting**: Định dạng toàn bộ báo cáo, kiểm tra tính nhất quán

### **Phước: Chương 2**
- **Chương 2**: Cơ sở lý thuyết & Phân tích yêu cầu (3-4 trang)

### **Nhung: Chương 3**
- **Chương 3**: Thiết kế hệ thống (4-5 trang)

### **Tài: Chương 4**
- **Chương 4**: Hiện thực & Triển khai (4-5 trang)

### **Đạt: Chương 5**
- **Chương 5**: Kiểm thử & Đánh giá (3-4 trang)

---

## 📖 KHUNG CHI TIẾT CHO TỪNG CHƯƠNG

---

## CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI

### 📌 Yêu cầu chung
- **Độ dài**: 2-3 trang
- **Mục tiêu**: Giới thiệu vấn đề, domain, mục tiêu và hạn chế

### 📝 Khung nội dung

#### 1.1. Domain và Bối cảnh
**Nội dung cần có**:
- **Domain**: Intersection/Freeway traffic incident detection
- **Bối cảnh thực tế**: 
  - Tình trạng giao thông tại Việt Nam
  - Tầm quan trọng của phát hiện sự cố sớm
  - Ứng dụng trong ITS (Intelligent Transportation Systems)
- **Ví dụ cụ thể**: Tai nạn, xe hỏng, sự kiện đặc biệt trên đường cao tốc/giao lộ

**Tài liệu tham khảo**:
- File: `README.md` - Mô tả hệ thống
- File: `docs/ARCHITECTURE.md` - Kiến trúc hệ thống
- File: `he_thong.bat` - Các chức năng hệ thống

#### 1.2. Vấn đề cần giải quyết (Clear & Measurable)
**Nội dung cần có**:
- **Vấn đề chính**: 
  - Phát hiện sự cố giao thông từ ảnh/video camera tự động
  - Giảm thời gian phản ứng (MTTD - Mean Time To Detection)
  - Giảm False Alarm Rate (FAR)
- **Metrics đo lường**:
  - **Accuracy**: Độ chính xác phát hiện (target: >90%)
  - **Recall**: Tỷ lệ phát hiện đúng sự cố (target: >85%)
  - **False Alarm Rate**: Tỷ lệ báo động sai (target: <10%)
  - **Latency**: Thời gian xử lý (target: <200ms per frame)
  - **FPS**: Frames per second xử lý được (target: >5 FPS)

**Thông tin từ hệ thống**:
- File: `docs/EVALUATION_PROTOCOL.md` - Metrics và evaluation
- File: `src/serving/temporal_confirmation.py` - Temporal confirmation để giảm FAR
- File: `docs/ARCHITECTURE.md` - Latency targets

#### 1.3. Mục tiêu & Yêu cầu tổng quan
**Nội dung cần có**:
- **Mục tiêu chính**:
  1. Xây dựng hệ thống phát hiện sự cố giao thông tự động
  2. Sử dụng CNN với Transfer Learning (MobileNetV2, ResNet50, VGG16)
  3. Tích hợp Temporal Confirmation để giảm false alarm
  4. Xây dựng Dashboard (Streamlit) và API (FastAPI)
  5. Lưu trữ và quản lý incidents trong PostgreSQL

- **Yêu cầu chức năng**:
  - Phát hiện sự cố từ ảnh/video
  - Huấn luyện mô hình CNN
  - Giao diện web để upload và xem kết quả
  - API để tích hợp với hệ thống khác
  - Lưu trữ incidents vào database

- **Yêu cầu phi chức năng**:
  - Latency: <200ms per frame (CPU), <50ms (GPU)
  - Accuracy: >90%
  - Scalability: Hỗ trợ nhiều camera đồng thời
  - Reliability: Hệ thống ổn định, có logging và monitoring

**Thông tin từ hệ thống**:
- File: `he_thong.bat` - Các chức năng: GUI, API, Training, Testing
- File: `src/models/cnn.py` - CNN models với Transfer Learning
- File: `src/serving/api.py` - FastAPI endpoints
- File: `app.py` hoặc `run_streamlit.py` - Streamlit dashboard

#### 1.4. Hạn chế thực tế
**Nội dung cần có**:
- **Camera & Thiết bị**:
  - Chất lượng camera (resolution, góc quay)
  - Điều kiện ánh sáng (sáng/tối, ngày/đêm)
  - Vị trí lắp đặt camera

- **Ánh sáng**:
  - Ảnh hưởng của ánh sáng tự nhiên (ngày/đêm)
  - Phản xạ, bóng đổ
  - Weather conditions (mưa, sương mù)

- **Latency**:
  - Xử lý trên CPU vs GPU
  - Network latency (nếu camera remote)
  - Database write latency

- **Thiết bị**:
  - Yêu cầu phần cứng (CPU/GPU, RAM)
  - Edge deployment (Jetson, Coral) vs Cloud

- **Dữ liệu**:
  - Số lượng ảnh training (hiện tại: normal/incident folders)
  - Chất lượng dữ liệu (labeling, diversity)
  - Data augmentation để tăng dataset

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Section "ĐIỂM NGHẼN LATENCY VÀ CÁCH TỐI ƯU"
- File: `data/images/` - Cấu trúc dữ liệu training
- File: `src/data_processing/image_processor.py` - Image preprocessing

#### 1.5. Cấu trúc báo cáo
**Nội dung cần có**:
- Tóm tắt các chương tiếp theo
- Mối liên hệ giữa các chương

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & PHÂN TÍCH YÊU CẦU

### 📌 Yêu cầu chung
- **Độ dài**: 3-4 trang
- **Mục tiêu**: Trình bày lý thuyết và phân tích yêu cầu chi tiết

### 📝 Khung nội dung

#### 2.1. Lý thuyết liên quan

##### 2.1.1. Computer Vision
**Nội dung cần có**:
- **CNN (Convolutional Neural Network)**:
  - Kiến trúc CNN cơ bản (Convolution, Pooling, Fully Connected)
  - Transfer Learning: MobileNetV2, ResNet50, VGG16
  - Tại sao dùng Transfer Learning (giảm training time, tận dụng pre-trained weights)
  
- **Image Classification**:
  - Binary classification (normal vs incident)
  - Image preprocessing (resize, normalize, augmentation)
  - Data augmentation techniques

**Thông tin từ hệ thống**:
- File: `src/models/cnn.py` - Implementation CNN với Transfer Learning
- File: `src/data_processing/image_processor.py` - Image preprocessing
- File: `train_cnn.py` - Training script

##### 2.1.2. Deep Learning Models
**Nội dung cần có**:
- **CNN Model**:
  - MobileNetV2: Lightweight, phù hợp real-time
  - ResNet50: Deeper network, accuracy cao hơn
  - VGG16: Classic architecture
  - So sánh ưu/nhược điểm

- **Neural Networks khác** (nếu có):
  - ANN (Feed-forward): Dữ liệu mô phỏng
  - RNN (LSTM/GRU): Dữ liệu temporal
  - RBFNN: Radial Basis Function Network

**Thông tin từ hệ thống**:
- File: `src/models/cnn.py` - CNN implementation
- File: `src/models/ann.py` - ANN model
- File: `src/models/rnn.py` - RNN model
- File: `src/models/rbfnn.py` - RBFNN model
- File: `pipelines/training_pipeline.py` - Training pipeline cho các models

##### 2.1.3. Temporal Processing
**Nội dung cần có**:
- **Temporal Confirmation**:
  - Vấn đề: False alarm từ single frame
  - Giải pháp: Xác nhận qua nhiều frames
  - Methods:
    - K-frames confirmation (cần K frames liên tiếp có incident)
    - Moving average window
    - Cooldown period (tránh spam alerts)

**Thông tin từ hệ thống**:
- File: `src/serving/temporal_confirmation.py` - Temporal confirmation implementation
- File: `docs/ARCHITECTURE.md` - Section "Temporal Confirmation Layer"

##### 2.1.4. Image Processing
**Nội dung cần có**:
- **Preprocessing**:
  - Resize to 224x224 (input size cho CNN)
  - Normalization (0-1 range)
  - Data augmentation (rotation, flip, brightness, contrast)

- **OpenCV & Pillow**:
  - Sử dụng OpenCV cho video processing
  - Pillow cho image manipulation

**Thông tin từ hệ thống**:
- File: `src/data_processing/image_processor.py` - Image processing
- File: `src/data_processing/preprocessors.py` - Preprocessing functions

#### 2.2. Phân tích yêu cầu

##### 2.2.1. Functional Requirements (FR)
**Nội dung cần có**:

**FR1: Phát hiện sự cố từ ảnh**
- Input: Ảnh (JPG, PNG, WEBP)
- Output: Probability (0-1), Classification (normal/incident)
- Accuracy: >90%

**FR2: Phát hiện sự cố từ video**
- Input: Video file (MP4, AVI) hoặc RTSP stream
- Output: Frame-by-frame predictions, Temporal confirmation
- FPS: >5 FPS

**FR3: Huấn luyện mô hình**
- Input: Dataset (normal/incident images)
- Output: Trained model (.keras file)
- Features: Configurable epochs, batch size, image size

**FR4: Giao diện Web**
- Upload ảnh/video
- Xem kết quả prediction
- Xem metrics và training history
- Quản lý incidents

**FR5: API Service**
- REST API endpoints
- Predict image/video
- Get incidents
- Health check

**FR6: Database Storage**
- Lưu incidents
- Lưu predictions (audit trail)
- Lưu model runs (MLflow)

**Thông tin từ hệ thống**:
- File: `he_thong.bat` - Menu chức năng: [1] GUI, [2] API, [3] Training, [4] Test
- File: `src/serving/api.py` - API endpoints
- File: `app.py` hoặc `run_streamlit.py` - Streamlit UI
- File: `src/database/models.py` - Database schema

##### 2.2.2. Non-Functional Requirements (NFR)
**Nội dung cần có**:

**NFR1: Performance**
- Latency: <200ms per frame (CPU), <50ms (GPU)
- Throughput: >5 FPS
- Model size: <50MB (để deploy edge)

**NFR2: Accuracy**
- Accuracy: >90%
- Recall: >85%
- False Alarm Rate: <10%

**NFR3: Reliability**
- System uptime: >99%
- Error handling: Graceful degradation
- Logging: Structured logs (JSON)

**NFR4: Scalability**
- Hỗ trợ nhiều camera đồng thời
- Horizontal scaling (multiple API instances)
- Database connection pooling

**NFR5: Maintainability**
- Code organization (src/ structure)
- Documentation (docstrings, README)
- Testing (unit tests)

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Performance targets, scalability
- File: `src/utils/logger.py` - Logging system
- File: `tests/` - Test files

##### 2.2.3. Actors & User Scenarios
**Nội dung cần có**:

**Actors**:
1. **Traffic Management Center (TMC) Operator**:
   - Xem incidents real-time
   - Confirm/false alarm incidents
   - Xem analytics

2. **System Administrator**:
   - Train models
   - Monitor system health
   - Configure settings

3. **API Consumer** (External System):
   - Gọi API để predict
   - Lấy incidents data

**User Scenarios**:

**Scenario 1: Phát hiện sự cố từ camera**
1. Camera stream → System
2. System preprocess frames
3. CNN inference → Probability
4. Temporal confirmation → Incident event
5. Save to database
6. Send alert to TMC
7. TMC xem trên dashboard

**Scenario 2: Huấn luyện mô hình mới**
1. Admin upload dataset (normal/incident)
2. Configure training parameters
3. Start training
4. Monitor metrics (loss, accuracy)
5. Save model
6. Deploy model

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Pipeline flow
- File: `he_thong.bat` - User workflows
- File: `app.py` - Streamlit UI workflows

---

## CHƯƠNG 3: THIẾT KẾ HỆ THỐNG

### 📌 Yêu cầu chung
- **Độ dài**: 4-5 trang
- **Mục tiêu**: Thiết kế kiến trúc, components, data flow, algorithms

### 📝 Khung nội dung

#### 3.1. System Architecture
**Nội dung cần có**:

##### 3.1.1. High-Level Architecture
- **Pipeline tổng quan**:
  ```
  Camera/Video → Preprocessing → CNN Inference → Temporal Confirmation → 
  Incident Service → Alert Service → Storage → Dashboard
  ```

- **Components chính**:
  1. Data Ingestion Layer
  2. Preprocessing Layer
  3. Inference Layer (CNN)
  4. Temporal Confirmation Layer
  5. Incident Service
  6. Alert Service
  7. Storage Layer (PostgreSQL + Object Storage)
  8. Dashboard Layer (Streamlit)

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Section "PIPELINE TỔNG QUAN" và "KIẾN TRÚC CHI TIẾT"
- File: `docs/ARCHITECTURE.md` - Mermaid diagram

##### 3.1.2. Component Architecture
**Mô tả từng component**:

**1. Data Ingestion Layer**
- Input: RTSP stream, Video files, Image files
- Output: Frames (numpy arrays, 224x224x3)
- Technology: OpenCV, FFmpeg

**2. Preprocessing Layer**
- Functions: Resize, Normalize, Augmentation
- Latency: ~5-10ms per frame

**3. Inference Layer**
- Model: MobileNetV2-based CNN
- Input: Preprocessed frame (224x224x3)
- Output: Probability (0-1)
- Latency: CPU ~200-300ms, GPU ~20-50ms

**4. Temporal Confirmation Layer**
- Input: Stream of probabilities
- Methods: K-frames confirmation, Moving average, Cooldown
- Output: Incident events

**5. Incident Service**
- Functions: Create incidents, Link media, Update status
- Storage: PostgreSQL

**6. Alert Service**
- Channels: Email, SMS, Push, Webhook

**7. Storage Layer**
- Database: PostgreSQL (incidents, predictions, model_runs)
- Object Storage: Local filesystem / S3 (images, videos)

**8. Dashboard Layer**
- Technology: Streamlit
- Features: Real-time feed, Incident details, Analytics

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Chi tiết từng component
- File: `src/` - Source code structure

#### 3.2. Data Flow Diagram
**Nội dung cần có**:

##### 3.2.1. Data Flow cho Prediction
```
[Camera] → [Video Ingest] → [Preprocessing] → [CNN Inference] → 
[Probability] → [Temporal Confirmation] → [Incident Event] → 
[Incident Service] → [Database] + [Object Storage] → [Alert Service] → [Dashboard]
```

##### 3.2.2. Data Flow cho Training
```
[Dataset] → [Image Loader] → [Preprocessing + Augmentation] → 
[CNN Model] → [Training Loop] → [Validation] → [Model Save] → [MLflow Tracking]
```

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Pipeline diagrams
- File: `train_cnn.py` - Training flow
- File: `src/serving/predictor.py` - Prediction flow

#### 3.3. Component/Module Design
**Nội dung cần có**:

##### 3.3.1. Module Structure
```
src/
├── models/          # ML models (CNN, ANN, RNN, RBFNN)
├── training/        # Training pipeline
├── data_processing/ # Image processing, preprocessing
├── serving/         # API, predictor, temporal confirmation
├── database/        # Database models, migrations
└── utils/          # Config, logger
```

##### 3.3.2. Key Modules

**1. CNN Model (`src/models/cnn.py`)**
- Class: `CNNModel`
- Methods: `build()`, `train()`, `predict()`, `save()`, `load()`
- Transfer Learning: MobileNetV2, ResNet50, VGG16

**2. Image Processor (`src/data_processing/image_processor.py`)**
- Functions: Resize, Normalize, Augment

**3. Temporal Confirmation (`src/serving/temporal_confirmation.py`)**
- Class: `TemporalConfirmation`
- Methods: `confirm()`, `update()`, `reset()`

**4. API (`src/serving/api.py`)**
- Endpoints: `/predict/image`, `/predict/video`, `/incidents`, `/health`

**5. Database Models (`src/database/models.py`)**
- Tables: `Incident`, `Prediction`, `ModelRun`

**Thông tin từ hệ thống**:
- File: `src/` - Source code structure
- File: `README.md` - Section "Cấu trúc Dự án"

#### 3.4. Decision Logic & Algorithms
**Nội dung cần có**:

##### 3.4.1. CNN Inference Algorithm
```
1. Load pre-trained CNN model
2. Preprocess image (resize to 224x224, normalize)
3. Model.predict(image) → probability
4. If probability > threshold (0.5):
     → Incident detected
   Else:
     → Normal
```

##### 3.4.2. Temporal Confirmation Algorithm
```
1. Maintain sliding window of probabilities [p_1, p_2, ..., p_K]
2. For each new probability p_t:
   a. Add to window
   b. Calculate moving average
   c. If moving_average > threshold AND K consecutive frames > threshold:
        → Confirm incident
   d. Apply cooldown period (avoid spam)
```

**Thông tin từ hệ thống**:
- File: `src/serving/temporal_confirmation.py` - Algorithm implementation
- File: `src/serving/predictor.py` - Inference logic

##### 3.4.3. Training Algorithm
```
1. Load dataset (normal/incident images)
2. Split: train/validation (80/20)
3. Data augmentation (rotation, flip, brightness)
4. Build CNN model (Transfer Learning)
5. Compile model (optimizer, loss, metrics)
6. Train model (epochs, batch_size)
7. Evaluate on validation set
8. Save best model
9. Log to MLflow
```

**Thông tin từ hệ thống**:
- File: `train_cnn.py` - Training script
- File: `src/training/trainer.py` - Training logic
- File: `src/models/cnn.py` - Model building

#### 3.5. Database Schema Design
**Nội dung cần có**:

##### 3.5.1. Tables

**Table: incidents**
- `id` (PK)
- `timestamp` (datetime)
- `camera_id` (string)
- `confidence_score` (float)
- `model_version` (string)
- `status` (string: detected/confirmed/false_alarm/resolved)
- `image_path` (string)
- `metadata` (JSON)

**Table: predictions**
- `id` (PK)
- `timestamp` (datetime)
- `camera_id` (string)
- `probability` (float)
- `prediction` (string: normal/incident)
- `frame_number` (int)

**Table: model_runs**
- `id` (PK)
- `run_id` (string, MLflow)
- `model_version` (string)
- `training_date` (datetime)
- `metrics` (JSON)
- `model_path` (string)

**Thông tin từ hệ thống**:
- File: `src/database/models.py` - SQLAlchemy models
- File: `src/database/migrations/001_initial_schema.sql` - SQL schema

---

## CHƯƠNG 4: HIỆN THỰC & TRIỂN KHAI

### 📌 Yêu cầu chung
- **Độ dài**: 4-5 trang
- **Mục tiêu**: Mô tả implementation, tools, code structure, screenshots

### 📝 Khung nội dung

#### 4.1. Công cụ & Công nghệ sử dụng
**Nội dung cần có**:

##### 4.1.1. Programming Language & Framework
- **Python 3.11**: Main language
- **TensorFlow/Keras**: Deep Learning framework
- **FastAPI**: REST API framework
- **Streamlit**: Web dashboard framework

##### 4.1.2. Libraries & Tools
- **Computer Vision**: OpenCV, Pillow
- **Data Processing**: NumPy, Pandas
- **MLOps**: MLflow (experiment tracking)
- **Database**: SQLAlchemy, PostgreSQL
- **Utilities**: python-dotenv, pyyaml, python-json-logger

**Thông tin từ hệ thống**:
- File: `requirements.txt` - All dependencies
- File: `README.md` - Section "Công nghệ sử dụng"

##### 4.1.3. Development Tools
- **IDE**: VS Code, PyCharm
- **Version Control**: Git
- **Virtual Environment**: venv (Python 3.11)
- **Package Manager**: pip

#### 4.2. Mô tả các Module chính

##### 4.2.1. CNN Model Module (`src/models/cnn.py`)
**Nội dung cần có**:
- **Class**: `CNNModel`
- **Key Methods**:
  - `build()`: Xây dựng model với Transfer Learning
  - `train()`: Huấn luyện model
  - `predict()`: Dự đoán từ ảnh
  - `save()` / `load()`: Lưu/tải model

- **Implementation details**:
  - Transfer Learning với MobileNetV2/ResNet50/VGG16
  - Custom top layers (GlobalAveragePooling, Dense, Dropout)
  - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

**Code snippets** (từ hệ thống):
- File: `src/models/cnn.py` - Copy relevant code sections

**Thông tin từ hệ thống**:
- File: `src/models/cnn.py` - Full implementation
- File: `src/models/base_model.py` - Base class

##### 4.2.2. Image Processing Module (`src/data_processing/image_processor.py`)
**Nội dung cần có**:
- **Functions**:
  - `resize_image()`: Resize to 224x224
  - `normalize_image()`: Normalize to 0-1 range
  - `augment_image()`: Data augmentation (rotation, flip, brightness)

**Code snippets**:
- File: `src/data_processing/image_processor.py`

##### 4.2.3. Temporal Confirmation Module (`src/serving/temporal_confirmation.py`)
**Nội dung cần có**:
- **Class**: `TemporalConfirmation`
- **Methods**:
  - `confirm()`: Xác nhận incident qua K frames
  - `update()`: Cập nhật sliding window
  - `reset()`: Reset state

- **Algorithm**:
  - Sliding window of probabilities
  - Moving average calculation
  - Cooldown period

**Code snippets**:
- File: `src/serving/temporal_confirmation.py`

##### 4.2.4. API Module (`src/serving/api.py`)
**Nội dung cần có**:
- **Endpoints**:
  - `POST /predict/image`: Predict từ ảnh
  - `POST /predict/video`: Predict từ video
  - `GET /incidents`: Lấy danh sách incidents
  - `GET /health`: Health check

- **Request/Response formats**:
  - Input: Image file hoặc image path
  - Output: JSON với probability, prediction, confidence

**Code snippets**:
- File: `src/serving/api.py`

##### 4.2.5. Training Pipeline (`train_cnn.py`, `src/training/trainer.py`)
**Nội dung cần có**:
- **Workflow**:
  1. Load dataset từ `data/images/normal/` và `data/images/incident/`
  2. Split train/validation
  3. Data augmentation
  4. Build model
  5. Train với callbacks
  6. Evaluate
  7. Save model to `models/CNN_model/model.keras`
  8. Log to MLflow

**Code snippets**:
- File: `train_cnn.py`
- File: `src/training/trainer.py`

##### 4.2.6. Streamlit Dashboard (`app.py` hoặc `run_streamlit.py`)
**Nội dung cần có**:
- **Features**:
  - Upload ảnh/video
  - Xem prediction results
  - Training interface
  - Metrics visualization
  - Incident management

**Screenshots cần có**:
- Giao diện upload ảnh
- Kết quả prediction
- Training interface
- Metrics charts

**Thông tin từ hệ thống**:
- File: `app.py` hoặc `run_streamlit.py` - Streamlit app
- Chạy `python run_streamlit.py` để chụp screenshots

#### 4.3. Design → Code Mapping
**Nội dung cần có**:

##### 4.3.1. Architecture → Implementation
- **Data Ingestion Layer** → `src/data_processing/collectors.py`
- **Preprocessing Layer** → `src/data_processing/image_processor.py`, `preprocessors.py`
- **Inference Layer** → `src/models/cnn.py`, `src/serving/predictor.py`
- **Temporal Confirmation** → `src/serving/temporal_confirmation.py`
- **Incident Service** → `src/serving/api.py` (incident endpoints)
- **Storage Layer** → `src/database/models.py`, PostgreSQL
- **Dashboard** → `app.py`, `run_streamlit.py`

##### 4.3.2. Data Flow → Code Flow
- **Prediction Flow**:
  ```
  API endpoint → predictor.py → cnn.py → temporal_confirmation.py → 
  database/models.py → API response
  ```

- **Training Flow**:
  ```
  train_cnn.py → trainer.py → cnn.py → MLflow → model save
  ```

#### 4.4. Flowchart thực tế
**Nội dung cần có**:

##### 4.4.1. Prediction Flowchart
```
[Start] → [Load Image] → [Preprocess] → [CNN Inference] → 
[Probability > 0.5?] → Yes → [Temporal Confirmation] → 
[K frames confirmed?] → Yes → [Create Incident] → [Save DB] → 
[Send Alert] → [End]
                    ↓ No
              [Normal] → [End]
```

##### 4.4.2. Training Flowchart
```
[Start] → [Load Dataset] → [Split Train/Val] → [Augment] → 
[Build Model] → [Compile] → [Train Loop] → [Validate] → 
[Save Best Model] → [Log MLflow] → [End]
```

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Pipeline diagrams
- Code flow trong source files

#### 4.5. Screenshots & Result Images
**Nội dung cần có**:

##### 4.5.1. System Screenshots
1. **Streamlit Dashboard**:
   - Home page
   - Upload image interface
   - Prediction results
   - Training interface
   - Metrics visualization

2. **API Documentation**:
   - Swagger UI (`http://localhost:8000/docs`)

3. **Training Process**:
   - Training progress
   - Loss/Accuracy curves
   - Model summary

##### 4.5.2. Result Images
1. **Prediction Examples**:
   - Normal image → Prediction: Normal (confidence: 0.15)
   - Incident image → Prediction: Incident (confidence: 0.92)

2. **Model Performance**:
   - Confusion matrix
   - ROC curve
   - Training history plots

**Cách lấy screenshots**:
- Chạy `python run_streamlit.py` → Chụp màn hình
- Chạy `python start_api.py` → Mở `http://localhost:8000/docs` → Chụp màn hình
- Test với ảnh trong `data/images/` → Chụp kết quả

---

## CHƯƠNG 5: KIỂM THỬ & ĐÁNH GIÁ

### 📌 Yêu cầu chung
- **Độ dài**: 3-4 trang
- **Mục tiêu**: Test cases, metrics, evaluation, limitations

### 📝 Khung nội dung

#### 5.1. Test Cases
**Nội dung cần có**:

##### 5.1.1. Functional Testing

**Test Case 1: Phát hiện sự cố từ ảnh**
- **Input**: Ảnh có sự cố (từ `data/images/incident/`)
- **Expected**: Prediction = "Incident", Confidence > 0.5
- **Actual**: [Ghi kết quả thực tế]
- **Status**: Pass/Fail

**Test Case 2: Phát hiện ảnh bình thường**
- **Input**: Ảnh bình thường (từ `data/images/normal/`)
- **Expected**: Prediction = "Normal", Confidence < 0.5
- **Actual**: [Ghi kết quả thực tế]
- **Status**: Pass/Fail

**Test Case 3: Phát hiện từ video**
- **Input**: Video file
- **Expected**: Frame-by-frame predictions, Temporal confirmation
- **Actual**: [Ghi kết quả thực tế]
- **Status**: Pass/Fail

**Test Case 4: API Endpoints**
- **Test**: `POST /predict/image`
- **Expected**: JSON response với probability, prediction
- **Actual**: [Ghi kết quả thực tế]
- **Status**: Pass/Fail

**Thông tin từ hệ thống**:
- File: `test_cnn_image.py` - Test với ảnh
- File: `test_cnn_video.py` - Test với video
- File: `test_api.py` - Test API
- File: `tests/unit/test_preprocessors.py` - Unit tests

##### 5.1.2. Test Cases theo Bối cảnh

**Bối cảnh 1: Điều kiện ánh sáng sáng (ngày)**
- Test với ảnh sáng
- Expected: Accuracy tốt
- Results: [Ghi kết quả]

**Bối cảnh 2: Điều kiện ánh sáng tối (đêm)**
- Test với ảnh tối
- Expected: Accuracy có thể giảm
- Results: [Ghi kết quả]

**Bối cảnh 3: Giao thông đông**
- Test với ảnh có nhiều xe
- Expected: Có thể có false alarm
- Results: [Ghi kết quả]

**Bối cảnh 4: Giao thông vắng**
- Test với ảnh ít xe
- Expected: Accuracy tốt
- Results: [Ghi kết quả]

**Bối cảnh 5: Occlusion (che khuất)**
- Test với ảnh có vật che khuất
- Expected: Có thể miss detection
- Results: [Ghi kết quả]

**Cách test**:
- Sử dụng ảnh trong `data/images/` (nếu có đa dạng)
- Hoặc thu thập thêm ảnh test
- Chạy `python test_cnn_image.py <image_path>`

#### 5.2. Metrics & Evaluation

##### 5.2.1. Model Performance Metrics

**Accuracy**
- **Definition**: Tỷ lệ dự đoán đúng
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Target**: >90%
- **Actual**: [Ghi kết quả từ validation set]

**Recall (Sensitivity)**
- **Definition**: Tỷ lệ phát hiện đúng sự cố
- **Formula**: TP / (TP + FN)
- **Target**: >85%
- **Actual**: [Ghi kết quả]

**Precision**
- **Definition**: Tỷ lệ dự đoán incident là đúng
- **Formula**: TP / (TP + FP)
- **Target**: >85%
- **Actual**: [Ghi kết quả]

**False Alarm Rate (FAR)**
- **Definition**: Tỷ lệ báo động sai
- **Formula**: FP / (FP + TN)
- **Target**: <10%
- **Actual**: [Ghi kết quả]
- **Note**: Temporal Confirmation giúp giảm FAR

**F1-Score**
- **Definition**: Harmonic mean của Precision và Recall
- **Formula**: 2 * (Precision * Recall) / (Precision + Recall)
- **Target**: >85%
- **Actual**: [Ghi kết quả]

**Thông tin từ hệ thống**:
- File: `src/training/evaluator.py` - Evaluation functions
- File: `docs/EVALUATION_PROTOCOL.md` - Evaluation protocol
- File: `docs/BASELINE_COMPARISON.md` - Baseline comparison

##### 5.2.2. System Performance Metrics

**Latency (Inference Time)**
- **CPU**: ~200-300ms per frame
- **GPU**: ~20-50ms per frame
- **Target**: <200ms (CPU), <50ms (GPU)
- **Actual**: [Ghi kết quả]

**FPS (Frames Per Second)**
- **Target**: >5 FPS
- **Actual**: [Ghi kết quả]

**Model Size**
- **Target**: <50MB (để deploy edge)
- **Actual**: [Ghi kích thước file `models/CNN_model/model.keras`]

**Thông tin từ hệ thống**:
- File: `docs/ARCHITECTURE.md` - Latency targets
- Measure bằng cách chạy inference và đo thời gian

##### 5.2.3. Evaluation Protocol

**Dataset Split**
- **Train**: 80%
- **Validation**: 20%
- **Test**: (nếu có)
- **Note**: Tránh data leakage (không shuffle trước khi split)

**Evaluation Method**
- K-fold Cross Validation (nếu có đủ data)
- Hoặc Train/Validation split

**Thông tin từ hệ thống**:
- File: `docs/EVALUATION_PROTOCOL.md` - Chi tiết protocol
- File: `train_cnn.py` - Split logic

#### 5.3. Kết quả thực nghiệm

##### 5.3.1. Training Results
- **Epochs**: [Số epochs đã train]
- **Final Accuracy**: [Kết quả]
- **Final Loss**: [Kết quả]
- **Training Time**: [Thời gian]
- **Best Model**: Saved tại `models/CNN_model/model.keras`

**Visualizations**:
- Loss curve (training vs validation)
- Accuracy curve
- Confusion matrix
- ROC curve (nếu có)

**Thông tin từ hệ thống**:
- File: `src/training/visualizer.py` - Visualization functions
- MLflow tracking: `http://localhost:5000` (nếu chạy MLflow)

##### 5.3.2. Test Results
- **Test Set Size**: [Số ảnh test]
- **Accuracy**: [Kết quả]
- **Recall**: [Kết quả]
- **Precision**: [Kết quả]
- **FAR**: [Kết quả]

**Confusion Matrix**:
```
                Predicted
              Normal  Incident
Actual Normal   [TN]    [FP]
       Incident [FN]    [TP]
```

#### 5.4. Hạn chế

##### 5.4.1. Dataset Limitations
- **Size**: Dataset nhỏ (hiện tại: normal/incident folders)
- **Diversity**: Thiếu đa dạng (điều kiện ánh sáng, thời tiết, góc camera)
- **Labeling**: Cần labeling chính xác

##### 5.4.2. Model Limitations
- **Transfer Learning**: Phụ thuộc vào pre-trained weights (ImageNet)
- **Binary Classification**: Chỉ phân loại Normal vs Incident (chưa phân loại loại sự cố)
- **Single Frame**: Dựa trên single frame (chưa tận dụng temporal information đầy đủ)

##### 5.4.3. System Limitations
- **Latency**: Chậm trên CPU (cần GPU để real-time)
- **Scalability**: Chưa test với nhiều camera đồng thời
- **Edge Deployment**: Chưa optimize cho edge devices (Jetson, Coral)

**Thông tin từ hệ thống**:
- File: `docs/ROADMAP.md` - Future improvements
- File: `docs/ARCHITECTURE.md` - Limitations và optimizations

#### 5.5. Điều kiện triển khai thực tế

##### 5.5.1. Hardware Requirements
- **CPU**: Multi-core (4+ cores)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **GPU**: Optional (nhưng khuyến nghị cho real-time)
- **Storage**: Đủ để lưu models và data

##### 5.5.2. Software Requirements
- **OS**: Windows, Linux, macOS
- **Python**: 3.9, 3.10, hoặc 3.11
- **Dependencies**: Xem `requirements.txt`

##### 5.5.3. Deployment Considerations
- **Camera Setup**: Cần camera với resolution đủ, góc quay phù hợp
- **Network**: Nếu camera remote, cần network ổn định
- **Database**: PostgreSQL server (local hoặc cloud)
- **Monitoring**: Cần monitoring system health

**Thông tin từ hệ thống**:
- File: `README.md` - Section "Yêu cầu Hệ thống"
- File: `docs/ARCHITECTURE.md` - Deployment section

#### 5.6. Tiềm năng cải tiến

##### 5.6.1. Model Improvements
- **Multi-class Classification**: Phân loại loại sự cố (tai nạn, xe hỏng, ...)
- **Object Detection**: Sử dụng YOLO để detect objects trước
- **Temporal Models**: Sử dụng 3D CNN hoặc LSTM cho video
- **Ensemble**: Kết hợp nhiều models

##### 5.6.2. System Improvements
- **Edge Deployment**: Optimize cho Jetson, Coral
- **Real-time Streaming**: RTSP stream processing
- **Multi-camera**: Hỗ trợ nhiều camera đồng thời
- **Cloud Integration**: Deploy lên cloud (AWS, GCP)

##### 5.6.3. Data Improvements
- **Dataset Expansion**: Thu thập thêm data đa dạng
- **Data Augmentation**: Tăng cường augmentation techniques
- **Synthetic Data**: Tạo synthetic data (GAN, ...)

**Thông tin từ hệ thống**:
- File: `docs/ROADMAP.md` - Roadmap 3 phase (MVP → Hybrid → Production)

---

## CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 📌 Yêu cầu chung
- **Độ dài**: 2-3 trang
- **Mục tiêu**: Tóm tắt, ý nghĩa, hướng phát triển

### 📝 Khung nội dung

#### 6.1. Tóm tắt đề tài
**Nội dung cần có**:

##### 6.1.1. Vấn đề đã giải quyết
- Xây dựng hệ thống phát hiện sự cố giao thông tự động
- Sử dụng CNN với Transfer Learning (MobileNetV2, ResNet50, VGG16)
- Tích hợp Temporal Confirmation để giảm false alarm
- Xây dựng Dashboard (Streamlit) và API (FastAPI)
- Lưu trữ incidents trong PostgreSQL

##### 6.1.2. Kết quả đạt được
- **Model Performance**: Accuracy >90%, Recall >85%, FAR <10%
- **System Performance**: Latency <200ms (CPU), FPS >5
- **Features**: GUI, API, Training pipeline, Database storage

##### 6.1.3. Đóng góp chính
- Ứng dụng Deep Learning vào ITS
- Temporal Confirmation để giảm false alarm
- Hệ thống end-to-end (từ camera đến dashboard)

#### 6.2. Ý nghĩa đối với ITS
**Nội dung cần có**:

##### 6.2.1. Ứng dụng thực tế
- **Traffic Management Centers (TMC)**: Phát hiện sự cố sớm, phản ứng nhanh
- **Highway Management**: Giám sát đường cao tốc tự động
- **Smart Cities**: Tích hợp vào hệ thống thành phố thông minh

##### 6.2.2. Lợi ích
- **Giảm thời gian phản ứng**: Phát hiện sự cố sớm hơn
- **Giảm chi phí**: Tự động hóa, giảm nhân lực
- **Tăng an toàn**: Cảnh báo sớm cho người tham gia giao thông

##### 6.2.3. Tác động
- Cải thiện hiệu quả quản lý giao thông
- Hỗ trợ quyết định real-time
- Tích hợp với các hệ thống ITS khác (V2X, traffic lights, ...)

#### 6.3. Hướng phát triển

##### 6.3.1. Model Improvements
- **Multi-class Classification**: Phân loại loại sự cố
- **Object Detection**: YOLO để detect vehicles, people
- **Temporal Models**: 3D CNN, LSTM cho video sequences
- **Ensemble Methods**: Kết hợp nhiều models

##### 6.3.2. System Scaling
- **Edge Computing**: Deploy trên edge devices (Jetson, Coral)
- **Cloud Deployment**: AWS, GCP, Azure
- **Horizontal Scaling**: Multiple API instances, load balancing
- **Real-time Streaming**: RTSP stream processing

##### 6.3.3. Integration
- **V2X Communication**: Tích hợp với vehicle-to-everything
- **Traffic Light Control**: Tích hợp với hệ thống đèn giao thông
- **Navigation Systems**: Cảnh báo cho navigation apps
- **Emergency Services**: Tích hợp với 911/emergency services

##### 6.3.4. Dataset & Data
- **Dataset Expansion**: Thu thập thêm data đa dạng
- **Public Datasets**: Sử dụng public ITS datasets
- **Synthetic Data**: GAN, data augmentation nâng cao
- **Active Learning**: Tự động labeling, continuous learning

##### 6.3.5. Deployment
- **Production Deployment**: Deploy lên production environment
- **Monitoring & Logging**: Prometheus, Grafana, CloudWatch
- **CI/CD**: Automated testing, deployment pipeline
- **Security**: Authentication, encryption, access control

**Thông tin từ hệ thống**:
- File: `docs/ROADMAP.md` - Roadmap 3 phase
- File: `docs/ARCHITECTURE.md` - Future improvements

#### 6.4. Kết luận
**Nội dung cần có**:
- Tóm tắt lại toàn bộ đề tài
- Nhấn mạnh đóng góp và kết quả
- Kết luận về tính khả thi và ứng dụng thực tế

---

## 📋 CHECKLIST CHO TỪNG THÀNH VIÊN

### ✅ Hùng (Chương 1 & Chương 6)
- [ ] Viết Chương 1: Tổng quan đề tài (2-3 trang)
- [ ] Viết Chương 6: Kết luận & Hướng phát triển (2-3 trang)
- [ ] Đọc và tham khảo: `README.md`, `docs/ARCHITECTURE.md`, `he_thong.bat`
- [ ] Tham khảo code: `src/models/cnn.py`, `src/serving/temporal_confirmation.py`
- [ ] Đọc: `docs/ROADMAP.md` - Roadmap và hướng phát triển
- [ ] Tổng hợp toàn bộ báo cáo
- [ ] Kiểm tra formatting (font, spacing, numbering)
- [ ] Kiểm tra tính nhất quán (terminology, style)
- [ ] Tạo mục lục, danh sách hình ảnh, bảng biểu
- [ ] Kiểm tra references và citations

### ✅ Phước (Chương 2)
- [ ] Viết Chương 2: Cơ sở lý thuyết & Phân tích yêu cầu (3-4 trang)
- [ ] Đọc và tham khảo: `README.md`, `docs/ARCHITECTURE.md`, `he_thong.bat`
- [ ] Tham khảo code: `src/models/cnn.py`, `src/serving/temporal_confirmation.py`
- [ ] Tham khảo: `src/models/ann.py`, `src/models/rnn.py`, `src/models/rbfnn.py`
- [ ] Kiểm tra metrics: `docs/EVALUATION_PROTOCOL.md`
- [ ] Đọc: `src/data_processing/image_processor.py` - Image preprocessing

### ✅ Nhung (Chương 3)
- [ ] Viết Chương 3: Thiết kế hệ thống (4-5 trang)
- [ ] Vẽ sơ đồ kiến trúc (có thể dùng Mermaid từ `docs/ARCHITECTURE.md`)
- [ ] Vẽ data flow diagram
- [ ] Mô tả components và modules
- [ ] Thiết kế database schema
- [ ] Đọc: `docs/ARCHITECTURE.md`, `src/database/models.py`
- [ ] Đọc: `src/database/migrations/001_initial_schema.sql` - SQL schema
- [ ] Đọc: `src/serving/temporal_confirmation.py` - Algorithm implementation

### ✅ Tài (Chương 4)
- [ ] Viết Chương 4: Hiện thực & Triển khai (4-5 trang)
- [ ] Mô tả các module chính với code snippets
- [ ] Chụp screenshots: Streamlit UI, API docs, Training interface
- [ ] Chụp result images: Prediction examples, Metrics plots
- [ ] Vẽ flowchart thực tế
- [ ] Đọc code: `src/models/cnn.py`, `src/serving/api.py`, `train_cnn.py`
- [ ] Đọc: `src/data_processing/image_processor.py` - Image processing
- [ ] Đọc: `src/training/trainer.py` - Training logic

### ✅ Đạt (Chương 5)
- [ ] Viết Chương 5: Kiểm thử & Đánh giá (3-4 trang)
- [ ] Viết test cases (functional, theo bối cảnh)
- [ ] Tính toán metrics (Accuracy, Recall, Precision, FAR)
- [ ] Đo latency, FPS
- [ ] Ghi kết quả thực nghiệm
- [ ] Phân tích hạn chế
- [ ] Chạy tests: `test_cnn_image.py`, `test_cnn_video.py`, `test_api.py`
- [ ] Đọc: `src/training/evaluator.py` - Evaluation functions
- [ ] Đọc: `docs/EVALUATION_PROTOCOL.md` - Evaluation protocol
- [ ] Đọc: `docs/BASELINE_COMPARISON.md` - Baseline comparison

---

## 📚 TÀI LIỆU THAM KHẢO QUAN TRỌNG

### Files trong dự án
1. **README.md** - Tổng quan hệ thống
2. **docs/ARCHITECTURE.md** - Kiến trúc chi tiết
3. **docs/EVALUATION_PROTOCOL.md** - Evaluation metrics
4. **docs/BASELINE_COMPARISON.md** - Model comparison
5. **docs/ROADMAP.md** - Roadmap và hướng phát triển
6. **he_thong.bat** - Menu hệ thống (chức năng)

### Source Code
1. **src/models/cnn.py** - CNN model implementation
2. **src/serving/temporal_confirmation.py** - Temporal confirmation
3. **src/serving/api.py** - FastAPI endpoints
4. **src/database/models.py** - Database schema
5. **train_cnn.py** - Training script
6. **app.py** hoặc **run_streamlit.py** - Streamlit dashboard

### Testing
1. **test_cnn_image.py** - Test với ảnh
2. **test_cnn_video.py** - Test với video
3. **test_api.py** - Test API

---

## 🎯 LƯU Ý QUAN TRỌNG

1. **Dựa trên hệ thống hiện tại**: Tất cả nội dung phải dựa trên code và hệ thống thực tế
2. **Screenshots thực tế**: Chụp màn hình từ hệ thống đang chạy
3. **Code snippets**: Copy từ source code thực tế (có thể rút gọn nhưng phải chính xác)
4. **Metrics thực tế**: Chạy tests và ghi kết quả thực tế
5. **Consistency**: Đảm bảo terminology nhất quán giữa các chương
6. **References**: Trích dẫn đúng format (theo yêu cầu của trường)

---

## 📞 HỖ TRỢ

Nếu có thắc mắc về:
- **Hệ thống**: Đọc `README.md`, `docs/HUONG_DAN_SU_DUNG.md`
- **Kiến trúc**: Đọc `docs/ARCHITECTURE.md`
- **Code**: Đọc docstrings trong source code
- **Testing**: Chạy các test scripts và xem kết quả

---

**Chúc các bạn hoàn thành báo cáo tốt! 🚀**

