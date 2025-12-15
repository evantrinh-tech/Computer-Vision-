# 📚 HƯỚNG DẪN SỬ DỤNG CÁC TÀI LIỆU VÀ MODULE MỚI

## 📋 TỔNG QUAN

Tài liệu này hướng dẫn cách sử dụng các tài liệu và module mới đã được tạo để hoàn thiện dự án phát hiện sự cố giao thông.

---

## 📁 CẤU TRÚC TÀI LIỆU

```
docs/
├── ROADMAP.md                    # Roadmap 3 phase (MVP → Hybrid → Production)
├── EVALUATION_PROTOCOL.md         # Protocol đánh giá (split, threshold, MTTD)
├── BASELINE_COMPARISON.md         # Baseline & Model Comparison (tách rõ Vision/Sensor/Hybrid)
├── ARCHITECTURE.md                # Kiến trúc hệ thống (diagram + giải thích)
├── BAO_CAO_CUOI.md                # Outline báo cáo cuối (10-15 trang)
└── HUONG_DAN_SU_DUNG.md           # File này

src/
├── serving/
│   └── temporal_confirmation.py   # Module temporal confirmation
└── database/
    ├── models.py                   # SQLAlchemy models
    └── migrations/
        └── 001_initial_schema.sql # Migration script
```

---

## 🚀 CÁCH SỬ DỤNG

### 1. Roadmap (docs/ROADMAP.md)

**Mục đích**: Kế hoạch phát triển 3 phase với metrics và task breakdown

**Cách sử dụng**:
1. Đọc để hiểu roadmap tổng thể
2. Theo dõi tiến độ theo từng phase
3. Điều chỉnh task nếu cần

**Ví dụ**:
```bash
# Xem roadmap
cat docs/ROADMAP.md
```

---

### 2. Evaluation Protocol (docs/EVALUATION_PROTOCOL.md)

**Mục đích**: Protocol chuẩn để đánh giá model (tránh data leakage, threshold tuning)

**Cách sử dụng**:
1. **Chia dữ liệu**: Sử dụng code trong section 1 để chia train/val/test
2. **Tune threshold**: Sử dụng function `tune_threshold_on_validation()` trong section 2
3. **Tính MTTD**: Sử dụng function `calculate_mttd()` trong section 3
4. **Vẽ biểu đồ**: Sử dụng function `generate_all_evaluation_plots()` trong section 4

**Ví dụ**:
```python
from docs.EVALUATION_PROTOCOL import tune_threshold_on_validation

# Tune threshold trên validation
best_params = tune_threshold_on_validation(
    y_val_proba=y_val_proba,
    y_val_true=y_val_true,
    target_recall=0.9,
    target_far=0.01
)
print(f"Best threshold: {best_params['threshold']}")
```

---

### 3. Baseline Comparison (docs/BASELINE_COMPARISON.md)

**Mục đích**: Tài liệu so sánh baseline, tách rõ Vision/Sensor/Hybrid

**Cách sử dụng**:
1. Đọc để hiểu cách so sánh công bằng
2. Sử dụng trong báo cáo để tránh lỗi "so sánh khác loại dữ liệu"
3. Tham khảo bảng so sánh để chọn model phù hợp

**Lưu ý**: 
- ✅ So sánh Vision models với nhau
- ✅ So sánh Sensor models với nhau
- ❌ KHÔNG so sánh Vision với Sensor (khác loại dữ liệu)

---

### 4. Temporal Confirmation Module (src/serving/temporal_confirmation.py)

**Mục đích**: Module giảm false alarm bằng cách xác nhận theo thời gian

**Cách sử dụng**:

#### 4.1. Basic Usage

```python
from src.serving.temporal_confirmation import TemporalConfirmation, IncidentStatus

# Khởi tạo
confirmer = TemporalConfirmation(
    k_frames=5,              # Cần 5 frames liên tiếp
    window_size=10,          # Window size cho moving average
    threshold=0.5,          # Threshold probability
    cooldown_seconds=30.0,   # Cooldown 30 giây
    fps=30.0                # FPS của video
)

# Xử lý stream probabilities
probabilities = [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5]
events = confirmer.process_stream(probabilities)

# Xem kết quả
for event in events:
    print(f"Event: {event.status}, Frame: {event.start_frame}, Prob: {event.avg_probability:.3f}")
```

#### 4.2. Frame-by-frame Processing

```python
# Xử lý từng frame
for frame_num, prob in enumerate(probabilities):
    event = confirmer.process_frame(frame_num, prob)
    if event is not None:
        print(f"Incident confirmed at frame {frame_num}!")
```

#### 4.3. Tune Parameters

```python
from src.serving.temporal_confirmation import tune_temporal_params

# Tune trên validation set
best_params = tune_temporal_params(
    probabilities=y_val_proba,
    ground_truth=y_val_true,
    fps=30.0,
    k_range=(3, 10),
    window_range=(5, 20),
    threshold_range=(0.3, 0.7),
    cooldown_range=(10.0, 60.0)
)

print(f"Best params: {best_params}")
```

#### 4.4. Integrate với Video Processing

```python
from src.data_processing.image_processor import VideoProcessor
from src.models.cnn import CNNModel

# Load model
model = CNNModel()
model.load("models/CNN_model/model.keras")

# Process video
video_processor = VideoProcessor()
confirmer = TemporalConfirmation(k_frames=5, threshold=0.5)

cap = cv2.VideoCapture("video.mp4")
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess và predict
    processed = video_processor.preprocess_image(frame)
    prob = model.predict_proba(processed.reshape(1, 224, 224, 3))[0]
    
    # Temporal confirmation
    event = confirmer.process_frame(frame_num, prob)
    if event is not None:
        print(f"Incident confirmed at frame {frame_num}!")
    
    frame_num += 1
```

---

### 5. Database Schema (src/database/models.py)

**Mục đích**: SQLAlchemy models cho PostgreSQL

**Cách sử dụng**:

#### 5.1. Setup Database

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, Incident

# Tạo engine
engine = create_engine("postgresql://user:password@localhost:5432/traffic_db")

# Tạo tables
Base.metadata.create_all(engine)

# Tạo session
Session = sessionmaker(bind=engine)
session = Session()
```

#### 5.2. Tạo Incident

```python
from src.database.models import Incident
from datetime import datetime

# Tạo incident mới
incident = Incident(
    timestamp=datetime.now(),
    camera_id="camera_001",
    location="Highway 1, km 50",
    latitude=10.762622,
    longitude=106.660172,
    incident_type="accident",
    severity="high",
    confidence_score=0.87,
    model_version="CNN_v1.0",
    threshold=0.5,
    rule_version="temporal_v1.0",
    confirmation_method="k_frames",
    status="confirmed",
    image_path="data/incidents/2024/01/incident_001.jpg",
    media_storage_type="local",
    latency_ms=450.0,
    processing_time_ms=200.0
)

session.add(incident)
session.commit()
```

#### 5.3. Query Incidents

```python
# Lấy incidents theo camera
incidents = session.query(Incident).filter(
    Incident.camera_id == "camera_001",
    Incident.status == "confirmed"
).all()

# Lấy incidents trong khoảng thời gian
from datetime import datetime, timedelta
start_time = datetime.now() - timedelta(days=7)
incidents = session.query(Incident).filter(
    Incident.timestamp >= start_time
).order_by(Incident.timestamp.desc()).all()
```

#### 5.4. Migration

```bash
# Chạy migration script
psql -U user -d traffic_db -f src/database/migrations/001_initial_schema.sql
```

---

### 6. Architecture Diagram (docs/ARCHITECTURE.md)

**Mục đích**: Mô tả kiến trúc hệ thống

**Cách sử dụng**:
1. Đọc để hiểu kiến trúc tổng thể
2. Sử dụng trong báo cáo
3. Tham khảo khi thiết kế features mới

---

### 7. Báo cáo Cuối (docs/BAO_CAO_CUOI.md)

**Mục đích**: Outline báo cáo cuối (10-15 trang)

**Cách sử dụng**:
1. Điền thông tin vào các section
2. Thêm biểu đồ và kết quả
3. Format theo yêu cầu của trường/khoa

---

## 🔧 TÍCH HỢP VÀO DỰ ÁN

### 1. Integrate Temporal Confirmation vào API

```python
# src/serving/api.py
from src.serving.temporal_confirmation import TemporalConfirmation

# Trong prediction endpoint
confirmer = TemporalConfirmation(k_frames=5, threshold=0.5)

# Sau khi predict
prob = model.predict_proba(image)
event = confirmer.process_frame(frame_num, prob)

if event and event.status == IncidentStatus.CONFIRMED:
    # Tạo incident record
    incident = create_incident(event, image_path)
```

### 2. Integrate Database vào Training

```python
# src/training/trainer.py
from src.database.models import ModelRun

# Sau khi train xong
model_run = ModelRun(
    model_name="CNN",
    model_version="v1.0",
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
    status="completed"
)
session.add(model_run)
session.commit()
```

---

## 📝 CHECKLIST SỬ DỤNG

- [ ] Đã đọc và hiểu roadmap
- [ ] Đã implement evaluation protocol
- [ ] Đã sử dụng temporal confirmation
- [ ] Đã setup database schema
- [ ] Đã tích hợp vào code hiện tại
- [ ] Đã viết báo cáo cuối

---

## 🆘 HỖ TRỢ

Nếu có vấn đề, xem:
- Code examples trong các file
- Comments trong source code
- Tài liệu tham khảo trong docs/

---

*Cập nhật lần cuối: [Ngày hiện tại]*

