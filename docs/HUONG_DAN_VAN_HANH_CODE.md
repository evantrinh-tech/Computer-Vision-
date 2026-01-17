# HƯỚNG DẪN VẬN HÀNH CODE HỆ THỐNG ITS
## Intelligent Transportation System - Phát hiện Sự cố Giao thông

**Phiên bản**: 1.0  
**Ngày cập nhật**: 15/01/2026  
**Mục đích**: Giải thích cấu trúc code, cách vận hành các file/folder trong hệ thống

---

## 📂 1. TỔNG QUAN CẤU TRÚC THỦ MỤC

```
ITS/
├── 📁 src/                      # Source code chính
│   ├── 📁 models/               # Các mô hình ML/DL
│   ├── 📁 data_processing/      # Xử lý dữ liệu
│   ├── 📁 serving/              # API và serving
│   ├── 📁 training/             # Training pipeline
│   ├── 📁 database/             # Database models
│   └── 📁 utils/                # Utilities
│
├── 📁 data/                     # Dữ liệu
│   ├── 📁 images/               # Dataset ảnh
│   ├── 📁 processed/            # Dữ liệu đã xử lý
│   └── 📁 raw/                  # Dữ liệu thô
│
├── 📁 models/                   # Mô hình đã train
│   └── 📁 CNN_model/
│       └── model.keras
│
├── 📁 docs/                     # Tài liệu
├── 📁 tests/                    # Unit tests
├── 📁 configs/                  # Configuration files
├── 📁 pipelines/                # Training pipelines
├── 📁 scripts/                  # Utility scripts
├── 📁 logs/                     # Log files
│
├── 📄 app.py                    # Streamlit dashboard (Entry point chính)
├── 📄 train_cnn.py              # Training script
├── 📄 start_api.py              # API server entry point
├── 📄 test_cnn_image.py         # Test với ảnh
├── 📄 test_cnn_video.py         # Test với video
├── 📄 test_api.py               # Test API
├── 📄 run_streamlit.py          # Helper chạy Streamlit
├── 📄 he_thong.bat              # Menu hệ thống (Windows)
└── 📄 requirements.txt          # Dependencies
```

---

## 📦 2. THƯ MỤC `src/` - SOURCE CODE CHÍNH

### 2.1. `src/models/` - Các Mô hình Machine Learning

#### 📄 `base_model.py`
**Chức năng**: Base class cho tất cả các models

**Nội dung chính**:
```python
class BaseModel:
    def __init__(self, model_type: str, config: Optional[Dict] = None)
    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None
    def train(self, X_train, y_train, X_val, y_val, **kwargs) -> Dict
    def predict(self, X: np.ndarray) -> np.ndarray
    def save(self, path: Path) -> None
    def load(self, path: Path) -> None
```

**Khi nào dùng**: Khi tạo model mới, kế thừa class này

---

#### 📄 `cnn.py` - CNN Model (QUAN TRỌNG NHẤT)
**Chức năng**: Convolutional Neural Network với Transfer Learning

**Class chính**: `CNNModel`

**Parameters quan trọng**:
- `use_transfer_learning`: True/False (mặc định True)
- `base_model`: 'MobileNetV2' | 'ResNet50' | 'VGG16'
- `image_size`: (224, 224) hoặc (128, 128)
- `learning_rate`: 0.001 (mặc định)

**Cách sử dụng**:
```python
from src.models.cnn import CNNModel

# Tạo model
model = CNNModel(
    use_transfer_learning=True,
    base_model='MobileNetV2',
    image_size=(224, 224),
    learning_rate=0.001
)

# Build model
model.build(input_shape=(224, 224, 3))

# Train
history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32
)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Save
model.save('models/CNN_model/')

# Load
model.load('models/CNN_model/model.keras')
```

**Transfer Learning Flow**:
1. Load pre-trained base model (ImageNet weights)
2. Freeze base layers
3. Add custom top layers (Dense, Dropout)
4. Train top layers
5. Unfreeze base layers và fine-tune

**Khi nào dùng**: Đây là model chính của hệ thống, dùng cho detection

---

#### 📄 `ann.py` - Artificial Neural Network
**Chức năng**: Feed-forward Neural Network đơn giản

**Class**: `ANNModel`

**Cách dùng**: Tương tự CNN, nhưng nhận input là features vector thay vì ảnh

---

#### 📄 `rnn.py` - Recurrent Neural Network
**Chức năng**: LSTM/GRU cho temporal patterns

**Class**: `RNNModel`

**Khi nào dùng**: Khi cần phân tích time-series data (ví dụ: sensor data theo thời gian)

---

#### 📄 `rbfnn.py` - Radial Basis Function Neural Network
**Chức năng**: RBFNN cho classification

**Class**: `RBFNNModel`

**Khi nào dùng**: Alternative approach, ít dùng trong project này

---

#### 📄 `segmentation.py` - U-Net Segmentation
**Chức năng**: U-Net architecture cho pixel-level segmentation

**Status**: Đã thiết kế kiến trúc, chưa implement hoàn chỉnh

**Khi nào dùng**: Khi cần phân vùng (segmentation) vùng sự cố trong ảnh

---

### 2.2. `src/data_processing/` - Xử lý Dữ liệu

#### 📄 `image_processor.py`
**Chức năng**: Xử lý ảnh (resize, normalize, augmentation)

**Class**: `ImageProcessor`

**Methods chính**:
```python
def load_image(image_path: str, target_size=(224, 224)) -> np.ndarray
def preprocess_image(img: np.ndarray) -> np.ndarray
def normalize(img: np.ndarray) -> np.ndarray
def augment(img: np.ndarray) -> np.ndarray
```

**Cách dùng**:
```python
from src.data_processing.image_processor import ImageProcessor

processor = ImageProcessor()
img = processor.load_image('path/to/image.jpg', target_size=(224, 224))
img_processed = processor.preprocess_image(img)
```

---

#### 📄 `preprocessors.py`
**Chức năng**: Preprocessing tổng quát cho data

**Functions**:
- `load_dataset(data_dir)`: Load ảnh từ thư mục
- `split_data(X, y, test_size)`: Split train/validation
- `create_data_generator()`: Tạo data generator cho training

**Cách dùng**:
```python
from src.data_processing.preprocessors import load_dataset, split_data

# Load dataset
X, y, class_names = load_dataset('data/images/')

# Split
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)
```

---

#### 📄 `collectors.py`
**Chức năng**: Thu thập dữ liệu từ nhiều nguồn

**Khi nào dùng**: Khi cần crawl/collect thêm data

---

#### 📄 `validators.py`
**Chức năng**: Validate chất lượng dữ liệu

**Functions**:
- `validate_image(img_path)`: Kiểm tra ảnh có hợp lệ
- `check_dataset_balance(y)`: Kiểm tra dataset có balanced
- `detect_duplicates(X)`: Tìm ảnh trùng lặp

---

### 2.3. `src/serving/` - API và Serving

#### 📄 `api.py` - FastAPI Server (QUAN TRỌNG)
**Chức năng**: REST API endpoints cho hệ thống

**Endpoints chính**:

1. **GET `/`**: Root endpoint
2. **GET `/health`**: Health check
3. **POST `/predict`**: Prediction endpoint
4. **GET `/metrics`**: Monitoring metrics
5. **POST `/model/reload`**: Reload model
6. **GET `/model/info`**: Model information

**Cách chạy**:
```bash
# Cách 1: Trực tiếp
python start_api.py

# Cách 2: Qua uvicorn
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Cách 3: Qua menu
he_thong.bat -> [2] Chạy API Server
```

**Swagger docs**: http://localhost:8000/docs

**Cách test**:
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"timestamp": "2024-01-15", "detector_id": "001", "volume": 100, "speed": 60, "occupancy": 0.5}]}'
```

---

#### 📄 `predictor.py` - Prediction Logic
**Chức năng**: Load model và thực hiện prediction

**Class**: `ModelPredictor`

**Methods**:
```python
def load_model(model_path: Path)
def predict(data: pd.DataFrame) -> List[Dict]
def predict_image(image_path: str) -> Dict
def is_model_loaded() -> bool
def get_model_version() -> str
```

**Cách dùng**:
```python
from src.serving.predictor import ModelPredictor

predictor = ModelPredictor()
predictor.load_model('models/CNN_model/model.keras')

result = predictor.predict_image('path/to/image.jpg')
print(result)  # {'prediction': 'incident', 'probability': 0.92}
```

---

#### 📄 `temporal_confirmation.py` - Temporal Confirmation
**Chức năng**: Giảm false alarms bằng xác nhận qua nhiều frames

**Class**: `TemporalConfirmation`

**Parameters**:
- `k_frames`: Số frames để xác nhận (mặc định 5)
- `threshold`: Threshold probability (mặc định 0.7)
- `cooldown`: Cooldown period (mặc định 30 frames)

**Algorithm**:
```python
# Pseudo-code
if moving_average(probabilities[-k_frames:]) > threshold:
    if not in_cooldown:
        trigger_incident()
        start_cooldown()
```

**Cách dùng**:
```python
from src.serving.temporal_confirmation import TemporalConfirmation

tc = TemporalConfirmation(k_frames=5, threshold=0.7)

for frame in video_frames:
    prob = model.predict(frame)
    incident = tc.process(prob)
    if incident:
        print(f"INCIDENT CONFIRMED at frame {frame_number}")
```

---

#### 📄 `monitoring.py`
**Chức năng**: System monitoring và metrics collection

**Class**: `MetricsCollector`

---

### 2.4. `src/training/` - Training Pipeline

#### 📄 `trainer.py`
**Chức năng**: Training logic và pipeline

**Class**: `Trainer`

**Methods**:
- `train_model()`: Main training function
- `setup_callbacks()`: Setup callbacks (EarlyStopping, etc.)
- `log_metrics()`: Log to MLflow

---

#### 📄 `evaluator.py`
**Chức năng**: Model evaluation

**Functions**:
- `evaluate_model(model, X_test, y_test)`: Evaluate và tính metrics
- `calculate_metrics(y_true, y_pred)`: Calculate Precision, Recall, F1
- `generate_confusion_matrix()`: Tạo confusion matrix

---

#### 📄 `visualizer.py`
**Chức năng**: Visualization cho training

**Functions**:
- `plot_training_history()`: Plot loss và accuracy curves
- `plot_confusion_matrix()`: Vẽ confusion matrix
- `plot_roc_curve()`: Vẽ ROC curve

---

### 2.5. `src/database/` - Database

#### 📄 `models.py` - SQLAlchemy Models
**Chức năng**: Database schema definitions

**Models chính**:

```python
class Incident(Base):
    __tablename__ = 'incidents'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    camera_id = Column(String)
    confidence_score = Column(Float)
    status = Column(String)  # detected, confirmed, false_alarm
    image_path = Column(String)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    prediction = Column(String)
    probability = Column(Float)
    model_version = Column(String)
```

**Cách dùng**:
```python
from src.database.models import Incident, Prediction
from src.utils.config import get_db_session

session = get_db_session()

# Create incident
incident = Incident(
    timestamp=datetime.now(),
    camera_id='CAM001',
    confidence_score=0.92,
    status='detected'
)
session.add(incident)
session.commit()

# Query
incidents = session.query(Incident).filter(
    Incident.status == 'confirmed'
).all()
```

---

### 2.6. `src/utils/` - Utilities

#### 📄 `config.py`
**Chức năng**: Configuration management

**Settings**:
```python
class Settings:
    # API settings
    api_host = "0.0.0.0"
    api_port = 8000
    api_workers = 4
    
    # Model paths
    model_dir = Path("models/")
    default_model_path = model_dir / "CNN_model" / "model.keras"
    
    # Database
    database_url = "postgresql://user:pass@localhost/traffic_db"
```

---

#### 📄 `logger.py`
**Chức năng**: Logging configuration

**Cách dùng**:
```python
from src.utils.logger import logger

logger.info("Model training started")
logger.warning("Low confidence prediction")
logger.error("Failed to load model")
```

---

## 📄 3. ROOT LEVEL FILES - ENTRY POINTS

### 📄 `app.py` - Streamlit Dashboard (⭐ ENTRY POINT CHÍNH)

**Chức năng**: Giao diện web dashboard tương tác

**Các Tab**:
1. ** Trang chủ**: Overview hệ thống
2. ** Test mô hình**: Upload ảnh/video, predict
3. ** Huấn luyện mô hình**: Training interface
4. ** Xem kết quả**: Metrics visualization
5. **🚨 Quản lý Incidents**: Incident management

**Cách chạy**:
```bash
# Cách 1
python run_streamlit.py

# Cách 2
streamlit run app.py

# Cách 3
he_thong.bat -> [1] Giao diện Web
```

**URL**: http://localhost:8501

**Cấu trúc code**:
```python
import streamlit as st
from src.models.cnn import CNNModel
from src.data_processing.preprocessors import load_dataset

st.title(" Hệ thống Phát hiện Sự cố Giao thông")

# Sidebar
page = st.sidebar.selectbox("Chọn trang", ["Trang chủ", "Test", "Huấn luyện"])

if page == "Test":
    uploaded_file = st.file_uploader("Upload ảnh")
    if uploaded_file:
        # Process and predict
        result = predict_image(uploaded_file)
        st.write(f"Kết quả: {result}")

elif page == "Huấn luyện":
    epochs = st.slider("Epochs", 10, 100, 50)
    batch_size = st.selectbox("Batch size", [16, 32, 64])
    
    if st.button("Bắt đầu huấn luyện"):
        # Training logic
        train_model(epochs, batch_size)
```

---

### 📄 `train_cnn.py` - Training Script

**Chức năng**: Script để train CNN model từ command line

**Cách chạy**:
```bash
# Basic
python train_cnn.py

# Với arguments (nếu có)
python train_cnn.py --epochs 50 --batch_size 32 --image_size 224

# Qua menu
he_thong.bat -> [3] Huấn luyện mô hình -> [1] CNN
```

**Flow**:
1. Load dataset từ `data/images/`
2. Split train/validation
3. Create CNN model
4. Train với data augmentation
5. Evaluate
6. Save model to `models/CNN_model/`
7. Log metrics to MLflow

**Output**:
- Model file: `models/CNN_model/model.keras`
- Training history: `logs/training_history.json`
- Plots: `logs/plots/`

---

### 📄 `start_api.py` - API Server Entry Point

**Chức năng**: Start FastAPI server

**Cách chạy**:
```bash
python start_api.py

# Hoặc
he_thong.bat -> [2] Chạy API Server
```

**Code**:
```python
from src.serving.api import app, main

if __name__ == '__main__':
    main()  # Load model và start server
```

---

### 📄 `test_cnn_image.py` - Test với Ảnh

**Chức năng**: Test model với single image

**Cách chạy**:
```bash
python test_cnn_image.py path/to/image.jpg

# Hoặc
he_thong.bat -> [4] Test mô hình -> [1] Test với ảnh
```

**Output**:
```
Loading model from models/CNN_model/model.keras...
Processing image: path/to/image.jpg

Results:
  Prediction: INCIDENT
  Probability: 0.92
  Confidence: 92%
```

---

### 📄 `test_cnn_video.py` - Test với Video

**Chức năng**: Test model với video file

**Cách chạy**:
```bash
python test_cnn_video.py path/to/video.mp4

# Hoặc
he_thong.bat -> [4] Test mô hình -> [2] Test với video
```

**Flow**:
1. Load video
2. Extract frames
3. Predict each frame
4. Apply temporal confirmation
5. Generate incident timeline

**Output**:
```
Processing video: path/to/video.mp4
Total frames: 300
Processing... [=====>] 100%

Results:
  Total frames: 300
  Incidents detected: 5
  False alarm rate: 8%
  
Incident Timeline:
  Frame 45-52: INCIDENT (prob=0.92)
  Frame 130-138: INCIDENT (prob=0.87)
  ...
```

---

### 📄 `test_api.py` - Test API Endpoints

**Chức năng**: Test tất cả API endpoints

**Cách chạy**:
```bash
python test_api.py

# Hoặc
he_thong.bat -> [4] Test mô hình -> [4] Test API
```

**Tests**:
- Health check endpoint
- Predict endpoint
- Model info endpoint

---

### 📄 `run_streamlit.py` - Streamlit Helper

**Chức năng**: Helper script để chạy Streamlit

**Code**:
```python
import os
os.system('streamlit run app.py')
```

---

### 📄 `he_thong.bat` - Menu Hệ thống (Windows)

**Chức năng**: Menu tương tác để quản lý hệ thống

**Options**:
1. Giao diện Web (Streamlit)
2. Chạy API Server
3. Huấn luyện mô hình
4. Test mô hình
5. Kiểm tra trạng thái
6. Tạo Virtual Environment
7. Setup Database
8. Dọn dẹp hệ thống
9. Quick Start

**Cách dùng**:
```bash
he_thong.bat
```

---

##  4. DATA ORGANIZATION

### 4.1. `data/images/` - Dataset

**Cấu trúc**:
```
data/images/
├── normal/          # Ảnh giao thông bình thường
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── incident/        # Ảnh có sự cố
    ├── img001.jpg
    ├── img002.jpg
    └── ...
```

**Yêu cầu**:
- Tối thiểu 50 ảnh/class
- Khuyến nghị 200+ ảnh/class
- Format: `.jpg`, `.jpeg`, `.png`

**Cách add thêm data**:
1. Copy ảnh vào thư mục tương ứng (`normal/` hoặc `incident/`)
2. Chạy lại training

---

### 4.2. `models/CNN_model/` - Trained Models

**Files**:
- `model.keras`: Model đã train (Keras format)
- `weights.h5`: Model weights (optional)
- `training_history.json`: Training history

**Cách load model**:
```python
from tensorflow import keras

model = keras.models.load_model('models/CNN_model/model.keras')
```

---

## 🔄 5. WORKFLOW THÔNG DỤNG

### 5.1. Workflow Training Model Mới

```bash
# 1. Chuẩn bị data
# Copy ảnh vào data/images/normal/ và data/images/incident/

# 2. Train model
python train_cnn.py
# Hoặc qua Streamlit: app.py -> Tab "Huấn luyện"

# 3. Model sẽ được lưu tại models/CNN_model/model.keras

# 4. Test model
python test_cnn_image.py data/images/incident/test.jpg

# 5. Deploy model (start API)
python start_api.py
```

---

### 5.2. Workflow Test Hệ thống

```bash
# 1. Start Streamlit dashboard
python run_streamlit.py

# 2. Mở browser: http://localhost:8501

# 3. Upload ảnh test trong tab "Test mô hình"

# 4. Xem kết quả prediction
```

---

### 5.3. Workflow Sử dụng API

```bash
# 1. Start API server
python start_api.py

# 2. Test health
curl http://localhost:8000/health

# 3. Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @request.json

# 4. Xem Swagger docs
# Browser: http://localhost:8000/docs
```

---

### 5.4. Workflow Development

```bash
# 1. Activate virtual environment
venv311\Scripts\activate  # Windows
source venv311/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make changes to code

# 4. Test
pytest tests/

# 5. Run application
python app.py  # hoặc python start_api.py
```

---

## 🛠️ 6. CONFIGURATION FILES

### 📄 `requirements.txt`
**Chức năng**: Python dependencies

**Cài đặt**:
```bash
pip install -r requirements.txt
```

**Main dependencies**:
- tensorflow
- streamlit
- fastapi
- opencv-python
- sqlalchemy
- mlflow

---

### 📄 `configs/training_config.yaml`
**Chức năng**: Training configuration

**Content**:
```yaml
training:
  epochs: 50
  batch_size: 32
  image_size: [224, 224]
  learning_rate: 0.001
  base_model: "MobileNetV2"
  validation_split: 0.2
```

**Cách dùng**:
```python
import yaml

with open('configs/training_config.yaml') as f:
    config = yaml.safe_load(f)
    
epochs = config['training']['epochs']
```

---

##  7. IMPORT PATHS - QUAN TRỌNG

### Correct Import Examples

```python
# Models
from src.models.cnn import CNNModel
from src.models.ann import ANNModel
from src.models.segmentation import UNetSegmentation

# Data Processing
from src.data_processing.image_processor import ImageProcessor
from src.data_processing.preprocessors import load_dataset, split_data

# Serving
from src.serving.predictor import ModelPredictor
from src.serving.api import app
from src.serving.temporal_confirmation import TemporalConfirmation

# Training
from src.training.trainer import Trainer
from src.training.evaluator import evaluate_model

# Utils
from src.utils.config import settings
from src.utils.logger import logger

# Database
from src.database.models import Incident, Prediction
```

### PYTHONPATH Setup

**Windows**:
```bash
set PYTHONPATH=%PYTHONPATH%;C:\path\to\ITS
```

**Linux/Mac**:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ITS
```

**Hoặc dùng script**:
```bash
# Windows
set_pythonpath.bat

# Linux/Mac
source set_pythonpath.sh
```

---

## 📖 8. CODE EXAMPLES

### 8.1. Train Model từ Code

```python
from src.models.cnn import CNNModel
from src.data_processing.preprocessors import load_dataset, split_data

# 1. Load dataset
X, y, class_names = load_dataset('data/images/')
print(f"Loaded {len(X)} images")
print(f"Classes: {class_names}")

# 2. Split data
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2)

# 3. Create model
model = CNNModel(
    use_transfer_learning=True,
    base_model='MobileNetV2',
    image_size=(224, 224),
    learning_rate=0.001
)

# 4. Train
history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=32,
    verbose=1
)

# 5. Save
model.save('models/CNN_model/')
print("Model saved!")
```

---

### 8.2. Predict từ Code

```python
from src.models.cnn import CNNModel
from src.data_processing.image_processor import ImageProcessor

# 1. Load model
model = CNNModel()
model.load('models/CNN_model/model.keras')

# 2. Load và preprocess image
processor = ImageProcessor()
img = processor.load_image('path/to/image.jpg', target_size=(224, 224))
img = processor.preprocess_image(img)
img = img.reshape(1, 224, 224, 3)  # Add batch dimension

# 3. Predict
prediction = model.predict(img)[0]
probability = model.predict_proba(img)[0]

# 4. Interpret
class_names = ['normal', 'incident']
result = class_names[prediction]
print(f"Prediction: {result}")
print(f"Probability: {probability:.2%}")
```

---

### 8.3. Process Video

```python
import cv2
from src.models.cnn import CNNModel
from src.serving.temporal_confirmation import TemporalConfirmation

# 1. Load model
model = CNNModel()
model.load('models/CNN_model/model.keras')

# 2. Setup temporal confirmation
tc = TemporalConfirmation(k_frames=5, threshold=0.7)

# 3. Open video
cap = cv2.VideoCapture('path/to/video.mp4')

frame_number = 0
incidents = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    frame_batch = frame_normalized.reshape(1, 224, 224, 3)
    
    # Predict
    prob = model.predict_proba(frame_batch)[0]
    
    # Temporal confirmation
    incident = tc.process(prob)
    
    if incident:
        incidents.append({
            'frame': frame_number,
            'probability': prob
        })
        print(f"INCIDENT at frame {frame_number}")
    
    frame_number += 1

cap.release()

print(f"\nTotal incidents: {len(incidents)}")
```

---

## 🐛 9. DEBUGGING & TROUBLESHOOTING

### 9.1. Model không load được

**Lỗi**: `FileNotFoundError: Không tìm thấy model`

**Giải pháp**:
```bash
# Kiểm tra file model có tồn tại
ls models/CNN_model/model.keras

# Nếu không có, train lại
python train_cnn.py
```

---

### 9.2. Import Error

**Lỗi**: `ModuleNotFoundError: No module named 'src'`

**Giải pháp**:
```bash
# Set PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;C:\path\to\ITS  # Windows
export PYTHONPATH=$PYTHONPATH:/path/to/ITS   # Linux

# Hoặc chạy từ root directory
cd ITS/
python -m src.models.cnn
```

---

### 9.3. TensorFlow lỗi

**Lỗi**: `Could not load dynamic library 'cudart64_110.dll'`

**Giải pháp**: Đang dùng TensorFlow-GPU nhưng không có CUDA. Cài TensorFlow CPU:
```bash
pip uninstall tensorflow-gpu
pip install tensorflow
```

---

### 9.4. Streamlit không chạy

**Lỗi**: `streamlit: command not found`

**Giải pháp**:
```bash
# Cài lại streamlit
pip install --upgrade streamlit

# Hoặc chạy trực tiếp
python -m streamlit run app.py
```

---

## 📚 10. BEST PRACTICES

### 10.1. Code Organization

 **NÊN**:
- Đặt code trong `src/` theo modules
- Sử dụng absolute imports (`from src.models.cnn import ...`)
- Docstrings cho tất cả functions/classes
- Type hints cho parameters

❌ **KHÔNG NÊN**:
- Relative imports (`from ..models import`)
- Hardcode paths
- Code trực tiếp trong root files

---

### 10.2. Training

 **NÊN**:
- Save model sau mỗi epoch tốt nhất (ModelCheckpoint)
- Log metrics vào MLflow
- Validate trên validation set
- Early stopping để tránh overfit

❌ **KHÔNG NÊN**:
- Train quá nhiều epochs mà không EarlyStopping
- Quên validate
- Train trên toàn bộ dataset (không split)

---

### 10.3. Testing

 **NÊN**:
- Test model trên test set riêng
- Sử dụng confusion matrix
- Calculate nhiều metrics (Precision, Recall, F1)

❌ **KHÔNG NÊN**:
- Test trên training data
- Chỉ nhìn accuracy

---

## 📞 11. HỖ TRỢ

### Tài liệu tham khảo
- `README.md` - Overview hệ thống
- `BAO_CAO_TIEN_DO_HE_THONG.md` - Báo cáo chi tiết
- `docs/ARCHITECTURE.md` - Kiến trúc
- `docs/HUONG_DAN_SU_DUNG.md` - Hướng dẫn sử dụng

### Code documentation
- Docstrings trong mỗi file `.py`
- Comments inline cho logic phức tạp
- Type hints

---

**Chúc bạn code hiệu quả! **

---

*File tạo ngày: 15/01/2026*  
*Version: 1.0*  
*Cập nhật: Khi có thay đổi lớn trong codebase*
