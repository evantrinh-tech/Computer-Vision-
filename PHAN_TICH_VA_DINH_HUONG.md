# 📊 PHÂN TÍCH VÀ ĐỊNH HƯỚNG PHÁT TRIỂN HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG

## 📋 MỤC LỤC

1. [Xác định Baseline và So sánh các Mô hình](#1-xác-định-baseline-và-so-sánh-các-mô-hình)
2. [Testing và Validating với Diagram và Phân tích](#2-testing-và-validating-với-diagram-và-phân-tích)
3. [Phát triển Features: What For, Where Store, Who Use](#3-phát-triển-features-what-for-where-store-who-use)

---

## 1. XÁC ĐỊNH BASELINE VÀ SO SÁNH CÁC MÔ HÌNH

### 1.1. Xác định Baseline: CNN Model

**CNN (Convolutional Neural Network) được chọn làm Baseline** vì các lý do sau:

#### ✅ Lý do chọn CNN làm Baseline:

1. **Phù hợp với dữ liệu ảnh**: CNN được thiết kế đặc biệt để xử lý dữ liệu hình ảnh, có khả năng tự động trích xuất các đặc trưng (features) từ ảnh mà không cần feature engineering thủ công.

2. **Transfer Learning**: CNN model trong hệ thống sử dụng Transfer Learning với các pre-trained models như:
   - **MobileNetV2** (mặc định): Nhẹ, nhanh, phù hợp cho real-time
   - **ResNet50**: Độ chính xác cao hơn
   - **VGG16**: Kiến trúc đơn giản, dễ hiểu

3. **Kiến trúc Baseline CNN**:
   ```python
   # Kiến trúc CNN Baseline (từ src/models/cnn.py)
   - Input: Ảnh 224x224x3 (RGB)
   - Base Model: MobileNetV2 (pre-trained trên ImageNet)
   - Global Average Pooling
   - Dropout (0.2)
   - Dense Layer (128 neurons, ReLU)
   - Dropout (0.2)
   - Output: Dense(1, sigmoid) - Binary Classification
   ```

4. **Metrics Baseline**:
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam (learning_rate=0.001)
   - Metrics: Accuracy, Precision, Recall
   - Data Augmentation: Rotation, Shift, Flip, Zoom

### 1.2. So sánh các Mô hình với Baseline (CNN)

Hệ thống hiện tại có **4 mô hình** để so sánh:

| Mô hình | Loại dữ liệu | Kiến trúc | Ưu điểm | Nhược điểm | So với CNN |
|---------|--------------|-----------|---------|------------|------------|
| **CNN** (Baseline) | Ảnh | Transfer Learning (MobileNetV2) | ✅ Tối ưu cho ảnh<br>✅ Transfer Learning<br>✅ Tự động feature extraction | ❌ Cần GPU để train nhanh<br>❌ Yêu cầu nhiều dữ liệu ảnh | **Baseline** |
| **ANN** | Sensor data (mô phỏng) | Feed-forward Neural Network | ✅ Đơn giản, nhanh<br>✅ Phù hợp dữ liệu số | ❌ Không xử lý được ảnh<br>❌ Cần feature engineering | Khác loại dữ liệu |
| **RNN** (LSTM/GRU) | Time-series sensor data | LSTM/GRU layers | ✅ Nắm bắt temporal patterns<br>✅ Phù hợp dữ liệu chuỗi thời gian | ❌ Chậm hơn ANN<br>❌ Không xử lý được ảnh | Khác loại dữ liệu |
| **RBFNN** | Sensor data (mô phỏng) | Radial Basis Function + Wavelet | ✅ Xử lý non-linear tốt<br>✅ Wavelet transform | ❌ Phức tạp hơn<br>❌ Không xử lý được ảnh | Khác loại dữ liệu |

### 1.3. Phân tích Chi tiết So sánh

#### 🔍 **CNN vs ANN**

**CNN (Baseline)**:
- **Input**: Ảnh (224x224x3)
- **Architecture**: Convolutional layers → Feature extraction tự động
- **Use case**: Phát hiện sự cố từ camera/ảnh
- **Performance**: Tối ưu cho computer vision tasks

**ANN**:
- **Input**: Sensor data (volume, speed, occupancy, ...)
- **Architecture**: Dense layers (64 → 32 → 1)
- **Use case**: Phát hiện sự cố từ cảm biến giao thông
- **Performance**: Nhanh nhưng cần feature engineering

**Kết luận**: CNN và ANN xử lý **khác loại dữ liệu** nên không thể so sánh trực tiếp. CNN là baseline cho **image-based detection**, ANN là baseline cho **sensor-based detection**.

#### 🔍 **CNN vs RNN**

**RNN (LSTM/GRU)**:
- **Input**: Time-series sensor data (chuỗi thời gian)
- **Architecture**: LSTM/GRU layers để nắm bắt temporal dependencies
- **Use case**: Phát hiện sự cố dựa trên pattern theo thời gian
- **Performance**: Tốt cho dữ liệu có tính tuần tự

**Kết luận**: RNN bổ sung cho CNN bằng cách xử lý **temporal patterns** trong sensor data, trong khi CNN xử lý **spatial patterns** trong ảnh.

#### 🔍 **CNN vs RBFNN**

**RBFNN**:
- **Input**: Sensor data với Wavelet transform
- **Architecture**: Radial Basis Function + Wavelet decomposition
- **Use case**: Phát hiện sự cố với non-linear patterns phức tạp
- **Performance**: Tốt cho dữ liệu có nhiễu và patterns phức tạp

**Kết luận**: RBFNN là một approach khác cho sensor data, sử dụng wavelet để xử lý tín hiệu tốt hơn.

### 1.4. Kết luận về Baseline

**CNN là Baseline chính** cho hệ thống vì:
1. ✅ Xử lý dữ liệu ảnh - nguồn dữ liệu chính của hệ thống
2. ✅ Sử dụng Transfer Learning - tận dụng kiến thức từ ImageNet
3. ✅ Tự động feature extraction - không cần feature engineering thủ công
4. ✅ Hiệu suất tốt với dữ liệu ảnh

**Các mô hình khác (ANN, RNN, RBFNN)** là **bổ sung** cho CNN, xử lý các loại dữ liệu khác (sensor data) để tạo hệ thống **hybrid detection** hoàn chỉnh.

---

## 2. TESTING VÀ VALIDATING VỚI DIAGRAM VÀ PHÂN TÍCH

### 2.1. Các Metrics được Đánh giá

Hệ thống sử dụng **ModelEvaluator** (`src/training/evaluator.py`) để tính toán các metrics sau:

#### 📊 **Primary Metrics**:

1. **Accuracy** (Độ chính xác)
   - Công thức: `(TP + TN) / (TP + TN + FP + FN)`
   - Ý nghĩa: Tỷ lệ dự đoán đúng tổng thể

2. **Precision** (Độ chính xác dự đoán)
   - Công thức: `TP / (TP + FP)`
   - Ý nghĩa: Trong số các dự đoán "có sự cố", bao nhiêu là đúng

3. **Recall** (Độ nhạy / Detection Rate)
   - Công thức: `TP / (TP + FN)`
   - Ý nghĩa: Trong số các sự cố thực tế, bao nhiêu được phát hiện

4. **F1-Score** (Harmonic Mean)
   - Công thức: `2 * (Precision * Recall) / (Precision + Recall)`
   - Ý nghĩa: Cân bằng giữa Precision và Recall

#### 📊 **Secondary Metrics**:

5. **ROC-AUC Score**
   - Ý nghĩa: Khả năng phân biệt giữa "có sự cố" và "không có sự cố"

6. **False Alarm Rate** (Tỷ lệ báo động sai)
   - Công thức: `FP / (FP + TN)`
   - Ý nghĩa: Tỷ lệ báo động sai trong các trường hợp bình thường

7. **Mean Time To Detection (MTTD)**
   - Ý nghĩa: Thời gian trung bình để phát hiện sự cố sau khi xảy ra

8. **Confusion Matrix**
   - Bao gồm: TP, TN, FP, FN

### 2.2. Diagram và Visualization

#### 📈 **1. Training History Diagrams**

**Loss Curve (Training & Validation)**:
```
Epoch → Loss
- Training Loss: Giảm dần theo epochs
- Validation Loss: Giảm dần, có thể tăng nếu overfitting
- Early Stopping: Dừng khi validation loss không cải thiện
```

**Accuracy Curve (Training & Validation)**:
```
Epoch → Accuracy
- Training Accuracy: Tăng dần
- Validation Accuracy: Tăng dần, có thể giảm nếu overfitting
- Gap giữa train và val: Chỉ số overfitting
```

**Precision & Recall Curves**:
```
Epoch → Metric Value
- Precision: Tăng dần
- Recall: Tăng dần
- Cân bằng giữa Precision và Recall quan trọng
```

#### 📊 **2. Model Comparison Diagrams**

**Bar Chart - Metrics Comparison**:
```
Model → Metric Value
- So sánh Accuracy, Precision, Recall, F1-Score giữa các models
- CNN (Baseline) vs các models khác
```

**Confusion Matrix Heatmap**:
```
        Predicted
        No    Yes
Actual No  [TN]  [FP]
      Yes  [FN]  [TP]
```

**ROC Curve**:
```
True Positive Rate vs False Positive Rate
- AUC Score càng cao càng tốt (0.5 = random, 1.0 = perfect)
```

#### 📉 **3. Performance Metrics Diagrams**

**F1-Score Comparison**:
```
Model → F1-Score
- F1-Score cao = Cân bằng tốt giữa Precision và Recall
```

**False Alarm Rate**:
```
Model → False Alarm Rate
- Tỷ lệ báo động sai càng thấp càng tốt
```

**Prediction Time**:
```
Model → Average Prediction Time (ms)
- Quan trọng cho real-time applications
```

### 2.3. Phân tích và Giải thích (Reasoning)

#### 🔬 **Phân tích Loss Curve**

**Kịch bản 1: Training Loss giảm, Validation Loss giảm**
- ✅ **Tốt**: Model đang học tốt, không overfitting
- **Reasoning**: Model học được patterns chung, không chỉ ghi nhớ training data

**Kịch bản 2: Training Loss giảm, Validation Loss tăng**
- ❌ **Overfitting**: Model ghi nhớ training data quá mức
- **Reasoning**: Model quá phức tạp hoặc training data quá ít
- **Giải pháp**: 
  - Tăng Dropout rate
  - Data Augmentation
  - Early Stopping
  - Giảm model complexity

**Kịch bản 3: Cả hai Loss đều không giảm**
- ❌ **Underfitting**: Model quá đơn giản
- **Reasoning**: Model không đủ khả năng học patterns
- **Giải pháp**:
  - Tăng model complexity
  - Tăng số epochs
  - Tăng learning rate (cẩn thận)
  - Feature engineering tốt hơn

#### 🔬 **Phân tích Accuracy vs Precision vs Recall**

**High Accuracy, Low Precision, Low Recall**:
- **Tình huống**: Dataset mất cân bằng (imbalanced)
- **Reasoning**: Model dự đoán đa số là class chiếm ưu thế
- **Ví dụ**: 90% normal, 10% incident → Model luôn dự đoán "normal" → Accuracy cao nhưng không phát hiện được incident
- **Giải pháp**: 
  - Class weights
  - SMOTE oversampling
  - Focal Loss

**High Precision, Low Recall**:
- **Tình huống**: Model thận trọng, chỉ dự đoán "có sự cố" khi rất chắc chắn
- **Reasoning**: Ít False Positives nhưng bỏ sót nhiều sự cố thực tế
- **Ứng dụng**: Khi False Alarm tốn kém (ví dụ: gọi cảnh sát)
- **Giải pháp**: Giảm threshold (từ 0.5 xuống 0.3-0.4)

**Low Precision, High Recall**:
- **Tình huống**: Model nhạy cảm, dự đoán "có sự cố" nhiều
- **Reasoning**: Phát hiện được nhiều sự cố nhưng có nhiều False Positives
- **Ứng dụng**: Khi bỏ sót sự cố nguy hiểm hơn False Alarm
- **Giải pháp**: Tăng threshold (từ 0.5 lên 0.6-0.7)

**High Precision, High Recall (High F1-Score)**:
- ✅ **Lý tưởng**: Cân bằng tốt
- **Reasoning**: Model vừa chính xác vừa nhạy cảm
- **Đạt được bằng**: 
  - Đủ dữ liệu training
  - Model architecture phù hợp
  - Hyperparameter tuning tốt

#### 🔬 **Phân tích Confusion Matrix**

**High TP, Low FP, Low FN**:
- ✅ **Tốt**: Phát hiện được nhiều sự cố, ít báo động sai, ít bỏ sót

**High FP (False Positives)**:
- **Vấn đề**: Nhiều báo động sai
- **Nguyên nhân**: 
  - Model quá nhạy cảm
  - Threshold quá thấp
  - Training data có nhiều noise
- **Giải pháp**: Tăng threshold, cải thiện data quality

**High FN (False Negatives)**:
- **Vấn đề**: Bỏ sót nhiều sự cố
- **Nguyên nhân**:
  - Model không đủ nhạy cảm
  - Threshold quá cao
  - Thiếu dữ liệu training cho một số loại sự cố
- **Giải pháp**: Giảm threshold, thu thập thêm dữ liệu

#### 🔬 **Phân tích ROC-AUC Score**

**AUC = 0.5**:
- ❌ **Random**: Model không tốt hơn đoán ngẫu nhiên
- **Reasoning**: Model không học được gì từ dữ liệu

**0.5 < AUC < 0.7**:
- ⚠️ **Yếu**: Model có khả năng phân biệt nhưng chưa tốt
- **Reasoning**: Cần cải thiện model hoặc features

**0.7 ≤ AUC < 0.9**:
- ✅ **Tốt**: Model có khả năng phân biệt tốt
- **Reasoning**: Model học được patterns hữu ích

**AUC ≥ 0.9**:
- ✅✅ **Rất tốt**: Model phân biệt rất tốt
- **Reasoning**: Model học được patterns rõ ràng và nhất quán

#### 🔬 **Phân tích False Alarm Rate**

**False Alarm Rate cao**:
- **Vấn đề**: Nhiều báo động sai → Tốn tài nguyên, mất niềm tin
- **Reasoning**: 
  - Model quá nhạy cảm
  - Training data có nhiều edge cases
  - Thiếu dữ liệu "normal" đa dạng
- **Giải pháp**: 
  - Tăng threshold
  - Thu thập thêm dữ liệu "normal" đa dạng
  - Post-processing: Cần nhiều frames liên tiếp để xác nhận

**False Alarm Rate thấp**:
- ✅ **Tốt**: Ít báo động sai
- **Lưu ý**: Đảm bảo không hy sinh Recall

### 2.4. Validation Strategy

#### 📋 **1. Train/Validation/Test Split**

```
Tổng dữ liệu (100%)
├── Training Set (70%): Huấn luyện model
├── Validation Set (10%): Tune hyperparameters, early stopping
└── Test Set (20%): Đánh giá cuối cùng, không được sử dụng trong training
```

**Lý do chia như vậy**:
- **Training (70%)**: Đủ lớn để model học được patterns
- **Validation (10%)**: Đủ để đánh giá performance mà không lãng phí dữ liệu
- **Test (20%)**: Đủ để đánh giá chính xác, đại diện cho real-world performance

#### 📋 **2. Cross-Validation (K-Fold)**

**Khi nào sử dụng**:
- Dataset nhỏ (< 1000 samples)
- Cần đánh giá chính xác hơn

**Cách thực hiện**:
```
K = 5 folds
Fold 1: [Train] [Val] [Train] [Train] [Train]
Fold 2: [Train] [Train] [Val] [Train] [Train]
...
Fold 5: [Train] [Train] [Train] [Train] [Val]
→ Lấy trung bình metrics từ 5 folds
```

#### 📋 **3. Stratified Split**

**Quan trọng cho imbalanced dataset**:
- Đảm bảo tỷ lệ "normal" và "incident" giống nhau trong train/val/test
- Tránh trường hợp test set chỉ có "normal" → Accuracy cao giả tạo

### 2.5. Code để Tạo Diagrams

**Ví dụ code để visualize metrics** (có thể thêm vào `src/training/evaluator.py`):

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_training_history(history):
    """Vẽ loss và accuracy curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history['precision'], label='Train Precision')
    axes[1, 0].plot(history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Precision Curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history['recall'], label='Train Recall')
    axes[1, 1].plot(history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Recall Curve')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Vẽ confusion matrix heatmap"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Incident', 'Incident'],
                yticklabels=['No Incident', 'Incident'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return plt.gcf()

def plot_roc_curve(y_true, y_proba):
    """Vẽ ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt.gcf()

def plot_metrics_comparison(models_metrics):
    """So sánh metrics giữa các models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = list(models_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [models_metrics[m].get(metric, 0) for m in models]
        axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[i].set_title(f'{metric.upper()} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim([0, 1])
        axes[i].grid(True, axis='y', alpha=0.3)
        
        # Thêm giá trị trên mỗi cột
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', 
                        ha='center', va='bottom')
    
    plt.tight_layout()
    return fig
```

---

## 3. PHÁT TRIỂN FEATURES: WHAT FOR, WHERE STORE, WHO USE

### 3.1. WHAT FOR - Phát hiện Sự cố để Làm Gì?

#### 🎯 **Mục đích Chính**:

1. **Cảnh báo Sớm (Early Warning)**
   - **Mục đích**: Phát hiện sự cố ngay khi xảy ra để có phản ứng nhanh
   - **Lợi ích**: 
     - Giảm thời gian phản ứng (response time)
     - Giảm thiểu hậu quả (tắc đường, tai nạn thứ cấp)
     - Cứu sống người trong trường hợp khẩn cấp

2. **Quản lý Giao thông Tự động (Automated Traffic Management)**
   - **Mục đích**: Tự động điều chỉnh đèn giao thông, biển báo
   - **Lợi ích**:
     - Giảm tắc đường
     - Tối ưu luồng giao thông
     - Giảm chi phí vận hành

3. **Phân tích và Báo cáo (Analytics & Reporting)**
   - **Mục đích**: Thu thập dữ liệu về sự cố để phân tích xu hướng
   - **Lợi ích**:
     - Xác định điểm đen (black spots) thường xảy ra sự cố
     - Phân tích nguyên nhân (thời tiết, giờ cao điểm, ...)
     - Lập kế hoạch cải thiện hạ tầng

4. **Tích hợp với Hệ thống Khác (Integration)**
   - **Mục đích**: Kết nối với hệ thống cảnh sát, cứu thương, bảo hiểm
   - **Lợi ích**:
     - Tự động gọi cảnh sát/cứu thương
     - Tạo báo cáo bảo hiểm tự động
     - Phối hợp giữa các cơ quan

#### 🔄 **Workflow sau khi Phát hiện Sự cố**:

```
Phát hiện Sự cố
    ↓
Xác nhận (Confirmation)
    ↓
Phân loại (Classification)
    ├── Tai nạn nghiêm trọng → Gọi cảnh sát + cứu thương
    ├── Xe hỏng → Gọi cứu hộ
    ├── Tắc đường → Điều chỉnh đèn giao thông
    └── Sự kiện đặc biệt → Thông báo cho cơ quan liên quan
    ↓
Lưu trữ (Storage)
    ↓
Phân tích (Analytics)
    ↓
Báo cáo (Reporting)
```

### 3.2. WHERE STORE - Hệ thống Lưu trữ và Tracking

#### 💾 **1. Lưu trữ Dữ liệu Sự cố (Incident Storage)**

**Database Schema** (Đề xuất):

```sql
-- Bảng lưu trữ sự cố
CREATE TABLE incidents (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    location VARCHAR(255),  -- Vị trí (GPS, địa chỉ)
    camera_id VARCHAR(100),  -- ID camera phát hiện
    incident_type VARCHAR(50), -- 'accident', 'breakdown', 'congestion', 'event'
    confidence_score FLOAT,   -- Độ tin cậy (0-1)
    status VARCHAR(20),       -- 'detected', 'confirmed', 'resolved', 'false_alarm'
    image_path TEXT,          -- Đường dẫn ảnh/video
    metadata JSONB,           -- Thông tin bổ sung (model version, processing time, ...)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Bảng lưu trữ metrics và performance
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    dataset_type VARCHAR(20), -- 'train', 'val', 'test'
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Bảng lưu trữ predictions (để phân tích sau)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    model_name VARCHAR(50),
    prediction BOOLEAN,
    probability FLOAT,
    processing_time_ms FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

**File System Storage**:

```
data/
├── incidents/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── incident_20240115_143022_001.jpg
│   │   │   ├── incident_20240115_143022_001_metadata.json
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── videos/
│   └── [tương tự cấu trúc incidents]
└── models/
    ├── CNN_model/
    │   ├── model.keras
    │   ├── weights.h5
    │   └── metadata.json
    └── ...
```

**Cloud Storage** (Đề xuất cho production):
- **AWS S3** hoặc **Google Cloud Storage**
- **Lợi ích**: 
  - Scalable
  - Backup tự động
  - CDN cho truy cập nhanh
  - Chi phí thấp

#### 📊 **2. MLflow Tracking**

**Hiện tại hệ thống đã có MLflow** (`src/training/trainer.py`):

```python
# MLflow tracking đã được tích hợp
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)

# Log parameters, metrics, models
mlflow.log_params(model_config)
mlflow.log_metric('train_accuracy', train_metrics['accuracy'])
mlflow.log_metric('val_accuracy', val_metrics['accuracy'])
mlflow.tensorflow.log_model(model, "model")
```

**MLflow lưu trữ**:
- **Parameters**: Hyperparameters (learning_rate, batch_size, ...)
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Loss, ...
- **Artifacts**: Model files, plots, configs
- **Runs**: Mỗi lần training là một run

**Cấu trúc MLflow**:
```
mlruns/
├── 0/
│   ├── meta.yaml (experiment info)
│   └── [run_id]/
│       ├── meta.yaml
│       ├── metrics/
│       ├── params/
│       ├── artifacts/
│       └── ...
```

#### 🔍 **3. Monitoring và Logging**

**Application Logs** (`logs/app.log`):
- Log mỗi prediction
- Log errors và exceptions
- Log performance metrics

**Metrics Collector** (`src/serving/monitoring.py`):
- Tổng số predictions
- Tổng số incidents detected
- Average processing time
- Throughput (predictions/second)

**Real-time Monitoring** (Đề xuất):
- **Prometheus + Grafana**: Monitor system metrics
- **ELK Stack** (Elasticsearch, Logstash, Kibana): Log analysis
- **Custom Dashboard**: Streamlit dashboard để xem metrics real-time

### 3.3. WHO USE - Các Use Cases và Người Dùng

#### 👥 **1. Người Dùng và Use Cases**

##### **A. Nhân viên Quản lý Giao thông (Traffic Management Staff)**

**Use Case 1: Giám sát Real-time**
- **Mô tả**: Xem dashboard real-time để theo dõi tình trạng giao thông
- **Workflow**:
  1. Mở dashboard Streamlit
  2. Xem danh sách incidents đang xảy ra
  3. Xem ảnh/video của incident
  4. Xác nhận hoặc đánh dấu false alarm
  5. Gửi cảnh báo đến các cơ quan liên quan

**Use Case 2: Phân tích Xu hướng**
- **Mô tả**: Phân tích dữ liệu incidents để tìm patterns
- **Workflow**:
  1. Truy vấn database incidents theo thời gian, địa điểm
  2. Xem biểu đồ xu hướng (số incidents theo giờ, ngày, tuần)
  3. Xác định điểm đen (black spots)
  4. Tạo báo cáo

**Features cần có**:
- Dashboard real-time với map
- Filter và search incidents
- Export báo cáo (PDF, Excel)
- Alert notifications

##### **B. Cảnh sát Giao thông (Traffic Police)**

**Use Case 1: Nhận Cảnh báo Tự động**
- **Mô tả**: Nhận thông báo khi phát hiện sự cố nghiêm trọng
- **Workflow**:
  1. Hệ thống phát hiện incident (confidence > 0.8)
  2. Tự động gửi notification (SMS, Email, App push)
  3. Cảnh sát xem thông tin (vị trí, ảnh, loại sự cố)
  4. Phản ứng (điều phối lực lượng, gọi cứu thương)

**Use Case 2: Xem Lịch sử Sự cố**
- **Mô tả**: Xem lại các incidents đã xảy ra để điều tra
- **Workflow**:
  1. Tìm kiếm incidents theo thời gian, địa điểm
  2. Xem ảnh/video và metadata
  3. Tải về để làm bằng chứng

**Features cần có**:
- Mobile app hoặc web app responsive
- Push notifications
- GPS integration
- Export evidence (ảnh, video)

##### **C. Nhà Phân tích Dữ liệu (Data Analyst)**

**Use Case 1: Phân tích Hiệu suất Model**
- **Mô tả**: Đánh giá và cải thiện model
- **Workflow**:
  1. Xem metrics trong MLflow
  2. So sánh các model versions
  3. Phân tích confusion matrix, ROC curve
  4. Xác định cần cải thiện gì (thu thập thêm dữ liệu, tune hyperparameters)

**Use Case 2: Phân tích Xu hướng Sự cố**
- **Mô tả**: Tìm patterns trong dữ liệu incidents
- **Workflow**:
  1. Query database
  2. Phân tích thống kê (correlation, trends)
  3. Tạo visualizations
  4. Viết báo cáo insights

**Features cần có**:
- MLflow UI
- Jupyter notebooks integration
- SQL query interface
- Data export (CSV, JSON)

##### **D. Developer/Engineer**

**Use Case 1: Training và Deploy Model**
- **Mô tả**: Huấn luyện model mới và deploy
- **Workflow**:
  1. Chuẩn bị dữ liệu training
  2. Chạy training script
  3. Đánh giá model
  4. Deploy model mới (A/B testing)
  5. Monitor performance

**Use Case 2: Debug và Troubleshooting**
- **Mô tả**: Sửa lỗi và tối ưu hệ thống
- **Workflow**:
  1. Xem logs
  2. Reproduce issues
  3. Fix bugs
  4. Test và deploy

**Features cần có**:
- Command-line tools
- API documentation
- Logging và debugging tools
- Testing framework

#### 🔄 **2. Integration với Hệ thống Khác**

##### **A. Hệ thống Đèn Giao thông (Traffic Light System)**

**Integration**:
- API endpoint để gửi incident alerts
- Tự động điều chỉnh đèn giao thông khi có sự cố
- **Protocol**: REST API hoặc MQTT

##### **B. Hệ thống Cảnh báo (Alert System)**

**Integration**:
- Gửi SMS/Email khi phát hiện sự cố nghiêm trọng
- Push notifications đến mobile app
- **Protocol**: Webhooks, SMS Gateway API

##### **C. Hệ thống Bảo hiểm (Insurance System)**

**Integration**:
- Tự động tạo claim khi phát hiện tai nạn
- Gửi ảnh/video làm bằng chứng
- **Protocol**: REST API

##### **D. Hệ thống Bản đồ (Mapping System)**

**Integration**:
- Hiển thị incidents trên bản đồ (Google Maps, OpenStreetMap)
- Tính toán route tránh incidents
- **Protocol**: REST API, GeoJSON

### 3.4. Roadmap Phát triển Features

#### 🚀 **Phase 1: Core Features (Hiện tại)**

✅ **Đã có**:
- CNN model training
- Image/video prediction
- Basic API
- MLflow tracking
- Streamlit dashboard

#### 🚀 **Phase 2: Storage & Tracking (Đề xuất)**

📋 **Cần phát triển**:
1. **Database Integration**
   - PostgreSQL/MySQL cho incidents storage
   - Schema design và migration
   - ORM (SQLAlchemy)

2. **File Storage System**
   - Organize images/videos theo cấu trúc
   - Backup strategy
   - Cloud storage integration

3. **Enhanced MLflow**
   - Model versioning
   - Model registry
   - Experiment comparison UI

#### 🚀 **Phase 3: User Features (Đề xuất)**

📋 **Cần phát triển**:
1. **Real-time Dashboard**
   - Map integration (Google Maps/Leaflet)
   - Live incident feed
   - Alert notifications

2. **Mobile App**
   - React Native hoặc Flutter
   - Push notifications
   - Offline mode

3. **Analytics Dashboard**
   - Trend analysis
   - Statistical reports
   - Export functionality

#### 🚀 **Phase 4: Integration (Đề xuất)**

📋 **Cần phát triển**:
1. **External System Integration**
   - Traffic light system
   - Alert system (SMS/Email)
   - Insurance system
   - Mapping system

2. **API Enhancements**
   - Webhooks
   - Authentication & Authorization
   - Rate limiting
   - API versioning

---

## 📝 KẾT LUẬN

### Tóm tắt:

1. **Baseline**: CNN model là baseline chính cho image-based detection, các models khác (ANN, RNN, RBFNN) bổ sung cho sensor-based detection.

2. **Testing & Validation**: Cần tạo các diagrams (loss curves, confusion matrix, ROC curve, metrics comparison) và phân tích chi tiết với reasoning.

3. **Features Development**:
   - **What For**: Cảnh báo sớm, quản lý giao thông tự động, phân tích xu hướng, tích hợp hệ thống
   - **Where Store**: Database (PostgreSQL), File system, MLflow, Cloud storage
   - **Who Use**: Traffic management staff, Police, Data analysts, Developers

### Hướng phát triển tiếp:

1. ✅ Implement database schema và storage system
2. ✅ Tạo visualization tools cho metrics
3. ✅ Phát triển real-time dashboard
4. ✅ Tích hợp với external systems
5. ✅ Mobile app development

---

**Tài liệu này được tạo để làm rõ các vấn đề mà thầy giáo đã nhận xét và định hướng.**

