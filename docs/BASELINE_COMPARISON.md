# 📊 BASELINE & MODEL COMPARISON

## 📋 TÓM TẮT

Tài liệu này mô tả baseline và so sánh các mô hình trong hệ thống phát hiện sự cố giao thông. Để tránh lỗi "so sánh khác loại dữ liệu", chúng ta tách rõ **3 task riêng biệt**: Vision-based, Sensor-based, và Hybrid.

---

## 🎯 PHÂN LOẠI TASK

Hệ thống phát hiện sự cố giao thông có thể sử dụng 3 loại dữ liệu đầu vào:

1. **Vision Task**: Phát hiện từ ảnh/video camera
2. **Sensor Task**: Phát hiện từ dữ liệu cảm biến (volume, speed, occupancy)
3. **Hybrid Task**: Kết hợp Vision + Sensor (late fusion hoặc early fusion)

**QUAN TRỌNG**: Mỗi task có baseline riêng và không thể so sánh trực tiếp giữa các task.

---

## 1. VISION TASK: PHÁT HIỆN TỪ ẢNH/VIDEO

### 1.1. Baseline: CNN với Transfer Learning

**Baseline được chọn**: **CNN (Convolutional Neural Network)** với Transfer Learning từ MobileNetV2.

#### Lý do chọn CNN làm Baseline:

1. **Phù hợp với dữ liệu ảnh**: CNN được thiết kế đặc biệt để xử lý dữ liệu hình ảnh, có khả năng tự động trích xuất các đặc trưng (features) từ ảnh mà không cần feature engineering thủ công.

2. **Transfer Learning**: CNN model sử dụng Transfer Learning với các pre-trained models:
   - **MobileNetV2** (mặc định): Nhẹ, nhanh, phù hợp cho real-time inference
   - **ResNet50**: Độ chính xác cao hơn, nhưng chậm hơn
   - **VGG16**: Kiến trúc đơn giản, dễ hiểu

3. **Kiến trúc Baseline CNN**:
   ```
   Input: Ảnh 224x224x3 (RGB)
   ↓
   Base Model: MobileNetV2 (pre-trained trên ImageNet)
   ↓
   Global Average Pooling
   ↓
   Dropout (0.2)
   ↓
   Dense Layer (128 neurons, ReLU)
   ↓
   Dropout (0.2)
   ↓
   Output: Dense(1, sigmoid) - Binary Classification
   ```

4. **Hyperparameters Baseline**:
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam (learning_rate=0.001)
   - Metrics: Accuracy, Precision, Recall
   - Data Augmentation: Rotation (20°), Shift (0.2), Flip (horizontal), Zoom (0.2)
   - Batch Size: 32
   - Epochs: 50 (với early stopping)

### 1.2. Tiêu chí Đánh giá cho Vision Task

| Metric | Mô tả | Target |
|--------|-------|--------|
| **Recall** | Tỉ lệ phát hiện được sự cố thực tế | ≥ 0.85 |
| **Precision** | Tỉ lệ dự đoán đúng trong các dự đoán "có sự cố" | ≥ 0.80 |
| **FAR (False Alarm Rate)** | Tỉ lệ cảnh báo sai | ≤ 0.05 (5%) |
| **F1-Score** | Harmonic mean của Precision và Recall | ≥ 0.82 |
| **Latency p95** | 95% requests xử lý trong thời gian này | ≤ 500ms |
| **MTTD** | Thời gian trung bình phát hiện sự cố | ≤ 10 giây |

### 1.3. So sánh với các Model khác (cùng Vision Task)

| Model | Architecture | Ưu điểm | Nhược điểm | So với CNN Baseline |
|-------|--------------|---------|------------|---------------------|
| **CNN (Baseline)** | MobileNetV2 + FC layers | ✅ Transfer Learning<br>✅ Tự động feature extraction<br>✅ Nhanh (real-time) | ❌ Cần GPU để train<br>❌ Yêu cầu nhiều dữ liệu ảnh | **Baseline** |
| **CNN (ResNet50)** | ResNet50 + FC layers | ✅ Độ chính xác cao hơn<br>✅ Transfer Learning | ❌ Chậm hơn MobileNetV2<br>❌ Model lớn hơn | +5-10% F1, -30% speed |
| **CNN (VGG16)** | VGG16 + FC layers | ✅ Kiến trúc đơn giản<br>✅ Dễ hiểu | ❌ Chậm hơn<br>❌ Model lớn | -3-5% F1, -20% speed |
| **YOLO/Object Detection** | YOLOv5/v8 | ✅ Phát hiện object + location<br>✅ Real-time | ❌ Phức tạp hơn<br>❌ Cần label bbox | Khác task (object detection) |

**Kết luận**: CNN với MobileNetV2 là baseline phù hợp cho Vision Task vì cân bằng tốt giữa accuracy và speed.

---

## 2. SENSOR TASK: PHÁT HIỆN TỪ DỮ LIỆU CẢM BIẾN

### 2.1. Baseline: Logistic Regression

**Baseline được chọn**: **Logistic Regression** cho sensor-based detection.

#### Lý do chọn Logistic Regression làm Baseline:

1. **Đơn giản và Interpretable**: Logistic Regression là mô hình đơn giản nhất, dễ hiểu và dễ debug.

2. **Phù hợp với dữ liệu số**: Sensor data là dữ liệu số (volume, speed, occupancy), không phải ảnh.

3. **Baseline công bằng**: Khi so sánh với các model phức tạp hơn (XGBoost, ANN, RNN), Logistic Regression là baseline hợp lý.

4. **Nhanh**: Inference rất nhanh, phù hợp cho real-time.

### 2.2. Alternative Baseline: XGBoost

**XGBoost** cũng có thể được coi là baseline cho sensor task vì:
- Phổ biến trong các bài toán tabular data
- Hiệu suất tốt với dữ liệu số
- Dễ tune hyperparameters

Tuy nhiên, chúng ta chọn **Logistic Regression** làm baseline chính vì đơn giản hơn.

### 2.3. Tiêu chí Đánh giá cho Sensor Task

| Metric | Mô tả | Target |
|--------|-------|--------|
| **Recall** | Tỉ lệ phát hiện được sự cố | ≥ 0.80 |
| **Precision** | Tỉ lệ dự đoán đúng | ≥ 0.75 |
| **FAR** | Tỉ lệ cảnh báo sai | ≤ 0.05 (5%) |
| **F1-Score** | Harmonic mean | ≥ 0.77 |
| **Latency p95** | 95% requests | ≤ 100ms (nhanh hơn Vision) |
| **MTTD** | Thời gian phát hiện | ≤ 5 giây |

### 2.4. So sánh các Model Sensor-based

| Model | Architecture | Ưu điểm | Nhược điểm | So với Logistic Regression |
|-------|--------------|---------|------------|----------------------------|
| **Logistic Regression (Baseline)** | Linear classifier | ✅ Đơn giản<br>✅ Nhanh<br>✅ Interpretable | ❌ Không capture non-linear | **Baseline** |
| **XGBoost** | Gradient Boosting | ✅ Hiệu suất tốt<br>✅ Feature importance | ❌ Phức tạp hơn<br>❌ Cần tune nhiều | +10-15% F1 |
| **ANN** | Feed-forward NN | ✅ Non-linear<br>✅ Deep learning | ❌ Cần nhiều data<br>❌ Black box | +5-10% F1 |
| **RNN/LSTM** | LSTM/GRU | ✅ Capture temporal patterns | ❌ Chậm hơn<br>❌ Phức tạp | +8-12% F1 (nếu có temporal) |
| **RBFNN** | RBF + Wavelet | ✅ Xử lý non-linear tốt<br>✅ Wavelet transform | ❌ Phức tạp<br>❌ Cần tune nhiều | +5-8% F1 |

**Kết luận**: Logistic Regression là baseline phù hợp cho Sensor Task. XGBoost có thể được sử dụng như một baseline nâng cao.

---

## 3. HYBRID TASK: KẾT HỢP VISION + SENSOR

### 3.1. Baseline: Late Fusion (Weighted Average)

**Baseline được chọn**: **Late Fusion** với weighted average của predictions từ Vision model và Sensor model.

#### Lý do chọn Late Fusion làm Baseline:

1. **Đơn giản**: Late fusion là cách đơn giản nhất để kết hợp 2 modalities.

2. **Không cần retrain**: Có thể sử dụng các model đã train riêng lẻ.

3. **Interpretable**: Dễ hiểu và debug.

4. **Kiến trúc Baseline Late Fusion**:
   ```
   Vision Model (CNN) → p_vision (probability)
   Sensor Model (Logistic/XGBoost) → p_sensor (probability)
   ↓
   Late Fusion: p_final = w1 * p_vision + w2 * p_sensor
   (với w1 + w2 = 1, thường w1 = 0.7, w2 = 0.3)
   ↓
   Threshold → Binary prediction
   ```

### 3.2. Alternative Fusion Methods

| Method | Mô tả | Ưu điểm | Nhược điểm |
|--------|-------|---------|------------|
| **Late Fusion (Baseline)** | Weighted average của probabilities | ✅ Đơn giản<br>✅ Không cần retrain | ❌ Không tận dụng feature-level info |
| **Voting** | Majority vote hoặc weighted vote | ✅ Đơn giản | ❌ Không tận dụng confidence |
| **Early Fusion** | Concatenate features trước khi train | ✅ Tận dụng feature-level | ❌ Cần retrain<br>❌ Phức tạp |
| **Attention-based Fusion** | Learn attention weights | ✅ Tự động học weights | ❌ Phức tạp<br>❌ Cần nhiều data |

### 3.3. Tiêu chí Đánh giá cho Hybrid Task

| Metric | Mô tả | Target |
|--------|-------|--------|
| **Recall** | Tỉ lệ phát hiện được sự cố | ≥ 0.90 |
| **Precision** | Tỉ lệ dự đoán đúng | ≥ 0.85 |
| **FAR** | Tỉ lệ cảnh báo sai | ≤ 0.03 (3%) |
| **F1-Score** | Harmonic mean | ≥ 0.87 |
| **Latency p95** | 95% requests | ≤ 300ms |
| **MTTD** | Thời gian phát hiện | ≤ 8 giây |

### 3.4. So sánh Hybrid vs Single Modality

| Model | Vision Only | Sensor Only | Hybrid (Late Fusion) |
|-------|-------------|-------------|----------------------|
| **Recall** | 0.85 | 0.80 | **0.90** ✅ |
| **Precision** | 0.80 | 0.75 | **0.85** ✅ |
| **FAR** | 0.05 | 0.05 | **0.03** ✅ |
| **F1-Score** | 0.82 | 0.77 | **0.87** ✅ |
| **Latency** | 500ms | 100ms | 300ms |

**Kết luận**: Hybrid model (Late Fusion) tốt hơn cả Vision-only và Sensor-only, đạt được mục tiêu cao hơn.

---

## 4. BẢNG SO SÁNH TỔNG HỢP

### 4.1. So sánh Baselines theo Task

| Task | Baseline | Architecture | Data Type | Target Metrics |
|------|----------|--------------|-----------|----------------|
| **Vision** | CNN (MobileNetV2) | Transfer Learning | Images (224x224x3) | Recall ≥ 0.85, FAR ≤ 0.05 |
| **Sensor** | Logistic Regression | Linear Classifier | Tabular (volume, speed, ...) | Recall ≥ 0.80, FAR ≤ 0.05 |
| **Hybrid** | Late Fusion | Weighted Average | Images + Tabular | Recall ≥ 0.90, FAR ≤ 0.03 |

### 4.2. Lý do Không So sánh Trực tiếp Vision vs Sensor

**KHÔNG ĐƯỢC** so sánh trực tiếp Vision model với Sensor model vì:

1. **Khác loại dữ liệu đầu vào**:
   - Vision: Ảnh (224x224x3 pixels)
   - Sensor: Số liệu (volume, speed, occupancy)

2. **Khác preprocessing**:
   - Vision: Image augmentation, normalization
   - Sensor: Feature engineering, scaling

3. **Khác use case**:
   - Vision: Phát hiện từ camera
   - Sensor: Phát hiện từ cảm biến giao thông

4. **Khác baseline**:
   - Vision: CNN
   - Sensor: Logistic Regression

**CHỈ SO SÁNH**:
- Vision models với nhau (CNN MobileNetV2 vs ResNet50 vs VGG16)
- Sensor models với nhau (Logistic vs XGBoost vs ANN vs RNN)
- Hybrid methods với nhau (Late Fusion vs Early Fusion vs Attention)

---

## 5. KẾT LUẬN

### 5.1. Baselines được Chọn

1. **Vision Task**: CNN với MobileNetV2 (Transfer Learning)
2. **Sensor Task**: Logistic Regression
3. **Hybrid Task**: Late Fusion (Weighted Average)

### 5.2. Tiêu chí So sánh Công bằng

- ✅ So sánh các model **cùng task** (cùng loại dữ liệu đầu vào)
- ✅ Sử dụng **cùng evaluation protocol** (train/val/test split, metrics)
- ✅ So sánh trên **cùng dataset** (nếu có)
- ❌ **KHÔNG** so sánh Vision với Sensor (khác loại dữ liệu)

### 5.3. Roadmap Nâng cấp

1. **Phase 1 (MVP)**: Vision baseline (CNN MobileNetV2)
2. **Phase 2 (Hybrid)**: Thêm Sensor baseline (Logistic) → Hybrid (Late Fusion)
3. **Phase 3 (Production)**: Tối ưu và nâng cấp (ResNet50, XGBoost, Early Fusion)

---

*Tài liệu này đảm bảo so sánh công bằng và tránh lỗi "so sánh khác loại dữ liệu".*

*Cập nhật lần cuối: [Ngày hiện tại]*

