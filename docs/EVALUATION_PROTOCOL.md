#  PROTOCOL ĐÁNH GIÁ HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG

## 📋 MỤC LỤC

1. [Chia dữ liệu Train/Val/Test](#1-chia-dữ-liệu-trainvaltest)
2. [Chọn Threshold trên Validation](#2-chọn-threshold-trên-validation)
3. [Định nghĩa và Tính MTTD](#3-định-nghĩa-và-tính-mttd)
4. [Biểu đồ Bắt buộc](#4-biểu-đồ-bắt-buộc)
5. [Checklist Kiểm tra Sai lầm](#5-checklist-kiểm-tra-sai-lầm)

---

## 1. CHIA DỮ LIỆU TRAIN/VAL/TEST

### 1.1. Nguyên tắc Chia dữ liệu

**QUAN TRỌNG**: Phải tránh data leakage theo 3 chiều:
- **Incident-level**: Các frames của cùng một incident phải cùng một split
- **Camera-level**: Dữ liệu từ cùng một camera nên cùng một split
- **Time-level**: Dữ liệu theo thời gian phải được chia tuần tự (không random)

### 1.2. Phương pháp Chia cho Image Data

```python
def split_image_data_by_incident(
    data_path: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Chia dữ liệu ảnh theo incident để tránh leakage
    
    Strategy:
    1. Group ảnh theo incident (nếu có metadata)
    2. Hoặc group theo thư mục con (nếu mỗi thư mục là một incident)
    3. Chia incidents thành train/val/test
    4. Tất cả ảnh của một incident cùng một split
    """
    # Load tất cả ảnh
    normal_images = list((data_path / "normal").glob("*.jpg"))
    incident_images = list((data_path / "incident").glob("*.jpg"))
    
    # Group theo incident (giả sử có metadata hoặc naming convention)
    # Ví dụ: incident_001_frame_001.jpg, incident_001_frame_002.jpg
    incident_groups = {}
    for img in incident_images:
        # Extract incident ID từ tên file
        incident_id = extract_incident_id(img.name)  # Cần implement
        if incident_id not in incident_groups:
            incident_groups[incident_id] = []
        incident_groups[incident_id].append(img)
    
    # Chia incidents (không phải ảnh)
    incident_ids = list(incident_groups.keys())
    np.random.seed(42)
    np.random.shuffle(incident_ids)
    
    n_total = len(incident_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_incidents = incident_ids[:n_train]
    val_incidents = incident_ids[n_train:n_train+n_val]
    test_incidents = incident_ids[n_train+n_val:]
    
    # Collect images theo split
    train_images = []
    val_images = []
    test_images = []
    
    for incident_id in train_incidents:
        train_images.extend(incident_groups[incident_id])
    for incident_id in val_incidents:
        val_images.extend(incident_groups[incident_id])
    for incident_id in test_incidents:
        test_images.extend(incident_groups[incident_id])
    
    # Normal images: chia random (không có incident grouping)
    np.random.shuffle(normal_images)
    n_normal = len(normal_images)
    train_normal = normal_images[:int(n_normal * train_ratio)]
    val_normal = normal_images[int(n_normal * train_ratio):int(n_normal * (train_ratio + val_ratio))]
    test_normal = normal_images[int(n_normal * (train_ratio + val_ratio)):]
    
    train_images.extend(train_normal)
    val_images.extend(val_normal)
    test_images.extend(test_normal)
    
    return train_images, val_images, test_images
```

### 1.3. Phương pháp Chia cho Video Data

```python
def split_video_data_by_time(
    video_list: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Chia video theo thời gian (time-based split)
    
    Strategy:
    1. Sort videos theo timestamp
    2. Chia tuần tự: train (70%) → val (15%) → test (15%)
    3. Không random để tránh future leakage
    """
    # Sort theo timestamp (từ metadata hoặc filename)
    sorted_videos = sort_videos_by_time(video_list)
    
    n_total = len(sorted_videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_videos = sorted_videos[:n_train]
    val_videos = sorted_videos[n_train:n_train+n_val]
    test_videos = sorted_videos[n_train+n_val:]
    
    return train_videos, val_videos, test_videos
```

### 1.4. Validation Checklist

- [ ] Không có incident nào xuất hiện ở cả train và test
- [ ] Không có camera nào xuất hiện ở cả train và test (nếu có metadata camera)
- [ ] Test set được lấy từ thời gian sau train/val (time-based split)
- [ ] Tỉ lệ class (normal/incident) tương đương giữa train/val/test
- [ ] Kích thước test set ≥ 20% tổng dữ liệu

---

## 2. CHỌN THRESHOLD TRÊN VALIDATION

### 2.1. Mục tiêu Vận hành

Tùy vào use case, chọn threshold theo mục tiêu:

| Use Case | Mục tiêu | Strategy |
|----------|----------|----------|
| **An toàn cao** | Recall ≥ 0.9, FAR chấp nhận được | Ưu tiên Recall, threshold thấp (0.3-0.4) |
| **Giảm False Alarm** | FAR ≤ 1%, Recall ≥ 0.85 | Cân bằng, threshold trung bình (0.5-0.6) |
| **Precision cao** | Precision ≥ 0.95 | Ưu tiên Precision, threshold cao (0.7-0.8) |

### 2.2. Phương pháp Tune Threshold

```python
def tune_threshold_on_validation(
    y_val_proba: np.ndarray,
    y_val_true: np.ndarray,
    target_recall: float = 0.9,
    target_far: float = 0.01
) -> Dict[str, Any]:
    """
    Tune threshold trên validation set
    
    Args:
        y_val_proba: Xác suất từ model trên validation set
        y_val_true: Nhãn thực tế
        target_recall: Mục tiêu Recall (default 0.9)
        target_far: Mục tiêu FAR (default 0.01 = 1%)
        
    Returns:
        Dict chứa best threshold và metrics
    """
    from sklearn.metrics import recall_score, precision_score, confusion_matrix
    
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_metrics = None
    best_score = -1
    
    for threshold in thresholds:
        y_pred = (y_val_proba >= threshold).astype(int)
        
        recall = recall_score(y_val_true, y_pred, zero_division=0)
        precision = precision_score(y_val_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_val_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            far = 0.0
        
        # Score: đạt cả 2 mục tiêu
        score = 0
        if recall >= target_recall:
            score += 1
        if far <= target_far:
            score += 1
        
        # Nếu đạt cả 2, ưu tiên F1 cao
        if score == 2:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
                best_metrics = {
                    "threshold": threshold,
                    "recall": recall,
                    "precision": precision,
                    "f1_score": f1,
                    "far": far,
                    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
                }
    
    if best_metrics is None:
        # Không tìm thấy threshold đạt cả 2 mục tiêu
        # Chọn threshold có Recall gần target nhất
        for threshold in thresholds:
            y_pred = (y_val_proba >= threshold).astype(int)
            recall = recall_score(y_val_true, y_pred, zero_division=0)
            if abs(recall - target_recall) < abs(best_metrics.get("recall", 1.0) - target_recall):
                # Tính lại metrics
                precision = precision_score(y_val_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_val_true, y_pred)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                else:
                    far = 0.0
                
                best_threshold = threshold
                best_metrics = {
                    "threshold": threshold,
                    "recall": recall,
                    "precision": precision,
                    "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
                    "far": far,
                    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
                }
    
    return best_metrics
```

### 2.3. Validation Curve

Vẽ biểu đồ **FAR vs Recall** và **Precision vs Recall** để chọn threshold:

```python
def plot_threshold_curves(
    y_val_proba: np.ndarray,
    y_val_true: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Vẽ biểu đồ threshold curves
    """
    import matplotlib.pyplot as plt
    
    thresholds = np.arange(0.1, 0.95, 0.01)
    recalls = []
    precisions = []
    fars = []
    
    for threshold in thresholds:
        y_pred = (y_val_proba >= threshold).astype(int)
        recall = recall_score(y_val_true, y_pred, zero_division=0)
        precision = precision_score(y_val_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_val_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            far = 0.0
        
        recalls.append(recall)
        precisions.append(precision)
        fars.append(far)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # FAR vs Recall
    axes[0].plot(recalls, fars, 'b-', linewidth=2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('False Alarm Rate (FAR)')
    axes[0].set_title('FAR vs Recall Curve')
    axes[0].grid(True)
    
    # Precision vs Recall (PR Curve)
    axes[1].plot(recalls, precisions, 'r-', linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 3. ĐỊNH NGHĨA VÀ TÍNH MTTD

### 3.1. Định nghĩa MTTD (Mean Time To Detection)

**MTTD** = Thời gian trung bình từ khi sự cố xảy ra đến khi hệ thống phát hiện được.

**Đơn vị**: Giây (seconds)

### 3.2. Cách tính MTTD từ Frame/Video Data

```python
def calculate_mttd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    fps: float = 30.0
) -> float:
    """
    Tính Mean Time To Detection (MTTD)
    
    Args:
        y_true: Nhãn thực tế (0/1) theo frame
        y_pred: Predictions (0/1) theo frame
        timestamps: Timestamps thực tế (giây), nếu None sẽ tính từ frame_number/fps
        fps: Frames per second
        
    Returns:
        MTTD (giây)
    """
    if timestamps is None:
        timestamps = np.arange(len(y_true)) / fps
    
    detection_times = []
    
    # Tìm các incident thực tế
    incident_starts = []
    incident_ends = []
    
    in_incident = False
    start_idx = None
    
    for i, label in enumerate(y_true):
        if label == 1 and not in_incident:
            # Bắt đầu incident
            in_incident = True
            start_idx = i
        elif label == 0 and in_incident:
            # Kết thúc incident
            incident_starts.append(start_idx)
            incident_ends.append(i - 1)
            in_incident = False
    
    # Nếu incident kéo dài đến cuối
    if in_incident:
        incident_starts.append(start_idx)
        incident_ends.append(len(y_true) - 1)
    
    # Tính thời gian detection cho mỗi incident
    for start_idx, end_idx in zip(incident_starts, incident_ends):
        incident_start_time = timestamps[start_idx]
        
        # Tìm frame đầu tiên model phát hiện được (trong window)
        detection_idx = None
        window = int(fps * 10)  # Tìm trong 10 giây sau khi incident bắt đầu
        
        search_start = start_idx
        search_end = min(start_idx + window, len(y_pred))
        
        for i in range(search_start, search_end):
            if y_pred[i] == 1:
                detection_idx = i
                break
        
        if detection_idx is not None:
            detection_time = timestamps[detection_idx] - incident_start_time
            detection_times.append(detection_time)
        else:
            # Không phát hiện được (False Negative)
            # Có thể bỏ qua hoặc tính là MTTD = infinity
            pass
    
    if len(detection_times) == 0:
        return 0.0  # Hoặc return np.inf nếu muốn báo không phát hiện được
    
    return np.mean(detection_times)
```

### 3.3. MTTD theo Event

Nếu dữ liệu được label theo **event** (không phải frame-by-frame):

```python
def calculate_mttd_by_events(
    incident_events: List[Dict],  # [{"start_time": 10.5, "end_time": 15.2}, ...]
    detection_events: List[Dict],  # [{"detected_time": 11.2}, ...]
    max_detection_window: float = 10.0  # Giây
) -> float:
    """
    Tính MTTD theo events
    
    Args:
        incident_events: List các incident thực tế
        detection_events: List các detection từ model
        max_detection_window: Window tối đa để match detection với incident
        
    Returns:
        MTTD (giây)
    """
    detection_times = []
    matched_detections = set()
    
    for incident in incident_events:
        incident_start = incident["start_time"]
        best_detection = None
        best_time_diff = float('inf')
        
        for i, detection in enumerate(detection_events):
            if i in matched_detections:
                continue
            
            detection_time = detection["detected_time"]
            time_diff = detection_time - incident_start
            
            if 0 <= time_diff <= max_detection_window:
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_detection = i
        
        if best_detection is not None:
            detection_times.append(best_time_diff)
            matched_detections.add(best_detection)
    
    if len(detection_times) == 0:
        return 0.0
    
    return np.mean(detection_times)
```

---

## 4. BIỂU ĐỒ BẮT BUỘC

### 4.1. Danh sách Biểu đồ

1. **PR Curve (Precision-Recall Curve)**
2. **ROC Curve (Receiver Operating Characteristic)**
3. **Confusion Matrix**
4. **FAR vs Recall Curve**
5. **Latency Histogram**
6. **Loss Curves (Training/Validation)**
7. **MTTD Distribution**

### 4.2. Code Template

```python
def generate_all_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    latencies: List[float],
    mttd_values: List[float],
    save_dir: Path
):
    """
    Tạo tất cả biểu đồ đánh giá
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_dir / "pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. FAR vs Recall
    thresholds = np.arange(0.1, 0.95, 0.01)
    recalls = []
    fars = []
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred_thresh, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_thresh)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            far = 0.0
        recalls.append(recall)
        fars.append(far)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, fars, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('False Alarm Rate (FAR)')
    plt.title('FAR vs Recall Curve')
    plt.grid(True)
    plt.savefig(save_dir / "far_vs_recall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Latency Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(latencies, bins=50, edgecolor='black')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution')
    plt.axvline(np.percentile(latencies, 95), color='r', linestyle='--', 
                label=f'p95: {np.percentile(latencies, 95):.2f}ms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "latency_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. MTTD Distribution
    if mttd_values:
        plt.figure(figsize=(8, 6))
        plt.hist(mttd_values, bins=30, edgecolor='black')
        plt.xlabel('Time to Detection (seconds)')
        plt.ylabel('Frequency')
        plt.title('MTTD Distribution')
        plt.axvline(np.mean(mttd_values), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(mttd_values):.2f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "mttd_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
```

---

## 5. CHECKLIST KIỂM TRA SAI LẦM

### 5.1. Data Leakage

- [ ] **Kiểm tra**: Có incident nào xuất hiện ở cả train và test không?
  - **Cách kiểm tra**: So sánh metadata/ID của incidents
  - **Cách sửa**: Chia lại theo incident-level

- [ ] **Kiểm tra**: Có camera nào xuất hiện ở cả train và test không?
  - **Cách kiểm tra**: Group theo camera_id, kiểm tra overlap
  - **Cách sửa**: Chia theo camera-level

- [ ] **Kiểm tra**: Test set có dữ liệu từ tương lai không?
  - **Cách kiểm tra**: So sánh timestamps
  - **Cách sửa**: Chia time-based (train trước, test sau)

### 5.2. Class Imbalance

- [ ] **Kiểm tra**: Tỉ lệ normal/incident trong train/val/test
  - **Cách kiểm tra**: `np.bincount(y_train)`, `np.bincount(y_test)`
  - **Cách sửa**: Stratified split, class weights, SMOTE

- [ ] **Kiểm tra**: Model có bias về class đa số không?
  - **Cách kiểm tra**: Confusion matrix, xem TP/TN/FP/FN
  - **Cách sửa**: Class weights, focal loss, resampling

### 5.3. Threshold Issues

- [ ] **Kiểm tra**: Threshold có được tune trên validation không?
  - **Cách kiểm tra**: Xem code có `tune_threshold_on_validation()` không
  - **Cách sửa**: Implement threshold tuning

- [ ] **Kiểm tra**: Threshold có phù hợp với mục tiêu vận hành không?
  - **Cách kiểm tra**: Vẽ FAR vs Recall curve, xem có đạt target không
  - **Cách sửa**: Điều chỉnh target hoặc threshold

### 5.4. Label Noise

- [ ] **Kiểm tra**: Có label sai không?
  - **Cách kiểm tra**: Xem sample FP/FN, kiểm tra manual
  - **Cách sửa**: Relabel, loại bỏ noisy samples

### 5.5. Train/Val Mismatch

- [ ] **Kiểm tra**: Distribution của train và val có khác nhau không?
  - **Cách kiểm tra**: So sánh statistics (mean, std) của features
  - **Cách sửa**: Chia lại data, normalize chung

### 5.6. Overfitting

- [ ] **Kiểm tra**: Train accuracy >> Val accuracy?
  - **Cách kiểm tra**: So sánh metrics train vs val
  - **Cách sửa**: Dropout, regularization, early stopping, thêm data

### 5.7. Metrics Calculation

- [ ] **Kiểm tra**: FAR có được tính đúng không?
  - **Cách kiểm tra**: `FAR = FP / (FP + TN)`, không phải `FP / (FP + TP)`
  - **Cách sửa**: Sửa công thức

- [ ] **Kiểm tra**: MTTD có được tính đúng không?
  - **Cách kiểm tra**: Manual check với sample incidents
  - **Cách sửa**: Sửa logic tính MTTD

### 5.8. Temporal Confirmation

- [ ] **Kiểm tra**: Temporal confirmation có giảm FAR không?
  - **Cách kiểm tra**: So sánh FAR trước và sau khi apply temporal confirmation
  - **Cách sửa**: Tune parameters (K, window, threshold, cooldown)

---

## 📝 TÓM TẮT

1. **Chia data**: Theo incident/camera/time, không random
2. **Tune threshold**: Trên validation, theo mục tiêu vận hành
3. **Tính MTTD**: Theo event hoặc frame, đúng định nghĩa
4. **Vẽ biểu đồ**: PR, ROC, Confusion Matrix, FAR vs Recall, Latency, MTTD
5. **Kiểm tra**: Data leakage, imbalance, threshold, label noise, overfitting

---

*Cập nhật lần cuối: [Ngày hiện tại]*

