import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import time

from src.models.base_model import BaseModel
from src.utils.logger import logger

class ModelEvaluator:
    """
    ModelEvaluator: Tính toán các chỉ số đánh giá độ chính xác của mô hình.
    
    Metrics:
    - Accuracy: Độ chính xác tổng thể.
    - Precision: Tỷ lệ phát hiện đúng trong số các lần báo động.
    - Recall (Detection Rate): Tỷ lệ phát hiện được các sự cố thực tế.
    - F1-Score: Trung bình điều hòa của Precision và Recall.
    - False Alarm Rate (FAR): Tỷ lệ báo động giả.
    - Mean Time To Detection (MTTD): Thời gian trung bình để phát hiện sự cố.
    """

    def __init__(self):

        pass

    def evaluate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y_true: np.ndarray,
        calculate_mttd: bool = False
    ) -> Dict[str, float]:
        """
        Đánh giá model trên tập dữ liệu X.
        """

        start_time = time.time()
        y_pred = model.predict(X)
        prediction_time = time.time() - start_time

        try:
            y_proba = model.predict_proba(X)
        except:
            y_proba = None

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        detection_rate = recall

        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0

        mttd = self._calculate_mttd(y_true, y_pred) if calculate_mttd else None

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'roc_auc': roc_auc,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'prediction_time': prediction_time
        }

        if mttd is not None:
            metrics['mean_time_to_detection'] = mttd

        return metrics

    def _calculate_mttd(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_interval: float = 1.0
    ) -> float:

        detection_times = []

        incident_indices = np.where(y_true == 1)[0]

        for idx in incident_indices:
            detection_idx = None

            window = 10
            search_start = max(0, idx - window)
            search_end = min(len(y_pred), idx + window)

            for i in range(search_start, search_end):
                if y_pred[i] == 1:
                    detection_idx = i
                    break

            if detection_idx is not None:
                time_to_detect = abs(detection_idx - idx) * time_interval
                detection_times.append(time_to_detect)

        if len(detection_times) == 0:
            return 0.0

        return np.mean(detection_times)

    def print_classification_report(
        self,
        model: BaseModel,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> None:

        y_pred = model.predict(X)

        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=['No Incident', 'Incident']))
        print("="*50)

        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              No    Yes")
        print(f"Actual No    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Yes    {cm[1,0]:4d}  {cm[1,1]:4d}")
        print()