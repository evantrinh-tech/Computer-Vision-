from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from src.utils.logger import logger

class BaseModel(ABC):
    """
    BaseModel: Lớp cơ sở trừu tượng (Abstract Base Class) cho tất cả các mô hình AI trong dự án.
    
    Mục đích:
    - Định nghĩa một giao diện chung (interface) bắt buộc cho mọi model (CNN, RNN, ANN...).
    - Đảm bảo tính nhất quán: Bất kể dùng thuật toán nào, code bên ngoài chỉ cần gọi .build(), .train(), .predict() theo chuẩn này.
    - Áp dụng nguyên lý đa hình (Polymorphism) trong OOP.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo common attributes cho mọi model.
        Args:
            name: Tên của model (ví dụ: 'MobileNetV2_Custom', 'LSTM_Network').
            config: Dictionary chứa các tham số cấu hình (learning rate, units, dropout...).
        """
        self.name = name
        self.config = config or {}
        self.model = None        # Đối tượng model thực tế (ví dụ: keras.Model)
        self.is_trained = False  # Cờ đánh dấu model đã train chưa
        self.history = None      # Lưu lịch sử training (loss, accuracy qua các epochs)

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Phương thức Abstract: Xây dựng kiến trúc model.
        Các class con (như CNNModel) BẮT BUỘC phải implement hàm này.
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Phương thức Abstract: Huấn luyện model.
        Args:
            X_train, y_train: Dữ liệu huấn luyện.
            X_val, y_val: Dữ liệu validation (để kiểm tra overfitting).
        Returns:
            Dictionary chứa kết quả training (loss, accuracy, history).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Phương thức Abstract: Dự đoán kết quả từ dữ liệu mới.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Phương thức Abstract: Lưu model xuống đĩa cứng.
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Phương thức Abstract: Load model từ đĩa cứng lên RAM.
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất (probability).
        Mặc định gọi hàm predict(). Có thể override nếu cần behavior khác.
        """
        predictions = self.predict(X)
        if predictions.ndim == 1:
            return predictions
        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin metadata của model.
        """
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'config': self.config
        }