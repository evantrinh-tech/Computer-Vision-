import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

from src.models.base_model import BaseModel
from src.utils.logger import logger

class RBFNNModel(BaseModel):
    """
    RBFNNModel: Mạng Nơ-ron Radial Basis Function.
    
    Đặc điểm:
    - Sử dụng hàm cơ sở xuyên tâm (Radial Basis Function - thường là Gaussian) làm hàm kích hoạt.
    - Huấn luyện qua 2 giai đoạn:
      1. Unsupervised: Tìm các Centers bằng thuật toán K-Means Clustering.
      2. Supervised: Tính toán weights kết nối từ lớp ẩn đến lớp output (dùng Linear Regression hoặc Pseudo-inverse).
    
    Ưu điểm:
    - Train rất nhanh (không cần Backpropagation phức tạp).
    - Khả năng xấp xỉ hàm tốt.
    """

    def __init__(
        self,
        n_centers: int = 20,
        sigma: float = 1.0,
        learning_rate: float = 0.01,
        use_wavelet: bool = True,
        wavelet: str = 'db4',
        wavelet_level: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo RBFNN Model.
        Args:
            n_centers: Số lượng tâm (clusters) cho lớp ẩn. Tương đương số nơ-ron ẩn.
            sigma: Độ rộng (spread) của hàm Gaussian.
            use_wavelet: Có sử dụng biến đổi Wavelet để trích xuất đặc trưng trước không.
        """

        super().__init__("RBFNN", config)
        self.n_centers = n_centers
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.use_wavelet = use_wavelet
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level

        self.centers = None
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _gaussian_rbf(self, X: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Hàm Gaussian Radial Basis Function.
        Công thức: exp(-||x - center||^2 / (2 * sigma^2))
        """
        distances = np.sum((X - center) ** 2, axis=1)
        return np.exp(-distances / (2 * self.sigma ** 2))

    def _apply_wavelet(self, X: np.ndarray) -> np.ndarray:

        if not self.use_wavelet:
            return X

        try:
            import pywt
            X_wavelet = []

            for sample in X:
                features = []
                for feature in sample:
                    coeffs = pywt.wavedec(feature, self.wavelet, level=self.wavelet_level)
                    flat_coeffs = np.concatenate(coeffs)
                    features.extend(flat_coeffs)

                X_wavelet.append(features)

            return np.array(X_wavelet)
        except ImportError:
            logger.warning("PyWavelets không được cài đặt. Bỏ qua wavelet transform.")
            return X

    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:

        logger.info(f"RBFNN model sẽ được khởi tạo với {self.n_centers} centers")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện RBFNN.
        
        Quy trình:
        1. K-Means Clustering: Tìm k tâm (centers) đại diện cho dữ liệu.
        2. Tính RBF Activations: Tính khoảng cách từ mỗi điểm dữ liệu đến các tâm.
        3. Linear Regression: Tính weights tối ưu bằng phương pháp bình phương tối thiểu (Least Squares).
        """

        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.use_wavelet:
            X_train_scaled = self._apply_wavelet(X_train_scaled)

        logger.info(f"Đang tìm {self.n_centers} RBF centers bằng K-means...")
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        self.centers = kmeans.cluster_centers_

        logger.info("Đang tính toán RBF activations...")
        n_samples = X_train_scaled.shape[0]
        rbf_outputs = np.zeros((n_samples, self.n_centers))

        for i, center in enumerate(self.centers):
            rbf_outputs[:, i] = self._gaussian_rbf(X_train_scaled, center)

        rbf_with_bias = np.hstack([np.ones((n_samples, 1)), rbf_outputs])

        try:
            self.weights = np.linalg.lstsq(
                rbf_with_bias,
                y_train,
                rcond=None
            )[0]
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(rbf_with_bias) @ y_train

        self.bias = self.weights[0]
        self.weights = self.weights[1:]

        self.is_trained = True
        logger.info("Đã hoàn thành training RBFNN model")

        train_pred = self.predict(X_train)
        train_acc = np.mean(train_pred == y_train)

        history = {
            'loss': [0.0],
            'accuracy': [train_acc]
        }

        return {'history': history, 'final_accuracy': train_acc}

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise ValueError("Model chưa được train.")

        X_scaled = self.scaler.transform(X)

        if self.use_wavelet:
            X_scaled = self._apply_wavelet(X_scaled)

        n_samples = X_scaled.shape[0]
        rbf_outputs = np.zeros((n_samples, self.n_centers))

        for i, center in enumerate(self.centers):
            rbf_outputs[:, i] = self._gaussian_rbf(X_scaled, center)

        outputs = rbf_outputs @ self.weights + self.bias

        probabilities = 1 / (1 + np.exp(-outputs))
        predictions = (probabilities > 0.5).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise ValueError("Model chưa được train.")

        X_scaled = self.scaler.transform(X)

        if self.use_wavelet:
            X_scaled = self._apply_wavelet(X_scaled)

        n_samples = X_scaled.shape[0]
        rbf_outputs = np.zeros((n_samples, self.n_centers))

        for i, center in enumerate(self.centers):
            rbf_outputs[:, i] = self._gaussian_rbf(X_scaled, center)

        outputs = rbf_outputs @ self.weights + self.bias

        probabilities = 1 / (1 + np.exp(-outputs))

        return probabilities

    def save(self, path: Path) -> None:

        if not self.is_trained:
            raise ValueError("Model chưa được train.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'centers': self.centers,
            'weights': self.weights,
            'bias': self.bias,
            'scaler': self.scaler,
            'n_centers': self.n_centers,
            'sigma': self.sigma,
            'use_wavelet': self.use_wavelet,
            'wavelet': self.wavelet,
            'wavelet_level': self.wavelet_level
        }

        joblib.dump(model_data, path)
        logger.info(f"Đã lưu RBFNN model tại {path}")

    def load(self, path: Path) -> None:

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        model_data = joblib.load(path)

        self.centers = model_data['centers']
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.scaler = model_data['scaler']
        self.n_centers = model_data['n_centers']
        self.sigma = model_data['sigma']
        self.use_wavelet = model_data.get('use_wavelet', True)
        self.wavelet = model_data.get('wavelet', 'db4')
        self.wavelet_level = model_data.get('wavelet_level', 3)

        self.is_trained = True
        logger.info(f"Đã load RBFNN model từ {path}")