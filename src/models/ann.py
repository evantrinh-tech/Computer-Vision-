import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from src.models.base_model import BaseModel
from src.utils.logger import logger

class ANNModel(BaseModel):
    """
    ANNModel: Mạng Nơ-ron Nhân tạo (Artificial Neural Network - Feed Forward).
    
    Đặc điểm:
    - Là mô hình cơ bản nhất (Multilayer Perceptron).
    - Dùng cho dữ liệu dạng bảng (tabular data) hoặc feature vectors đã được trích xuất.
    - Cấu trúc gồm Input Layer -> Hidden Layers (Dense) -> Output Layer.
    """

    def __init__(
        self,
        hidden_layers: list = [64, 32],
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo ANN Model.
        Args:
            hidden_layers: List số lượng nơ-ron cho từng lớp ẩn (VD: [64, 32]).
            activation: Hàm kích hoạt (relu, tanh, sigmoid...).
            dropout_rate: Tỷ lệ Dropout để chống overfitting.
        """

        super().__init__("ANN", config)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.is_trained = False

    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Xây dựng kiến trúc mạng MLP (Multi-layer Perceptron).
        """
        input_dim = input_shape[0] if len(input_shape) > 0 else input_shape

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for n_neurons in self.hidden_layers:
            x = layers.Dense(n_neurons, activation=self.activation)(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        logger.info(f"Đã build ANN model với {len(self.hidden_layers)} hidden layers")
        logger.info(f"Input shape: {input_shape}, Total params: {self.model.count_params()}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình ANN với dữ liệu dạng vectors.
        """

        if self.model is None:
            self.build(X_train.shape[1:])

        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        self.is_trained = True
        logger.info("Đã hoàn thành training ANN model")

        return {
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_accuracy': self.history.history['accuracy'][-1]
        }

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise ValueError("Model chưa được train. Gọi train() trước.")

        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise ValueError("Model chưa được train.")

        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: Path) -> None:

        if self.model is None:
            raise ValueError("Model chưa được build.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        logger.info(f"Đã lưu model tại {path}")

    def load(self, path: Path) -> None:

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        self.model = keras.models.load_model(str(path))
        self.is_trained = True
        logger.info(f"Đã load model từ {path}")