import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, applications

from src.models.base_model import BaseModel
from src.utils.logger import logger

class CNNModel(BaseModel):
    """
    CNNModel: Mô hình Convolutional Neural Network sử dụng Transfer Learning.
    
    Đặc điểm:
    - Sử dụng Backbone (MobileNetV2, ResNet50...) để trích xuất đặc trưng ảnh.
    - Thêm các lớp Classification Head (Dense, Dropout) để phân loại (Normal/Incident).
    - Hỗ trợ Fine-tuning để cải thiện độ chính xác trên tập dữ liệu cụ thể.
    """

    def __init__(
        self,
        use_transfer_learning: bool = True,
        base_model: str = 'MobileNetV2',
        image_size: Tuple[int, int] = (224, 224),
        learning_rate: float = 0.001,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Khởi tạo CNN Model.
        Args:
            use_transfer_learning: Có dùng model pre-trained không (True/False).
            base_model: Tên model nền (MobileNetV2, ResNet50, VGG16).
            image_size: Kích thước ảnh đầu vào (width, height).
            learning_rate: Tốc độ học ban đầu.
        """

        super().__init__("CNN", config)
        self.use_transfer_learning = use_transfer_learning
        self.base_model_name = base_model
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.model = None
        self.is_trained = False

    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:
        """
        Xây dựng kiến trúc mạng (Architecture).
        
        Quy trình:
        1. Nếu dùng Transfer Learning:
           - Load Pre-trained Base Model (ImageNet weights).
           - Freeze (đóng băng) các layers của Base Model.
           - Thêm GlobalAveragePooling để giảm chiều dữ liệu.
           - Thêm Dense & Dropout để phân loại.
        2. Nếu tự build (use_transfer_learning=False):
           - Tạo mạng CNN thuần túy với Conv2D và MaxPooling2D.
           - Phù hợp cho bài tập lớn nhỏ hoặc khi dữ liệu rất đơn giản.
        """

        if self.use_transfer_learning:
            if self.base_model_name == 'MobileNetV2':
                base = applications.MobileNetV2(
                    input_shape=(*self.image_size, 3),
                    include_top=False,
                    weights='imagenet'
                )
            elif self.base_model_name == 'ResNet50':
                base = applications.ResNet50(
                    input_shape=(*self.image_size, 3),
                    include_top=False,
                    weights='imagenet'
                )
            elif self.base_model_name == 'VGG16':
                base = applications.VGG16(
                    input_shape=(*self.image_size, 3),
                    include_top=False,
                    weights='imagenet'
                )
            else:
                raise ValueError(f"Unknown base model: {self.base_model_name}")

            base.trainable = False

            inputs = keras.Input(shape=(*self.image_size, 3))
            
            # MobileNetV2 expects input in range [-1, 1], but ImageProcessor returns [0, 1]
            # Rescaling: scale=2. (0->0, 1->2), offset=-1. (0->-1, 2->1) => [-1, 1]
            x = layers.Rescaling(scale=2., offset=-1.)(inputs)
            
            x = base(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)

            self.model = models.Model(inputs, outputs)
        else:
            inputs = keras.Input(shape=(*self.image_size, 3))

            x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = layers.MaxPooling2D(2, 2)(x)
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D(2, 2)(x)
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D(2, 2)(x)

            x = layers.Flatten()(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)

            self.model = models.Model(inputs, outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        logger.info(f"Đã build CNN model (Transfer Learning: {self.use_transfer_learning})")
        logger.info(f"Total params: {self.model.count_params()}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình.
        
        Quy trình:
        1. Sử dụng ImageDataGenerator để Augmentation (làm giàu dữ liệu: xoay, lật...).
        2. Train giai đoạn 1: Chỉ train các lớp Dense mới (Base mode bị đóng băng).
        3. Train giai đoạn 2 (Fine-tuning): Mở khóa một phần Base model để train tiếp với learning rate nhỏ hơn.
        
        Returns:
            Dictionary chứa lịch sử huấn luyện (loss, accuracy).
        """

        if self.model is None:
            self.build(X_train.shape[1:])

        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

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
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callback_list,
            verbose=verbose,
            steps_per_epoch=len(X_train) // batch_size
        )

        if self.use_transfer_learning:
            logger.info("Bắt đầu fine-tuning...")
            self.model.layers[1].trainable = True

            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate / 10)
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )

            self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=validation_data,
                epochs=epochs // 2,
                callbacks=callback_list,
                verbose=verbose,
                steps_per_epoch=len(X_train) // batch_size
            )

        self.is_trained = True
        logger.info("Đã hoàn thành training CNN model")

        return {
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_accuracy': self.history.history['accuracy'][-1]
        }

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise ValueError("Model chưa được train.")

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

        if path.suffix == '':
            path.mkdir(parents=True, exist_ok=True)
            model_file = path / "model.keras"
            self.model.save(str(model_file))
            logger.info(f"Đã lưu CNN model tại {model_file}")

            try:
                weights_file = path / "weights.h5"
                self.model.save_weights(str(weights_file))
                logger.info(f"Đã lưu weights tại {weights_file}")
            except Exception as e:
                logger.warning(f"Không thể lưu weights riêng: {e}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            logger.info(f"Đã lưu CNN model tại {path}")

    def load(self, path: Path) -> None:

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {path}")

        if path.is_dir():
            model_file = path / "model.keras"
            if not model_file.exists():
                keras_files = list(path.glob("*.keras"))
                h5_files = list(path.glob("*.h5"))
                if keras_files:
                    model_file = keras_files[0]
                elif h5_files:
                    model_file = h5_files[0]
                else:
                    raise FileNotFoundError(f"Không tìm thấy file model (.keras/.h5) trong {path}")
            path = model_file

        self.model = keras.models.load_model(str(path))
        self.is_trained = True
        logger.info(f"Đã load CNN model từ {path}")