import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Cấu hình đường dẫn
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "merged" 
IMAGE_SIZE = 224
BATCH_SIZE = 32

class ResNetTrainer:
    def __init__(self, data_path: Path, run_name: str = "traffic_resnet50"):
        self.data_path = data_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT_DIR / "models" / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập Logging
        log_file = self.run_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Kết quả sẽ được lưu tại: {self.run_dir}")
        self.logger.info("Kiến trúc sử dụng: ResNet-50 (Pre-trained on ImageNet)")
        
        # Khởi tạo Base Model ResNet50
        self.base_model = ResNet50(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        self.model = self._build_classifier()

    def _build_classifier(self):
        model = models.Sequential([
            layers.Lambda(preprocess_input), # Sử dụng chuẩn hóa chuẩn của ResNet50
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def prepare_data(self):
        if not self.data_path.exists():
            self.logger.error(f"Thư mục dữ liệu không tồn tại: {self.data_path}")
            return False
        
        train_dir = self.data_path / "train"
        val_dir = self.data_path / "val"
        test_dir = self.data_path / "test"
        
        # Load datasets
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
            label_mode='binary', shuffle=True, seed=42
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
            label_mode='binary', shuffle=False
        )
        
        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
            label_mode='binary', shuffle=False
        )

        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        self.train_ds = self.train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        self.val_ds = self.val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return True

    def fit(self, stage_name: str, epochs: int, lr: float, freeze: bool = True):
        self.logger.info(f"\nSTARTING STAGE: {stage_name}")
        self.base_model.trainable = not freeze
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        best_model_path = self.run_dir / f"best_{stage_name.lower()}.h5"
        
        cbs = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            callbacks.ModelCheckpoint(filepath=str(best_model_path), monitor='val_accuracy', save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]
        
        history = self.model.fit(
            self.train_ds, validation_data=self.val_ds,
            epochs=epochs, callbacks=cbs
        )
        
        self._plot_history(history.history, stage_name)
        return history

    def evaluate_test(self):
        self.logger.info("\nEvaluating on Test Set...")
        y_true, y_pred = [], []
        for images, labels in self.test_ds:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend((preds > 0.5).astype(int).flatten())
        
        report = classification_report(y_true, y_pred, target_names=['Incident', 'Normal'])
        self.logger.info("\nClassification Report:\n" + report)
        
        # Save results
        with open(self.run_dir / "test_results.txt", 'w') as f:
            f.write(report)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Incident', 'Normal'], yticklabels=['Incident', 'Normal'])
        plt.title('ResNet-50 Confusion Matrix')
        plt.savefig(self.run_dir / 'confusion_matrix.png')
        plt.close()

    def _plot_history(self, history, title):
        def smooth(points, factor=0.8):
            smoothed = []
            for p in points:
                if smoothed: smoothed.append(smoothed[-1] * factor + p * (1 - factor))
                else: smoothed.append(p)
            return smoothed

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(smooth(history['accuracy']), label='Train')
        axes[0].plot(smooth(history['val_accuracy']), label='Val')
        axes[0].set_title(f'{title} - Accuracy')
        axes[0].legend()
        
        axes[1].plot(smooth(history['loss']), label='Train')
        axes[1].plot(smooth(history['val_loss']), label='Val')
        axes[1].set_title(f'{title} - Loss')
        axes[1].legend()
        
        plt.savefig(self.run_dir / f"{title.lower()}_metrics.png")
        plt.close()

if __name__ == '__main__':
    trainer = ResNetTrainer(DATA_DIR)
    if trainer.prepare_data():
        # Stage 1: Transfer Learning
        # Với ResNet, Stage 1 cần train kỹ hơn một chút để head bắt kịp backbone
        trainer.fit("Stage1_Transfer", epochs=30, lr=1e-3, freeze=True)
        # Stage 2: Fine Tuning
        trainer.fit("Stage2_FineTune", epochs=50, lr=1e-5, freeze=False)
        trainer.evaluate_test()
        trainer.model.save(trainer.run_dir / "final_resnet50.h5")
        print(f"\nResNet-50 Training Done. Results in: {trainer.run_dir}")
