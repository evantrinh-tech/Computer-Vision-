import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime

# Using relative paths for better portability
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "images"
IMAGE_SIZE = 224
BATCH_SIZE = 16

class ModelTrainerTF:
    def __init__(self, data_path: Path, run_name: str = "traffic_pro_tf"):
        self.data_path = data_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT_DIR / "models" / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.run_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Results (TF) will be saved to: {self.run_dir}")
        
        # Build Model
        self.base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        self.model = self._build_classifier()

    def _build_classifier(self):
        model = models.Sequential([
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def prepare_data(self):
        if not self.data_path.exists():
            self.logger.error(f"Data directory not found: {self.data_path}")
            return False
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='binary'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='binary'
        )

        # Augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ])

        self.train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        self.logger.info("Data preparation completed.")
        return True

    def fit(self, stage_name: str, epochs: int, lr: float, freeze: bool = True):
        self.logger.info(f"STARTING STAGE: {stage_name}")
        
        self.base_model.trainable = not freeze
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        best_model_path = self.run_dir / f"best_{stage_name.lower().replace(' ', '_')}.h5"
        
        # Callbacks
        cbs = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
            callbacks.ModelCheckpoint(filepath=str(best_model_path), save_best_only=True, verbose=1),
            callbacks.CSVLogger(str(self.run_dir / f"history_{stage_name.lower()}.csv"))
        ]
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=cbs
        )
        
        self._plot_smooth(history.history, stage_name)

    def _plot_smooth(self, history, title):
        def smooth(data, w=0.6):
            res = [data[0]]
            for p in data[1:]: 
                res.append(res[-1]*w + p*(1-w))
            return res

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(smooth(history['accuracy']), label='Train')
        plt.plot(smooth(history['val_accuracy']), label='Val')
        plt.title(f"{title} - Accuracy")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(smooth(history['loss']), label='Train')
        plt.plot(smooth(history['val_loss']), label='Val')
        plt.title(f"{title} - Loss")
        plt.legend()
        
        chart_path = self.run_dir / f"{title.lower().replace(' ', '_')}_chart.png"
        plt.savefig(chart_path)
        self.logger.info(f"Chart saved to {chart_path}")
        plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("ITS - TRAFFIC INCIDENT DETECTION (TENSORFLOW VERSION)")
    print("=" * 60)
    
    trainer = ModelTrainerTF(DATA_DIR)
    if trainer.prepare_data():
        # Start training process
        # Stage 1: Transfer Learning (Heads only)
        trainer.fit("Transfer_Learning", epochs=20, lr=1e-3, freeze=True)
        
        # Stage 2: Fine Tuning (Whole model)
        trainer.fit("Fine_Tuning", epochs=50, lr=1e-5, freeze=False)
        
        print("\nTraining completed successfully.")
        print(f"Models (.h5) and logs are available in: {trainer.run_dir}")
    else:
        print("\nTraining failed due to missing data.")
