import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "merged" 
IMAGE_SIZE = 224
BATCH_SIZE = 32

class ModelTrainerTF:
    def __init__(self, data_path: Path, run_name: str = "traffic_incident_tf"):
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
        self.logger.info(f"Results will be saved to: {self.run_dir}")
        
        # Build Model with MobileNetV2
        self.base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        self.model = self._build_classifier()

    def _build_classifier(self):
        model = models.Sequential([
            layers.Rescaling(1./255),  # Normalize to [0,1]
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def prepare_data(self):
        """Load train, val, test datasets from merged directory"""
        if not self.data_path.exists():
            self.logger.error(f"Data directory not found: {self.data_path}")
            return False
        
        # Load train dataset
        train_dir = self.data_path / "train"
        val_dir = self.data_path / "val"
        test_dir = self.data_path / "test"
        
        if not all([train_dir.exists(), val_dir.exists(), test_dir.exists()]):
            self.logger.error("Missing train/val/test directories in merged dataset")
            return False
        
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='binary',
            shuffle=True,
            seed=42
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='binary',
            shuffle=False
        )
        
        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='binary',
            shuffle=False
        )

        # Data augmentation for training
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])

        self.train_ds = self.train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        self.val_ds = self.val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Log dataset info
        train_count = sum(1 for _ in self.train_ds.unbatch())
        val_count = sum(1 for _ in self.val_ds.unbatch())
        test_count = sum(1 for _ in self.test_ds.unbatch())
        
        self.logger.info(f"Dataset loaded: Train={train_count}, Val={val_count}, Test={test_count}")
        return True

    def fit(self, stage_name: str, epochs: int, lr: float, freeze: bool = True):
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING STAGE: {stage_name}")
        self.logger.info(f"{'='*60}")
        
        self.base_model.trainable = not freeze
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        best_model_path = self.run_dir / f"best_{stage_name.lower().replace(' ', '_')}.h5"
        
        # Callbacks
        cbs = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12, 
                restore_best_weights=True, 
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(best_model_path), 
                monitor='val_accuracy',
                save_best_only=True, 
                verbose=1
            ),
            callbacks.CSVLogger(str(self.run_dir / f"history_{stage_name.lower().replace(' ', '_')}.csv")),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=cbs
        )
        
        self._plot_history(history.history, stage_name)
        return history

    def evaluate_test(self):
        """Evaluate on test set and generate metrics"""
        self.logger.info("\n" + "="*60)
        self.logger.info("EVALUATING ON TEST SET")
        self.logger.info("="*60)
        
        # Get predictions
        y_true = []
        y_pred = []
        
        for images, labels in self.test_ds:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend((predictions > 0.5).astype(int).flatten())
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Normal', 'Incident'])
        self.logger.info("\nClassification Report:\n" + report)
        
        # Save report
        with open(self.run_dir / "test_results.txt", 'w') as f:
            f.write("TEST SET EVALUATION\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Incident'],
                    yticklabels=['Normal', 'Incident'])
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.run_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {self.run_dir / 'confusion_matrix.png'}")

    def _plot_history(self, history, title):
        """Plot training history with smoothing for better Signal-to-Noise Ratio (SNR)"""
        def smooth_curve(points, factor=0.75):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(smooth_curve(history['accuracy']), label='Train (Smoothed)', linewidth=2)
        axes[0, 0].plot(smooth_curve(history['val_accuracy']), label='Validation (Smoothed)', linewidth=2)
        axes[0, 0].plot(history['accuracy'], alpha=0.2, color='blue') # Raw data
        axes[0, 0].plot(history['val_accuracy'], alpha=0.2, color='orange') # Raw data
        axes[0, 0].set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(smooth_curve(history['loss']), label='Train (Smoothed)', linewidth=2)
        axes[0, 1].plot(smooth_curve(history['val_loss']), label='Validation (Smoothed)', linewidth=2)
        axes[0, 1].plot(history['loss'], alpha=0.2, color='blue')
        axes[0, 1].plot(history['val_loss'], alpha=0.2, color='orange')
        axes[0, 1].set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if 'precision' in history or 'Precision' in history:
            metric_key = 'precision' if 'precision' in history else 'Precision'
            val_key = f'val_{metric_key}'
            axes[1, 0].plot(smooth_curve(history[metric_key]), label='Train (Smoothed)', linewidth=2)
            if val_key in history:
                axes[1, 0].plot(smooth_curve(history[val_key]), label='Validation (Smoothed)', linewidth=2)
            axes[1, 0].set_title(f'{title} - Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history or 'Recall' in history:
            metric_key = 'recall' if 'recall' in history else 'Recall'
            val_key = f'val_{metric_key}'
            axes[1, 1].plot(smooth_curve(history[metric_key]), label='Train (Smoothed)', linewidth=2)
            if val_key in history:
                axes[1, 1].plot(smooth_curve(history[val_key]), label='Validation (Smoothed)', linewidth=2)
            axes[1, 1].set_title(f'{title} - Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.run_dir / f"{title.lower().replace(' ', '_')}_metrics.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Metrics chart saved to {chart_path}")
        plt.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "TRAFFIC INCIDENT DETECTION - TRAINING")
    print("="*70)
    print(f"\nDataset: {DATA_DIR}")
    print(f"Model: MobileNetV2 + Transfer Learning")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*70 + "\n")
    
    trainer = ModelTrainerTF(DATA_DIR)
    
    if trainer.prepare_data():
        # Stage 1: Transfer Learning (freeze backbone)
        print("\n[STAGE 1] Transfer Learning - Training classifier head only...")
        trainer.fit("Stage1_Transfer_Learning", epochs=30, lr=5e-4, freeze=True)
        
        # Stage 2: Fine Tuning (unfreeze backbone)
        print("\n[STAGE 2] Fine Tuning - Training entire model...")
        trainer.fit("Stage2_Fine_Tuning", epochs=60, lr=1e-5, freeze=False)
        
        # Evaluate on test set
        trainer.evaluate_test()
        
        # Save final model
        final_model_path = trainer.run_dir / "final_model.h5"
        trainer.model.save(final_model_path)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n Results saved to: {trainer.run_dir}")
        print(f" Model file: {final_model_path}")
        print("\nFiles generated:")
        print("  - final_model.h5 (trained model)")
        print("  - confusion_matrix.png")
        print("  - stage1_transfer_learning_metrics.png")
        print("  - stage2_fine_tuning_metrics.png")
        print("  - test_results.txt")
        print("  - train.log")
        print("="*70 + "\n")
    else:
        print("\n Training failed: Data preparation error")
        print(f"Please check if {DATA_DIR} exists and contains train/val/test folders\n")
