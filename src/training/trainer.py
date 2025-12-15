import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import cv2

from src.models import ANNModel, RBFNNModel, CNNModel, RNNModel
from src.data_processing.preprocessors import DataPreprocessor, TimeSeriesPreprocessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.image_processor import ImageProcessor
from src.training.evaluator import ModelEvaluator
from src.utils.config import settings
from src.utils.logger import logger

class ModelTrainer:

    def __init__(self, model_type: str = 'ANN', config_path: Optional[Path] = None):

        self.model_type = model_type.upper()
        self.config = self._load_config(config_path) if config_path else {}
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.evaluator = ModelEvaluator()
        self.use_mlflow = True

        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            logger.info(f"✅ MLflow initialized: {settings.mlflow_tracking_uri}")
        except Exception as e:
            self.use_mlflow = False
            logger.warning(f"⚠️ MLflow không khả dụng (training vẫn tiếp tục): {str(e)[:100]}")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_model(self) -> Any:

        model_config = self.config.get('model', {})

        if self.model_type == 'ANN':
            return ANNModel(
                hidden_layers=model_config.get('hidden_layers', [64, 32]),
                activation=model_config.get('activation', 'relu'),
                dropout_rate=model_config.get('dropout_rate', 0.2),
                learning_rate=model_config.get('learning_rate', 0.001)
            )
        elif self.model_type == 'RBFNN':
            return RBFNNModel(
                n_centers=model_config.get('n_centers', 20),
                sigma=model_config.get('sigma', 1.0),
                learning_rate=model_config.get('learning_rate', 0.01),
                use_wavelet=model_config.get('use_wavelet', True)
            )
        elif self.model_type == 'CNN':
            return CNNModel(
                use_transfer_learning=model_config.get('use_transfer_learning', True),
                image_size=tuple(model_config.get('image_size', [224, 224])),
                learning_rate=model_config.get('learning_rate', 0.001)
            )
        elif self.model_type == 'RNN':
            return RNNModel(
                rnn_type=model_config.get('rnn_type', 'LSTM'),
                hidden_units=model_config.get('hidden_units', [64, 32]),
                dropout_rate=model_config.get('dropout_rate', 0.2),
                learning_rate=model_config.get('learning_rate', 0.001)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_data(
        self,
        df: Optional[pd.DataFrame] = None,
        data_path: Optional[Path] = None,
        target_col: str = 'has_incident',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if self.model_type == 'CNN' and data_path is not None:
            return self._prepare_image_data(data_path, test_size, val_size)

        if df is None:
            raise ValueError("Cần cung cấp df cho sensor data models hoặc data_path cho CNN model")

        from sklearn.model_selection import train_test_split

        self.feature_engineer = FeatureEngineer()
        use_wavelet = self.model_type == 'RBFNN'
        df_features = self.feature_engineer.create_all_features(df, include_wavelet=use_wavelet)

        feature_cols = [c for c in df_features.columns if c not in [target_col, 'timestamp', 'detector_id']]
        X = df_features[feature_cols].values
        y = df_features[target_col].values if target_col in df_features.columns else np.zeros(len(df))

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        self.preprocessor = DataPreprocessor(
            scaling_method=self.config.get('preprocessing', {}).get('scaling_method', 'standard')
        )
        X_train = self.preprocessor.fit_transform(pd.DataFrame(X_train, columns=feature_cols))
        X_val = self.preprocessor.transform(pd.DataFrame(X_val, columns=feature_cols))
        X_test = self.preprocessor.transform(pd.DataFrame(X_test, columns=feature_cols))

        logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _prepare_image_data(
        self,
        data_path: Path,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        from sklearn.model_selection import train_test_split

        data_path = Path(data_path)
        image_processor = ImageProcessor(image_size=(224, 224))

        if data_path.is_file():
            data_path = data_path.parent
            logger.info(f"Đã nhận file, sử dụng thư mục: {data_path}")

        normal_dir = None
        incident_dir = None

        if (data_path / "normal").exists():
            normal_dir = data_path / "normal"
        if (data_path / "incident").exists():
            incident_dir = data_path / "incident"

        if data_path.name in ["normal", "incident"]:
            parent = data_path.parent
            if (parent / "normal").exists():
                normal_dir = parent / "normal"
            if (parent / "incident").exists():
                incident_dir = parent / "incident"

        images_dir = data_path
        if data_path.name != "images":
            for parent in [data_path, data_path.parent, data_path.parent.parent]:
                if (parent / "images").exists():
                    images_dir = parent / "images"
                    break

        if not normal_dir and (images_dir / "normal").exists():
            normal_dir = images_dir / "normal"
        if not incident_dir and (images_dir / "incident").exists():
            incident_dir = images_dir / "incident"

        images = []
        labels = []

        if normal_dir and normal_dir.exists():
            image_files = (
                list(normal_dir.glob("*.jpg")) +
                list(normal_dir.glob("*.jpeg")) +
                list(normal_dir.glob("*.png")) +
                list(normal_dir.glob("*.webp")) +
                list(normal_dir.glob("*.gif"))
            )
            logger.info(f"Tìm thấy {len(image_files)} ảnh trong {normal_dir}")
            for img_file in image_files:
                try:
                    image = image_processor.load_image(img_file)
                    processed = image_processor.preprocess_image(image)
                    images.append(processed)
                    labels.append(0)
                except Exception as e:
                    logger.warning(f"Không thể load ảnh {img_file}: {e}")

        if incident_dir and incident_dir.exists():
            image_files = (
                list(incident_dir.glob("*.jpg")) +
                list(incident_dir.glob("*.jpeg")) +
                list(incident_dir.glob("*.png")) +
                list(incident_dir.glob("*.webp")) +
                list(incident_dir.glob("*.gif"))
            )
            logger.info(f"Tìm thấy {len(image_files)} ảnh trong {incident_dir}")
            for img_file in image_files:
                try:
                    image = image_processor.load_image(img_file)
                    processed = image_processor.preprocess_image(image)
                    images.append(processed)
                    labels.append(1)
                except Exception as e:
                    logger.warning(f"Không thể load ảnh {img_file}: {e}")

        if len(images) == 0:
            raise ValueError(
                f"Không tìm thấy ảnh trong {data_path}\n"
                f"Vui lòng đảm bảo có thư mục normal/ và incident/ chứa ảnh"
            )

        logger.info(f"Đã load {len(images)} ảnh ({sum(1 for l in labels if l == 0)} normal, {sum(1 for l in labels if l == 1)} incident)")

        X = np.array(images)
        y = np.array(labels)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        logger.info(f"Image data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:

        self.model = self._create_model()
        if self.model_type == 'CNN':
            self.model.build(X_train.shape[1:])
        else:
            self.model.build((X_train.shape[1],))

        training_config = self.config.get('training', {})
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=training_config.get('epochs', 100),
            batch_size=training_config.get('batch_size', 32)
        )

        train_metrics = self.evaluator.evaluate(self.model, X_train, y_train)
        val_metrics = self.evaluator.evaluate(self.model, X_val, y_val) if X_val is not None else {}

        model_path = settings.model_registry_path / f"{self.model_type}_model"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)

        if self.use_mlflow:
            try:
                import uuid
                default_run_name = run_name or f"{self.model_type}_{uuid.uuid4().hex[:8]}"
                with mlflow.start_run(run_name=default_run_name):
                    mlflow.log_params(self.config.get('model', {}))
                    mlflow.log_param('model_type', self.model_type)
                    mlflow.log_param('n_train_samples', len(X_train))
                    mlflow.log_param('n_features', X_train.shape[1] if len(X_train.shape) > 1 else X_train.shape[0])

                    for metric, value in train_metrics.items():
                        mlflow.log_metric(f'train_{metric}', value)
                    for metric, value in val_metrics.items():
                        mlflow.log_metric(f'val_{metric}', value)

                    try:
                        if self.model_type in ['ANN', 'CNN', 'RNN']:
                            mlflow.tensorflow.log_model(self.model.model, "model")
                        else:
                            mlflow.sklearn.log_model(self.model, "model")
                        mlflow.log_artifacts(str(model_path.parent), "models")
                    except Exception as e:
                        logger.warning(f"Không thể log model vào MLflow: {str(e)[:100]}")
            except Exception as e:
                logger.warning(f"Không thể log vào MLflow (training vẫn thành công): {str(e)[:100]}")
        else:
            logger.info("Training hoàn thành (không có MLflow tracking)")

        logger.info(f"Training completed. Train accuracy: {train_metrics.get('accuracy', 0):.4f}")

        return {
            'history': history,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_path': model_path
        }

    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:

        if self.model is None:
            raise ValueError("Model chưa được train.")

        metrics = self.evaluator.evaluate(self.model, X_test, y_test)

        logger.info("Test set evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Train traffic incident detection model')
    parser.add_argument('--model', type=str, default='ANN', choices=['ANN', 'RBFNN', 'CNN', 'RNN'])
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--run-name', type=str, help='MLflow run name')

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    trainer = ModelTrainer(model_type=args.model, config_path=Path(args.config) if args.config else None)
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(df)

    trainer.train(X_train, y_train, X_val, y_val, run_name=args.run_name)

    trainer.evaluate_on_test(X_test, y_test)

    logger.info("Training pipeline completed!")

if __name__ == '__main__':
    main()