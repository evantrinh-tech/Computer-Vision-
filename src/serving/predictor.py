import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.models.base_model import BaseModel
from src.models.rbfnn import RBFNNModel

try:
    from src.models.ann import ANNModel
    from src.models.cnn import CNNModel
    from src.models.rnn import RNNModel
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    ANNModel = None
    CNNModel = None
    RNNModel = None
from src.data_processing.preprocessors import DataPreprocessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.utils.config import settings
from src.utils.logger import logger

class ModelPredictor:
    """
    ModelPredictor: Wrapper class để load model và thực hiện dự đoán (Inference).
    
    Chức năng:
    - Tự động nhận diện loại model (ANN, CNN, RNN...) dựa trên tên file.
    - Load model từ file (.h5, .keras, .pkl).
    - Tiền xử lý dữ liệu đầu vào (Preprocessing) giống hệt lúc train.
    - Thực hiện dự đoán (Predict).
    """

    def __init__(self):

        self.model = None
        self.model_type = None
        self.model_version = "unknown"
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        self.is_loaded = False

    def load_model(
        self,
        model_path: Path,
        model_type: Optional[str] = None
    ) -> None:

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_type is None:
            filename = model_path.name.lower()
            if 'ann' in filename:
                model_type = 'ANN'
            elif 'rbfnn' in filename or 'rbf' in filename:
                model_type = 'RBFNN'
            elif 'cnn' in filename:
                model_type = 'CNN'
            elif 'rnn' in filename or 'lstm' in filename or 'gru' in filename:
                model_type = 'RNN'
            else:
                model_type = 'ANN'

        self.model_type = model_type

        if model_type == 'ANN':
            if not HAS_TENSORFLOW or ANNModel is None:
                raise ImportError("ANN model requires TensorFlow. Please install TensorFlow or use RBFNN model.")
            self.model = ANNModel()
        elif model_type == 'RBFNN':
            self.model = RBFNNModel()
        elif model_type == 'CNN':
            if not HAS_TENSORFLOW or CNNModel is None:
                raise ImportError("CNN model requires TensorFlow. Please install TensorFlow or use RBFNN model.")
            self.model = CNNModel()
        elif model_type == 'RNN':
            if not HAS_TENSORFLOW or RNNModel is None:
                raise ImportError("RNN model requires TensorFlow. Please install TensorFlow or use RBFNN model.")
            self.model = RNNModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load(model_path)
        self.model_version = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_loaded = True

        logger.info(f"Loaded {model_type} model from {model_path}")

    def load_default_model(self) -> None:

        demo_model_path = Path("models/rbfnn_demo_model.pkl")
        if demo_model_path.exists():
            self.load_model(demo_model_path, model_type='RBFNN')
            return

        model_path = settings.model_registry_path / "ANN_model"

        if not model_path.exists():
            model_files = list(settings.model_registry_path.glob("*_model*"))
            if model_files:
                model_path = model_files[0]
            else:
                raise FileNotFoundError("No model found in registry")

        self.load_model(model_path)

    def is_model_loaded(self) -> bool:

        return self.is_loaded and self.model is not None

    def get_model_version(self) -> str:

        return self.model_version

    def get_model_info(self) -> Dict[str, Any]:

        if not self.is_model_loaded():
            return {}

        return self.model.get_model_info()

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:

        if not self.is_model_loaded():
            raise ValueError("Model chưa được load. Gọi load_model() trước.")

        df_copy = df.copy()

        min_samples = 5
        if len(df_copy) < min_samples:
            n_duplicates = (min_samples // len(df_copy)) + 1
            df_copy = pd.concat([df_copy] * n_duplicates, ignore_index=True)
            df_copy = df_copy.head(min_samples)

        df_features = self.feature_engineer.create_all_features(
            df_copy,
            include_wavelet=False
        )

        original_len = len(df)
        if len(df_features) > original_len:
            df_features = df_features.head(original_len)
        elif len(df_features) < original_len:
            last_row = df_features.iloc[-1:].copy()
            n_needed = original_len - len(df_features)
            df_features = pd.concat([df_features, pd.concat([last_row] * n_needed, ignore_index=True)], ignore_index=True)

        exclude_cols = ['timestamp', 'detector_id', 'has_incident', 'incident', 'label']
        feature_cols = [
            c for c in df_features.columns
            if c not in exclude_cols and df_features[c].dtype in ['int64', 'float64']
        ]

        X = df_features[feature_cols].values

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'prediction': int(pred),
                'probability': float(prob),
                'sample_index': i
            })

        return results