from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from src.utils.logger import logger

class BaseModel(ABC):

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):

        self.name = name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.history = None

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], **kwargs) -> None:

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

        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:

        pass

    @abstractmethod
    def save(self, path: Path) -> None:

        pass

    @abstractmethod
    def load(self, path: Path) -> None:

        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        predictions = self.predict(X)
        if predictions.ndim == 1:
            return predictions
        return predictions

    def get_model_info(self) -> Dict[str, Any]:

        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'config': self.config
        }