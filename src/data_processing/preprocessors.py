import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

from src.utils.logger import logger

class DataPreprocessor:

    def __init__(
        self,
        scaling_method: str = 'standard',
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0
    ):

        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold

        self.scaler = None
        self.imputer = None
        self.feature_columns = None

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> None:

        if feature_columns is None:
            exclude_cols = ['timestamp', 'detector_id', 'has_incident', 'incident', 'label']
            feature_columns = [
                col for col in df.columns
                if col not in exclude_cols and df[col].dtype in ['int64', 'float64']
            ]

        self.feature_columns = feature_columns

        self.imputer = SimpleImputer(strategy='median')
        X = df[feature_columns].copy()
        self.imputer.fit(X)

        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)

        logger.info(f"Đã fit preprocessor với {len(feature_columns)} features")

    def transform(self, df: pd.DataFrame) -> np.ndarray:

        if self.feature_columns is None:
            raise ValueError("Preprocessor chưa được fit. Gọi fit() trước.")

        X = df[self.feature_columns].copy()

        if self.handle_outliers:
            X = self._handle_outliers(X)

        X_imputed = self.imputer.transform(X)

        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def fit_transform(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:

        self.fit(df, feature_columns)
        return self.transform(df)

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        df_clean = df.copy()

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.outlier_threshold

                if outliers.any():
                    median_value = df[col].median()
                    df_clean.loc[outliers, col] = median_value
                    logger.debug(f"Đã xử lý {outliers.sum()} outliers trong cột {col}")

        return df_clean

    def get_feature_names(self) -> List[str]:

        return self.feature_columns or []

class TimeSeriesPreprocessor:

    def __init__(self, window_size: int = 10, step_size: int = 1):

        self.window_size = window_size
        self.step_size = step_size

    def create_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        sequences = []
        sequence_labels = []

        for i in range(0, len(data) - self.window_size + 1, self.step_size):
            seq = data[i:i + self.window_size]
            sequences.append(seq)

            if labels is not None:
                sequence_labels.append(labels[i + self.window_size - 1])

        X = np.array(sequences)
        y = np.array(sequence_labels) if labels is not None else None

        return X, y