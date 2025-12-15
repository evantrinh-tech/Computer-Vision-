import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import pywt

from src.utils.logger import logger

class FeatureEngineer:

    def __init__(self):

        pass

    def create_upstream_downstream_features(
        self,
        df: pd.DataFrame,
        detector_id_col: str = 'detector_id',
        volume_col: str = 'volume',
        speed_col: str = 'speed',
        occupancy_col: str = 'occupancy'
    ) -> pd.DataFrame:

        df = df.copy()
        df = df.sort_values(['timestamp', detector_id_col])

        df['volume_diff'] = df.groupby('timestamp')[volume_col].diff()
        df['volume_diff_abs'] = df['volume_diff'].abs()

        df['speed_diff'] = df.groupby('timestamp')[speed_col].diff()
        df['speed_diff_abs'] = df['speed_diff'].abs()

        df['occupancy_diff'] = df.groupby('timestamp')[occupancy_col].diff()
        df['occupancy_diff_abs'] = df['occupancy_diff'].abs()

        df['volume_ratio'] = df.groupby('timestamp')[volume_col].pct_change() + 1
        df['speed_ratio'] = df.groupby('timestamp')[speed_col].pct_change() + 1

        df = df.fillna(0)

        logger.info("Đã tạo upstream-downstream features")
        return df

    def create_statistical_features(
        self,
        df: pd.DataFrame,
        window_size: int = 5,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:

        df = df.copy()
        df = df.sort_values('timestamp')

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c not in ['has_incident', 'incident', 'label']]

        for col in columns:
            if col in df.columns:
                df[f'{col}_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_std'] = df[col].rolling(window=window_size, min_periods=1).std()
                df[f'{col}_min'] = df[col].rolling(window=window_size, min_periods=1).min()
                df[f'{col}_max'] = df[col].rolling(window=window_size, min_periods=1).max()
                df[f'{col}_median'] = df[col].rolling(window=window_size, min_periods=1).median()

        df = df.fillna(method='bfill').fillna(0)

        logger.info(f"Đã tạo statistical features cho {len(columns)} cột")
        return df

    def create_wavelet_features(
        self,
        signal: np.ndarray,
        wavelet: str = 'db4',
        level: int = 3
    ) -> Dict[str, np.ndarray]:

        try:
            coeffs = pywt.wavedec(signal, wavelet, level=level)

            features = {
                'approximation': coeffs[0],
                'details': coeffs[1:]
            }

            features['approx_mean'] = np.mean(coeffs[0])
            features['approx_std'] = np.std(coeffs[0])
            features['approx_energy'] = np.sum(coeffs[0] ** 2)

            for i, detail in enumerate(coeffs[1:], 1):
                features[f'detail_{i}_mean'] = np.mean(detail)
                features[f'detail_{i}_std'] = np.std(detail)
                features[f'detail_{i}_energy'] = np.sum(detail ** 2)

            return features

        except Exception as e:
            logger.error(f"Lỗi trong wavelet transform: {e}")
            return {}

    def create_temporal_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        logger.info("Đã tạo temporal features")
        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        include_wavelet: bool = False
    ) -> pd.DataFrame:

        df = df.copy()

        df = self.create_temporal_features(df)

        df = self.create_upstream_downstream_features(df)

        df = self.create_statistical_features(df)

        if include_wavelet:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:
                if col not in ['has_incident', 'incident', 'label']:
                    signal = df[col].values
                    wavelet_features = self.create_wavelet_features(signal)

                    for key, value in wavelet_features.items():
                        if isinstance(value, np.ndarray):
                            if len(value) == len(df):
                                df[f'{col}_{key}'] = value
                        else:
                            df[f'{col}_{key}'] = value

        logger.info(f"Đã tạo tất cả features. Tổng số features: {len(df.columns)}")
        return df