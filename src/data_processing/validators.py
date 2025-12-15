import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import logger

class DataValidator:

    def __init__(
        self,
        schema: Optional[Dict] = None,
        range_checks: Optional[Dict] = None
    ):

        self.schema = schema or self._default_schema()
        self.range_checks = range_checks or self._default_range_checks()
        self.validation_errors = []

    def _default_schema(self) -> Dict:

        return {
            'timestamp': 'datetime64[ns]',
            'detector_id': 'object',
            'volume': 'float64',
            'speed': 'float64',
            'occupancy': 'float64'
        }

    def _default_range_checks(self) -> Dict:

        return {
            'volume': (0, 10000),
            'speed': (0, 200),
            'occupancy': (0, 1)
        }

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:

        errors = []

        for col, dtype in self.schema.items():
            if col not in df.columns:
                errors.append(f"Thiếu cột bắt buộc: {col}")
            else:
                expected_dtype = str(dtype)
                actual_dtype = str(df[col].dtype)

                if 'int' in expected_dtype and 'float' in actual_dtype:
                    pass
                elif 'float' in expected_dtype and 'int' in actual_dtype:
                    pass
                elif expected_dtype not in actual_dtype and actual_dtype not in expected_dtype:
                    errors.append(
                        f"Cột {col}: kiểu dữ liệu không khớp. "
                        f"Kỳ vọng: {expected_dtype}, Thực tế: {actual_dtype}"
                    )

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Schema validation failed: {errors}")

        return is_valid, errors

    def validate_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:

        errors = []

        for col, (min_val, max_val) in self.range_checks.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                n_outliers = out_of_range.sum()

                if n_outliers > 0:
                    errors.append(
                        f"Cột {col}: {n_outliers} giá trị ngoài range "
                        f"[{min_val}, {max_val}]"
                    )

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Range validation failed: {errors}")

        return is_valid, errors

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        max_missing_ratio: float = 0.5
    ) -> Tuple[bool, List[str]]:

        errors = []

        for col in df.columns:
            missing_ratio = df[col].isna().sum() / len(df)

            if missing_ratio > max_missing_ratio:
                errors.append(
                    f"Cột {col}: tỷ lệ missing quá cao ({missing_ratio:.2%})"
                )

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Missing values validation failed: {errors}")

        return is_valid, errors

    def validate_timestamp(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> Tuple[bool, List[str]]:

        errors = []

        if timestamp_col not in df.columns:
            errors.append(f"Không tìm thấy cột timestamp: {timestamp_col}")
            return False, errors

        try:
            timestamps = pd.to_datetime(df[timestamp_col])
        except Exception as e:
            errors.append(f"Lỗi parse timestamp: {e}")
            return False, errors

        now = datetime.now()
        future_timestamps = timestamps > now
        if future_timestamps.any():
            n_future = future_timestamps.sum()
            errors.append(f"Có {n_future} timestamp trong tương lai")

        one_year_ago = now.replace(year=now.year - 1)
        old_timestamps = timestamps < one_year_ago
        if old_timestamps.any():
            n_old = old_timestamps.sum()
            errors.append(f"Có {n_old} timestamp quá cũ (>1 năm)")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_all(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[bool, Dict[str, List[str]]]:

        all_errors = {}

        is_valid, errors = self.validate_schema(df)
        all_errors['schema'] = errors
        if strict and not is_valid:
            return False, all_errors

        is_valid, errors = self.validate_ranges(df)
        all_errors['ranges'] = errors
        if strict and not is_valid:
            return False, all_errors

        is_valid, errors = self.validate_missing_values(df)
        all_errors['missing'] = errors
        if strict and not is_valid:
            return False, all_errors

        is_valid, errors = self.validate_timestamp(df)
        all_errors['timestamp'] = errors
        if strict and not is_valid:
            return False, all_errors

        total_errors = sum(len(errors) for errors in all_errors.values())
        is_valid = total_errors == 0

        if is_valid:
            logger.info("Tất cả validation đều pass")
        else:
            logger.warning(f"Validation failed với {total_errors} lỗi")

        return is_valid, all_errors