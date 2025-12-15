import pytest
import numpy as np
import pandas as pd

from src.data_processing.preprocessors import DataPreprocessor, TimeSeriesPreprocessor

class TestDataPreprocessor:

    def test_fit_transform(self):

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'label': [0, 1, 0, 1, 0]
        })

        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(df, feature_columns=['feature1', 'feature2'])

        assert X.shape == (5, 2)
        assert not np.isnan(X).any()

    def test_handle_outliers(self):

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 1000, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        preprocessor = DataPreprocessor(handle_outliers=True, outlier_threshold=3.0)
        preprocessor.fit(df, feature_columns=['feature1', 'feature2'])
        X = preprocessor.transform(df)

        assert X.shape == (5, 2)

class TestTimeSeriesPreprocessor:

    def test_create_sequences(self):

        data = np.random.randn(100, 5)
        labels = np.random.randint(0, 2, 100)

        preprocessor = TimeSeriesPreprocessor(window_size=10, step_size=1)
        X_seq, y_seq = preprocessor.create_sequences(data, labels)

        assert X_seq.shape[0] == 91
        assert X_seq.shape[1] == 10
        assert X_seq.shape[2] == 5
        assert len(y_seq) == 91