import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("DEMO ĐẦY ĐỦ: Hệ thống Phát hiện Sự cố Giao thông")
print("=" * 70)
print()

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} đã được cài đặt")
    HAS_TENSORFLOW = True
except ImportError:
    print("⚠️  TensorFlow chưa được cài đặt")
    print("   Chỉ có thể chạy RBFNN model")
    HAS_TENSORFLOW = False
    print()

from src.data_processing.collectors import SimulatedDataCollector
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.preprocessors import DataPreprocessor
from src.models.rbfnn import RBFNNModel
from src.training.evaluator import ModelEvaluator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if HAS_TENSORFLOW:
    try:
        from src.models.ann import ANNModel
        from src.models.rnn import RNNModel
        HAS_TF_MODELS = True
    except ImportError as e:
        print(f"⚠️  Không thể import TensorFlow models: {e}")
        HAS_TF_MODELS = False
else:
    HAS_TF_MODELS = False

print("\n[1/6] Đang tạo dữ liệu mô phỏng...")
collector = SimulatedDataCollector(seed=42)
df_normal = collector.generate_sensor_data(n_samples=800, has_incident=False)
df_incident = collector.generate_sensor_data(n_samples=200, has_incident=True)
df = pd.concat([df_normal, df_incident], ignore_index=True)
print(f"✓ Đã tạo {len(df)} samples ({df['has_incident'].sum()} có sự cố)")

print("\n[2/6] Đang tạo features...")
feature_engineer = FeatureEngineer()
df_features = feature_engineer.create_all_features(df, include_wavelet=False)
print(f"✓ Đã tạo {len(df_features.columns)} features")

print("\n[3/6] Đang chuẩn bị dữ liệu...")
exclude_cols = ['timestamp', 'detector_id', 'has_incident']
feature_cols = [c for c in df_features.columns if c not in exclude_cols and df_features[c].dtype in ['int64', 'float64']]

X = df_features[feature_cols].values
y = df_features['has_incident'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

preprocessor = DataPreprocessor(scaling_method='standard')
X_train = preprocessor.fit_transform(pd.DataFrame(X_train, columns=feature_cols))
X_val = preprocessor.transform(pd.DataFrame(X_val, columns=feature_cols))
X_test = preprocessor.transform(pd.DataFrame(X_test, columns=feature_cols))

print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

evaluator = ModelEvaluator()
results = {}

print("\n[4/6] Đang train RBFNN model...")
model_rbfnn = RBFNNModel(n_centers=15, sigma=1.0, use_wavelet=False)
model_rbfnn.build(X_train.shape[1:])
model_rbfnn.train(X_train, y_train, X_val, y_val, epochs=1)
test_metrics = evaluator.evaluate(model_rbfnn, X_test, y_test)
results['RBFNN'] = test_metrics
print(f"✓ RBFNN - Accuracy: {test_metrics['accuracy']:.4f}, DR: {test_metrics['detection_rate']:.4f}")

if HAS_TF_MODELS:
    print("\n[5/6] Đang train ANN model...")
    try:
        model_ann = ANNModel(hidden_layers=[64, 32], dropout_rate=0.2)
        model_ann.build(X_train.shape[1:])
        model_ann.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)
        test_metrics = evaluator.evaluate(model_ann, X_test, y_test)
        results['ANN'] = test_metrics
        print(f"✓ ANN - Accuracy: {test_metrics['accuracy']:.4f}, DR: {test_metrics['detection_rate']:.4f}")
    except Exception as e:
        print(f"⚠️  Lỗi train ANN: {e}")

    print("\n[6/6] Đang train RNN model...")
    try:
        from src.data_processing.preprocessors import TimeSeriesPreprocessor
        ts_preprocessor = TimeSeriesPreprocessor(window_size=10, step_size=1)
        X_train_seq, y_train_seq = ts_preprocessor.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = ts_preprocessor.create_sequences(X_test, y_test)

        if len(X_train_seq) > 0:
            model_rnn = RNNModel(rnn_type='LSTM', hidden_units=[64, 32])
            model_rnn.build(X_train_seq.shape[1:])
            model_rnn.train(X_train_seq, y_train_seq, epochs=30, batch_size=32, verbose=0)
            test_metrics = evaluator.evaluate(model_rnn, X_test_seq, y_test_seq)
            results['RNN'] = test_metrics
            print(f"✓ RNN - Accuracy: {test_metrics['accuracy']:.4f}, DR: {test_metrics['detection_rate']:.4f}")
        else:
            print("⚠️  Không đủ dữ liệu để tạo sequences cho RNN")
    except Exception as e:
        print(f"⚠️  Lỗi train RNN: {e}")
else:
    print("\n[5-6/6] Bỏ qua ANN và RNN (cần TensorFlow)")

print("\n" + "=" * 70)
print("KẾT QUẢ SO SÁNH CÁC MODELS")
print("=" * 70)
print(f"\n{'Model':<10} {'Accuracy':<12} {'DR':<12} {'FAR':<12} {'F1-Score':<12}")
print("-" * 70)

for model_name, metrics in results.items():
    print(f"{model_name:<10} {metrics['accuracy']:<12.4f} {metrics['detection_rate']:<12.4f} "
          f"{metrics['false_alarm_rate']:<12.4f} {metrics['f1_score']:<12.4f}")

print("\n" + "=" * 70)
print("DEMO HOÀN THÀNH!")
print("=" * 70)

if not HAS_TENSORFLOW:
    print("\n💡 Để chạy đầy đủ tất cả models, vui lòng cài TensorFlow:")
    print("   1. Cài Python 3.11")
    print("   2. Chạy: .\\setup_tensorflow.ps1")
    print("   3. Chạy lại script này")