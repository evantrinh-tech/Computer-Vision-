import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.collectors import SimulatedDataCollector
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.preprocessors import DataPreprocessor
from src.models.rbfnn import RBFNNModel
from src.training.evaluator import ModelEvaluator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 60)
print("DEMO: Hệ thống Phát hiện Sự cố Giao thông")
print("=" * 60)

print("\n[1/5] Đang tạo dữ liệu mô phỏng...")
collector = SimulatedDataCollector(seed=42)
df_normal = collector.generate_sensor_data(n_samples=800, has_incident=False)
df_incident = collector.generate_sensor_data(n_samples=200, has_incident=True)
df = pd.concat([df_normal, df_incident], ignore_index=True)
print(f"✓ Đã tạo {len(df)} samples ({df['has_incident'].sum()} có sự cố)")

print("\n[2/5] Đang tạo features...")
feature_engineer = FeatureEngineer()
df_features = feature_engineer.create_all_features(df, include_wavelet=False)
print(f"✓ Đã tạo {len(df_features.columns)} features")

print("\n[3/5] Đang chuẩn bị dữ liệu...")
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

print("\n[4/5] Đang train RBFNN model...")
model = RBFNNModel(
    n_centers=15,
    sigma=1.0,
    use_wavelet=False
)
model.build(X_train.shape[1:])

training_results = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=1
)
print("✓ Training hoàn thành!")

print("\n[5/5] Đang đánh giá mô hình...")
evaluator = ModelEvaluator()

train_metrics = evaluator.evaluate(model, X_train, y_train)
val_metrics = evaluator.evaluate(model, X_val, y_val)
test_metrics = evaluator.evaluate(model, X_test, y_test)

print("\n" + "=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ")
print("=" * 60)
print(f"\n📊 Training Set:")
print(f"   - Accuracy: {train_metrics['accuracy']:.4f}")
print(f"   - Detection Rate: {train_metrics['detection_rate']:.4f}")
print(f"   - False Alarm Rate: {train_metrics['false_alarm_rate']:.4f}")

print(f"\n📊 Validation Set:")
print(f"   - Accuracy: {val_metrics['accuracy']:.4f}")
print(f"   - Detection Rate: {val_metrics['detection_rate']:.4f}")
print(f"   - False Alarm Rate: {val_metrics['false_alarm_rate']:.4f}")

print(f"\n📊 Test Set:")
print(f"   - Accuracy: {test_metrics['accuracy']:.4f}")
print(f"   - Detection Rate: {test_metrics['detection_rate']:.4f}")
print(f"   - False Alarm Rate: {test_metrics['false_alarm_rate']:.4f}")
print(f"   - Precision: {test_metrics['precision']:.4f}")
print(f"   - Recall: {test_metrics['recall']:.4f}")
print(f"   - F1-Score: {test_metrics['f1_score']:.4f}")

model_path = Path("models/rbfnn_demo_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(model_path)
print(f"\n✓ Đã lưu model tại: {model_path}")

print("\n" + "=" * 60)
print("DEMO HOÀN THÀNH!")
print("=" * 60)
print("\nBạn có thể:")
print("1. Xem model đã được lưu tại: models/rbfnn_demo_model.pkl")
print("2. Chạy API server: uvicorn src.serving.api:app --reload")
print("3. Xem documentation tại: docs/")