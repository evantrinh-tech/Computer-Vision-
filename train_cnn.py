import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ModelTrainer
from src.utils.logger import logger
from src.utils.config import settings
import mlflow

def main():

    print("=" * 60)
    print("HUẤN LUYỆN MÔ HÌNH CNN - PHÁT HIỆN SỰ CỐ GIAO THÔNG")
    print("=" * 60)
    print()

    data_path = Path("data/images")

    if not data_path.exists():
        print(f"❌ Lỗi: Không tìm thấy folder {data_path}")
        print("Vui lòng đảm bảo có folder data/images/ với 2 subfolder:")
        print("  - data/images/normal/ (chứa ảnh bình thường)")
        print("  - data/images/incident/ (chứa ảnh có sự cố)")
        return

    normal_dir = data_path / "normal"
    incident_dir = data_path / "incident"

    if not normal_dir.exists():
        print(f"❌ Lỗi: Không tìm thấy folder {normal_dir}")
        return

    if not incident_dir.exists():
        print(f"❌ Lỗi: Không tìm thấy folder {incident_dir}")
        return

    normal_images = (
        list(normal_dir.glob("*.jpg")) +
        list(normal_dir.glob("*.jpeg")) +
        list(normal_dir.glob("*.png")) +
        list(normal_dir.glob("*.webp")) +
        list(normal_dir.glob("*.gif"))
    )
    incident_images = (
        list(incident_dir.glob("*.jpg")) +
        list(incident_dir.glob("*.jpeg")) +
        list(incident_dir.glob("*.png")) +
        list(incident_dir.glob("*.webp")) +
        list(incident_dir.glob("*.gif"))
    )

    print(f"📁 Đã tìm thấy:")
    print(f"   - {len(normal_images)} ảnh bình thường (normal)")
    print(f"   - {len(incident_images)} ảnh có sự cố (incident)")
    print(f"   - Tổng cộng: {len(normal_images) + len(incident_images)} ảnh")
    print()

    if len(normal_images) == 0 and len(incident_images) == 0:
        print("❌ Lỗi: Không tìm thấy ảnh nào trong các folder")
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    config_path = Path("configs/training_config.yaml")
    if not config_path.exists():
        config_path = None
        print("⚠️  Không tìm thấy config file, sử dụng cấu hình mặc định")

    print("🔧 Đang khởi tạo trainer...")
    trainer = ModelTrainer(model_type='CNN', config_path=config_path)
    print("✅ Đã khởi tạo trainer")
    print()

    print("📊 Đang chuẩn bị dữ liệu...")
    print("   (Đang load và xử lý ảnh, có thể mất vài phút...)")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
            data_path=data_path,
            test_size=0.2,
            val_size=0.1
        )
        print("✅ Đã chuẩn bị xong dữ liệu")
        print(f"   - Training set: {len(X_train)} ảnh")
        print(f"   - Validation set: {len(X_val)} ảnh")
        print(f"   - Test set: {len(X_test)} ảnh")
        print()
    except Exception as e:
        print(f"❌ Lỗi khi chuẩn bị dữ liệu: {e}")
        logger.exception("Lỗi khi chuẩn bị dữ liệu")
        return

    print("🚀 Bắt đầu huấn luyện mô hình...")
    print("   (Quá trình này có thể mất nhiều thời gian tùy vào số lượng ảnh và cấu hình)")
    print()

    try:
        training_results = trainer.train(
            X_train, y_train,
            X_val, y_val,
            run_name="CNN_training_from_images"
        )

        print()
        print("✅ Đã hoàn thành huấn luyện!")
        print()

        print("📈 Đang đánh giá mô hình trên test set...")
        test_metrics = trainer.evaluate_on_test(X_test, y_test)

        print()
        print("=" * 60)
        print("KẾT QUẢ HUẤN LUYỆN")
        print("=" * 60)
        print()
        print("📊 Metrics trên Test Set:")
        for metric, value in test_metrics.items():
            print(f"   - {metric}: {value:.4f}")
        print()

        model_path = training_results.get('model_path')
        if model_path:
            print(f"💾 Mô hình đã được lưu tại: {model_path}")
        print()

        print("✅ Hoàn tất!")

    except Exception as e:
        print(f"❌ Lỗi khi huấn luyện: {e}")
        logger.exception("Lỗi khi huấn luyện")
        return

if __name__ == '__main__':
    main()