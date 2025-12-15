import sys
import io
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.image_processor import ImageProcessor
from src.models.cnn import CNNModel
from src.utils.logger import logger

print("=" * 70)
print("TEST CNN MODEL VỚI ẢNH")
print("=" * 70)
print()

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} OK")
except ImportError:
    print("❌ TensorFlow chưa được cài đặt")
    print("Vui lòng cài Python 3.11 và TensorFlow trước")
    sys.exit(1)

image_processor = ImageProcessor(image_size=(224, 224))

model_path = Path("models/CNN_model")
if not model_path.exists():
    print()
    print("⚠️  Chưa có CNN model được train")
    print()
    print("Bạn có 2 lựa chọn:")
    print("1. Train CNN model với dữ liệu ảnh của bạn")
    print("2. Sử dụng pre-trained model (nếu có)")
    print()
    print("Để train CNN model:")
    print("  python pipelines/training_pipeline.py --model CNN --data <path_to_images>")
    print()

    print("Đang tạo model mẫu để demo...")
    model = CNNModel(use_transfer_learning=True, image_size=(224, 224))
    model.build((224, 224, 3))
    print("✓ Đã tạo model mẫu (chưa train, chỉ để demo)")
else:
    print(f"Đang load model từ {model_path}...")
    model = CNNModel()
    model.load(model_path)
    print("✓ Đã load model")

print()

print("=" * 70)
print("HƯỚNG DẪN SỬ DỤNG")
print("=" * 70)
print()
print("1. Chuẩn bị ảnh:")
print("   - Đặt ảnh vào thư mục: data/images/")
print("   - Format: JPG, PNG")
print("   - Ảnh từ camera giao thông")
print()
print("2. Chạy test:")
print("   python test_cnn_image.py <path_to_image>")
print()
print("3. Hoặc test với tất cả ảnh trong thư mục:")
print("   python test_cnn_image.py data/images/")
print()

if len(sys.argv) > 1:
    input_path = Path(sys.argv[1])

    if input_path.is_file():
        print(f"Đang test với ảnh: {input_path}")
        try:
            image = image_processor.load_image(input_path)
            processed = image_processor.preprocess_image(image)

            prediction = model.predict(np.array([processed]))
            probability = model.predict_proba(np.array([processed]))

            print()
            print("=" * 70)
            print("KẾT QUẢ")
            print("=" * 70)
            print(f"File: {input_path.name}")
            print(f"Có sự cố: {'CÓ' if prediction[0] else 'KHÔNG'}")
            print(f"Xác suất: {probability[0]:.4f}")
            print()

        except Exception as e:
            print(f"❌ Lỗi: {e}")

    elif input_path.is_dir():
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

        if not image_files:
            print(f"❌ Không tìm thấy ảnh trong {input_path}")
            sys.exit(1)

        print(f"Tìm thấy {len(image_files)} ảnh")
        print("Đang xử lý...")
        print()

        results = []
        for img_file in image_files:
            try:
                image = image_processor.load_image(img_file)
                processed = image_processor.preprocess_image(image)

                prediction = model.predict(np.array([processed]))
                probability = model.predict_proba(np.array([processed]))

                results.append({
                    'file': img_file.name,
                    'has_incident': bool(prediction[0]),
                    'probability': float(probability[0])
                })

                status = "⚠️ SỰ CỐ" if prediction[0] else "✓ Bình thường"
                print(f"{status} | {img_file.name} | Xác suất: {probability[0]:.4f}")

            except Exception as e:
                print(f"❌ Lỗi xử lý {img_file.name}: {e}")

        print()
        print("=" * 70)
        print("TỔNG KẾT")
        print("=" * 70)
        n_incidents = sum(1 for r in results if r['has_incident'])
        print(f"Tổng số ảnh: {len(results)}")
        print(f"Phát hiện sự cố: {n_incidents}")
        print(f"Bình thường: {len(results) - n_incidents}")
        print()