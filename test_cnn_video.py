import sys
import io
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.image_processor import ImageProcessor, VideoProcessor
from src.models.cnn import CNNModel
from src.utils.logger import logger

print("=" * 70)
print("TEST CNN MODEL VỚI VIDEO")
print("=" * 70)
print()

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} OK")
except ImportError:
    print("❌ TensorFlow chưa được cài đặt")
    sys.exit(1)

video_processor = VideoProcessor(image_size=(224, 224), frame_skip=5)

model_path = Path("models/CNN_model")
if not model_path.exists():
    print("⚠️  Chưa có CNN model được train")
    print("Vui lòng train CNN model trước")
    sys.exit(1)

print(f"Đang load model từ {model_path}...")
model = CNNModel()
model.load(model_path)
print("✓ Đã load model")
print()

if len(sys.argv) < 2:
    print("=" * 70)
    print("HƯỚNG DẪN SỬ DỤNG")
    print("=" * 70)
    print()
    print("Cách dùng:")
    print("  python test_cnn_video.py <path_to_video>")
    print()
    print("Ví dụ:")
    print("  python test_cnn_video.py data/videos/traffic_incident.mp4")
    print()
    print("Video sẽ được xử lý và hiển thị:")
    print("  - Các frame có sự cố")
    print("  - Timestamp của sự cố")
    print("  - Xác suất phát hiện")
    print()
    sys.exit(0)

video_path = Path(sys.argv[1])

if not video_path.exists():
    print(f"❌ Không tìm thấy file video: {video_path}")
    sys.exit(1)

print(f"Đang xử lý video: {video_path.name}")
print("(Có thể mất vài phút tùy độ dài video)")
print()

incidents = video_processor.detect_incidents_in_video(
    video_path,
    model,
    threshold=0.5
)

print()
print("=" * 70)
print("KẾT QUẢ PHÁT HIỆN SỰ CỐ")
print("=" * 70)
print()

if not incidents:
    print("✓ Không phát hiện sự cố trong video")
else:
    print(f"⚠️  Phát hiện {len(incidents)} sự cố:")
    print()
    print(f"{'Frame':<10} {'Thời gian (s)':<15} {'Xác suất':<10}")
    print("-" * 70)

    for incident in incidents:
        print(f"{incident['frame_number']:<10} "
              f"{incident['timestamp_seconds']:<15.2f} "
              f"{incident['probability']:<10.4f}")

print()
print("=" * 70)