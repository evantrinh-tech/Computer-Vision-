import sys
from pathlib import Path

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("KIỂM TRA MÔ HÌNH")
print("=" * 60)
print()

models_dir = Path("models")

print("📁 Kiểm tra thư mục models/...")
if not models_dir.exists():
    print("❌ Không tìm thấy thư mục models/")
else:
    print(f"✅ Tìm thấy thư mục: {models_dir.absolute()}")
    print()

print("🔍 Kiểm tra CNN Model...")
cnn_path = models_dir / "CNN_model"
if cnn_path.exists():
    files = list(cnn_path.glob("*"))
    if files:
        print(f"✅ Tìm thấy CNN model tại: {cnn_path.absolute()}")
        print(f"   Số file: {len(files)}")
        print("   Các file:")
        for f in files[:5]:
            print(f"     - {f.name}")
        if len(files) > 5:
            print(f"     ... và {len(files) - 5} file khác")

        try:
            print()
            print("🔄 Đang thử load model...")
            from src.models.cnn import CNNModel
            model = CNNModel()
            model.load(cnn_path)
            print("✅ Model load thành công!")
            print(f"   Model đã được train: {model.is_trained}")
        except Exception as e:
            print(f"⚠️  Không thể load model: {e}")
    else:
        print(f"⚠️  Thư mục {cnn_path} tồn tại nhưng trống")
else:
    print("❌ Chưa có CNN model được huấn luyện")
    print(f"   Đường dẫn mong đợi: {cnn_path.absolute()}")

print()

print("🔍 Kiểm tra các model khác...")
other_models = {
    "ANN": models_dir / "ANN_model",
    "RNN": models_dir / "RNN_model",
    "RBFNN": models_dir / "rbfnn_demo_model.pkl"
}

for model_name, model_path in other_models.items():
    if model_path.exists():
        print(f"✅ Tìm thấy {model_name} model: {model_path.name}")
    else:
        print(f"❌ Chưa có {model_name} model")

print()
print("=" * 60)
print("KẾT LUẬN")
print("=" * 60)

if cnn_path.exists() and list(cnn_path.glob("*")):
    print("✅ Đã có mô hình CNN được huấn luyện")
    print("   Bạn có thể sử dụng chức năng Test mô hình")
else:
    print("⚠️  Chưa có mô hình CNN được huấn luyện")
    print("   Vui lòng huấn luyện mô hình trước:")
    print("   - Chạy he_thong.bat → [3] → [1] (Train CNN)")
    print("   - Hoặc sử dụng giao diện web → Trang 'Huấn luyện mô hình'")

print()