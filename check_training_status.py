import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Đang kiểm tra trạng thái training...")
print()

model_path = Path("models/CNN_model")
if model_path.exists():
    files = list(model_path.glob("*"))
    if files:
        print(f"✅ Mô hình đã được tạo!")
        print(f"   Số file: {len(files)}")
        print(f"   Đường dẫn: {model_path.absolute()}")

        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        mod_time = time.ctime(latest_file.stat().st_mtime)
        print(f"   File mới nhất: {latest_file.name}")
        print(f"   Thời gian sửa đổi: {mod_time}")
    else:
        print("⚠️  Thư mục tồn tại nhưng trống")
else:
    print("❌ Mô hình chưa được tạo")
    print("   Quá trình training có thể đang chạy hoặc đã bị lỗi")

print()
print("Để kiểm tra lại, chạy: python check_training_status.py")