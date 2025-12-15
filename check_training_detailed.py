import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("KIỂM TRA TRẠNG THÁI TRAINING")
print("=" * 60)
print()

print("1️⃣ KIỂM TRA MÔ HÌNH")
print("-" * 60)
model_path = Path("models/CNN_model")
model_file = model_path / "model.keras" if model_path.is_dir() else model_path

if model_path.exists():
    if model_path.is_dir():
        files = list(model_path.glob("*"))
        model_files = [f for f in files if f.suffix in ['.keras', '.h5']]

        if model_files or files:
            print(f"✅ Mô hình đã được tạo!")
            print(f"   Đường dẫn: {model_path.absolute()}")
            print(f"   Số file: {len(files)}")

            if model_files:
                print(f"   File model: {', '.join([f.name for f in model_files])}")

            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            mod_time = latest_file.stat().st_mtime
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   File mới nhất: {latest_file.name}")
            print(f"   Thời gian sửa đổi: {mod_time_str}")

            time_diff = time.time() - mod_time
            if time_diff < 3600:
                print(f"   ⏱️  Mô hình được tạo cách đây: {int(time_diff/60)} phút")
            else:
                print(f"   ⏱️  Mô hình được tạo cách đây: {int(time_diff/3600)} giờ")
        else:
            print("⚠️  Thư mục tồn tại nhưng trống")
    else:
        print(f"✅ Mô hình đã được tạo!")
        print(f"   File: {model_path.absolute()}")
        mod_time = model_path.stat().st_mtime
        mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"   Thời gian sửa đổi: {mod_time_str}")
else:
    print("❌ Mô hình chưa được tạo")
    print("   Quá trình training có thể đang chạy hoặc đã bị lỗi")

print()

print("2️⃣ KIỂM TRA DỮ LIỆU")
print("-" * 60)
data_path = Path("data/images")
normal_dir = data_path / "normal"
incident_dir = data_path / "incident"

def load_image_files(folder_path: Path):

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif']
    image_files = []
    for ext in extensions:
        image_files.extend(list(folder_path.glob(ext)))
        image_files.extend(list(folder_path.glob(ext.upper ())))
    return sorted(list(set(image_files)))

normal_count = 0
incident_count = 0

if normal_dir.exists():
    normal_files = load_image_files(normal_dir)
    normal_count = len(normal_files)
    print(f"✅ Thư mục normal: {normal_count} ảnh")
    if normal_count > 0:
        extensions_found = set([f.suffix.lower() for f in normal_files])
        print(f"   Định dạng: {', '.join(sorted(extensions_found))}")
else:
    print("❌ Không tìm thấy thư mục normal")

if incident_dir.exists():
    incident_files = load_image_files(incident_dir)
    incident_count = len(incident_files)
    print(f"✅ Thư mục incident: {incident_count} ảnh")
    if incident_count > 0:
        extensions_found = set([f.suffix.lower() for f in incident_files])
        print(f"   Định dạng: {', '.join(sorted(extensions_found))}")
else:
    print("❌ Không tìm thấy thư mục incident")

total_images = normal_count + incident_count
print(f"   📊 Tổng số ảnh: {total_images}")
if total_images > 0:
    balance = abs(normal_count - incident_count) / total_images * 100
    if balance < 20:
        print(f"   ✅ Tỷ lệ cân bằng: {normal_count}/{incident_count} (chênh lệch {balance:.1f}%)")
    else:
        print(f"   ⚠️  Dữ liệu không cân bằng: {normal_count}/{incident_count} (chênh lệch {balance:.1f}%)")

    if total_images < 20:
        print(f"   ⚠️  CẢNH BÁO: Số lượng ảnh quá ít ({total_images} ảnh)")
        print(f"      Khuyến nghị: Cần ít nhất 50-100 ảnh mỗi loại để training hiệu quả")
    elif total_images < 50:
        print(f"   ⚠️  Số lượng ảnh hơi ít ({total_images} ảnh)")
        print(f"      Có thể training nhưng kết quả có thể không tối ưu")
    elif total_images < 100:
        print(f"   ✅ Số lượng ảnh đủ để training cơ bản ({total_images} ảnh)")
    else:
        print(f"   ✅ Số lượng ảnh tốt cho training ({total_images} ảnh)")

print()

print("3️⃣ KIỂM TRA MLFLOW")
print("-" * 60)
try:
    import mlflow
    from src.utils.config import settings

    print(f"   Tracking URI: {settings.mlflow_tracking_uri}")
    print(f"   Experiment: {settings.mlflow_experiment_name}")

    try:
        import socket
        import urllib.parse

        parsed_uri = urllib.parse.urlparse(settings.mlflow_tracking_uri)
        host = parsed_uri.hostname or "localhost"
        port = parsed_uri.port or 5000

        print(f"   Đang kiểm tra kết nối tới {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            print(f"⚠️  MLflow server không chạy tại {host}:{port}")
            print("   (Training vẫn hoạt động bình thường, chỉ không có tracking)")
        else:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            try:
                experiments = mlflow.search_experiments()
                print(f"✅ Kết nối MLflow thành công! Số experiments: {len(experiments)}")

                exp = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
                if exp:
                    print(f"✅ Tìm thấy experiment: {settings.mlflow_experiment_name}")
                    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
                    if len(runs) > 0:
                        print(f"   Số runs gần đây: {len(runs)}")
                        latest_run = runs.iloc[0]
                        run_name = latest_run.get('tags.mlflow.runName', latest_run.get('run_id', 'N/A'))
                        print(f"   Run mới nhất: {run_name}")
                    else:
                        print("   ⚠️  Chưa có runs nào")
                else:
                    print(f"⚠️  Không tìm thấy experiment: {settings.mlflow_experiment_name}")
            except Exception as mlflow_error:
                print(f"⚠️  Lỗi khi truy vấn MLflow: {str(mlflow_error)[:100]}")
                print("   (Training vẫn hoạt động bình thường)")

    except socket.timeout:
        print(f"⚠️  Timeout khi kết nối MLflow server")
        print("   (Training vẫn hoạt động bình thường, chỉ không có tracking)")
    except Exception as e:
        error_msg = str(e)
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"⚠️  Timeout khi kết nối MLflow: {error_msg[:80]}")
        else:
            print(f"⚠️  Không thể kết nối MLflow: {error_msg[:80]}")
        print("   (Có thể MLflow server chưa chạy, nhưng training vẫn hoạt động)")
except ImportError:
    print("⚠️  MLflow chưa được cài đặt")
except Exception as e:
    print(f"⚠️  Lỗi khi kiểm tra MLflow: {str(e)[:80]}")

print()

print("4️⃣ KIỂM TRA LOGS")
print("-" * 60)
log_path = Path("logs/app.log")
if log_path.exists():
    print(f"✅ File log tồn tại: {log_path}")
    log_size = log_path.stat().st_size
    print(f"   Kích thước: {log_size / 1024:.2f} KB")

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                print(f"   Dòng cuối cùng: {lines[-1].strip()[:80]}...")
    except Exception as e:
        print(f"   ⚠️  Không thể đọc log: {e}")
else:
    print("⚠️  Không tìm thấy file log")

print()
print("=" * 60)
print("KẾT LUẬN & KHUYẾN NGHỊ")
print("=" * 60)

has_model = model_path.exists() and list(model_path.glob("*"))
has_data = normal_count > 0 and incident_count > 0

if has_model:
    print("✅ Hệ thống đã sẵn sàng! Mô hình đã được tạo.")
    print("   Bạn có thể sử dụng mô hình để test hoặc predict.")
elif has_data:
    print("✅ Dữ liệu đã sẵn sàng để training!")
    print()
    print("📝 BƯỚC TIẾP THEO:")
    print("   1. Mở giao diện web Streamlit:")
    print("      - Chạy: he_thong.bat -> chọn [1] Giao diện web")
    print("      - Hoặc: streamlit run app.py")
    print("   2. Truy cập: http://localhost:8501")
    print("   3. Vào trang '🎓 Huấn luyện mô hình CNN'")
    print("   4. Điều chỉnh tham số (epochs, batch_size) nếu cần")
    print("   5. Nhấn nút '🚀 Bắt đầu huấn luyện'")
    print()
    print("   ⏱️  Thời gian training dự kiến:")
    if total_images < 50:
        print(f"      - Với {total_images} ảnh: 5-15 phút")
    else:
        print(f"      - Với {total_images} ảnh: 10-30 phút")
else:
    print("⚠️  Chưa sẵn sàng để training.")
    if normal_count == 0:
        print("   ❌ Thiếu ảnh normal trong data/images/normal/")
    if incident_count == 0:
        print("   ❌ Thiếu ảnh incident trong data/images/incident/")
    print()
    print("📝 Cần chuẩn bị dữ liệu trước khi training!")

print()
print("=" * 60)
print("Lệnh kiểm tra lại: python check_training_detailed.py")
print("=" * 60)