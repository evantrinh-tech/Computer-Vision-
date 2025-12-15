import os
import shutil
from pathlib import Path
import sys
import io

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def get_size_mb(path):

    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except:
            pass
        return total / (1024 * 1024)
    return 0

def delete_safely(path):

    try:
        if path.is_file():
            path.unlink()
            return True
        elif path.is_dir():
            shutil.rmtree(path)
            return True
    except Exception as e:
        print(f"  ⚠️  Không thể xóa {path}: {e}")
        return False
    return False

def main():
    print("=" * 70)
    print("DỌN DẸP VÀ SẮP XẾP LẠI HỆ THỐNG")
    print("=" * 70)
    print()

    base_path = Path(__file__).parent
    total_freed = 0
    deleted_items = []

    print("1️⃣ XÓA __pycache__ TRONG SOURCE CODE")
    print("-" * 70)
    pycache_dirs = [
        base_path / "__pycache__",
        base_path / "src" / "__pycache__",
        base_path / "src" / "models" / "__pycache__",
        base_path / "src" / "training" / "__pycache__",
        base_path / "src" / "data_processing" / "__pycache__",
        base_path / "src" / "serving" / "__pycache__",
        base_path / "src" / "utils" / "__pycache__",
    ]

    for pycache_dir in pycache_dirs:
        if pycache_dir.exists():
            size = get_size_mb(pycache_dir)
            if delete_safely(pycache_dir):
                total_freed += size
                deleted_items.append(f"  ✅ {pycache_dir} ({size:.2f} MB)")
                print(f"  ✅ Đã xóa: {pycache_dir} ({size:.2f} MB)")

    print()

    print("2️⃣ XÓA FILE TEMP CỦA OFFICE")
    print("-" * 70)
    temp_files = list(base_path.glob("~$*"))
    for temp_file in temp_files:
        if temp_file.exists():
            size = get_size_mb(temp_file)
            if delete_safely(temp_file):
                total_freed += size
                deleted_items.append(f"  ✅ {temp_file.name} ({size:.2f} MB)")
                print(f"  ✅ Đã xóa: {temp_file.name} ({size:.2f} MB)")

    if not temp_files:
        print("  ℹ️  Không có file temp nào")
    print()

    print("3️⃣ XÓA FILE KHÔNG RÕ MỤC ĐÍCH")
    print("-" * 70)
    suspicious_files = [
        base_path / "1.28.0",
    ]

    for file in suspicious_files:
        if file.exists():
            size = get_size_mb(file)
            print(f"  ⚠️  Tìm thấy file đáng ngờ: {file.name}")
            if delete_safely(file):
                total_freed += size
                deleted_items.append(f"  ✅ {file.name} ({size:.2f} MB)")
                print(f"  ✅ Đã xóa: {file.name}")

    if not any(f.exists() for f in suspicious_files):
        print("  ℹ️  Không có file đáng ngờ nào")

    print()

    print("4️⃣ KIỂM TRA .gitignore")
    print("-" * 70)
    gitignore_path = base_path / ".gitignore"
    if not gitignore_path.exists():
        print("  ⚠️  Không tìm thấy .gitignore, đang tạo...")
    else:
        print("  ✅ .gitignore đã tồn tại")
    print()

    print("5️⃣ TỔ CHỨC LẠI DOCUMENTATION")
    print("-" * 70)
    docs_dir = base_path / "docs"
    docs_dir.mkdir(exist_ok=True)

    markdown_files = {
        "HUONG_DAN_*.md": "docs/huong_dan/",
        "TONG_KET_*.md": "docs/tong_ket/",
        "CAI_DAT_*.md": "docs/cai_dat/",
        "INSTALL*.md": "docs/cai_dat/",
        "SO_LUONG_*.md": "docs/",
        "DVC_CONFIG.md": "docs/",
        "PROMPT_*.md": "docs/",
        "chuan_bi_*.md": "docs/",
    }

    moved_count = 0
    for pattern, target_dir in markdown_files.items():
        target_path = base_path / target_dir
        target_path.mkdir(parents=True, exist_ok=True)

        if "*" in pattern:
            prefix = pattern.split("*")[0]
            for file in base_path.glob(pattern):
                if file.is_file() and file.name.startswith(prefix):
                    try:
                        new_path = target_path / file.name
                        if not new_path.exists():
                            shutil.move(str(file), str(new_path))
                            moved_count += 1
                            print(f"  ✅ Đã di chuyển: {file.name} -> {target_dir}")
                    except Exception as e:
                        print(f"  ⚠️  Không thể di chuyển {file.name}: {e}")
        else:
            file = base_path / pattern
            if file.exists() and file.is_file():
                try:
                    new_path = target_path / file.name
                    if not new_path.exists():
                        shutil.move(str(file), str(new_path))
                        moved_count += 1
                        print(f"  ✅ Đã di chuyển: {file.name} -> {target_dir}")
                except Exception as e:
                    print(f"  ⚠️  Không thể di chuyển {file.name}: {e}")

    if moved_count == 0:
        print("  ℹ️  Không có file nào cần di chuyển")
    print()

    print("5b. TỔ CHỨC DEMO SCRIPTS")
    print("-" * 70)
    examples_dir = base_path / "docs" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    demo_scripts = [
        base_path / "run_demo.py",
        base_path / "run_full_demo.py",
    ]

    demo_moved = 0
    for script in demo_scripts:
        if script.exists():
            try:
                new_path = examples_dir / script.name
                if not new_path.exists():
                    shutil.move(str(script), str(new_path))
                    demo_moved += 1
                    print(f"  ✅ Đã di chuyển: {script.name} -> docs/examples/")
            except Exception as e:
                print(f"  ⚠️  Không thể di chuyển {script.name}: {e}")

    if demo_moved == 0:
        print("  ℹ️  Không có demo script nào cần di chuyển")
    print()

    print("6️⃣ KIỂM TRA CẤU TRÚC THƯ MỤC")
    print("-" * 70)
    required_dirs = [
        base_path / "logs",
        base_path / "data" / "raw",
        base_path / "data" / "processed",
        base_path / "models",
    ]

    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists() and not any(dir_path.iterdir()):
            gitkeep.touch()
            print(f"  ✅ Đã tạo: {dir_path} (với .gitkeep)")

    print()

    print("=" * 70)
    print("TÓM TẮT")
    print("=" * 70)
    print(f"📊 Tổng dung lượng đã giải phóng: {total_freed:.2f} MB")
    print(f"📁 Số file/folder đã xóa: {len(deleted_items)}")
    print(f"📝 Số file đã di chuyển: {moved_count + demo_moved}")
    print()

    if deleted_items:
        print("Các file đã xóa:")
        for item in deleted_items:
            print(item)

    print()
    print("✅ Hoàn tất dọn dẹp!")
    print()
    print("📋 CẤU TRÚC SAU KHI DỌN DẸP:")
    print("  - Tài liệu: docs/")
    print("  - Source code: src/")
    print("  - Scripts chính: app.py, start_api.py, train_cnn.py")
    print("  - Scripts tiện ích: check_*.py, cleanup_system.py")
    print()
    print("Lưu ý:")
    print("  - Các file __pycache__ trong venv/ và venv311/ được giữ lại")
    print("  - Virtual environments được giữ nguyên")
    print("  - Dữ liệu và models được giữ nguyên")
    print("  - Xem CAU_TRUC_DU_AN.md để biết cấu trúc chi tiết")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)