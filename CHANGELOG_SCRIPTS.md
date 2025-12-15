# Changelog - Sắp xếp lại Scripts

## Thay đổi

### 1. Tổ chức lại Batch Scripts

**Trước:**
- Các file `.bat` nằm ở thư mục gốc
- `he_thong.bat`, `tao_venv.bat` ở root

**Sau:**
- Tất cả batch scripts được di chuyển vào `scripts/`
- Cấu trúc rõ ràng hơn, dễ quản lý

### 2. File mới được tạo

- `scripts/he_thong.bat` - Menu chính (đã cập nhật)
- `scripts/tao_venv.bat` - Tạo venv (đã cập nhật)
- `scripts/setup_tensorflow.ps1` - Setup TensorFlow (đã cập nhật)
- `scripts/quick_start.bat` - Quick start mới
- `scripts/cleanup.bat` - Dọn dẹp hệ thống mới
- `scripts/README.md` - Hướng dẫn scripts

### 3. Cập nhật Menu chính

Menu chính (`scripts/he_thong.bat`) đã được cập nhật với:
- ✅ Thêm option "Kiểm tra trạng thái hệ thống"
- ✅ Thêm option "Setup Database"
- ✅ Thêm test "Temporal Confirmation"
- ✅ Tất cả paths đã được cập nhật để hoạt động từ thư mục scripts

### 4. Cách sử dụng mới

**Trước:**
```bash
he_thong.bat
```

**Sau:**
```bash
scripts\he_thong.bat
```

Hoặc Quick Start:
```bash
scripts\quick_start.bat
```

### 5. Lưu ý

- Tất cả scripts trong `scripts/` đều tự động chuyển về thư mục gốc (`cd /d "%~dp0\.."`)
- Không cần di chuyển file cũ, có thể xóa sau khi test
- README.md đã được cập nhật với cấu trúc mới

---

*Cập nhật: [Ngày hiện tại]*

