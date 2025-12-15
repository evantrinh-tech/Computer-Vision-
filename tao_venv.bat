@echo off
chcp 65001 >nul
title Tạo Virtual Environment - Python 3.11
color 0E

echo ========================================
echo   TẠO VIRTUAL ENVIRONMENT
echo ========================================
echo.

echo Đang kiểm tra Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ Lỗi: Không tìm thấy Python 3.11
    echo.
    echo Vui lòng:
    echo 1. Tải Python 3.11.7 từ python.org
    echo 2. Cài đặt và chọn "Add Python to PATH"
    echo 3. Chạy lại script này
    echo.
    pause
    exit /b 1
)

py -3.11 --version
echo ✓ Python 3.11 đã được cài đặt
echo.

if exist "venv311" (
    echo ⚠️  venv311 đã tồn tại
    echo.
    set /p recreate="Bạn có muốn xóa và tạo lại? (y/n): "
    if /i "%recreate%"=="y" (
        echo.
        echo Đang xóa venv311 cũ...
        rmdir /s /q venv311
    ) else (
        echo.
        echo Giữ nguyên venv311 hiện có
        goto INSTALL_DEPS
    )
)

echo.
echo Đang tạo virtual environment...
py -3.11 -m venv venv311
if errorlevel 1 (
    echo.
    echo ❌ Lỗi: Không thể tạo venv311
    pause
    exit /b 1
)
echo ✓ Đã tạo venv311
echo.

:INSTALL_DEPS
echo Đang kích hoạt virtual environment...
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    exit /b 1
)
echo ✓ Đã kích hoạt venv311
echo.

echo Đang cập nhật pip...
python -m pip install --upgrade pip
echo.

echo Đang cài đặt TensorFlow (có thể mất vài phút)...
pip install tensorflow
if errorlevel 1 (
    echo.
    echo ❌ Lỗi: Không thể cài TensorFlow
    echo Vui lòng kiểm tra Python version (phải là 3.9-3.11)
    pause
    exit /b 1
)
echo ✓ Đã cài TensorFlow
echo.

echo Đang cài đặt các dependencies khác...
pip install mlflow fastapi uvicorn pandas scikit-learn pywavelets kafka-python python-dotenv pyyaml python-json-logger pydantic-settings
echo.

echo ========================================
echo   HOÀN THÀNH!
echo ========================================
echo.
echo ✓ Virtual environment đã được tạo
echo ✓ TensorFlow đã được cài đặt
echo ✓ Tất cả dependencies đã được cài đặt
echo.
echo Bây giờ bạn có thể chạy:
echo   - chay_he_thong.bat
echo   - chay_demo.bat
echo   - chay_api.bat
echo.
pause

