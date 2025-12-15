@echo off
chcp 65001 >nul
title Hệ thống Phát hiện Sự cố Giao thông
color 0A

cd /d "%~dp0"

:MENU
cls
echo ========================================
echo   HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG
echo ========================================
echo.
echo Chọn chức năng:
echo.
echo [1]  Giao diện Web (Streamlit) 
echo [2]  Chạy API Server
echo [3]  Huấn luyện mô hình
echo [4]  Test mô hình
echo [5]  Kiểm tra trạng thái hệ thống
echo [6]   Tạo Virtual Environment
echo [7]   Setup Database
echo [8]  Dọn dẹp hệ thống
echo [9]  Quick Start (Tự động setup và chạy)
echo [V]   Verify hệ thống (check imports)
echo [0]  Thoát
echo.
set /p choice="Nhập lựa chọn (0-9 hoặc V): "

if /i "%choice%"=="V" goto VERIFY_SYSTEM

if "%choice%"=="1" goto GUI
if "%choice%"=="2" goto API_SERVER
if "%choice%"=="3" goto TRAIN_MENU
if "%choice%"=="4" goto TEST_MENU
if "%choice%"=="5" goto CHECK_STATUS
if "%choice%"=="6" goto CREATE_VENV
if "%choice%"=="7" goto SETUP_DB
if "%choice%"=="8" goto CLEANUP
if "%choice%"=="9" goto QUICK_START
if "%choice%"=="0" goto EXIT
goto MENU

REM ========================================
REM GIAO DIỆN WEB (STREAMLIT)
REM ========================================
:GUI
cls
echo ========================================
echo   GIAO DIỆN WEB (STREAMLIT)
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo  Lỗi: Không tìm thấy venv311
    echo Vui lòng chọn [6] để tạo virtual environment
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo  Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo  Đã kích hoạt virtual environment
echo.
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo   Streamlit chưa được cài đặt
    echo Đang cài đặt Streamlit...
    pip install streamlit>=1.28.0
    if errorlevel 1 (
        echo  Lỗi: Không thể cài đặt Streamlit
        pause
        goto MENU
    )
    echo  Đã cài đặt Streamlit
    echo.
)
echo  Đang khởi động giao diện web...
echo.
echo  Giao diện sẽ mở tại: http://localhost:8501
echo  Nhấn Ctrl+C để dừng server
echo.
if not exist ".streamlit" mkdir .streamlit
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python run_streamlit.py
pause
goto MENU

REM ========================================
REM API SERVER
REM ========================================
:API_SERVER
cls
echo ========================================
echo   CHẠY API SERVER
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo  Lỗi: Không tìm thấy venv311
    echo Vui lòng chọn [6] để tạo virtual environment
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo  Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo API Server sẽ chạy tại: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo.
echo Nhấn Ctrl+C để dừng server
echo.
python start_api.py
pause
goto MENU

REM ========================================
REM MENU HUẤN LUYỆN
REM ========================================
:TRAIN_MENU
cls
echo ========================================
echo   HUẤN LUYỆN MÔ HÌNH
echo ========================================
echo.
echo Chọn model để train:
echo.
echo [1] CNN (Convolutional Neural Network) - Với ảnh
echo [2] ANN (Feed-forward Neural Network) - Dữ liệu mô phỏng
echo [3] RNN (LSTM/GRU) - Dữ liệu mô phỏng
echo [4] RBFNN (Radial Basis Function) - Dữ liệu mô phỏng
echo [5] Quay lại menu chính
echo.
set /p train_choice="Nhập lựa chọn (1-5): "

if "%train_choice%"=="1" goto TRAIN_CNN
if "%train_choice%"=="2" goto TRAIN_ANN
if "%train_choice%"=="3" goto TRAIN_RNN
if "%train_choice%"=="4" goto TRAIN_RBFNN
if "%train_choice%"=="5" goto MENU
goto TRAIN_MENU

:TRAIN_CNN
cls
echo ========================================
echo   TRAIN CNN MODEL (VỚI ẢNH)
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo  Lỗi: Không tìm thấy venv311
    pause
    goto TRAIN_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo  Lỗi: Không thể kích hoạt venv311
    pause
    goto TRAIN_MENU
)
echo  Đã kích hoạt virtual environment
echo.
if not exist "data\images\normal" (
    echo  Lỗi: Không tìm thấy folder data\images\normal
    echo Vui lòng đảm bảo có folder data\images\normal chứa ảnh bình thường
    pause
    goto TRAIN_MENU
)
if not exist "data\images\incident" (
    echo  Lỗi: Không tìm thấy folder data\images\incident
    echo Vui lòng đảm bảo có folder data\images\incident chứa ảnh có sự cố
    pause
    goto TRAIN_MENU
)
echo 📁 Đã tìm thấy folder ảnh
echo.
echo  Bắt đầu huấn luyện mô hình CNN...
echo    (Quá trình này có thể mất nhiều thời gian)
echo.
python train_cnn.py
echo.
pause
goto TRAIN_MENU

:TRAIN_ANN
cls
echo ========================================
echo   TRAIN ANN MODEL
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo  Lỗi: Không tìm thấy venv311
    pause
    goto TRAIN_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo  Lỗi: Không thể kích hoạt venv311
    pause
    goto TRAIN_MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo Đang train ANN model với dữ liệu mô phỏng...
echo (Có thể mất vài phút)
echo.
set PYTHONPATH=%CD%
python pipelines\training_pipeline.py --model ANN --simulate
echo.
pause
goto TRAIN_MENU

:TRAIN_RNN
cls
echo ========================================
echo   TRAIN RNN MODEL
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto TRAIN_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto TRAIN_MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo Đang train RNN model với dữ liệu mô phỏng...
echo (Có thể mất vài phút)
echo.
set PYTHONPATH=%CD%
python pipelines\training_pipeline.py --model RNN --simulate
echo.
pause
goto TRAIN_MENU

:TRAIN_RBFNN
cls
echo ========================================
echo   TRAIN RBFNN MODEL
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto TRAIN_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto TRAIN_MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo Đang train RBFNN model với dữ liệu mô phỏng...
echo.
set PYTHONPATH=%CD%
python pipelines\training_pipeline.py --model RBFNN --simulate
echo.
pause
goto TRAIN_MENU

REM ========================================
REM MENU TEST
REM ========================================
:TEST_MENU
cls
echo ========================================
echo   TEST MÔ HÌNH
echo ========================================
echo.
echo Chọn loại test:
echo.
echo [1] Test CNN với ảnh
echo [2] Test CNN với video
echo [3] Test API
echo [4] Test Temporal Confirmation
echo [5] Quay lại menu chính
echo.
set /p test_choice="Nhập lựa chọn (1-5): "

if "%test_choice%"=="1" goto TEST_CNN_IMAGE
if "%test_choice%"=="2" goto TEST_CNN_VIDEO
if "%test_choice%"=="3" goto TEST_API
if "%test_choice%"=="4" goto TEST_TEMPORAL
if "%test_choice%"=="5" goto MENU
goto TEST_MENU

:TEST_CNN_IMAGE
cls
echo ========================================
echo   TEST CNN VỚI ẢNH
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto TEST_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto TEST_MENU
)
echo  Đã kích hoạt virtual environment
echo.
set /p image_path="Nhập đường dẫn ảnh hoặc thư mục (Enter để bỏ qua): "
if "%image_path%"=="" (
    echo Vui lòng nhập đường dẫn
    pause
    goto TEST_MENU
)
echo.
python test_cnn_image.py %image_path%
echo.
pause
goto TEST_MENU

:TEST_CNN_VIDEO
cls
echo ========================================
echo   TEST CNN VỚI VIDEO
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto TEST_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto TEST_MENU
)
echo  Đã kích hoạt virtual environment
echo.
set /p video_path="Nhập đường dẫn video: "
if "%video_path%"=="" (
    echo Vui lòng nhập đường dẫn video
    pause
    goto TEST_MENU
)
echo.
python test_cnn_video.py %video_path%
echo.
pause
goto TEST_MENU

:TEST_API
cls
echo ========================================
echo   TEST API
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto TEST_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto TEST_MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo Đang test API tại http://localhost:8000
echo.
echo Lưu ý: Đảm bảo API server đang chạy!
echo (Chạy [2] Chạy API Server trong menu chính)
echo.
pause
python test_api.py
echo.
pause
goto TEST_MENU

:TEST_TEMPORAL
cls
echo ========================================
echo   TEST TEMPORAL CONFIRMATION
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto TEST_MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto TEST_MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo Đang test Temporal Confirmation module...
echo.
python -c "from src.serving.temporal_confirmation import TemporalConfirmation; print(' Temporal Confirmation module OK')"
echo.
pause
goto TEST_MENU

REM ========================================
REM KIỂM TRA TRẠNG THÁI
REM ========================================
:CHECK_STATUS
cls
echo ========================================
echo   KIỂM TRA TRẠNG THÁI HỆ THỐNG
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo Đang kiểm tra trạng thái hệ thống...
echo.
python check_training_status.py
echo.
pause
goto MENU

REM ========================================
REM TẠO VIRTUAL ENVIRONMENT
REM ========================================
:CREATE_VENV
cls
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
    goto MENU
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
    goto MENU
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
    goto MENU
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
    goto MENU
)
echo ✓ Đã cài TensorFlow
echo.
echo Đang cài đặt các dependencies khác...
pip install mlflow fastapi uvicorn pandas scikit-learn pywavelets kafka-python python-dotenv pyyaml python-json-logger pydantic-settings sqlalchemy psycopg2-binary opencv-python pillow streamlit
echo.
echo ========================================
echo   HOÀN THÀNH!
echo ========================================
echo.
echo ✓ Virtual environment đã được tạo
echo ✓ TensorFlow đã được cài đặt
echo ✓ Tất cả dependencies đã được cài đặt
echo.
pause
goto MENU

REM ========================================
REM SETUP DATABASE
REM ========================================
:SETUP_DB
cls
echo ========================================
echo   SETUP DATABASE
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo  Đã kích hoạt virtual environment
echo.
echo 📌 Setup Database (PostgreSQL)
echo.
echo Lưu ý: Cần có PostgreSQL đã cài đặt và chạy
echo.
echo Bạn có thể:
echo 1. Chạy migration script: src\database\migrations\001_initial_schema.sql
echo 2. Hoặc sử dụng SQLAlchemy để tạo tables tự động
echo.
echo Đang kiểm tra SQLAlchemy...
python -c "from sqlalchemy import create_engine; print(' SQLAlchemy OK')" 2>nul
if errorlevel 1 (
    echo ⚠️  SQLAlchemy chưa được cài đặt
    echo Đang cài đặt...
    pip install sqlalchemy psycopg2-binary
)
echo.
echo  Database setup script sẵn sàng
echo Xem file: src\database\migrations\001_initial_schema.sql
echo.
pause
goto MENU

REM ========================================
REM DỌN DẸP HỆ THỐNG
REM ========================================
:CLEANUP
cls
echo ========================================
echo   DỌN DẸP HỆ THỐNG
echo ========================================
echo.
echo Cảnh báo: Script này sẽ xóa các file tạm và cache
echo.
set /p confirm="Bạn có chắc chắn? (y/n): "
if /i not "%confirm%"=="y" (
    echo Đã hủy
    pause
    goto MENU
)
echo.
echo Đang dọn dẹp...
echo.
echo [1/5] Xóa __pycache__...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo ✓ Đã xóa __pycache__
echo.
echo [2/5] Xóa .pytest_cache...
if exist ".pytest_cache" rmdir /s /q .pytest_cache
echo ✓ Đã xóa .pytest_cache
echo.
echo [3/5] Xóa .mypy_cache...
if exist ".mypy_cache" rmdir /s /q .mypy_cache
echo ✓ Đã xóa .mypy_cache
echo.
echo [4/5] Dọn dẹp logs...
if exist "logs" (
    forfiles /p logs /m *.log /d -7 /c "cmd /c del @path" 2>nul
)
echo ✓ Đã dọn dẹp logs
echo.
echo [5/5] Xóa file hệ thống...
del /s /q .DS_Store 2>nul
del /s /q Thumbs.db 2>nul
echo ✓ Đã xóa file hệ thống
echo.
echo ========================================
echo   HOÀN THÀNH!
echo ========================================
echo.
echo ✓ Đã dọn dẹp hệ thống
echo.
pause
goto MENU

REM ========================================
REM QUICK START
REM ========================================
:QUICK_START
cls
echo ========================================
echo   QUICK START
echo ========================================
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Virtual environment chưa được tạo
    echo.
    echo Đang tạo virtual environment...
    echo (Quá trình này có thể mất vài phút)
    echo.
    goto CREATE_VENV_FROM_QUICK
)
echo  Virtual environment đã sẵn sàng
echo.
echo Đang khởi động giao diện web...
echo.
call venv311\Scripts\activate.bat
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
if not exist ".streamlit" mkdir .streamlit
python run_streamlit.py
goto MENU

:CREATE_VENV_FROM_QUICK
echo Đang kiểm tra Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Lỗi: Không tìm thấy Python 3.11
    echo Vui lòng cài đặt Python 3.11 trước
    pause
    goto MENU
)
echo ✓ Python 3.11 OK
echo.
echo Đang tạo virtual environment...
py -3.11 -m venv venv311
if errorlevel 1 (
    echo ❌ Lỗi: Không thể tạo venv311
    pause
    goto MENU
)
echo ✓ Đã tạo venv311
echo.
echo Đang kích hoạt và cài đặt dependencies...
call venv311\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Đang cài đặt TensorFlow (có thể mất vài phút)...
pip install tensorflow
echo.
echo Đang cài đặt các dependencies khác...
pip install mlflow fastapi uvicorn pandas scikit-learn pywavelets kafka-python python-dotenv pyyaml python-json-logger pydantic-settings sqlalchemy psycopg2-binary opencv-python pillow streamlit
echo.
echo  Hoàn tất setup!
echo.
echo Đang khởi động giao diện web...
echo.
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
if not exist ".streamlit" mkdir .streamlit
python run_streamlit.py
goto MENU

REM ========================================
REM VERIFY SYSTEM
REM ========================================
:VERIFY_SYSTEM
cls
echo ========================================
echo   VERIFY HỆ THỐNG
echo ========================================
echo.
echo Đang kiểm tra cấu trúc imports và modules...
echo.
if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Lỗi: Không tìm thấy venv311
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo  Đã kích hoạt virtual environment
echo.
REM Set PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%
echo  PYTHONPATH: %CD%
echo.
echo Đang chạy verification script...
echo.
if exist "verify_and_fix_imports.py" (
    python verify_and_fix_imports.py
) else (
    echo ⚠️  Script verify_and_fix_imports.py không tồn tại
    echo.
    echo Đang kiểm tra imports cơ bản...
    python -c "import sys; sys.path.insert(0, '.'); from src.models.cnn import CNNModel; print(' CNN import OK')"
    python -c "import sys; sys.path.insert(0, '.'); from src.serving.api import app; print(' API import OK')"
    python -c "import sys; sys.path.insert(0, '.'); from src.data_processing.image_processor import ImageProcessor; print(' ImageProcessor import OK')"
)
echo.
pause
goto MENU

REM ========================================
REM THOÁT
REM ========================================
:EXIT
cls
echo.
echo Cảm ơn bạn đã sử dụng hệ thống!
echo.
timeout /t 2 >nul
exit
