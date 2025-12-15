@echo off
chcp 65001 >nul
title Hệ thống Phát hiện Sự cố Giao thông
color 0A

cd /d "%~dp0\.."

:MENU
cls
echo ========================================
echo   HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG
echo ========================================
echo.
echo Chọn chức năng:
echo.
echo [1] 🖥️  Giao diện Web (Streamlit) - KHUYẾN NGHỊ
echo [2] 🚀 Chạy API Server
echo [3] 🎓 Huấn luyện mô hình
echo [4] 🔍 Test mô hình
echo [5] 📊 Kiểm tra trạng thái hệ thống
echo [6] ⚙️  Tạo Virtual Environment
echo [7] 🗄️  Setup Database
echo [0] ❌ Thoát
echo.
set /p choice="Nhập lựa chọn (0-7): "

if "%choice%"=="1" goto GUI
if "%choice%"=="2" goto API_SERVER
if "%choice%"=="3" goto TRAIN_MENU
if "%choice%"=="4" goto TEST_MENU
if "%choice%"=="5" goto CHECK_STATUS
if "%choice%"=="6" goto CREATE_VENV
if "%choice%"=="7" goto SETUP_DB
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
    echo ❌ Lỗi: Không tìm thấy venv311
    echo Vui lòng chọn [6] để tạo virtual environment
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo ✅ Đã kích hoạt virtual environment
echo.
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ⚠️  Streamlit chưa được cài đặt
    echo Đang cài đặt Streamlit...
    pip install streamlit>=1.28.0
    if errorlevel 1 (
        echo ❌ Lỗi: Không thể cài đặt Streamlit
        pause
        goto MENU
    )
    echo ✅ Đã cài đặt Streamlit
    echo.
)
echo 🚀 Đang khởi động giao diện web...
echo.
echo 📌 Giao diện sẽ mở tại: http://localhost:8501
echo 📌 Nhấn Ctrl+C để dừng server
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
    echo ❌ Lỗi: Không tìm thấy venv311
    echo Vui lòng chọn [6] để tạo virtual environment
    pause
    goto MENU
)
call venv311\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Lỗi: Không thể kích hoạt venv311
    pause
    goto MENU
)
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
echo.
if not exist "data\images\normal" (
    echo ❌ Lỗi: Không tìm thấy folder data\images\normal
    echo Vui lòng đảm bảo có folder data\images\normal chứa ảnh bình thường
    pause
    goto TRAIN_MENU
)
if not exist "data\images\incident" (
    echo ❌ Lỗi: Không tìm thấy folder data\images\incident
    echo Vui lòng đảm bảo có folder data\images\incident chứa ảnh có sự cố
    pause
    goto TRAIN_MENU
)
echo 📁 Đã tìm thấy folder ảnh
echo.
echo 🚀 Bắt đầu huấn luyện mô hình CNN...
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
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
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
echo ✅ Đã kích hoạt virtual environment
echo.
echo Đang test Temporal Confirmation module...
echo.
python -c "from src.serving.temporal_confirmation import TemporalConfirmation; print('✅ Temporal Confirmation module OK')"
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
echo ✅ Đã kích hoạt virtual environment
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
if exist "venv311" (
    echo ⚠️  Virtual environment đã tồn tại
    set /p overwrite="Bạn có muốn tạo lại? (y/n): "
    if /i not "%overwrite%"=="y" goto MENU
    echo Đang xóa virtual environment cũ...
    rmdir /s /q venv311
)
echo Đang tạo virtual environment mới...
python -m venv venv311
if errorlevel 1 (
    echo ❌ Lỗi: Không thể tạo virtual environment
    pause
    goto MENU
)
echo ✅ Đã tạo virtual environment
echo.
echo Đang cài đặt dependencies...
call venv311\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️  Có một số lỗi khi cài đặt, nhưng có thể tiếp tục
)
echo.
echo ✅ Hoàn tất!
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
echo ✅ Đã kích hoạt virtual environment
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
python -c "from sqlalchemy import create_engine; print('✅ SQLAlchemy OK')" 2>nul
if errorlevel 1 (
    echo ⚠️  SQLAlchemy chưa được cài đặt
    echo Đang cài đặt...
    pip install sqlalchemy psycopg2-binary
)
echo.
echo ✅ Database setup script sẵn sàng
echo Xem file: src\database\migrations\001_initial_schema.sql
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

