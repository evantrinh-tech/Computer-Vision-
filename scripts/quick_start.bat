@echo off
chcp 65001 >nul
title Quick Start - Hệ thống Phát hiện Sự cố Giao thông
color 0B

cd /d "%~dp0\.."

echo ========================================
echo   QUICK START
echo ========================================
echo.

if not exist "venv311\Scripts\activate.bat" (
    echo ❌ Virtual environment chưa được tạo
    echo.
    echo Đang tạo virtual environment...
    call scripts\tao_venv.bat
    if errorlevel 1 (
        echo ❌ Lỗi tạo virtual environment
        pause
        exit /b 1
    )
)

echo  Virtual environment đã sẵn sàng
echo.
echo Đang khởi động giao diện web...
echo.
call venv311\Scripts\activate.bat
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python run_streamlit.py

