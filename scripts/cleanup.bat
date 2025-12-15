@echo off
chcp 65001 >nul
title Dọn dẹp Hệ thống
color 0C

cd /d "%~dp0\.."

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
    exit /b 0
)

echo.
echo Đang dọn dẹp...

REM Xóa __pycache__
echo [1/5] Xóa __pycache__...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo ✓ Đã xóa __pycache__

REM Xóa .pytest_cache
echo [2/5] Xóa .pytest_cache...
if exist ".pytest_cache" rmdir /s /q .pytest_cache
echo ✓ Đã xóa .pytest_cache

REM Xóa .mypy_cache
echo [3/5] Xóa .mypy_cache...
if exist ".mypy_cache" rmdir /s /q .mypy_cache
echo ✓ Đã xóa .mypy_cache

REM Xóa logs cũ (giữ lại 7 ngày gần nhất)
echo [4/5] Dọn dẹp logs...
if exist "logs" (
    forfiles /p logs /m *.log /d -7 /c "cmd /c del @path" 2>nul
)
echo ✓ Đã dọn dẹp logs

REM Xóa .DS_Store (macOS)
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

