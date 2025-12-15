# Script tự động cài đặt TensorFlow với Python 3.11
# Chạy: .\setup_tensorflow.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cài đặt TensorFlow cho Hệ thống" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python versions
Write-Host "[1/5] Kiểm tra Python versions..." -ForegroundColor Yellow
$pythonVersions = py -0 2>&1
Write-Host $pythonVersions

# Kiểm tra Python 3.11
$hasPython311 = $pythonVersions -match "3\.11"
if (-not $hasPython311) {
    Write-Host ""
    Write-Host "⚠️  Python 3.11 chưa được cài đặt!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Vui lòng cài đặt Python 3.11 từ:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3118/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Hoặc từ Microsoft Store: tìm 'Python 3.11'" -ForegroundColor Cyan
    Write-Host ""
    $continue = Read-Host "Bạn có muốn tiếp tục với Python hiện tại? (y/n)"
    if ($continue -ne "y") {
        exit
    }
    $pythonCmd = "python"
} else {
    Write-Host "✓ Tìm thấy Python 3.11" -ForegroundColor Green
    $pythonCmd = "py -3.11"
}

# Tạo venv mới
Write-Host ""
Write-Host "[2/5] Tạo virtual environment mới..." -ForegroundColor Yellow
if (Test-Path "venv311") {
    Write-Host "⚠️  venv311 đã tồn tại. Xóa và tạo lại? (y/n)" -ForegroundColor Yellow
    $recreate = Read-Host
    if ($recreate -eq "y") {
        Remove-Item -Recurse -Force venv311
    } else {
        Write-Host "Sử dụng venv311 hiện có" -ForegroundColor Cyan
        goto ActivateVenv
    }
}

& $pythonCmd -m venv venv311
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Lỗi tạo virtual environment!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Đã tạo venv311" -ForegroundColor Green

ActivateVenv:
# Kích hoạt venv
Write-Host ""
Write-Host "[3/5] Kích hoạt virtual environment..." -ForegroundColor Yellow
& .\venv311\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "[4/5] Cài đặt pip và dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "✓ pip đã được cập nhật" -ForegroundColor Green

# Cài TensorFlow
Write-Host ""
Write-Host "Đang cài đặt TensorFlow (có thể mất vài phút)..." -ForegroundColor Yellow
pip install tensorflow
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Lỗi cài đặt TensorFlow!" -ForegroundColor Red
    Write-Host "Thử cài tensorflow-cpu..." -ForegroundColor Yellow
    pip install tensorflow-cpu
}

# Cài các dependencies khác
Write-Host ""
Write-Host "Đang cài đặt các dependencies khác..." -ForegroundColor Yellow
pip install numpy pandas scikit-learn mlflow fastapi uvicorn pywavelets kafka-python python-dotenv pyyaml python-json-logger

# Kiểm tra cài đặt
Write-Host ""
Write-Host "[5/5] Kiểm tra cài đặt..." -ForegroundColor Yellow
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__} đã được cài đặt thành công!')"
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ CÀI ĐẶT THÀNH CÔNG!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Để sử dụng:" -ForegroundColor Cyan
    Write-Host "  .\venv311\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Để train model:" -ForegroundColor Cyan
    Write-Host "  python pipelines/training_pipeline.py --model ANN --simulate" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "❌ Có lỗi trong quá trình cài đặt!" -ForegroundColor Red
    Write-Host "Vui lòng kiểm tra lại Python version (phải là 3.9-3.11)" -ForegroundColor Yellow
}

