# Quick Start Guide - ITS System

##  Cài đặt và Chạy

### 1. Setup Environment

```bash
# Windows
.\tao_venv.bat

# Activate venv
venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set PYTHONPATH (Quan trọng!)

```bash
# Windows
set_pythonpath.bat

# Linux/Mac
source set_pythonpath.sh
```

### 3. Chạy Ứng dụng

#### Option A: Menu Hệ thống (Khuyến nghị)
```bash
he_thong.bat
# Chọn [1] để chạy Streamlit Dashboard
```

#### Option B: Direct Commands

**Streamlit Dashboard**:
```bash
python run_streamlit.py
# Hoặc
streamlit run app.py
```

**API Server**:
```bash
python start_api.py
```

**Training**:
```bash
python train_cnn.py
```

**Testing**:
```bash
# Test với ảnh
python test_cnn_image.py data/images/incident/img1.jpg

# Test với video
python test_cnn_video.py path/to/video.mp4
```

## 📁 Cấu trúc Imports

### Đúng 
```python
from src.models.cnn import CNNModel
from src.data_processing.image_processor import ImageProcessor
from src.serving.predictor import Predictor
```

### Sai ❌  
```python
from models.cnn import CNNModel  # Thiếu 'src.'
from cnn import CNNModel          # Sai hoàn toàn
```

## 🔧 Troubleshooting

### Lỗi: "No module named 'src'"

**Giải pháp 1**: Set PYTHONPATH
```bash
# Windows
set PYTHONPATH=%CD%;%PYTHONPATH%

# Linux/Mac
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

**Giải pháp 2**: Thêm vào đầu script
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Lỗi: "ModuleNotFoundError"

1. Kiểm tra file `__init__.py` có trong thư mục
2. Chạy: `python verify_and_fix_imports.py`
3. Kiểm tra PYTHONPATH: `echo %PYTHONPATH%` (Windows) hoặc `echo $PYTHONPATH` (Linux)

##  Verification

Chạy script verification:
```bash
python verify_and_fix_imports.py
```

Nếu tất cả là , hệ thống đã sẵn sàng!

---

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
