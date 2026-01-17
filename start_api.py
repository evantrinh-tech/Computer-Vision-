import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    import uvicorn
    from src.serving.api import app

    print("=" * 60)
    print("Khởi động API Server...")
    print("=" * 60)
    
    # API sẽ chạy ở địa chỉ 0.0.0.0 (chấp nhận kết nối từ mọi IP)
    # Port 8000
    # Auto-reload=False (để ổn định trên production)
    print("\nAPI sẽ chạy tại: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("\nNhấn Ctrl+C để dừng server\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )