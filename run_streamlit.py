import sys
import os
from pathlib import Path
import subprocess

os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    try:
        result = subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'app.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--server.headless=true',
            '--browser.gatherUsageStats=false',
            '--logger.level=debug'
        ], env=os.environ.copy(), cwd=str(Path(__file__).parent))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nĐã dừng server.")
        sys.exit(0)
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)