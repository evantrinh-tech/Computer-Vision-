import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import os

# Configuration
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "test" # Dùng tập Test để khách quan
MODEL_DIR = ROOT_DIR / "models"
IMAGE_SIZE = 224
BATCH_SIZE = 32

def load_latest_model():
    """Tìm và load model mới nhất"""
    subdirs = [d for d in MODEL_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        print("Không tìm thấy thư mục model nào!")
        return None, None
    
    # Sắp xếp lấy thư mục mới nhất
    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    model_path = latest_dir / "final_model.h5"
    if not model_path.exists():
        model_path = latest_dir / "best_stage2_fine_tuning.h5"
    
    if not model_path.exists():
        print(f"Không tìm thấy file model .h5 trong {latest_dir}")
        return None, None
        
    print(f"Đang load model: {model_path.name} từ {latest_dir.name}")
    return tf.keras.models.load_model(model_path), latest_dir

def generate_roc():
    print("\n" + "="*60)
    print("AI PERFORMANCE METRICS - ROC CURVE GENERATION")
    print("="*60)

    model, run_dir = load_latest_model()
    if not model: return

    # 1. Load Data
    print("Đang nạp dữ liệu Test...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False # Quan trọng: Giữ nguyên thứ tự để khớp nhãn
    )

    # 2. Dự đoán xác suất
    print("Đang chạy dự đoán (Inference)...")
    y_true = []
    y_probs = []

    for images, labels in test_ds:
        # Lưu ý: Model đã có tầng Rescaling(1./255) bên trong
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_probs.extend(probs.flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # 3. Tính toán ROC và AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # 4. Vẽ biểu đồ (STEM Elegance Style)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#d62728', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Báo động nhầm)', fontsize=12)
    plt.ylabel('True Positive Rate (Độ nhạy - Catch Rate)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) - Traffic Incident Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Thêm ghi chú về AUC
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=15, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#1f77b4'))

    # Lưu ảnh
    output_path = run_dir / "roc_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(ROOT_DIR / "roc_curve_latest.png", dpi=150, bbox_inches='tight') # Bản copy ở root để dễ tìm
    
    print("\n" + "-"*30)
    print(f"KẾT QUẢ ĐÁNH GIÁ")
    print("-"*30)
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"ROC Curve đã lưu tại: {output_path}")
    print("-"*30)
    print("\n[SUCCESS] Biểu đồ ROC đã sẵn sàng cho Slide của bạn.")
    
    plt.show()

if __name__ == "__main__":
    generate_roc()
