
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT_DIR = Path(r"d:\Computer Vision\Computer-Vision Project\Computer-Vision-")
DATA_DIR = ROOT_DIR / "data" / "images" / "normal" # Use a normal traffic image
OUTPUT_DIR = Path(r"d:\Computer Vision\Computer-Vision Project\docs\report\Images")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_lab2_figures():
    # 1. Load a sample image (Real traffic scene)
    # Get first jpg in normal dir
    img_path = next(DATA_DIR.glob("*.jpg"), None)
    if not img_path:
        img_path = next(DATA_DIR.glob("*.png"), None)
    
    if not img_path:
        print("No images found in data/images/normal")
        return

    print(f"Processing sample: {img_path.name}")
    
    # Read image
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. Sobel Vertical Edge ---
    # Sobel x=1, y=0 (Vertical edges)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_abs = np.absolute(sobelx)
    sobel_8u = np.uint8(sobel_abs)
    
    # Save Sobel
    # Typically Sobel output is grayscale, but for report usually shown as such
    plt.figure(figsize=(6, 4))
    plt.imshow(sobel_8u, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lab2_sobel.png", dpi=150, bbox_inches='tight')
    print(f"Generated lab2_sobel.png")

    # --- 3. Median Filter ---
    # Good for salt-and-pepper noise, but let's apply to original RGB
    median = cv2.medianBlur(img_rgb, 5) # kernel size 5
    
    plt.figure(figsize=(6, 4))
    plt.imshow(median)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lab2_median.jpg", dpi=150, bbox_inches='tight')
    print(f"Generated lab2_median.jpg")

    # --- 4. Bilateral Filter ---
    # Keeps edges sharp while smoothing
    # d=9, sigmaColor=75, sigmaSpace=75
    bilateral = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(bilateral)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lab2_bilateral.jpg", dpi=150, bbox_inches='tight')
    print(f"Generated lab2_bilateral.jpg")

    print("Done regenerating Lab 2 figures with real data.")

if __name__ == "__main__":
    generate_lab2_figures()
