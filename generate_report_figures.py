
import matplotlib.pyplot as plt
import os
import random
import shutil
from pathlib import Path
from PIL import Image

# Setup Paths
ROOT_PROJECT = Path(r"d:\Computer Vision\Computer-Vision Project\Computer-Vision-")
KAGGLE_DATA_DIR = ROOT_PROJECT / "data" / "kaggle" / "archive" / "data" # Source 1
CUSTOM_DATA_DIR = ROOT_PROJECT / "data" / "images" # Source 2 (Scraped/Custom)
REPORT_FIG_DIR = Path(r"d:\Computer Vision\Computer-Vision Project\docs\report\Images")
MODEL_DIR = ROOT_PROJECT / "models" / "traffic_incident_tf_20260127_232904"

# Create Figures Dir
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

def get_images_from_dir(base_dir, normal_name="normal", incident_name="incident"):
    """Helper to get image paths from a source directory"""
    normal_dir = base_dir / normal_name
    incident_dir = base_dir / incident_name
    
    # Check if dirs exist (case insensitive fallback could be added but simple is better)
    if not normal_dir.exists(): # Try capitalized if lowercase fails (e.g. Kaggle style)
         normal_dir = base_dir / "Non Accident"
    if not incident_dir.exists():
         incident_dir = base_dir / "Accident"

    normal_imgs = []
    incident_imgs = []
    
    if normal_dir.exists():
        normal_imgs = list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.png"))
    
    if incident_dir.exists():
        incident_imgs = list(incident_dir.glob("*.jpg")) + list(incident_dir.glob("*.png"))
        
    return normal_imgs, incident_imgs

def create_mixed_sample_grid(split_name, output_path, title, num_samples=9):
    """
    Create a grid with mixed samples from Kaggle and Custom data
    split_name: 'train' or 'test' (applies to Kaggle structure)
    """
    
    # 1. Get Kaggle Samples (Source 1)
    kaggle_split_dir = KAGGLE_DATA_DIR / split_name
    k_norm, k_inc = get_images_from_dir(kaggle_split_dir, "Non Accident", "Accident")
    
    # 2. Get Custom Samples (Source 2)
    # Assuming Custom dir structure is flat or has train/test subdirs? 
    # Based on previous code, it seemed flat: data/images/normal
    # We will sample from the root of custom dir for both train/test figures to ensure we show custom data
    c_norm, c_inc = get_images_from_dir(CUSTOM_DATA_DIR, "normal", "incident")
    
    print(f"[{split_name.upper()}] Source Kaggle: {len(k_norm)} Normal, {len(k_inc)} Incident")
    print(f"[{split_name.upper()}] Source Custom: {len(c_norm)} Normal, {len(c_inc)} Incident")

    samples = []
    
    # We want roughly equal split between Sources (Kaggle vs Custom)
    # And roughly equal split between Classes (Normal vs Incident)
    
    # Target: ~4-5 images total from Kaggle, ~4-5 from Custom
    num_kaggle = num_samples // 2 + 1 # 5
    num_custom = num_samples - num_kaggle # 4
    
    # Helper to pick balanced classes from a source
    def pick_balanced(img_pool_norm, img_pool_inc, count, source_tag):
        picked = []
        n_inc = count // 2
        n_norm = count - n_inc
        
        if len(img_pool_inc) >= n_inc:
            picked.extend([(img, "Incident", source_tag) for img in random.sample(img_pool_inc, n_inc)])
        elif img_pool_inc: # Take what we have
             picked.extend([(img, "Incident", source_tag) for img in random.sample(img_pool_inc, len(img_pool_inc))])
             
        if len(img_pool_norm) >= n_norm:
            picked.extend([(img, "Normal", source_tag) for img in random.sample(img_pool_norm, n_norm)])
        elif img_pool_norm:
             picked.extend([(img, "Normal", source_tag) for img in random.sample(img_pool_norm, len(img_pool_norm))])
        
        return picked

    # Pick samples
    samples.extend(pick_balanced(k_norm, k_inc, num_kaggle, "Kaggle"))
    samples.extend(pick_balanced(c_norm, c_inc, num_custom, "Custom"))
    
    # Shuffle
    random.shuffle(samples)
    
    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    for i, (img_path, label, source) in enumerate(samples):
        if i >= 9: break
        ax = axes[i // 3, i % 3]
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Color label based on class
            color = "red" if label == "Incident" else "green"
            
            # Title: Class (Source)
            ax.set_title(f"{label}\n({source})", color=color, fontsize=10, fontweight='bold')
            ax.axis("off")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Created {output_path} with {len(samples)} images mixed from Kaggle & Custom")


# Generate Figures
print("\n=== Generating Mixed Report Figures ===\n")
create_mixed_sample_grid("train", REPORT_FIG_DIR / "train_samples.png", "Training Data (Kaggle + Custom)")
create_mixed_sample_grid("test", REPORT_FIG_DIR / "test_samples.png", "Test Data (Kaggle + Custom)")

# Copy Confusion Matrix
print("\n=== Copying Confusion Matrix ===\n")
cm_src = MODEL_DIR / "confusion_matrix.png"
cm_dest = REPORT_FIG_DIR / "confusion_matrix.png"
if cm_src.exists():
    shutil.copy2(cm_src, cm_dest)
    print(f"✓ Copied confusion matrix to {cm_dest}")
else:
    print(f"⚠ Warning: Confusion matrix not found at {cm_src}")

print("\n=== Done! ===")
