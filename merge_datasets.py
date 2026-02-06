"""
Script to merge Kaggle CCTV dataset with existing dataset
Combines data/kaggle/archive/data with data/images
"""
import os
import shutil
from pathlib import Path

def merge_datasets():
    """Merge Kaggle dataset with existing dataset"""
    
    # Define paths
    base_dir = Path(__file__).parent / "data"
    kaggle_dir = base_dir / "kaggle" / "archive" / "data"
    existing_dir = base_dir / "images"
    merged_dir = base_dir / "merged"
    
    # Create merged directory structure
    for split in ["train", "val", "test"]:
        for class_name in ["incident", "normal"]:
            (merged_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MERGING DATASETS")
    print("=" * 60)
    
    # Copy Kaggle data
    print("\n[1/2] Copying Kaggle dataset...")
    kaggle_mapping = {
        "Accident": "incident",
        "Non Accident": "normal"
    }
    
    kaggle_counts = {"train": {}, "val": {}, "test": {}}
    
    for split in ["train", "val", "test"]:
        for kaggle_class, our_class in kaggle_mapping.items():
            src_dir = kaggle_dir / split / kaggle_class
            dst_dir = merged_dir / split / our_class
            
            if src_dir.exists():
                count = 0
                for img_file in src_dir.glob("*.jpg"):
                    # Rename to avoid conflicts
                    new_name = f"kaggle_{split}_{img_file.name}"
                    shutil.copy2(img_file, dst_dir / new_name)
                    count += 1
                kaggle_counts[split][our_class] = count
                print(f"  {split}/{our_class}: {count} images")
    
    # Copy existing data (only from images folder, not test)
    print("\n[2/2] Copying existing dataset...")
    existing_counts = {"train": {}, "val": {}, "test": {}}
    
    # Existing data goes to train/val (we'll split it)
    for class_name in ["incident", "normal"]:
        src_dir = existing_dir / class_name
        if src_dir.exists():
            images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
            
            # Split: 80% train, 20% val
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copy to train
            for img_file in train_images:
                new_name = f"existing_{img_file.name}"
                shutil.copy2(img_file, merged_dir / "train" / class_name / new_name)
            existing_counts["train"][class_name] = len(train_images)
            
            # Copy to val
            for img_file in val_images:
                new_name = f"existing_{img_file.name}"
                shutil.copy2(img_file, merged_dir / "val" / class_name / new_name)
            existing_counts["val"][class_name] = len(val_images)
            
            print(f"  train/{class_name}: {len(train_images)} images")
            print(f"  val/{class_name}: {len(val_images)} images")
    
    # Copy existing test data
    test_dir = base_dir / "test"
    if test_dir.exists():
        for class_name in ["incident", "normal"]:
            src_dir = test_dir / class_name
            if src_dir.exists():
                count = 0
                for img_file in src_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        new_name = f"existing_test_{img_file.name}"
                        shutil.copy2(img_file, merged_dir / "test" / class_name / new_name)
                        count += 1
                existing_counts["test"][class_name] = count
                print(f"  test/{class_name}: {count} images")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    
    total_counts = {"train": {}, "val": {}, "test": {}}
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        for class_name in ["incident", "normal"]:
            kaggle_count = kaggle_counts[split].get(class_name, 0)
            existing_count = existing_counts[split].get(class_name, 0)
            total = kaggle_count + existing_count
            total_counts[split][class_name] = total
            print(f"  {class_name}: {total} images (Kaggle: {kaggle_count}, Existing: {existing_count})")
    
    # Grand total
    print("\n" + "=" * 60)
    print("TOTAL DATASET:")
    grand_total = 0
    for split in ["train", "val", "test"]:
        split_total = sum(total_counts[split].values())
        grand_total += split_total
        print(f"  {split}: {split_total} images")
    print(f"\nGRAND TOTAL: {grand_total} images")
    print("=" * 60)
    
    print(f"\n✅ Merged dataset saved to: {merged_dir}")
    return merged_dir

if __name__ == "__main__":
    merged_dir = merge_datasets()
