
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import cv2
import os

# Define paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "merged" / "test"
MODEL_DIR = ROOT_DIR / "models"
IMAGE_SIZE = 224

def load_latest_model():
    """Find and load the latest trained model"""
    subdirs = [d for d in MODEL_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        print("No models found!")
        return None
    
    # Sort by creation time (newest first)
    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Loading model from: {latest_dir}")
    
    model_path = latest_dir / "final_model.h5"
    if not model_path.exists():
        model_path = latest_dir / "best_stage2_fine_tuning.h5"
        
    if not model_path.exists():
        print(f"No .h5 model file found in {latest_dir}")
        return None
        
    print(f"Loading: {model_path.name}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_and_visualize(model):
    """Pick random images and verify predictions"""
    if not DATA_DIR.exists():
        print(f"Test data not found at {DATA_DIR}")
        return

    # Get all test images
    incident_images = list((DATA_DIR / "incident").glob("*.jpg")) + list((DATA_DIR / "incident").glob("*.png"))
    normal_images = list((DATA_DIR / "normal").glob("*.jpg")) + list((DATA_DIR / "normal").glob("*.png"))
    
    if not incident_images and not normal_images:
        print("No images found in test folder")
        return

    # Select 20 random images (balanced if possible)
    selected_images = []
    labels = [] # 1 for incident, 0 for normal
    
    # Try to take 10 from each, or whatever is available
    k = min(10, len(incident_images))
    if k > 0:
        selected_images.extend(random.sample(incident_images, k))
        labels.extend([1] * k)
        
    k = min(10, len(normal_images))
    if k > 0:
        selected_images.extend(random.sample(normal_images, k))
        labels.extend([0] * k)
        
    # Shuffle
    combined = list(zip(selected_images, labels))
    random.shuffle(combined)
    selected_images, true_labels = zip(*combined)

    # Plot (4 cols x 5 rows)
    plt.figure(figsize=(20, 25))
    plt.suptitle(f"Model Prediction Check (20 Random Test Images)", fontsize=16)

    correct_count = 0
    
    print("\nRunning predictions...")
    for i, (img_path, true_label) in enumerate(zip(selected_images, true_labels)):
        # Load and preprocess image
        img = tf.keras.utils.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = tf.keras.utils.img_to_array(img)
        img_batch = tf.expand_dims(img_array, 0) # Create batch axis
        
        # Predict
        # Note: Model includes Rescaling layer, so we input [0, 255] values
        pred_prob = model.predict(img_batch, verbose=0)[0][0]
        pred_label = 1 if pred_prob > 0.5 else 0
        
        # Check correctness
        is_correct = (pred_label == true_label)
        if is_correct:
            correct_count += 1
            color = 'green'
        else:
            color = 'red'
            
        # Text info
        true_text = "ACCIDENT" if true_label == 1 else "NORMAL"
        pred_text = "ACCIDENT" if pred_label == 1 else "NORMAL"
        conf_text = f"{pred_prob:.1%}" if pred_label == 1 else f"{1-pred_prob:.1%}"
        
        # Subplot
        plt.subplot(5, 4, i+1)
        plt.imshow(img_array.astype("uint8"))
        plt.title(f"True: {true_text}\nPred: {pred_text} ({conf_text})", color=color, fontweight='bold')
        plt.axis('off')
        
    accuracy = correct_count / len(selected_images)
    print(f"\nMini-batch Accuracy: {accuracy:.1%}")
    
    output_path = ROOT_DIR / "prediction_demo.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Demo image saved to: {output_path}")
    
    # Also open the image automatically
    try:
        os.startfile(output_path)
    except:
        pass

if __name__ == "__main__":
    print("="*60)
    print("TRAFFIC INCIDENT DETECTION - VISUAL CHECK")
    print("="*60)
    
    model = load_latest_model()
    if model:
        predict_and_visualize(model)
    else:
        print("Could not load model. Please train first.")
