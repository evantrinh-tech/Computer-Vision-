
import sys
import os
from pathlib import Path
import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.cnn import CNNModel

def verify_mobilenet_build():
    print("Verifying MobileNetV2 Build...")
    try:
        model = CNNModel(use_transfer_learning=True, base_model='MobileNetV2')
        model.build(input_shape=(224, 224, 3))
        print("Model built successfully.")
        
        # Check if Rescaling layer exists
        found_rescaling = False
        for layer in model.model.layers:
            if 'rescaling' in layer.name.lower():
                found_rescaling = True
                print(f"Propably found Rescaling layer: {layer.name}")
                break
        
        if not found_rescaling:
             # TensorFlow adds Rescaling layer implicitly if we use keras.layers.Rescaling
             # It might be wrapped or named differently. Let's check the first layer config
             first_layer = model.model.layers[1] # 0 is Input
             print(f"Second layer type: {type(first_layer)}")
             if 'Rescaling' in str(type(first_layer)):
                 found_rescaling = True
                 print(f"Found Rescaling layer at index 1: {first_layer.name}")

        if found_rescaling:
            print("✅ Rescaling layer verification PASSED.")
        else:
            print("⚠️ Rescaling layer not explicitly found by name check (check model summary).")

        # dummy prediction
        dummy_input = np.zeros((1, 224, 224, 3))
        output = model.model.predict(dummy_input)
        print(f"Output shape: {output.shape}")
        
        if output.shape == (1, 1):
             print("✅ Output shape verification PASSED.")
        else:
             print(f"❌ Output shape mismatch. Expected (1, 1), got {output.shape}")
             
        model.model.summary()

    except Exception as e:
        print(f"❌ Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_mobilenet_build()
