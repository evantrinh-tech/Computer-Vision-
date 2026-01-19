import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# --- CẤU HÌNH ---
DATA_DIR = r"d:\Computer Vision\Computer-Vision Project\Computer-Vision-\data\images"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16 # Batch size nhỏ cho dataset ít ảnh
EPOCHS_STAGE1 = 20 # Train lớp cuối
EPOCHS_STAGE2 = 30 # Fine-tune toàn bộ
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5

def build_model():
    # 1. Load Base Model (MobileNetV2)
    base_model = applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Đóng băng base model ở giai đoạn 1
    base_model.trainable = False
    
    # 2. Thêm các lớp phân loại mới (Head)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') # Phân loại 2 lớp
    ])
    
    return model, base_model

def plot_history(history, title="Training Results"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # 1. Khởi tạo Data Generator với AUTO AUGMENTATION
    # Rất quan trọng để tránh overfitting với 180 ảnh
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,      # Xoay ảnh
        width_shift_range=0.2,  # Dịch ngang
        height_shift_range=0.2, # Dịch dọc
        shear_range=0.2,        # Biến dạng hình học
        zoom_range=0.2,         # Phóng to/nhỏ
        horizontal_flip=True,   # Lật ngang
        fill_mode='nearest',
        validation_split=0.2     # Chia 20% dữ liệu để test
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # 2. Build Model
    model, base_model = build_model()
    model.compile(optimizer=optimizers.Adam(learning_rate=LR_STAGE1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\n🚀 GIAI ĐOẠN 1: Huấn luyện các lớp phân loại mới...")
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_STAGE1,
        validation_data=validation_generator,
        verbose=1
    )
    plot_history(history1, "Stage 1: Transfer Learning")

    # 3. GIAI ĐOẠN 2: FINE-TUNING
    # Mở khóa base model để huấn luyện chuyên sâu hơn
    print("\n🔧 GIAI ĐOẠN 2: Fine-tuning toàn bộ mô hình...")
    base_model.trainable = True
    
    # Re-compile với learning rate cực nhỏ để không làm hỏng kiến thức cũ
    model.compile(optimizer=optimizers.Adam(learning_rate=LR_STAGE2),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history2 = model.fit(
        train_generator,
        epochs=EPOCHS_STAGE2,
        validation_data=validation_generator,
        verbose=1
    )
    plot_history(history2, "Stage 2: Fine-tuning")

    # 4. Lưu Model
    save_path = "models/FineTuned_MobileNetV2.h5"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"\n✅ HOÀN TẤT! Model đã được lưu tại: {save_path}")

if __name__ == "__main__":
    main()
