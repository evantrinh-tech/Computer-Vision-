import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List, Optional

# ==========================================
# --- 1. CẤU HÌNH (QUAN TRỌNG: ĐỊNH NGHĨA DATA_DIR Ở ĐÂY) ---
# ==========================================
DATA_DIR = r"d:\Computer Vision\Computer-Vision Project\Computer-Vision-\data\images"
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL DEFINITION (Sử dụng EfficientNet-B0) ---
class TrafficIncidentModel(nn.Module):
    def __init__(self, use_pretrained: bool = True):
        super(TrafficIncidentModel, self).__init__()
        
        # Load EfficientNet-B0 (Tốt hơn và hiện đại hơn MobileNetV2)
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights).features
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # EfficientNet-B0 có 1280 channels ở lớp cuối cùng giống MobileNetV2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256), # Tăng lên 256 để học sâu hơn
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# --- 2. EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss: float, model: nn.Module, path: Path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: Path):
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# --- 3. TRAINER CLASS ---
class ModelTrainer:
    def __init__(self, data_root: str, run_name: str = "run"):
        self.device = DEVICE
        self.data_root = Path(data_root)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f"models/{run_name}_{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        self.model = TrafficIncidentModel().to(self.device)
        self.criterion = nn.BCELoss()
        
    def _setup_logging(self):
        log_file = self.run_dir / "training.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, batch_size: int = 16):
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(self.data_root)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_set.dataset.transform = train_transform
        val_set.dataset.transform = val_transform

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        self.logger.info(f"Dữ liệu sẵn sàng: {len(train_set)} ảnh train, {len(val_set)} ảnh val.")

    def fit(self, stage_name: str, epochs: int, lr: float, freeze_backbone: bool = True, patience: int = 5):
        self.logger.info(f"\n--- {stage_name} ---")
        for param in self.model.backbone.parameters():
            param.requires_grad = not freeze_backbone
            
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stopping = EarlyStopping(patience=patience)
        best_path = self.run_dir / f"best_{stage_name.lower().replace(' ', '_')}.pth"
        
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'lr': []}

        for epoch in range(epochs):
            curr_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(curr_lr)
            
            self.model.train()
            t_loss, t_correct = 0.0, 0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False):
                images, labels = images.to(self.device), labels.to(self.device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * images.size(0)
                t_correct += torch.sum((outputs > 0.5).float() == labels.data)
            
            self.model.eval()
            v_loss, v_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device).float().view(-1, 1)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    v_loss += loss.item() * images.size(0)
                    v_correct += torch.sum((outputs > 0.5).float() == labels.data)
            
            metrics = {
                'loss': t_loss / len(self.train_loader.dataset),
                'acc': (t_correct.double() / len(self.train_loader.dataset)).item(),
                'val_loss': v_loss / len(self.val_loader.dataset),
                'val_acc': (v_correct.double() / len(self.val_loader.dataset)).item()
            }
            for k, v in metrics.items(): history[k].append(v)
            scheduler.step()
            self.logger.info(f"Epoch {epoch+1} | LR: {curr_lr:.2e} | Loss: {metrics['loss']:.4f} Acc: {metrics['acc']:.4f} | Val Loss: {metrics['val_loss']:.4f} Val Acc: {metrics['val_acc']:.4f}")
            
            early_stopping(metrics['val_loss'], self.model, best_path)
            if early_stopping.early_stop:
                self.logger.warning("Dừng sớm (Early Stopping) do không có tiến bộ.")
                break
        
        self.model.load_state_dict(torch.load(best_path, weights_only=True))
        self._plot_results(history, stage_name)

    def _plot_results(self, history: Dict[str, List[float]], title: str):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.plot(history['acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title(f'Accuracy - {title}'); plt.legend()
        plt.subplot(1, 3, 2); plt.plot(history['loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title(f'Loss - {title}'); plt.legend()
        plt.subplot(1, 3, 3); plt.plot(history['lr'], label='LR', color='orange'); plt.title('Learning Rate'); plt.yscale('log'); plt.legend()
        plt.tight_layout()
        plt.show()

# --- 4. CHẠY HUẤN LUYỆN ---
if __name__ == "__main__":
    # Đảm bảo DATA_DIR đã được định nghĩa ở trên đầu file
    trainer = ModelTrainer(data_root=DATA_DIR, run_name="traffic_efficientnet")
    trainer.prepare_data(batch_size=16)
    
    # Stage 1: Huấn luyện lớp phân loại (Freeze backbone)
    trainer.fit(stage_name="Stage 1 - Transfer Learning", epochs=15, lr=1e-3, freeze_backbone=True)
    
    # Stage 2: Fine-tuning toàn bộ (Unfreeze)
    trainer.fit(stage_name="Stage 2 - Fine Tuning", epochs=30, lr=1e-5, freeze_backbone=False)
    
    print("\n🎉 HOÀN TẤT HUẤN LUYỆN VỚI EFFICIENTNET-B0!")
