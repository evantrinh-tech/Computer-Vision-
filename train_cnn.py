import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Using relative paths for better portability
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "images"
IMAGE_SIZE = 224
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class ModelTrainer:
    def __init__(self, data_path: Path, run_name: str = "traffic_pro"):
        self.device = DEVICE
        self.data_path = data_path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT_DIR / "models" / f"{run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to both file and console
        log_file = self.run_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Results will be saved to: {self.run_dir}")
        
        # Load EfficientNet-B0
        try:
            from torchvision.models import EfficientNet_B0_Weights
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(self.device)
        except:
            # Fallback for older torchvision versions
            self.model = models.efficientnet_b0(pretrained=True).to(self.device)
            
        # Replace classifier for binary classification
        in_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_ftrs, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.criterion = nn.BCELoss()

    def prepare_data(self):
        if not self.data_path.exists():
            self.logger.error(f"Data directory not found: {self.data_path}")
            return False

        dataset = datasets.ImageFolder(self.data_path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Image transformations
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_set.dataset.transform = transform
        val_set.dataset.transform = transform
        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
        self.logger.info(f"Loaded {len(train_set)} training and {len(val_set)} validation images.")
        return True

    def fit(self, stage_name: str, epochs: int, lr: float, freeze: bool = True):
        self.logger.info(f"STARTING STAGE: {stage_name}")
        
        # Freeze or unfreeze backbone layers
        for name, param in self.model.named_parameters():
            if "classifier" not in name: 
                param.requires_grad = not freeze
            
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        early_stopping = EarlyStopping(patience=5, verbose=True)
        best_model_path = self.run_dir / f"best_{stage_name.lower().replace(' ', '_')}.pth"
        
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            t_loss, t_correct = 0, 0
            for imgs, lbls in tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False):
                imgs, lbls = imgs.to(self.device), lbls.to(self.device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * imgs.size(0)
                t_correct += torch.sum((outputs > 0.5).float() == lbls.data)
            
            # Validation phase
            self.model.eval()
            v_loss, v_correct = 0, 0
            with torch.no_grad():
                for imgs, lbls in self.val_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device).float().view(-1, 1)
                    outputs = self.model(imgs)
                    v_loss += self.criterion(outputs, lbls).item() * imgs.size(0)
                    v_correct += torch.sum((outputs > 0.5).float() == lbls.data)
            
            metrics = {
                'loss': t_loss / len(self.train_loader.dataset),
                'acc': (t_correct.double() / len(self.train_loader.dataset)).item(),
                'val_loss': v_loss / len(self.val_loader.dataset),
                'val_acc': (v_correct.double() / len(self.val_loader.dataset)).item()
            }
            for k in history: 
                history[k].append(metrics[k])
                
            self.logger.info(f"Epoch {epoch+1} | Loss: {metrics['loss']:.4f} Acc: {metrics['acc']:.4f} | Val Loss: {metrics['val_loss']:.4f} Val Acc: {metrics['val_acc']:.4f}")

            # Check for Early Stopping
            early_stopping(metrics['val_loss'], self.model, best_model_path)
            if early_stopping.early_stop:
                self.logger.warning("Early Stopping triggered. Halting training to prevent overfitting.")
                break

        self._plot_smooth(history, stage_name)

    def _plot_smooth(self, history, title):
        def smooth(data, w=0.6):
            res = [data[0]]
            for p in data[1:]: 
                res.append(res[-1]*w + p*(1-w))
            return res

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(smooth(history['acc']), label='Train')
        plt.plot(smooth(history['val_acc']), label='Val')
        plt.title(f"{title} - Accuracy")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(smooth(history['loss']), label='Train')
        plt.plot(smooth(history['val_loss']), label='Val')
        plt.title(f"{title} - Loss")
        plt.legend()
        
        chart_path = self.run_dir / f"{title.lower().replace(' ', '_')}_chart.png"
        plt.savefig(chart_path)
        self.logger.info(f"Chart saved to {chart_path}")
        plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("ITS - TRAFFIC INCIDENT DETECTION SYSTEM TRAINING")
    print("=" * 60)
    
    trainer = ModelTrainer(DATA_DIR)
    if trainer.prepare_data():
        # Start training process
        trainer.fit("Transfer_Learning", epochs=20, lr=1e-3, freeze=True)
        trainer.fit("Fine_Tuning", epochs=50, lr=1e-5, freeze=False)
        
        print("\nTraining completed successfully.")
        print(f"Models and logs are available in: {trainer.run_dir}")
    else:
        print("\nTraining failed due to missing data.")