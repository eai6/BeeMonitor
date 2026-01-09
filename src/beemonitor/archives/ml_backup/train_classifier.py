# """Train a lightweight CNN classifier for bee vs noise classification.

# Simple, fast model for filtering FG mask blobs before YOLO inference.
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from pathlib import Path
# import cv2
# import numpy as np
# from typing import Tuple, List
# import logging
# from tqdm import tqdm
# import argparse
# import json
# import random

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class BeeClassifierDataset(Dataset):
#     """Dataset for bee vs noise classification."""
    
#     def __init__(
#         self,
#         data_dir: Path,
#         split: str = 'train',
#         image_size: int = 64,
#         augment: bool = True,
#         balance_classes: bool = True
#     ):
#         """Initialize dataset.
        
#         Args:
#             data_dir: Root data directory (contains bee/ and noise/ subdirs)
#             split: Not used, included for compatibility
#             image_size: Target image size
#             augment: Apply data augmentation
#             balance_classes: Randomly undersample majority class to balance
#         """
#         self.data_dir = Path(data_dir)
#         self.image_size = image_size
#         self.augment = augment
        
#         # Collect samples by class
#         bee_samples = []
#         noise_samples = []
        
#         # Bee class (label 1)
#         bee_dir = self.data_dir / 'bee'
#         if bee_dir.exists():
#             for img_path in bee_dir.glob('*.jpg'):
#                 bee_samples.append((str(img_path), 1))
        
#         # Noise class (label 0)
#         noise_dir = self.data_dir / 'noise'
#         if noise_dir.exists():
#             for img_path in noise_dir.glob('*.jpg'):
#                 noise_samples.append((str(img_path), 0))
        
#         logger.info(f"Original counts - Bee: {len(bee_samples)}, Noise: {len(noise_samples)}")
        
#         # Balance classes if requested
#         if balance_classes:
#             min_count = min(len(bee_samples), len(noise_samples))
            
#             # Randomly sample to balance
#             if len(bee_samples) > min_count:
#                 bee_samples = random.sample(bee_samples, min_count)
#             if len(noise_samples) > min_count:
#                 noise_samples = random.sample(noise_samples, min_count)
            
#             logger.info(f"Balanced counts - Bee: {len(bee_samples)}, Noise: {len(noise_samples)}")
        
#         # Combine samples
#         self.samples = bee_samples + noise_samples
        
#         # Shuffle
#         random.shuffle(self.samples)
        
#         logger.info(f"Total samples: {len(self.samples)}")
        
#         # Transforms
#         if augment:
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(10),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2),
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#             ])
    
#     def __len__(self) -> int:
#         return len(self.samples)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
#         img_path, label = self.samples[idx]
        
#         # Load image
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Apply transforms
#         img = self.transform(img)
        
#         return img, label


# class SimpleBeeClassifier(nn.Module):
#     """Lightweight CNN for bee vs noise classification."""
    
#     def __init__(self, num_classes: int = 2):
#         super(SimpleBeeClassifier, self).__init__()
        
#         # Simple conv backbone
#         self.features = nn.Sequential(
#             # Conv block 1
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             # Conv block 2
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             # Conv block 3
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             # Conv block 4
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1)
#         )
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# class BeeClassifierTrainer:
#     """Trainer for bee classifier."""
    
#     def __init__(
#         self,
#         train_dir: Path,
#         val_dir: Path,
#         output_dir: Path,
#         image_size: int = 64,
#         batch_size: int = 32,
#         num_epochs: int = 20,
#         learning_rate: float = 0.001,
#         device: str = 'cuda',
#         balance_classes: bool = True,
#         checkpoint_interval: int = 5
#     ):
#         """Initialize trainer.
        
#         Args:
#             train_dir: Training data directory
#             val_dir: Validation data directory
#             output_dir: Output directory for models
#             image_size: Input image size
#             batch_size: Batch size
#             num_epochs: Number of epochs
#             learning_rate: Learning rate
#             device: Device to use ('cuda' or 'cpu')
#             balance_classes: Balance training data by undersampling
#             checkpoint_interval: Save checkpoint every N epochs (0 to disable)
#         """
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.num_epochs = num_epochs
#         self.checkpoint_interval = checkpoint_interval
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
#         logger.info(f"Using device: {self.device}")
        
#         # Create datasets
#         self.train_dataset = BeeClassifierDataset(
#             train_dir, image_size=image_size, augment=True,
#             balance_classes=balance_classes
#         )
#         self.val_dataset = BeeClassifierDataset(
#             val_dir, image_size=image_size, augment=False,
#             balance_classes=False  # Don't balance validation
#         )
        
#         # Create dataloaders
#         self.train_loader = DataLoader(
#             self.train_dataset, batch_size=batch_size,
#             shuffle=True, num_workers=4, pin_memory=True
#         )
#         self.val_loader = DataLoader(
#             self.val_dataset, batch_size=batch_size,
#             shuffle=False, num_workers=4, pin_memory=True
#         )
        
#         # Create model
#         self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
        
#         # Loss and optimizer
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode='max', patience=3, factor=0.5
#         )
        
#         # Training history
#         self.history = {
#             'train_loss': [],
#             'train_acc': [],
#             'val_loss': [],
#             'val_acc': []
#         }
    
#     def train_epoch(self) -> Tuple[float, float]:
#         """Train for one epoch."""
#         self.model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.train_loader, desc='Training')
#         for images, labels in pbar:
#             images = images.to(self.device)
#             labels = labels.to(self.device)
            
#             # Forward
#             self.optimizer.zero_grad()
#             outputs = self.model(images)
#             loss = self.criterion(outputs, labels)
            
#             # Backward
#             loss.backward()
#             self.optimizer.step()
            
#             # Stats
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
            
#             # Update progress bar
#             pbar.set_postfix({
#                 'loss': running_loss / (pbar.n + 1),
#                 'acc': 100. * correct / total
#             })
        
#         epoch_loss = running_loss / len(self.train_loader)
#         epoch_acc = 100. * correct / total
        
#         return epoch_loss, epoch_acc
    
#     def validate(self) -> Tuple[float, float]:
#         """Validate model."""
#         self.model.eval()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in tqdm(self.val_loader, desc='Validation'):
#                 images = images.to(self.device)
#                 labels = labels.to(self.device)
                
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, labels)
                
#                 running_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
        
#         val_loss = running_loss / len(self.val_loader)
#         val_acc = 100. * correct / total
        
#         return val_loss, val_acc
    
#     def train(self) -> None:
#         """Train the model."""
#         best_acc = 0.0
        
#         for epoch in range(self.num_epochs):
#             logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
#             # Train
#             train_loss, train_acc = self.train_epoch()
            
#             # Validate
#             val_loss, val_acc = self.validate()
            
#             # Update scheduler
#             self.scheduler.step(val_acc)
            
#             # Save history
#             self.history['train_loss'].append(train_loss)
#             self.history['train_acc'].append(train_acc)
#             self.history['val_loss'].append(val_loss)
#             self.history['val_acc'].append(val_acc)
            
#             logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
#             logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
#             # Save best model
#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 self._save_model('best_model.pth', epoch, val_acc)
#                 logger.info(f"New best model saved! Accuracy: {val_acc:.2f}%")
            
#             # Save latest
#             self._save_model('latest_model.pth', epoch, val_acc)
            
#             # Save interval checkpoint
#             if self.checkpoint_interval > 0 and (epoch + 1) % self.checkpoint_interval == 0:
#                 checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
#                 self._save_model(checkpoint_name, epoch, val_acc)
#                 logger.info(f"Checkpoint saved: {checkpoint_name}")
        
#         # Save final model
#         self._save_model('final_model.pth', self.num_epochs, val_acc)
        
#         # Save history
#         self._save_history()
        
#         logger.info(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    
#     def _save_model(self, filename: str, epoch: int, accuracy: float) -> None:
#         """Save model checkpoint."""
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'accuracy': accuracy,
#             'history': self.history
#         }
#         torch.save(checkpoint, self.output_dir / filename)
    
#     def _save_history(self) -> None:
#         """Save training history."""
#         with open(self.output_dir / 'history.json', 'w') as f:
#             json.dump(self.history, f, indent=2)


# def main():
#     trainer = BeeClassifierTrainer(
#         train_dir="/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/train",
#         val_dir="/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/eval",
#         output_dir='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/output',
#         image_size=64,
#         batch_size=32,
#         num_epochs=100,
#         learning_rate=0.001,
#         device='cpu',
#         balance_classes=True,
#         checkpoint_interval=10
#     )
    
#     trainer.train()

# if __name__ == '__main__':
#     main()







# """Train a lightweight CNN classifier for bee vs noise classification.

# Simple, fast model for filtering FG mask blobs before YOLO inference.
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from pathlib import Path
# import cv2
# import numpy as np
# from typing import Tuple, List
# import logging
# from tqdm import tqdm
# import argparse
# import json
# import random

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class BeeClassifierDataset(Dataset):
#     """Dataset for bee vs noise classification."""
    
#     def __init__(
#         self,
#         data_dir: Path,
#         split: str = 'train',
#         image_size: int = 64,
#         augment: bool = True,
#         balance_classes: bool = True
#     ):
#         """Initialize dataset.
        
#         Args:
#             data_dir: Root data directory (contains bee/ and noise/ subdirs)
#             split: Not used, included for compatibility
#             image_size: Target image size
#             augment: Apply data augmentation
#             balance_classes: Randomly undersample majority class to balance
#         """
#         self.data_dir = Path(data_dir)
#         self.image_size = image_size
#         self.augment = augment
        
#         # Collect samples by class
#         bee_samples = []
#         noise_samples = []
        
#         # Bee class (label 1)
#         bee_dir = self.data_dir / 'bee'
#         if bee_dir.exists():
#             for img_path in bee_dir.glob('*.jpg'):
#                 bee_samples.append((str(img_path), 1))
        
#         # Noise class (label 0)
#         noise_dir = self.data_dir / 'noise'
#         if noise_dir.exists():
#             for img_path in noise_dir.glob('*.jpg'):
#                 noise_samples.append((str(img_path), 0))
        
#         logger.info(f"Original counts - Bee: {len(bee_samples)}, Noise: {len(noise_samples)}")
        
#         # Balance classes if requested
#         if balance_classes:
#             min_count = min(len(bee_samples), len(noise_samples))
            
#             # Randomly sample to balance
#             if len(bee_samples) > min_count:
#                 bee_samples = random.sample(bee_samples, min_count)
#             if len(noise_samples) > min_count:
#                 noise_samples = random.sample(noise_samples, min_count)
            
#             logger.info(f"Balanced counts - Bee: {len(bee_samples)}, Noise: {len(noise_samples)}")
        
#         # Combine samples
#         self.samples = bee_samples + noise_samples
        
#         # Shuffle
#         random.shuffle(self.samples)
        
#         logger.info(f"Total samples: {len(self.samples)}")
        
#         # Transforms
#         if augment:
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(10),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2),
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#             ])
    
#     def __len__(self) -> int:
#         return len(self.samples)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
#         img_path, label = self.samples[idx]
        
#         # Load image
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Apply transforms
#         img = self.transform(img)
        
#         return img, label


# class SimpleBeeClassifier(nn.Module):
#     """Lightweight CNN for bee vs noise classification."""
    
#     def __init__(self, num_classes: int = 2):
#         super(SimpleBeeClassifier, self).__init__()
        
#         # Simple conv backbone
#         self.features = nn.Sequential(
#             # Conv block 1
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             # Conv block 2
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             # Conv block 3
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             # Conv block 4
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1)
#         )
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# class BeeClassifierTrainer:
#     """Trainer for bee classifier."""
    
#     def __init__(
#         self,
#         train_dir: Path,
#         val_dir: Path,
#         output_dir: Path,
#         image_size: int = 64,
#         batch_size: int = 32,
#         num_epochs: int = 20,
#         learning_rate: float = 0.001,
#         device: str = 'cuda',
#         balance_classes: bool = True,
#         checkpoint_interval: int = 5,
#         weight_decay: float = 1e-4,
#         early_stopping_patience: int = 10
#     ):
#         """Initialize trainer.
        
#         Args:
#             train_dir: Training data directory
#             val_dir: Validation data directory
#             output_dir: Output directory for models
#             image_size: Input image size
#             batch_size: Batch size
#             num_epochs: Number of epochs
#             learning_rate: Learning rate
#             device: Device to use ('cuda' or 'cpu')
#             balance_classes: Balance training data by undersampling
#             checkpoint_interval: Save checkpoint every N epochs (0 to disable)
#             weight_decay: L2 regularization strength
#             early_stopping_patience: Stop if no improvement for N epochs (0 to disable)
#         """
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.num_epochs = num_epochs
#         self.checkpoint_interval = checkpoint_interval
#         self.early_stopping_patience = early_stopping_patience
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
#         logger.info(f"Using device: {self.device}")
        
#         # Create datasets
#         self.train_dataset = BeeClassifierDataset(
#             train_dir, image_size=image_size, augment=True,
#             balance_classes=balance_classes
#         )
#         self.val_dataset = BeeClassifierDataset(
#             val_dir, image_size=image_size, augment=False,
#             balance_classes=False  # Don't balance validation
#         )
        
#         # Create dataloaders
#         self.train_loader = DataLoader(
#             self.train_dataset, batch_size=batch_size,
#             shuffle=True, num_workers=4, pin_memory=True
#         )
#         self.val_loader = DataLoader(
#             self.val_dataset, batch_size=batch_size,
#             shuffle=False, num_workers=4, pin_memory=True
#         )
        
#         # Create model
#         self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
        
#         # Loss and optimizer
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(
#             self.model.parameters(), 
#             lr=learning_rate,
#             weight_decay=weight_decay  # L2 regularization
#         )
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode='max', patience=3, factor=0.5
#         )
        
#         # Training history
#         self.history = {
#             'train_loss': [],
#             'train_acc': [],
#             'val_loss': [],
#             'val_acc': []
#         }
    
#     def train_epoch(self) -> Tuple[float, float]:
#         """Train for one epoch."""
#         self.model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.train_loader, desc='Training')
#         for images, labels in pbar:
#             images = images.to(self.device)
#             labels = labels.to(self.device)
            
#             # Forward
#             self.optimizer.zero_grad()
#             outputs = self.model(images)
#             loss = self.criterion(outputs, labels)
            
#             # Backward
#             loss.backward()
#             self.optimizer.step()
            
#             # Stats
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
            
#             # Update progress bar
#             pbar.set_postfix({
#                 'loss': running_loss / (pbar.n + 1),
#                 'acc': 100. * correct / total
#             })
        
#         epoch_loss = running_loss / len(self.train_loader)
#         epoch_acc = 100. * correct / total
        
#         return epoch_loss, epoch_acc
    
#     def validate(self) -> Tuple[float, float]:
#         """Validate model."""
#         self.model.eval()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for images, labels in tqdm(self.val_loader, desc='Validation'):
#                 images = images.to(self.device)
#                 labels = labels.to(self.device)
                
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, labels)
                
#                 running_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
        
#         val_loss = running_loss / len(self.val_loader)
#         val_acc = 100. * correct / total
        
#         return val_loss, val_acc
    
#     def train(self) -> None:
#         """Train the model."""
#         best_acc = 0.0
#         epochs_without_improvement = 0
        
#         for epoch in range(self.num_epochs):
#             logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
#             # Train
#             train_loss, train_acc = self.train_epoch()
            
#             # Validate
#             val_loss, val_acc = self.validate()
            
#             # Update scheduler
#             self.scheduler.step(val_acc)
            
#             # Save history
#             self.history['train_loss'].append(train_loss)
#             self.history['train_acc'].append(train_acc)
#             self.history['val_loss'].append(val_loss)
#             self.history['val_acc'].append(val_acc)
            
#             logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
#             logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
#             # Check for improvement
#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 epochs_without_improvement = 0
#                 self._save_model('best_model.pth', epoch, val_acc)
#                 logger.info(f"New best model saved! Accuracy: {val_acc:.2f}%")
#             else:
#                 epochs_without_improvement += 1
            
#             # Early stopping check
#             if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
#                 logger.info(f"\nEarly stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
#                 logger.info(f"Best validation accuracy: {best_acc:.2f}%")
#                 break
            
#             # Save latest
#             self._save_model('latest_model.pth', epoch, val_acc)
            
#             # Save interval checkpoint
#             if self.checkpoint_interval > 0 and (epoch + 1) % self.checkpoint_interval == 0:
#                 checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
#                 self._save_model(checkpoint_name, epoch, val_acc)
#                 logger.info(f"Checkpoint saved: {checkpoint_name}")
        
#         # Save final model
#         self._save_model('final_model.pth', epoch, val_acc)
        
#         # Save history
#         self._save_history()
        
#         logger.info(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    
#     def _save_model(self, filename: str, epoch: int, accuracy: float) -> None:
#         """Save model checkpoint."""
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'accuracy': accuracy,
#             'history': self.history
#         }
#         torch.save(checkpoint, self.output_dir / filename)
    
#     def _save_history(self) -> None:
#         """Save training history."""
#         with open(self.output_dir / 'history.json', 'w') as f:
#             json.dump(self.history, f, indent=2)









"""Train a lightweight CNN classifier for bee vs noise classification.

Simple, fast model for filtering FG mask blobs before YOLO inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, List
import logging
from tqdm import tqdm
import argparse
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeeClassifierDataset(Dataset):
    """Dataset for bee vs noise classification."""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        image_size: int = 64,
        augment: bool = True,
        balance_classes: bool = True,
        strong_augment: bool = False
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Root data directory (contains bee/ and noise/ subdirs)
            split: Not used, included for compatibility
            image_size: Target image size
            augment: Apply data augmentation
            balance_classes: Randomly undersample majority class to balance
            strong_augment: Apply stronger augmentation for better accuracy
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Collect samples by class
        bee_samples = []
        noise_samples = []
        
        # Bee class (label 1)
        bee_dir = self.data_dir / 'bee'
        if bee_dir.exists():
            for img_path in bee_dir.glob('*.jpg'):
                bee_samples.append((str(img_path), 1))
        
        # Noise class (label 0)
        noise_dir = self.data_dir / 'noise'
        if noise_dir.exists():
            for img_path in noise_dir.glob('*.jpg'):
                noise_samples.append((str(img_path), 0))
        
        logger.info(f"Original counts - Bee: {len(bee_samples)}, Noise: {len(noise_samples)}")
        
        # Balance classes if requested
        if balance_classes:
            min_count = min(len(bee_samples), len(noise_samples))
            
            # Randomly sample to balance
            if len(bee_samples) > min_count:
                bee_samples = random.sample(bee_samples, min_count)
            if len(noise_samples) > min_count:
                noise_samples = random.sample(noise_samples, min_count)
            
            logger.info(f"Balanced counts - Bee: {len(bee_samples)}, Noise: {len(noise_samples)}")
        
        # Combine samples
        self.samples = bee_samples + noise_samples
        
        # Shuffle
        random.shuffle(self.samples)
        
        logger.info(f"Total samples: {len(self.samples)}")
        
        # Transforms
        if augment:
            if strong_augment:
                # Stronger augmentation for better accuracy
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                # Standard augmentation
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img = self.transform(img)
        
        return img, label


class SimpleBeeClassifier(nn.Module):
    """Enhanced CNN for bee vs noise classification."""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleBeeClassifier, self).__init__()
        
        # Richer conv backbone with more filters and depth
        self.features = nn.Sequential(
            # Conv block 1 (64x64 -> 32x32)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 2 (32x32 -> 16x16)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 3 (16x16 -> 8x8)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 4 (8x8 -> 4x4)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Richer classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BeeClassifierTrainer:
    """Trainer for bee classifier."""
    
    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        output_dir: Path,
        image_size: int = 64,
        batch_size: int = 32,
        num_epochs: int = 20,
        learning_rate: float = 0.001,
        device: str = 'cuda',
        balance_classes: bool = True,
        checkpoint_interval: int = 5,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        strong_augment: bool = False
    ):
        """Initialize trainer.
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            output_dir: Output directory for models
            image_size: Input image size
            batch_size: Batch size
            num_epochs: Number of epochs
            learning_rate: Learning rate
            device: Device to use ('cuda' or 'cpu')
            balance_classes: Balance training data by undersampling
            checkpoint_interval: Save checkpoint every N epochs (0 to disable)
            weight_decay: L2 regularization strength
            early_stopping_patience: Stop if no improvement for N epochs (0 to disable)
            strong_augment: Use stronger data augmentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        if strong_augment:
            logger.info("Using STRONG data augmentation for better accuracy")
        
        # Create datasets
        self.train_dataset = BeeClassifierDataset(
            train_dir, image_size=image_size, augment=True,
            balance_classes=balance_classes, strong_augment=strong_augment
        )
        self.val_dataset = BeeClassifierDataset(
            val_dir, image_size=image_size, augment=False,
            balance_classes=False, strong_augment=False  # Never augment validation
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Create model
        self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay  # L2 regularization
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self) -> None:
        """Train the model."""
        best_acc = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Check for improvement
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvement = 0
                self._save_model('best_model.pth', epoch, val_acc)
                logger.info(f"New best model saved! Accuracy: {val_acc:.2f}%")
            else:
                epochs_without_improvement += 1
            
            # Early stopping check
            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                logger.info(f"Best validation accuracy: {best_acc:.2f}%")
                break
            
            # Save latest
            self._save_model('latest_model.pth', epoch, val_acc)
            
            # Save interval checkpoint
            if self.checkpoint_interval > 0 and (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                self._save_model(checkpoint_name, epoch, val_acc)
                logger.info(f"Checkpoint saved: {checkpoint_name}")
        
        # Save final model
        self._save_model('final_model.pth', epoch, val_acc)
        
        # Save history
        self._save_history()
        
        logger.info(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    
    def _save_model(self, filename: str, epoch: int, accuracy: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def _save_history(self) -> None:
        """Save training history."""
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    trainer = BeeClassifierTrainer(
        train_dir="/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/train",
        val_dir="/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/eval",
        output_dir='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/output2',
        image_size=64,
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001,
        device='cpu',
        balance_classes=True,
        checkpoint_interval=10,
        weight_decay=1e-4,
        early_stopping_patience=25,
        strong_augment=True,
    )
    
    trainer.train()

if __name__ == '__main__':
    main()