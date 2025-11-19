# """Test script for bee classifier with detailed metrics.

# Evaluates model on test set and reports metrics for both bee and noise classes.
# """

# import torch
# import torch.nn as nn
# from torchvision import transforms
# import cv2
# import numpy as np
# from pathlib import Path
# from typing import Tuple, List, Dict
# import logging
# from tqdm import tqdm
# import argparse
# import json
# from sklearn.metrics import (
#     accuracy_score, precision_recall_fscore_support,
#     confusion_matrix, classification_report
# )
# import matplotlib.pyplot as plt
# import seaborn as sns

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class SimpleBeeClassifier(nn.Module):
#     """Enhanced CNN for bee vs noise classification."""
    
#     def __init__(self, num_classes: int = 2):
#         super(SimpleBeeClassifier, self).__init__()
        
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.AdaptiveAvgPool2d(1)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.4),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# class BeeClassifierTester:
#     """Test bee classifier and generate detailed metrics."""
    
#     def __init__(
#         self,
#         model_path: str,
#         test_dir: Path,
#         output_dir: Path,
#         image_size: int = 64,
#         batch_size: int = 32,
#         device: str = 'cuda',
#         threshold: float = 0.5
#     ):
#         """Initialize tester.
        
#         Args:
#             model_path: Path to trained model checkpoint
#             test_dir: Test data directory (contains bee/ and noise/ subdirs)
#             output_dir: Output directory for results
#             image_size: Input image size
#             batch_size: Batch size for testing
#             device: Device to use
#             threshold: Classification threshold
#         """
#         self.test_dir = Path(test_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.image_size = image_size
#         self.batch_size = batch_size
#         self.threshold = threshold
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
#         logger.info(f"Using device: {self.device}")
        
#         # Load model
#         self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
#         checkpoint = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.eval()
        
#         logger.info(f"Loaded model from {model_path}")
#         logger.info(f"Model trained for {checkpoint['epoch']} epochs")
#         logger.info(f"Training accuracy: {checkpoint.get('accuracy', 'N/A')}")
        
#         # Transform
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#         ])
        
#         # Collect test samples
#         self.samples = []
#         self.class_names = ['noise', 'bee']
        
#         # Noise class (label 0)
#         noise_dir = self.test_dir / 'noise'
#         if noise_dir.exists():
#             for img_path in noise_dir.glob('*.jpg'):
#                 self.samples.append((str(img_path), 0))
        
#         # Bee class (label 1)
#         bee_dir = self.test_dir / 'bee'
#         if bee_dir.exists():
#             for img_path in bee_dir.glob('*.jpg'):
#                 self.samples.append((str(img_path), 1))
        
#         logger.info(f"Found {len(self.samples)} test samples")
#         logger.info(f"  Noise: {sum(1 for _, l in self.samples if l == 0)}")
#         logger.info(f"  Bee: {sum(1 for _, l in self.samples if l == 1)}")
    
#     def test(self) -> Dict:
#         """Run test and compute metrics."""
#         all_labels = []
#         all_predictions = []
#         all_probabilities = []
        
#         logger.info("Running inference on test set...")
        
#         with torch.no_grad():
#             for i in tqdm(range(0, len(self.samples), self.batch_size)):
#                 batch_samples = self.samples[i:i+self.batch_size]
                
#                 # Load batch
#                 images = []
#                 labels = []
                
#                 for img_path, label in batch_samples:
#                     img = cv2.imread(img_path)
#                     if img is None:
#                         continue
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     img_tensor = self.transform(img)
#                     images.append(img_tensor)
#                     labels.append(label)
                
#                 if not images:
#                     continue
                
#                 # Batch inference
#                 batch = torch.stack(images).to(self.device)
#                 outputs = self.model(batch)
#                 probabilities = torch.softmax(outputs, dim=1)
                
#                 # Get predictions
#                 bee_probs = probabilities[:, 1].cpu().numpy()
#                 predictions = (bee_probs > self.threshold).astype(int)
                
#                 all_labels.extend(labels)
#                 all_predictions.extend(predictions)
#                 all_probabilities.extend(bee_probs)
        
#         # Convert to numpy
#         all_labels = np.array(all_labels)
#         all_predictions = np.array(all_predictions)
#         all_probabilities = np.array(all_probabilities)
        
#         # Compute metrics
#         results = self._compute_metrics(all_labels, all_predictions, all_probabilities)
        
#         # Save results
#         self._save_results(results)
        
#         # Print results
#         self._print_results(results)
        
#         # Generate visualizations
#         self._generate_visualizations(all_labels, all_predictions, all_probabilities)
        
#         return results
    
#     def _compute_metrics(
#         self,
#         labels: np.ndarray,
#         predictions: np.ndarray,
#         probabilities: np.ndarray
#     ) -> Dict:
#         """Compute detailed metrics."""
        
#         # Overall accuracy
#         accuracy = accuracy_score(labels, predictions)
        
#         # Per-class metrics
#         precision, recall, f1, support = precision_recall_fscore_support(
#             labels, predictions, average=None
#         )
        
#         # Confusion matrix
#         cm = confusion_matrix(labels, predictions)
        
#         # Classification report
#         report = classification_report(
#             labels, predictions,
#             target_names=self.class_names,
#             output_dict=True
#         )
        
#         results = {
#             'overall': {
#                 'accuracy': float(accuracy),
#                 'total_samples': len(labels)
#             },
#             'per_class': {
#                 'noise': {
#                     'precision': float(precision[0]),
#                     'recall': float(recall[0]),
#                     'f1_score': float(f1[0]),
#                     'support': int(support[0])
#                 },
#                 'bee': {
#                     'precision': float(precision[1]),
#                     'recall': float(recall[1]),
#                     'f1_score': float(f1[1]),
#                     'support': int(support[1])
#                 }
#             },
#             'confusion_matrix': cm.tolist(),
#             'classification_report': report
#         }
        
#         return results
    
#     def _save_results(self, results: Dict) -> None:
#         """Save results to JSON file."""
#         output_file = self.output_dir / 'test_results.json'
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=2)
#         logger.info(f"Results saved to {output_file}")
    
#     def _print_results(self, results: Dict) -> None:
#         """Print results in a readable format."""
#         logger.info("\n" + "="*60)
#         logger.info("TEST RESULTS")
#         logger.info("="*60)
        
#         # Overall
#         logger.info(f"\nOverall Accuracy: {results['overall']['accuracy']*100:.2f}%")
#         logger.info(f"Total Samples: {results['overall']['total_samples']}")
        
#         # Per class
#         logger.info("\n" + "-"*60)
#         logger.info("PER-CLASS METRICS")
#         logger.info("-"*60)
        
#         for class_name in ['noise', 'bee']:
#             metrics = results['per_class'][class_name]
#             logger.info(f"\n{class_name.upper()}:")
#             logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
#             logger.info(f"  Recall:    {metrics['recall']*100:.2f}%")
#             logger.info(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
#             logger.info(f"  Support:   {metrics['support']} samples")
        
#         # Confusion matrix
#         logger.info("\n" + "-"*60)
#         logger.info("CONFUSION MATRIX")
#         logger.info("-"*60)
#         cm = np.array(results['confusion_matrix'])
#         logger.info("\n                Predicted")
#         logger.info("              Noise    Bee")
#         logger.info(f"Actual Noise   {cm[0][0]:4d}   {cm[0][1]:4d}")
#         logger.info(f"       Bee     {cm[1][0]:4d}   {cm[1][1]:4d}")
        
#         logger.info("\n" + "="*60)
    
#     def _generate_visualizations(
#         self,
#         labels: np.ndarray,
#         predictions: np.ndarray,
#         probabilities: np.ndarray
#     ) -> None:
#         """Generate and save visualizations."""
        
#         # Compute metrics for visualizations
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             labels, predictions, average=None
#         )
        
#         # 1. Confusion Matrix Heatmap
#         cm = confusion_matrix(labels, predictions)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(
#             cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=self.class_names,
#             yticklabels=self.class_names
#         )
#         plt.title('Confusion Matrix')
#         plt.ylabel('True Label')
#         plt.xlabel('Predicted Label')
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
#         plt.close()
        
#         logger.info(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")
        
#         # 2. Confidence Distribution
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
#         # Noise samples
#         noise_mask = labels == 0
#         axes[0].hist(probabilities[noise_mask], bins=50, alpha=0.7, color='blue', edgecolor='black')
#         axes[0].axvline(self.threshold, color='red', linestyle='--', label=f'Threshold={self.threshold}')
#         axes[0].set_xlabel('Bee Probability')
#         axes[0].set_ylabel('Frequency')
#         axes[0].set_title('Noise Samples - Confidence Distribution')
#         axes[0].legend()
        
#         # Bee samples
#         bee_mask = labels == 1
#         axes[1].hist(probabilities[bee_mask], bins=50, alpha=0.7, color='orange', edgecolor='black')
#         axes[1].axvline(self.threshold, color='red', linestyle='--', label=f'Threshold={self.threshold}')
#         axes[1].set_xlabel('Bee Probability')
#         axes[1].set_ylabel('Frequency')
#         axes[1].set_title('Bee Samples - Confidence Distribution')
#         axes[1].legend()
        
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=150)
#         plt.close()
        
#         logger.info(f"Confidence distribution saved to {self.output_dir / 'confidence_distribution.png'}")
        
#         # 3. Metrics Bar Chart
#         metrics_data = {
#             'Noise': [precision[0], recall[0], f1[0]],
#             'Bee': [precision[1], recall[1], f1[1]]
#         }
        
#         x = np.arange(3)
#         width = 0.35
        
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.bar(x - width/2, metrics_data['Noise'], width, label='Noise', color='blue', alpha=0.7)
#         ax.bar(x + width/2, metrics_data['Bee'], width, label='Bee', color='orange', alpha=0.7)
        
#         ax.set_ylabel('Score')
#         ax.set_title('Per-Class Metrics Comparison')
#         ax.set_xticks(x)
#         ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
#         ax.legend()
#         ax.set_ylim([0, 1.1])
#         ax.grid(axis='y', alpha=0.3)
        
#         # Add value labels on bars
#         for i, (noise_val, bee_val) in enumerate(zip(metrics_data['Noise'], metrics_data['Bee'])):
#             ax.text(i - width/2, noise_val + 0.02, f'{noise_val:.2f}', ha='center', va='bottom')
#             ax.text(i + width/2, bee_val + 0.02, f'{bee_val:.2f}', ha='center', va='bottom')
        
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=150)
#         plt.close()
        
#         logger.info(f"Metrics comparison saved to {self.output_dir / 'metrics_comparison.png'}")


# def main():
#     parser = argparse.ArgumentParser(description='Test bee classifier')
#     parser.add_argument('--model_path', type=str, help='Path to trained model', default='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/training_output2/best_model.pth', required=False)
#     parser.add_argument('--test_dir', type=str, help='Test data directory (contains bee/ and noise/)', default='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/test_2', required=False)
#     parser.add_argument('--output_dir', type=str, help='Output directory for results', default='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/test_output', required=False)
#     parser.add_argument('--image-size', type=int, default=64,
#                         help='Input image size', required=False)
#     parser.add_argument('--batch-size', type=int, default=32,
#                         help='Batch size', required=False)
#     parser.add_argument('--threshold', type=float, default=0.99,
#                         help='Classification threshold', required=False)
#     parser.add_argument('--device', type=str, default='cuda',
#                         help='Device (cuda/cpu)', required=False)
    
#     args = parser.parse_args()
    
#     tester = BeeClassifierTester(
#         model_path=args.model_path,
#         test_dir=Path(args.test_dir),
#         output_dir=Path(args.output_dir),
#         image_size=args.image_size,
#         batch_size=args.batch_size,
#         device=args.device,
#         threshold=args.threshold
#     )
    
#     results = tester.test()


# if __name__ == '__main__':
#     main()








"""Test script for bee classifier with detailed metrics.

Evaluates model on test set and reports metrics for both bee and noise classes.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from tqdm import tqdm
import argparse
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBeeClassifier(nn.Module):
    """Enhanced CNN for bee vs noise classification."""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleBeeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
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


class BeeClassifierTester:
    """Test bee classifier and generate detailed metrics."""
    
    def __init__(
        self,
        model_path: str,
        test_dir: Path,
        output_dir: Path,
        image_size: int = 64,
        batch_size: int = 32,
        device: str = 'cuda',
        threshold: float = 0.5,
        save_misclassified: bool = True
    ):
        """Initialize tester.
        
        Args:
            model_path: Path to trained model checkpoint
            test_dir: Test data directory (contains bee/ and noise/ subdirs)
            output_dir: Output directory for results
            image_size: Input image size
            batch_size: Batch size for testing
            device: Device to use
            threshold: Classification threshold
            save_misclassified: Save misclassified samples for review
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_misclassified = save_misclassified
        
        # Create misclassification directories
        if self.save_misclassified:
            self.false_positives_dir = self.output_dir / 'false_positives_noise_as_bee'
            self.false_negatives_dir = self.output_dir / 'false_negatives_bee_as_noise'
            self.false_positives_dir.mkdir(parents=True, exist_ok=True)
            self.false_negatives_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = image_size
        self.batch_size = batch_size
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model trained for {checkpoint['epoch']} epochs")
        logger.info(f"Training accuracy: {checkpoint.get('accuracy', 'N/A')}")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Collect test samples
        self.samples = []
        self.class_names = ['noise', 'bee']
        
        # Noise class (label 0)
        noise_dir = self.test_dir / 'noise'
        if noise_dir.exists():
            for img_path in noise_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 0))
        
        # Bee class (label 1)
        bee_dir = self.test_dir / 'bee'
        if bee_dir.exists():
            for img_path in bee_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 1))
        
        logger.info(f"Found {len(self.samples)} test samples")
        logger.info(f"  Noise: {sum(1 for _, l in self.samples if l == 0)}")
        logger.info(f"  Bee: {sum(1 for _, l in self.samples if l == 1)}")
    
    def test(self) -> Dict:
        """Run test and compute metrics."""
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_paths = []  # Track image paths for misclassification review
        
        logger.info("Running inference on test set...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.samples), self.batch_size)):
                batch_samples = self.samples[i:i+self.batch_size]
                
                # Load batch
                images = []
                labels = []
                paths = []
                
                for img_path, label in batch_samples:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
                    labels.append(label)
                    paths.append(img_path)
                
                if not images:
                    continue
                
                # Batch inference
                batch = torch.stack(images).to(self.device)
                outputs = self.model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predictions
                bee_probs = probabilities[:, 1].cpu().numpy()
                predictions = (bee_probs > self.threshold).astype(int)
                
                all_labels.extend(labels)
                all_predictions.extend(predictions)
                all_probabilities.extend(bee_probs)
                all_paths.extend(paths)
        
        # Convert to numpy
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Save misclassified samples
        if self.save_misclassified:
            self._save_misclassified_samples(
                all_paths, all_labels, all_predictions, all_probabilities
            )
        
        # Compute metrics
        results = self._compute_metrics(all_labels, all_predictions, all_probabilities)
        
        # Save results
        self._save_results(results)
        
        # Print results
        self._print_results(results)
        
        # Generate visualizations
        self._generate_visualizations(all_labels, all_predictions, all_probabilities)
        
        return results
    
    def _save_misclassified_samples(
        self,
        paths: List[str],
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> None:
        """Save misclassified samples for review.
        
        Args:
            paths: List of image paths
            labels: True labels
            predictions: Predicted labels
            probabilities: Bee probabilities
        """
        logger.info("\nSaving misclassified samples for review...")
        
        false_positives = []  # Noise predicted as bee
        false_negatives = []  # Bee predicted as noise
        
        for path, label, pred, prob in zip(paths, labels, predictions, probabilities):
            # False Positive: Noise (0) predicted as Bee (1)
            if label == 0 and pred == 1:
                false_positives.append((path, prob))
            
            # False Negative: Bee (1) predicted as Noise (0)
            elif label == 1 and pred == 0:
                false_negatives.append((path, prob))
        
        # Save False Positives (Noise called Bee)
        logger.info(f"\nFalse Positives (Noise predicted as Bee): {len(false_positives)}")
        for idx, (src_path, prob) in enumerate(false_positives):
            img = cv2.imread(src_path)
            if img is None:
                continue
            
            # Create informative filename
            orig_name = Path(src_path).stem
            dst_name = f"fp_{idx:04d}_prob{prob:.3f}_{orig_name}.jpg"
            dst_path = self.false_positives_dir / dst_name
            
            cv2.imwrite(str(dst_path), img)
        
        if len(false_positives) > 0:
            logger.info(f"  Saved to: {self.false_positives_dir}")
            logger.info(f"  ⚠️  REVIEW THESE: Labeled as 'noise' but model thinks 'bee'")
            logger.info(f"  ⚠️  Could be real bees that YOLO missed during labeling!")
        
        # Save False Negatives (Bee called Noise)
        logger.info(f"\nFalse Negatives (Bee predicted as Noise): {len(false_negatives)}")
        for idx, (src_path, prob) in enumerate(false_negatives):
            img = cv2.imread(src_path)
            if img is None:
                continue
            
            # Create informative filename
            orig_name = Path(src_path).stem
            dst_name = f"fn_{idx:04d}_prob{prob:.3f}_{orig_name}.jpg"
            dst_path = self.false_negatives_dir / dst_name
            
            cv2.imwrite(str(dst_path), img)
        
        if len(false_negatives) > 0:
            logger.info(f"  Saved to: {self.false_negatives_dir}")
            logger.info(f"  ⚠️  REVIEW THESE: Labeled as 'bee' but model thinks 'noise'")
            logger.info(f"  ⚠️  Could be classifier errors or mislabeled samples")
        
        # Create review report
        self._create_misclassification_report(false_positives, false_negatives)
    
    def _create_misclassification_report(
        self,
        false_positives: List[Tuple[str, float]],
        false_negatives: List[Tuple[str, float]]
    ) -> None:
        """Create a text report of misclassifications.
        
        Args:
            false_positives: List of (path, probability) for FPs
            false_negatives: List of (path, probability) for FNs
        """
        report_path = self.output_dir / 'misclassification_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MISCLASSIFICATION REVIEW REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Total misclassifications: {len(false_positives) + len(false_negatives)}\n\n")
            
            # False Positives
            f.write("-"*80 + "\n")
            f.write(f"FALSE POSITIVES (Noise predicted as Bee): {len(false_positives)}\n")
            f.write("-"*80 + "\n")
            f.write("These samples were labeled as 'noise' but classifier predicts 'bee'\n")
            f.write("IMPORTANT: Check if these are actually bees that YOLO missed!\n\n")
            
            if false_positives:
                # Sort by confidence (highest first)
                false_positives_sorted = sorted(false_positives, key=lambda x: x[1], reverse=True)
                
                f.write("Top 20 highest confidence false positives:\n")
                for idx, (path, prob) in enumerate(false_positives_sorted[:20]):
                    f.write(f"  {idx+1}. Bee prob: {prob:.4f} - {Path(path).name}\n")
                
                f.write(f"\nAll {len(false_positives)} saved to: {self.false_positives_dir}\n")
            
            f.write("\n")
            
            # False Negatives
            f.write("-"*80 + "\n")
            f.write(f"FALSE NEGATIVES (Bee predicted as Noise): {len(false_negatives)}\n")
            f.write("-"*80 + "\n")
            f.write("These samples were labeled as 'bee' but classifier predicts 'noise'\n")
            f.write("Could indicate classifier needs more training or mislabeled data\n\n")
            
            if false_negatives:
                # Sort by confidence (lowest first - most confident noise predictions)
                false_negatives_sorted = sorted(false_negatives, key=lambda x: x[1])
                
                f.write("Top 20 most confident misses (lowest bee probability):\n")
                for idx, (path, prob) in enumerate(false_negatives_sorted[:20]):
                    f.write(f"  {idx+1}. Bee prob: {prob:.4f} - {Path(path).name}\n")
                
                f.write(f"\nAll {len(false_negatives)} saved to: {self.false_negatives_dir}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            f.write("1. Review false positives (noise→bee) FIRST\n")
            f.write("   - If many look like real bees, YOLO labeling has false negatives\n")
            f.write("   - Need to regenerate training data with lower YOLO threshold\n\n")
            f.write("2. Review false negatives (bee→noise)\n")
            f.write("   - If they look like bees, classifier needs more training\n")
            f.write("   - If they don't look like bees, labels might be wrong\n\n")
            f.write("3. If >30% of false positives are actually bees:\n")
            f.write("   - Regenerate data with --yolo-conf 0.3 (lower threshold)\n")
            f.write("   - Current 'noise' samples contain real bees!\n")
            f.write("="*80 + "\n")
        
        logger.info(f"\nMisclassification report saved to: {report_path}")
    
    def _compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """Compute detailed metrics."""
        
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        results = {
            'overall': {
                'accuracy': float(accuracy),
                'total_samples': len(labels)
            },
            'per_class': {
                'noise': {
                    'precision': float(precision[0]),
                    'recall': float(recall[0]),
                    'f1_score': float(f1[0]),
                    'support': int(support[0])
                },
                'bee': {
                    'precision': float(precision[1]),
                    'recall': float(recall[1]),
                    'f1_score': float(f1[1]),
                    'support': int(support[1])
                }
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return results
    
    def _save_results(self, results: Dict) -> None:
        """Save results to JSON file."""
        output_file = self.output_dir / 'test_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    def _print_results(self, results: Dict) -> None:
        """Print results in a readable format."""
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS")
        logger.info("="*60)
        
        # Overall
        logger.info(f"\nOverall Accuracy: {results['overall']['accuracy']*100:.2f}%")
        logger.info(f"Total Samples: {results['overall']['total_samples']}")
        
        # Per class
        logger.info("\n" + "-"*60)
        logger.info("PER-CLASS METRICS")
        logger.info("-"*60)
        
        for class_name in ['noise', 'bee']:
            metrics = results['per_class'][class_name]
            logger.info(f"\n{class_name.upper()}:")
            logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
            logger.info(f"  Recall:    {metrics['recall']*100:.2f}%")
            logger.info(f"  F1-Score:  {metrics['f1_score']*100:.2f}%")
            logger.info(f"  Support:   {metrics['support']} samples")
        
        # Confusion matrix
        logger.info("\n" + "-"*60)
        logger.info("CONFUSION MATRIX")
        logger.info("-"*60)
        cm = np.array(results['confusion_matrix'])
        logger.info("\n                Predicted")
        logger.info("              Noise    Bee")
        logger.info(f"Actual Noise   {cm[0][0]:4d}   {cm[0][1]:4d}")
        logger.info(f"       Bee     {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        logger.info("\n" + "="*60)
    
    def _generate_visualizations(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> None:
        """Generate and save visualizations."""
        
        # Compute metrics for visualizations
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")
        
        # 2. Confidence Distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Noise samples
        noise_mask = labels == 0
        axes[0].hist(probabilities[noise_mask], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(self.threshold, color='red', linestyle='--', label=f'Threshold={self.threshold}')
        axes[0].set_xlabel('Bee Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Noise Samples - Confidence Distribution')
        axes[0].legend()
        
        # Bee samples
        bee_mask = labels == 1
        axes[1].hist(probabilities[bee_mask], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1].axvline(self.threshold, color='red', linestyle='--', label=f'Threshold={self.threshold}')
        axes[1].set_xlabel('Bee Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Bee Samples - Confidence Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=150)
        plt.close()
        
        logger.info(f"Confidence distribution saved to {self.output_dir / 'confidence_distribution.png'}")
        
        # 3. Metrics Bar Chart
        metrics_data = {
            'Noise': [precision[0], recall[0], f1[0]],
            'Bee': [precision[1], recall[1], f1[1]]
        }
        
        x = np.arange(3)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, metrics_data['Noise'], width, label='Noise', color='blue', alpha=0.7)
        ax.bar(x + width/2, metrics_data['Bee'], width, label='Bee', color='orange', alpha=0.7)
        
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (noise_val, bee_val) in enumerate(zip(metrics_data['Noise'], metrics_data['Bee'])):
            ax.text(i - width/2, noise_val + 0.02, f'{noise_val:.2f}', ha='center', va='bottom')
            ax.text(i + width/2, bee_val + 0.02, f'{bee_val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=150)
        plt.close()
        
        logger.info(f"Metrics comparison saved to {self.output_dir / 'metrics_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Test bee classifier')
    # parser.add_argument('--model_path', type=str, help='Path to trained model')
    # parser.add_argument('--test_dir', type=str, help='Test data directory (contains bee/ and noise/)')
    # parser.add_argument('--output_dir', type=str, help='Output directory for results')
    # parser.add_argument('--image-size', type=int, default=64,
    #                     help='Input image size')
    
    parser.add_argument('--model_path', type=str, help='Path to trained model', default='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/training_output2/best_model.pth', required=False)
    parser.add_argument('--test_dir', type=str, help='Test data directory (contains bee/ and noise/)', default='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/test_2', required=False)
    parser.add_argument('--output_dir', type=str, help='Output directory for results', default='/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/test_output_2', required=False)
    parser.add_argument('--image-size', type=int, default=64,
                        help='Input image size', required=False)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.99,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save-misclassified', action='store_true', default=True,
                        help='Save misclassified samples for review (default: True)')
  
    
    args = parser.parse_args()
    
    tester = BeeClassifierTester(
        model_path=args.model_path,
        test_dir=Path(args.test_dir),
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
        save_misclassified=args.save_misclassified
    )
    
    results = tester.test()


if __name__ == '__main__':
    main()