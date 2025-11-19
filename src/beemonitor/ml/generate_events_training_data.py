"""Helper script for preparing training data for ML event classifier.

This script helps create labeled training data from video annotations.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from beemonitor.core.config import Config
from beemonitor.processing.event_processor import EventProcessor


logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generate and manage training data for event classifier.
    
    This class helps create labeled datasets from annotated videos
    for training the ML models.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize generator.
        
        Args:
            config: Configuration object
        """
        self.config = config if config is not None else Config.default()
        self.classifier = EventProcessor(config)
    
    def create_training_data_from_annotations(
        self,
        motion_data: pd.DataFrame,
        annotations: Dict,
        nests: Dict
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """Create training data from manual annotations.
        
        Args:
            motion_data: DataFrame with tracking data
            annotations: Dictionary with manual labels:
                {
                    'track_id': {
                        'is_bee': bool,
                        'event_type': 'entry'|'exit'|'pass'|None
                    }
                }
            nests: Nest dictionary
            
        Returns:
            Tuple of (X_bee, y_bee, X_event, y_event)
        """
        # Extract movements
        movements = []
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        # Extract features
        features_list = []
        bee_labels = []
        event_labels = []
        
        for movement in movements:
            track_id = movement[0]
            
            # Skip if not annotated
            if track_id not in annotations:
                continue
            
            annotation = annotations[track_id]
            
            # Extract features
            features = self.classifier.trajectory_analyzer.extract_features(
                movement, nests
            )
            features_list.append(features)
            
            # Bee label (0=noise, 1=bee)
            bee_labels.append(1 if annotation['is_bee'] else 0)
            
            # Event label (0=exit, 1=entry, 2=pass/unknown)
            if annotation.get('event_type') == 'exit':
                event_labels.append(0)
            elif annotation.get('event_type') == 'entry':
                event_labels.append(1)
            else:
                event_labels.append(2)
        
        X = pd.DataFrame(features_list)
        y_bee = np.array(bee_labels)
        y_event = np.array(event_labels)
        
        # Split into bee and non-bee for event classifier
        X_bee = X[y_bee == 1]
        y_event_bee = y_event[y_bee == 1]
        
        logger.info(f"Generated training data:")
        logger.info(f"  Total samples: {len(X)}")
        logger.info(f"  Bee samples: {sum(y_bee)}")
        logger.info(f"  Entry samples: {sum(y_event_bee == 1)}")
        logger.info(f"  Exit samples: {sum(y_event_bee == 0)}")
        logger.info(f"  Pass samples: {sum(y_event_bee == 2)}")
        
        return X, y_bee, X_bee, y_event_bee
    
    def save_training_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        path: str,
        label_name: str = 'label'
    ):
        """Save training data to CSV.
        
        Args:
            X: Feature DataFrame
            y: Labels
            path: Output path
            label_name: Name for label column
        """
        df = X.copy()
        df[label_name] = y
        df.to_csv(path, index=False)
        logger.info(f"Training data saved to {path}")
    
    def load_training_data(
        self,
        path: str,
        label_name: str = 'label'
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Load training data from CSV.
        
        Args:
            path: Input path
            label_name: Name of label column
            
        Returns:
            Tuple of (X, y)
        """
        df = pd.read_csv(path)
        y = df[label_name].values
        X = df.drop(columns=[label_name])
        logger.info(f"Loaded training data from {path}: {len(X)} samples")
        return X, y


def create_annotation_template(
    motion_data: pd.DataFrame,
    output_path: str = 'annotations_template.json'
):
    """Create annotation template JSON for manual labeling.
    
    Args:
        motion_data: DataFrame with tracking data
        output_path: Where to save template
    """
    # Extract all track IDs
    track_ids = []
    for period in motion_data.tracks:
        for track in period:
            track_ids.append(track[0])
    
    # Create template
    template = {
        track_id: {
            'is_bee': None,  # True/False
            'event_type': None,  # 'entry'/'exit'/'pass'/None
            'notes': ''
        }
        for track_id in track_ids
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Annotation template saved to {output_path}")
    logger.info(f"Please manually label {len(track_ids)} tracks")
    print(f"\nAnnotation template created: {output_path}")
    print(f"Tracks to label: {len(track_ids)}")
    print("\nFor each track, set:")
    print("  - is_bee: true/false (is this a real bee?)")
    print("  - event_type: 'entry', 'exit', 'pass', or null")
    print("  - notes: any observations")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Training Data Generator")
    print("=" * 50)
    print("\nThis tool helps prepare training data for the ML classifier.")
    print("\nSteps:")
    print("1. Run video analysis to get motion_data")
    print("2. Create annotation template: create_annotation_template(motion_data)")
    print("3. Manually label tracks in the JSON file")
    print("4. Load annotations and create training data")
    print("5. Train classifier with event_classifier.py")
    print("\n" + "=" * 50)