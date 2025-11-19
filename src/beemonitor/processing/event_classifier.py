"""ML-based event classifier for bee tracking.

This module provides machine learning models to classify trajectories
into bee vs noise, and entry vs exit events.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, ML classifier disabled")

from beemonitor.core.config import Config
from beemonitor.processing.trajectory_analyzer import TrajectoryAnalyzer


logger = logging.getLogger(__name__)


class EventClassifier:
    """ML classifier for bee trajectory events.
    
    Uses Random Forest to classify:
    1. Bee vs Noise (is this a real bee trajectory?)
    2. Entry vs Exit vs Pass-through (what type of event?)
    3. Nest location prediction (where did event occur?)
    
    Attributes:
        config: Configuration object
        trajectory_analyzer: TrajectoryAnalyzer for feature extraction
        bee_classifier: Model for bee vs noise classification
        event_classifier: Model for entry/exit classification
        scaler: Feature scaler
        feature_columns: List of feature names
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize EventClassifier.
        
        Args:
            config: Configuration object (optional)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML classifier. Install: pip install scikit-learn")
        
        self.config = config if config is not None else Config.default()
        self.trajectory_analyzer = TrajectoryAnalyzer(self.config)
        
        # Models
        self.bee_classifier = None  # Bee vs Noise
        self.event_classifier = None  # Entry vs Exit vs Pass
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        logger.info("EventClassifier initialized (requires training)")
    
    def extract_features_from_movements(
        self,
        movements: List[Tuple],
        nests: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Extract features from multiple movements.
        
        Args:
            movements: List of movement tuples
            nests: Optional nest dictionary
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        for movement in movements:
            features = self.trajectory_analyzer.extract_features(movement, nests)
            features['track_id'] = movement[0]
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train_bee_classifier(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """Train classifier to distinguish bees from noise.
        
        Args:
            X: Feature DataFrame
            y: Labels (0=noise, 1=bee)
            test_size: Test split ratio
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.bee_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        self.bee_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.bee_classifier.score(X_train_scaled, y_train)
        test_score = self.bee_classifier.score(X_test_scaled, y_test)
        
        logger.info(f"Bee classifier trained - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def train_event_classifier(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """Train classifier to classify entry/exit/pass events.
        
        Args:
            X: Feature DataFrame (bee trajectories only)
            y: Labels (0=exit, 1=entry, 2=pass/unknown)
            test_size: Test split ratio
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        # Ensure features match
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale (use existing scaler if available)
        if self.bee_classifier is None:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.event_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        self.event_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.event_classifier.score(X_train_scaled, y_train)
        test_score = self.event_classifier.score(X_test_scaled, y_test)
        
        logger.info(f"Event classifier trained - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def predict_bee(
        self,
        X: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """Predict if trajectories are bees or noise.
        
        Args:
            X: Feature DataFrame
            return_proba: Return probabilities instead of classes
            
        Returns:
            Predictions (0=noise, 1=bee) or probabilities
        """
        if self.bee_classifier is None:
            raise ValueError("Bee classifier not trained yet")
        
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        if return_proba:
            return self.bee_classifier.predict_proba(X_scaled)[:, 1]  # Prob of bee
        else:
            return self.bee_classifier.predict(X_scaled)
    
    def predict_event(
        self,
        X: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """Predict event type (entry/exit/pass).
        
        Args:
            X: Feature DataFrame
            return_proba: Return probabilities instead of classes
            
        Returns:
            Predictions (0=exit, 1=entry, 2=pass) or probabilities
        """
        if self.event_classifier is None:
            raise ValueError("Event classifier not trained yet")
        
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        if return_proba:
            return self.event_classifier.predict_proba(X_scaled)
        else:
            return self.event_classifier.predict(X_scaled)
    
    def classify_trajectory(
        self,
        movement: Tuple,
        nests: Optional[Dict] = None,
        bee_threshold: float = 0.5,
        event_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Classify a single trajectory.
        
        Args:
            movement: Movement tuple
            nests: Nest dictionary
            bee_threshold: Confidence threshold for bee classification
            event_threshold: Confidence threshold for event classification
            
        Returns:
            Dictionary with classification results
        """
        if self.bee_classifier is None or self.event_classifier is None:
            raise ValueError("Classifiers not trained yet")
        
        # Extract features
        features = self.trajectory_analyzer.extract_features(movement, nests)
        X = pd.DataFrame([features])[self.feature_columns]
        
        # Predict bee vs noise
        bee_proba = self.predict_bee(X, return_proba=True)[0]
        is_bee = bee_proba >= bee_threshold
        
        if not is_bee:
            return {
                'is_bee': False,
                'bee_confidence': bee_proba,
                'event_type': None,
                'event_confidence': 0.0
            }
        
        # Predict event type
        event_proba = self.predict_event(X, return_proba=True)[0]
        event_class = np.argmax(event_proba)
        event_confidence = event_proba[event_class]
        
        event_map = {0: 'exit', 1: 'entry', 2: 'pass'}
        event_type = event_map[event_class]
        
        # Get event location (median of first/last N frames)
        trajectory = movement[1]
        if event_type == 'entry':
            # Use last 5 frames for entry
            end_points = trajectory[-min(5, len(trajectory)):]
            event_x = np.median([p[0] for p in end_points])
            event_y = np.median([p[1] for p in end_points])
        elif event_type == 'exit':
            # Use first 5 frames for exit
            start_points = trajectory[:min(5, len(trajectory))]
            event_x = np.median([p[0] for p in start_points])
            event_y = np.median([p[1] for p in start_points])
        else:
            # Pass-through - use middle point
            mid_idx = len(trajectory) // 2
            event_x, event_y = trajectory[mid_idx]
        
        return {
            'is_bee': True,
            'bee_confidence': bee_proba,
            'event_type': event_type if event_confidence >= event_threshold else None,
            'event_confidence': event_confidence,
            'event_location': (event_x, event_y)
        }
    
    def save_models(self, path: str):
        """Save trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.bee_classifier is not None:
            with open(path / 'bee_classifier.pkl', 'wb') as f:
                pickle.dump(self.bee_classifier, f)
        
        if self.event_classifier is not None:
            with open(path / 'event_classifier.pkl', 'wb') as f:
                pickle.dump(self.event_classifier, f)
        
        with open(path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(path / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk.
        
        Args:
            path: Directory path with saved models
        """
        path = Path(path)
        
        if (path / 'bee_classifier.pkl').exists():
            with open(path / 'bee_classifier.pkl', 'rb') as f:
                self.bee_classifier = pickle.load(f)
        
        if (path / 'event_classifier.pkl').exists():
            with open(path / 'event_classifier.pkl', 'rb') as f:
                self.event_classifier = pickle.load(f)
        
        with open(path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(path / 'feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        logger.info(f"Models loaded from {path}")
    
    def get_feature_importance(self, model_type: str = 'bee') -> pd.DataFrame:
        """Get feature importance from trained model.
        
        Args:
            model_type: 'bee' or 'event'
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if model_type == 'bee':
            model = self.bee_classifier
        elif model_type == 'event':
            model = self.event_classifier
        else:
            raise ValueError("model_type must be 'bee' or 'event'")
        
        if model is None:
            raise ValueError(f"{model_type} classifier not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df