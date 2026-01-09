"""Parameter optimization for tracking and event processing.

This module analyzes real-world tracking data to automatically optimize
parameters for better event detection accuracy.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from beemonitor.core.config import Config
from beemonitor.processing.event_processor import EventProcessor
from beemonitor.processing.trajectory_analyzer import TrajectoryAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, float]
    best_score: float
    initial_score: float
    improvement: float
    metrics: Dict[str, float]
    optimization_history: List[Dict]
    

class ParameterOptimizer:
    """Optimize tracking and event processing parameters using real data.
    
    This class analyzes ground truth data to find optimal parameter values
    that maximize event detection accuracy.
    
    Attributes:
        config: Base configuration
        ground_truth: Ground truth annotations
        video_data: Video and tracking data
        
    Example:
        >>> optimizer = ParameterOptimizer(
        ...     ground_truth_file="data/ground_truth.csv",
        ...     tracking_data_file="data/tracking_results.pkl"
        ... )
        >>> result = optimizer.optimize_all_parameters()
        >>> print(f"Improvement: {result.improvement:.1f}%")
    """
    
    def __init__(
        self,
        ground_truth_file: str,
        tracking_data_file: str,
        config: Optional[Config] = None,
        output_dir: str = "optimization_results"
    ):
        """Initialize ParameterOptimizer.
        
        Args:
            ground_truth_file: Path to ground truth annotations (CSV)
            tracking_data_file: Path to tracking results (pickle/CSV)
            config: Base configuration (optional)
            output_dir: Directory for saving optimization results
        """
        self.config = config if config is not None else Config.default()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading ground truth and tracking data...")
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.tracking_data = self._load_tracking_data(tracking_data_file)
        
        logger.info(f"Loaded {len(self.ground_truth)} ground truth events")
        logger.info(f"Loaded tracking data with {len(self.tracking_data)} trajectories")
        
        # Optimization history
        self.optimization_history = []
    
    def optimize_all_parameters(
        self,
        method: str = 'differential_evolution',
        max_iterations: int = 100
    ) -> OptimizationResult:
        """Optimize all parameters simultaneously.
        
        Args:
            method: Optimization method ('differential_evolution', 'bayesian', 'grid')
            max_iterations: Maximum iterations
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        logger.info(f"Starting parameter optimization using {method}...")
        
        # Define parameter bounds
        param_bounds = self._get_parameter_bounds()
        
        # Get initial score
        initial_score, initial_metrics = self._evaluate_parameters(self.config)
        logger.info(f"Initial F1 score: {initial_score:.4f}")
        
        # Optimize
        if method == 'differential_evolution':
            result = self._optimize_differential_evolution(param_bounds, max_iterations)
        elif method == 'grid':
            result = self._optimize_grid_search(param_bounds)
        elif method == 'bayesian':
            result = self._optimize_bayesian(param_bounds, max_iterations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Evaluate best parameters
        best_config = self._params_to_config(result['best_params'])
        best_score, best_metrics = self._evaluate_parameters(best_config)
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best F1 score: {best_score:.4f}")
        logger.info(f"Improvement: {(best_score - initial_score) / initial_score * 100:.1f}%")
        
        # Create result
        optimization_result = OptimizationResult(
            best_params=result['best_params'],
            best_score=best_score,
            initial_score=initial_score,
            improvement=(best_score - initial_score) / initial_score * 100,
            metrics=best_metrics,
            optimization_history=self.optimization_history
        )
        
        # Save results
        self._save_results(optimization_result)
        
        return optimization_result
    
    def optimize_tracking_parameters(self) -> Dict[str, float]:
        """Optimize only tracking parameters.
        
        Returns:
            Dictionary of optimized tracking parameters
        """
        logger.info("Optimizing tracking parameters...")
        
        param_bounds = {
            'distance_threshold': (50, 200),
            'association_threshold': (100, 400),
            'max_age': (10, 50)
        }
        
        def objective(params):
            config = self.config
            config.tracking.distance_threshold_base = params[0]
            config.tracking.association_threshold_base = params[1]
            config.tracking.max_age = int(params[2])
            
            score, _ = self._evaluate_parameters(config)
            return -score  # Minimize negative score
        
        bounds = [param_bounds[k] for k in ['distance_threshold', 'association_threshold', 'max_age']]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            seed=42,
            workers=1
        )
        
        optimized_params = {
            'distance_threshold_base': result.x[0],
            'association_threshold_base': result.x[1],
            'max_age': int(result.x[2])
        }
        
        logger.info("Tracking parameters optimized:")
        for key, value in optimized_params.items():
            logger.info(f"  {key}: {value}")
        
        return optimized_params
    
    def optimize_event_processing_parameters(self) -> Dict[str, float]:
        """Optimize only event processing parameters.
        
        Returns:
            Dictionary of optimized processing parameters
        """
        logger.info("Optimizing event processing parameters...")
        
        param_bounds = {
            'entry_padding': (5, 30),
            'exit_padding': (10, 40),
            'entry_window_size': (3, 10),
            'exit_window_size': (2, 8),
            'min_trajectory_length': (3, 15)
        }
        
        def objective(params):
            config = self.config
            config.processing.entry_padding_base = params[0]
            config.processing.exit_padding_base = params[1]
            config.processing.entry_window_size = int(params[2])
            config.processing.exit_window_size = int(params[3])
            config.processing.min_trajectory_length = int(params[4])
            
            score, _ = self._evaluate_parameters(config)
            return -score
        
        bounds = [param_bounds[k] for k in param_bounds.keys()]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            seed=42,
            workers=1
        )
        
        optimized_params = {
            'entry_padding_base': result.x[0],
            'exit_padding_base': result.x[1],
            'entry_window_size': int(result.x[2]),
            'exit_window_size': int(result.x[3]),
            'min_trajectory_length': int(result.x[4])
        }
        
        logger.info("Event processing parameters optimized:")
        for key, value in optimized_params.items():
            logger.info(f"  {key}: {value}")
        
        return optimized_params
    
    def analyze_parameter_sensitivity(
        self,
        parameter_name: str,
        value_range: Tuple[float, float],
        num_points: int = 20
    ) -> pd.DataFrame:
        """Analyze how a single parameter affects performance.
        
        Args:
            parameter_name: Name of parameter to analyze
            value_range: Tuple of (min_value, max_value)
            num_points: Number of points to evaluate
            
        Returns:
            DataFrame with parameter values and corresponding metrics
        """
        logger.info(f"Analyzing sensitivity of {parameter_name}...")
        
        values = np.linspace(value_range[0], value_range[1], num_points)
        results = []
        
        for value in tqdm(values, desc=f"Testing {parameter_name}"):
            config = self._create_config_with_param(parameter_name, value)
            score, metrics = self._evaluate_parameters(config)
            
            results.append({
                'parameter': parameter_name,
                'value': value,
                'f1_score': score,
                **metrics
            })
        
        df = pd.DataFrame(results)
        
        # Plot
        self._plot_sensitivity_analysis(df, parameter_name)
        
        return df
    
    def compare_event_classification_methods(self) -> pd.DataFrame:
        """Compare different event classification approaches.
        
        Returns:
            DataFrame comparing different methods
        """
        logger.info("Comparing event classification methods...")
        
        methods = {
            'speed_based': self._classify_speed_based,
            'position_based': self._classify_position_based,
            'trajectory_based': self._classify_trajectory_based,
            'hybrid': self._classify_hybrid
        }
        
        results = []
        
        for method_name, classify_fn in methods.items():
            score, metrics = self._evaluate_classification_method(classify_fn)
            
            results.append({
                'method': method_name,
                'f1_score': score,
                **metrics
            })
            
            logger.info(f"{method_name}: F1={score:.4f}")
        
        df = pd.DataFrame(results)
        return df
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds for optimization.
        
        Returns:
            Dictionary of parameter bounds
        """
        return {
            # Tracking parameters
            'distance_threshold': (50, 200),
            'association_threshold': (100, 400),
            'max_age': (10, 50),
            
            # Event processing parameters
            'entry_padding': (5, 30),
            'exit_padding': (10, 40),
            'entry_window_size': (3, 10),
            'exit_window_size': (2, 8),
            'min_trajectory_length': (3, 15),
        }
    
    def _optimize_differential_evolution(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int
    ) -> Dict:
        """Optimize using differential evolution.
        
        Args:
            param_bounds: Parameter bounds
            max_iterations: Maximum iterations
            
        Returns:
            Dictionary with optimization results
        """
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[k] for k in param_names]
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            config = self._params_to_config(param_dict)
            score, metrics = self._evaluate_parameters(config)
            
            # Store history
            self.optimization_history.append({
                'params': param_dict.copy(),
                'score': score,
                'metrics': metrics
            })
            
            return -score  # Minimize negative score
        
        logger.info("Running differential evolution...")
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            seed=42,
            workers=1,
            disp=True
        )
        
        best_params = dict(zip(param_names, result.x))
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'convergence': result.success
        }
    
    def _optimize_grid_search(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        num_points: int = 5
    ) -> Dict:
        """Optimize using grid search.
        
        Args:
            param_bounds: Parameter bounds
            num_points: Number of points per parameter
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Running grid search...")
        
        # Create parameter grid
        param_names = list(param_bounds.keys())
        param_grids = []
        
        for param_name in param_names:
            min_val, max_val = param_bounds[param_name]
            if 'window_size' in param_name or 'trajectory_length' in param_name or 'max_age' in param_name:
                # Integer parameters
                grid = np.linspace(min_val, max_val, num_points, dtype=int)
            else:
                # Float parameters
                grid = np.linspace(min_val, max_val, num_points)
            param_grids.append(grid)
        
        # Grid search
        best_score = -np.inf
        best_params = None
        
        from itertools import product
        total_combinations = np.prod([len(g) for g in param_grids])
        
        for param_values in tqdm(product(*param_grids), total=total_combinations, desc="Grid search"):
            param_dict = dict(zip(param_names, param_values))
            config = self._params_to_config(param_dict)
            score, metrics = self._evaluate_parameters(config)
            
            self.optimization_history.append({
                'params': param_dict.copy(),
                'score': score,
                'metrics': metrics
            })
            
            if score > best_score:
                best_score = score
                best_params = param_dict.copy()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'convergence': True
        }
    
    def _optimize_bayesian(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int
    ) -> Dict:
        """Optimize using Bayesian optimization.
        
        Args:
            param_bounds: Parameter bounds
            max_iterations: Maximum iterations
            
        Returns:
            Dictionary with optimization results
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            logger.warning("scikit-optimize not installed. Falling back to differential evolution.")
            return self._optimize_differential_evolution(param_bounds, max_iterations)
        
        logger.info("Running Bayesian optimization...")
        
        param_names = list(param_bounds.keys())
        space = []
        
        for param_name in param_names:
            min_val, max_val = param_bounds[param_name]
            if 'window_size' in param_name or 'trajectory_length' in param_name or 'max_age' in param_name:
                space.append(Integer(int(min_val), int(max_val), name=param_name))
            else:
                space.append(Real(min_val, max_val, name=param_name))
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            config = self._params_to_config(param_dict)
            score, metrics = self._evaluate_parameters(config)
            
            self.optimization_history.append({
                'params': param_dict.copy(),
                'score': score,
                'metrics': metrics
            })
            
            return -score
        
        result = gp_minimize(
            objective,
            space,
            n_calls=max_iterations,
            random_state=42,
            verbose=True
        )
        
        best_params = dict(zip(param_names, result.x))
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'convergence': True
        }
    
    def _params_to_config(self, params: Dict[str, float]) -> Config:
        """Convert parameter dictionary to Config object.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Config object with specified parameters
        """
        config = Config.from_dict(self.config.to_dict())
        
        # Update tracking parameters
        if 'distance_threshold' in params:
            config.tracking.distance_threshold_base = params['distance_threshold']
        if 'association_threshold' in params:
            config.tracking.association_threshold_base = params['association_threshold']
        if 'max_age' in params:
            config.tracking.max_age = int(params['max_age'])
        
        # Update processing parameters
        if 'entry_padding' in params:
            config.processing.entry_padding_base = params['entry_padding']
        if 'exit_padding' in params:
            config.processing.exit_padding_base = params['exit_padding']
        if 'entry_window_size' in params:
            config.processing.entry_window_size = int(params['entry_window_size'])
        if 'exit_window_size' in params:
            config.processing.exit_window_size = int(params['exit_window_size'])
        if 'min_trajectory_length' in params:
            config.processing.min_trajectory_length = int(params['min_trajectory_length'])
        
        return config
    
    def _evaluate_parameters(self, config: Config) -> Tuple[float, Dict[str, float]]:
        """Evaluate parameter configuration against ground truth.
        
        Args:
            config: Configuration to evaluate
            
        Returns:
            Tuple of (f1_score, metrics_dict)
        """
        # Process events with current config
        processor = EventProcessor(config)
        
        try:
            predicted_events = processor.process_tracks(
                self.tracking_data['motion_data'],
                self.tracking_data['nests']
            )
            
            # Compare with ground truth
            metrics = self._compute_metrics(predicted_events, self.ground_truth)
            f1_score = metrics['f1_score']
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            f1_score = 0.0
            metrics = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        return f1_score, metrics
    
    def _compute_metrics(
        self,
        predicted_events: pd.DataFrame,
        ground_truth: pd.DataFrame,
        time_tolerance: int = 30
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predicted_events: Predicted events DataFrame
            ground_truth: Ground truth events DataFrame
            time_tolerance: Frame tolerance for matching events
            
        Returns:
            Dictionary of metrics
        """
        # Match predicted events to ground truth
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        matched_gt = set()
        
        for _, pred_event in predicted_events.iterrows():
            # Find matching ground truth event
            matches = ground_truth[
                (ground_truth['action'] == pred_event['action']) &
                (ground_truth['nest'] == pred_event['nest']) &
                (abs(ground_truth['frame_number'] - pred_event['frame_number']) <= time_tolerance)
            ]
            
            if len(matches) > 0:
                # True positive
                tp += 1
                matched_gt.add(matches.iloc[0].name)
            else:
                # False positive
                fp += 1
        
        # False negatives (ground truth events not matched)
        fn = len(ground_truth) - len(matched_gt)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def _load_ground_truth(self, filepath: str) -> pd.DataFrame:
        """Load ground truth annotations.
        
        Args:
            filepath: Path to ground truth file
            
        Returns:
            DataFrame with ground truth events
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Validate required columns
        required_columns = ['action', 'nest', 'frame_number']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def _load_tracking_data(self, filepath: str) -> Dict:
        """Load tracking data.
        
        Args:
            filepath: Path to tracking data file
            
        Returns:
            Dictionary with tracking data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.pkl':
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return data
    
    def _create_config_with_param(self, param_name: str, value: float) -> Config:
        """Create config with a single parameter modified.
        
        Args:
            param_name: Parameter name
            value: Parameter value
            
        Returns:
            Modified Config object
        """
        config = Config.from_dict(self.config.to_dict())
        
        # Map parameter names to config attributes
        param_mapping = {
            'distance_threshold': ('tracking', 'distance_threshold_base'),
            'association_threshold': ('tracking', 'association_threshold_base'),
            'max_age': ('tracking', 'max_age'),
            'entry_padding': ('processing', 'entry_padding_base'),
            'exit_padding': ('processing', 'exit_padding_base'),
            'entry_window_size': ('processing', 'entry_window_size'),
            'exit_window_size': ('processing', 'exit_window_size'),
            'min_trajectory_length': ('processing', 'min_trajectory_length'),
        }
        
        if param_name in param_mapping:
            section, attr = param_mapping[param_name]
            if 'window_size' in param_name or 'trajectory_length' in param_name or 'max_age' in param_name:
                value = int(value)
            setattr(getattr(config, section), attr, value)
        
        return config
    
    def _plot_sensitivity_analysis(self, df: pd.DataFrame, parameter_name: str) -> None:
        """Plot sensitivity analysis results.
        
        Args:
            df: DataFrame with sensitivity analysis results
            parameter_name: Name of parameter analyzed
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sensitivity Analysis: {parameter_name}')
        
        # F1 Score
        axes[0, 0].plot(df['value'], df['f1_score'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel(parameter_name)
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('F1 Score')
        
        # Precision
        axes[0, 1].plot(df['value'], df['precision'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel(parameter_name)
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Precision')
        
        # Recall
        axes[1, 0].plot(df['value'], df['recall'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel(parameter_name)
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Recall')
        
        # Combined
        axes[1, 1].plot(df['value'], df['f1_score'], 'b-', label='F1', linewidth=2)
        axes[1, 1].plot(df['value'], df['precision'], 'g-', label='Precision', linewidth=2)
        axes[1, 1].plot(df['value'], df['recall'], 'r-', label='Recall', linewidth=2)
        axes[1, 1].set_xlabel(parameter_name)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('All Metrics')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'sensitivity_{parameter_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sensitivity plot saved: {plot_path}")
    
    def _save_results(self, result: OptimizationResult) -> None:
        """Save optimization results.
        
        Args:
            result: OptimizationResult object
        """
        # Save as JSON
        result_dict = asdict(result)
        result_path = self.output_dir / 'optimization_results.json'
        
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save best config
        best_config = self._params_to_config(result.best_params)
        config_path = self.output_dir / 'best_config.yaml'
        best_config.save_yaml(str(config_path))
        
        # Create report
        report_path = self.output_dir / 'optimization_report.txt'
        with open(report_path, 'w') as f:
            f.write("Parameter Optimization Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Initial Performance:\n")
            f.write(f"  F1 Score: {result.initial_score:.4f}\n\n")
            
            f.write("Optimized Performance:\n")
            f.write(f"  F1 Score: {result.best_score:.4f}\n")
            f.write(f"  Precision: {result.metrics['precision']:.4f}\n")
            f.write(f"  Recall: {result.metrics['recall']:.4f}\n")
            f.write(f"  Improvement: {result.improvement:.1f}%\n\n")
            
            f.write("Best Parameters:\n")
            for key, value in result.best_params.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        logger.info(f"Results saved to: {self.output_dir}")
    
    # Classification methods for comparison
    def _classify_speed_based(self, movement: Tuple) -> str:
        """Classify using speed thresholds."""
        analyzer = TrajectoryAnalyzer(self.config)
        if analyzer.is_entry_behavior(movement):
            return 'entry'
        elif analyzer.is_exit_behavior(movement):
            return 'exit'
        return 'unknown'
    
    def _classify_position_based(self, movement: Tuple) -> str:
        """Classify based on position relative to nests."""
        # Implement position-based classification
        # This is a placeholder - actual implementation would check
        # if trajectory starts/ends inside nest regions
        return 'unknown'
    
    def _classify_trajectory_based(self, movement: Tuple) -> str:
        """Classify based on trajectory shape and direction."""
        # Implement trajectory-based classification
        # This could use trajectory curvature, direction changes, etc.
        return 'unknown'
    
    def _classify_hybrid(self, movement: Tuple) -> str:
        """Classify using hybrid approach combining multiple methods."""
        # Combine speed, position, and trajectory features
        return 'unknown'
    
    def _evaluate_classification_method(self, classify_fn: callable) -> Tuple[float, Dict]:
        """Evaluate a classification method.
        
        Args:
            classify_fn: Classification function
            
        Returns:
            Tuple of (f1_score, metrics)
        """
        # Placeholder implementation
        return 0.0, {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}


class ImprovedEventClassifier:
    """Improved event classification that doesn't rely solely on speed thresholds.
    
    This classifier uses multiple features to determine entry/exit events:
    - Position relative to nest entrances
    - Trajectory direction and curvature
    - Movement patterns (acceleration, deceleration)
    - Dwell time inside nest regions
    """
    
    def __init__(self, config: Config):
        """Initialize classifier.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.trajectory_analyzer = TrajectoryAnalyzer(config)
    
    def classify_event(
        self,
        movement: Tuple,
        nest_bboxes: Dict,
        hotel_box: Optional[Dict] = None
    ) -> Optional[str]:
        """Classify a movement as entry, exit, or neither.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            nest_bboxes: Dictionary of nest bounding boxes
            hotel_box: Optional hotel box info
            
        Returns:
            'entry', 'exit', or None
        """
        trajectory = movement[1]  # centroids
        
        if len(trajectory) < 3:
            return None
        
        # Analyze position features
        position_features = self._analyze_position_features(trajectory, nest_bboxes)
        
        # Analyze motion features
        motion_features = self._analyze_motion_features(trajectory)
        
        # Analyze trajectory features
        trajectory_features = self._analyze_trajectory_features(trajectory)
        
        # Combined decision
        event_type = self._make_classification_decision(
            position_features,
            motion_features,
            trajectory_features
        )
        
        return event_type
    
    def _analyze_position_features(
        self,
        trajectory: List[Tuple[float, float]],
        nest_bboxes: Dict
    ) -> Dict:
        """Analyze position-based features.
        
        Args:
            trajectory: List of positions
            nest_bboxes: Nest bounding boxes
            
        Returns:
            Dictionary of position features
        """
        start_pos = trajectory[0]
        end_pos = trajectory[-1]
        
        # Check if start/end are inside nests
        start_in_nest = self._check_inside_any_nest(start_pos, nest_bboxes)
        end_in_nest = self._check_inside_any_nest(end_pos, nest_bboxes)
        
        # Calculate trajectory direction relative to nest
        direction_to_nest = self._calculate_direction_to_nearest_nest(
            start_pos, nest_bboxes
        )
        
        return {
            'start_in_nest': start_in_nest,
            'end_in_nest': end_in_nest,
            'direction_to_nest': direction_to_nest
        }
    
    def _analyze_motion_features(
        self,
        trajectory: List[Tuple[float, float]]
    ) -> Dict:
        """Analyze motion-based features.
        
        Args:
            trajectory: List of positions
            
        Returns:
            Dictionary of motion features
        """
        speeds = self.trajectory_analyzer.calculate_speed(trajectory)
        
        if not speeds:
            return {
                'avg_speed': 0.0,
                'start_speed': 0.0,
                'end_speed': 0.0,
                'decelerating': False,
                'accelerating': False
            }
        
        # Analyze speed pattern
        start_speed = np.mean(speeds[:min(3, len(speeds))])
        end_speed = np.mean(speeds[-min(3, len(speeds)):])
        avg_speed = np.mean(speeds)
        
        # Check for deceleration/acceleration patterns
        decelerating = end_speed < start_speed * 0.5
        accelerating = end_speed > start_speed * 2.0
        
        return {
            'avg_speed': avg_speed,
            'start_speed': start_speed,
            'end_speed': end_speed,
            'decelerating': decelerating,
            'accelerating': accelerating
        }
    
    def _analyze_trajectory_features(
        self,
        trajectory: List[Tuple[float, float]]
    ) -> Dict:
        """Analyze trajectory shape features.
        
        Args:
            trajectory: List of positions
            
        Returns:
            Dictionary of trajectory features
        """
        tortuosity = self.trajectory_analyzer.calculate_tortuosity(trajectory)
        path_length = self.trajectory_analyzer.calculate_trajectory_length(trajectory)
        displacement = self.trajectory_analyzer.calculate_displacement(trajectory)
        
        return {
            'tortuosity': tortuosity,
            'path_length': path_length,
            'displacement': displacement,
            'is_direct': tortuosity < 1.5  # Relatively straight path
        }
    
    def _make_classification_decision(
        self,
        position_features: Dict,
        motion_features: Dict,
        trajectory_features: Dict
    ) -> Optional[str]:
        """Make final classification decision based on all features.
        
        Args:
            position_features: Position-based features
            motion_features: Motion-based features
            trajectory_features: Trajectory-based features
            
        Returns:
            'entry', 'exit', or None
        """
        # Entry: trajectory ends inside a nest
        if position_features['end_in_nest'] and not position_features['start_in_nest']:
            # Additional checks for confidence
            if motion_features['decelerating'] or motion_features['end_speed'] < 15:
                return 'entry'
        
        # Exit: trajectory starts inside a nest
        elif position_features['start_in_nest'] and not position_features['end_in_nest']:
            # Additional checks for confidence
            if motion_features['accelerating'] or motion_features['start_speed'] < 15:
                return 'exit'
        
        return None
    
    def _check_inside_any_nest(
        self,
        position: Tuple[float, float],
        nest_bboxes: Dict
    ) -> bool:
        """Check if position is inside any nest.
        
        Args:
            position: (x, y) position
            nest_bboxes: Dictionary of nest bounding boxes
            
        Returns:
            True if inside any nest
        """
        x, y = position
        
        for nest_id, bbox in nest_bboxes.items():
            x_min, y_min, x_max, y_max = bbox
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        
        return False
    
    def _calculate_direction_to_nearest_nest(
        self,
        position: Tuple[float, float],
        nest_bboxes: Dict
    ) -> float:
        """Calculate direction angle to nearest nest.
        
        Args:
            position: (x, y) position
            nest_bboxes: Dictionary of nest bounding boxes
            
        Returns:
            Angle in radians
        """
        x, y = position
        
        min_distance = float('inf')
        nearest_nest_center = None
        
        for nest_id, bbox in nest_bboxes.items():
            x_min, y_min, x_max, y_max = bbox
            nest_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            
            distance = np.sqrt((nest_center[0] - x)**2 + (nest_center[1] - y)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_nest_center = nest_center
        
        if nearest_nest_center is None:
            return 0.0
        
        dx = nearest_nest_center[0] - x
        dy = nearest_nest_center[1] - y
        
        return np.arctan2(dy, dx)