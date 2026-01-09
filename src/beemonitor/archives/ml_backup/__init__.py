"""Machine Learning module for BeeMonitor.

This module provides tools for generating training data and fine-tuning
detection, tracking, and classification models.
"""

from beemonitor.ml.generate_image_training_data import TrainingDataGenerator
from beemonitor.ml.parameter_optimezer import ParameterOptimizer

__all__ = ['TrainingDataGenerator', 'ParameterOptimizer']