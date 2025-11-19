"""Example usage of BeeMonitor enhancements.

This script demonstrates:
1. Training data generation
2. Hotel box-aware configuration
3. Parameter optimization
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_generate_training_data():
    
    """Example 1: Generate training data from videos."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training Data Generation")
    print("="*60 + "\n")
    
    from beemonitor.ml.generate_image_training_data import TrainingDataGenerator
    
    # Initialize generator
    generator = TrainingDataGenerator(
        #video_folder="/Users/edwardamoah/Documents/GitHub/BeeMonitor/videos/mendels_2025/training_data",
        video_folder = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/videos/mendels_2025/generation_test",
        #output_folder="/Users/edwardamoah/Documents/GitHub/BeeMonitor/videos/mendels_2025/training_data/training_data",
        output_folder = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/videos/mendels_2025/generation_test/training_data",
        bee_detector_model="/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking.pt",
        interested_bee_labels=[0],  # Honey bee, Bumble bee, Mason bee
        min_detection_confidence=0.35,
        use_quality_filtering=False,
        include_empty_frames=False,
        use_motion_validation=False,
        use_temporal_validation=False,
        num_workers=8
    )

    print("Configuration:")
    print(f"  Motion validation: {generator.use_motion_validation}")
    print(f"  Temporal validation: {generator.use_temporal_validation}")
    print(f"  Min confidence: {generator.min_detection_confidence}")
    #print(f"  Model: {generator.bee_detector_model.ckpt_path}")

    # Generate balanced dataset
    stats = generator.generate_dataset(
        num_bee_like_frames=1000,
        num_non_bee_like_frames=300,
        min_frame_gap=0,
        diversity_threshold=0.0005,
        sample_rate=5,
        enforce_event_diversity=True,
        max_frames_per_session= None
    )

    # Visualize all annotations
    generator.visualize_annotations()

    
    #print(f"Dataset generated with {stats['bee_like_frames']} bee frames")
    print(f"Dataset statistics: {stats}")
    
    # Convert to YOLO format for training
    generator.generate_yolo_format(train_split=0.8)
    
    
     # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total detections: {stats['total_detections']}")
    print(f"Frames collected: {stats['bee_like_frames'] + stats['non_bee_like_frames']}")
    print(f"Bee-like frames: {stats['bee_like_frames']}")
    print(f"Unique events: {stats['unique_event_sessions_sampled']}")
    
    if stats['bee_like_frames'] > 0:
        acceptance_rate = (stats['bee_like_frames'] / stats['frames_analyzed'] * 100)
        print(f"\n✓ SUCCESS! Acceptance rate: {acceptance_rate:.1f}%")
        print(f"✓ This is a huge improvement from 0%!")
        
        # Visualize
        print("\nCreating visualizations...")
        generator.visualize_annotations(
            max_frames=100,
            show_metadata=True,
            bbox_thickness=3
        )
        
        print("\n✓ Check visualizations at:")
        print("  data/training_fixed/visualizations/")
        print("\n✓ Open index.html in a browser to review frames")
        
    else:
        print("\n❌ Still 0 frames collected!")
        print("   Check:")
        print("   1. Are there detections? (total_detections should be > 0)")
        print("   2. Do interested_bee_labels match model classes?")
        print("   3. Is min_detection_confidence too high?")
    
    print("\n" + "="*60)
    return stats


def example_2_hotel_box_configuration():
    """Example 2: Configure parameters based on hotel box position."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Hotel Box-Aware Configuration")
    print("="*60 + "\n")
    
    from beemonitor.core.config import Config
    
    # Scenario 1: Default configuration (1080p, standard distance)
    config1 = Config.default()
    config1.video.res_width = 1920
    config1.video.res_height = 1080
    
    print("Configuration 1: 1080p, standard distance")
    print(f"  Nest width: {config1.nest.nest_width(1920, 1080, config1.hotel_box)}px")
    print(f"  Distance threshold: {config1.tracking.distance_threshold(1920, 1080, config1.hotel_box):.1f}px")
    
    # Scenario 2: 4K resolution, hotel farther away
    config2 = Config.for_distance(1.5)  # 50% farther
    config2.video.res_width = 3840
    config2.video.res_height = 2160
    
    print("\nConfiguration 2: 4K, hotel 50% farther")
    print(f"  Nest width: {config2.nest.nest_width(3840, 2160, config2.hotel_box)}px")
    print(f"  Distance threshold: {config2.tracking.distance_threshold(3840, 2160, config2.hotel_box):.1f}px")
    
    # Scenario 3: Lower resolution, hotel closer
    config3 = Config.for_distance(0.7)  # 30% closer
    config3.video.res_width = 1280
    config3.video.res_height = 720
    
    print("\nConfiguration 3: 720p, hotel 30% closer")
    print(f"  Nest width: {config3.nest.nest_width(1280, 720, config3.hotel_box)}px")
    print(f"  Distance threshold: {config3.tracking.distance_threshold(1280, 720, config3.hotel_box):.1f}px")
    
    # Print full scaled parameters
    print("\nFull parameter scaling for Config 2:")
    config2.print_scaled_values()
    
    # Save configuration
    config2.save_yaml("config_4k_far.yaml")
    print("\nConfiguration saved to: config_4k_far.yaml")


def example_3_parameter_optimization():
    """Example 3: Optimize parameters using real data."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Parameter Optimization")
    print("="*60 + "\n")
    
    from beemonitor.ml.parameter_optimizer import ParameterOptimizer
    
    # Initialize optimizer with ground truth data
    optimizer = ParameterOptimizer(
        ground_truth_file="data/ground_truth_events.csv",
        tracking_data_file="data/tracking_results.pkl",
        output_dir="optimization_results"
    )
    
    print("Loaded ground truth and tracking data")
    
    # Optimize all parameters
    print("\nOptimizing all parameters...")
    result = optimizer.optimize_all_parameters(
        method='differential_evolution',
        max_iterations=50
    )
    
    print(f"\nOptimization Results:")
    print(f"  Initial F1 Score: {result.initial_score:.4f}")
    print(f"  Optimized F1 Score: {result.best_score:.4f}")
    print(f"  Improvement: {result.improvement:.1f}%")
    
    print(f"\nBest Parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value:.2f}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {result.metrics['precision']:.4f}")
    print(f"  Recall: {result.metrics['recall']:.4f}")
    print(f"  F1 Score: {result.metrics['f1_score']:.4f}")
    
    print("\nOptimization results saved to: optimization_results/")


def example_4_sensitivity_analysis():
    """Example 4: Analyze parameter sensitivity."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Parameter Sensitivity Analysis")
    print("="*60 + "\n")
    
    from beemonitor.ml.parameter_optimizer import ParameterOptimizer
    
    optimizer = ParameterOptimizer(
        ground_truth_file="data/ground_truth_events.csv",
        tracking_data_file="data/tracking_results.pkl",
        output_dir="sensitivity_analysis"
    )
    
    # Analyze sensitivity of entry_padding parameter
    print("Analyzing sensitivity of entry_padding...")
    df = optimizer.analyze_parameter_sensitivity(
        parameter_name='entry_padding',
        value_range=(5, 30),
        num_points=20
    )
    
    print(f"\nBest entry_padding value: {df.loc[df['f1_score'].idxmax(), 'value']:.1f}")
    print(f"Best F1 score: {df['f1_score'].max():.4f}")
    
    # Analyze sensitivity of distance_threshold
    print("\nAnalyzing sensitivity of distance_threshold...")
    df2 = optimizer.analyze_parameter_sensitivity(
        parameter_name='distance_threshold',
        value_range=(50, 200),
        num_points=20
    )
    
    print(f"\nBest distance_threshold value: {df2.loc[df2['f1_score'].idxmax(), 'value']:.1f}")
    print(f"Best F1 score: {df2['f1_score'].max():.4f}")
    
    print("\nSensitivity analysis plots saved to: sensitivity_analysis/")


def example_5_improved_event_classification():
    """Example 5: Use improved event classification."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Improved Event Classification")
    print("="*60 + "\n")
    
    from beemonitor.ml.parameter_optimizer import ImprovedEventClassifier
    from beemonitor.core.config import Config
    
    config = Config.default()
    classifier = ImprovedEventClassifier(config)
    
    # Example trajectory
    # This would normally come from your tracking system
    trajectory = [
        (100, 150), (105, 145), (110, 140), (115, 135), (120, 130)
    ]
    
    nest_bboxes = {
        '1': (115, 125, 135, 145),  # Nest 1
        '2': (145, 125, 165, 145),  # Nest 2
    }
    
    # Create movement tuple (track_id, centroids, bboxes, frame_numbers)
    movement = (
        1,  # track_id
        trajectory,  # centroids
        [(0, 0, 20, 20)] * len(trajectory),  # bboxes (placeholder)
        list(range(len(trajectory)))  # frame_numbers
    )
    
    # Classify event
    event_type = classifier.classify_event(movement, nest_bboxes)
    
    print(f"Trajectory classification: {event_type}")
    
    print("\nImproved classifier uses:")
    print("  - Position relative to nests (not just speed)")
    print("  - Trajectory shape and direction")
    print("  - Movement patterns (acceleration/deceleration)")
    print("  - More robust for walking bees!")


def example_6_complete_workflow():
    """Example 6: Complete workflow from video to optimized analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Complete Workflow")
    print("="*60 + "\n")
    
    from beemonitor.core.config import Config
    from beemonitor.core.video_analyzer import VideoAnalyzer
    
    # Step 1: Configure for your setup
    print("Step 1: Configuring for your setup...")
    config = Config.default()
    
    # Set video resolution
    config.video.res_width = 1920
    config.video.res_height = 1080
    
    # Set hotel box parameters (if known)
    config.hotel_box.distance_factor = 1.2  # Slightly farther than reference
    config.hotel_box.x_center = 0.5  # Centered horizontally
    config.hotel_box.y_center = 0.45  # Slightly above center
    
    print(f"  Resolution: {config.resolution}")
    print(f"  Distance factor: {config.hotel_box.distance_factor}")
    
    # Step 2: Process video
    print("\nStep 2: Processing video...")
    analyzer = VideoAnalyzer(config)
    
    # This would process your actual video
    # results = analyzer.analyze_video("path/to/video.mp4")
    print("  (Video processing would happen here)")
    
    # Step 3: Generate ground truth samples for optimization
    print("\nStep 3: Generate training data...")
    # from beemonitor.ml.generate_training_data import TrainingDataGenerator
    # generator = TrainingDataGenerator("videos/", "training_data/")
    # stats = generator.generate_dataset(num_bee_frames=500)
    print("  (Training data generation would happen here)")
    
    # Step 4: Optimize parameters with ground truth
    print("\nStep 4: Optimize parameters...")
    # from beemonitor.ml.parameter_optimizer import ParameterOptimizer
    # optimizer = ParameterOptimizer("ground_truth.csv", "tracking.pkl")
    # result = optimizer.optimize_all_parameters()
    print("  (Parameter optimization would happen here)")
    
    # Step 5: Re-analyze with optimized parameters
    print("\nStep 5: Re-analyze with optimized parameters...")
    # optimized_config = Config.from_yaml("optimization_results/best_config.yaml")
    # analyzer = VideoAnalyzer(optimized_config)
    # final_results = analyzer.analyze_video("path/to/video.mp4")
    print("  (Final analysis would happen here)")
    
    print("\nComplete workflow done!")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("BeeMonitor Enhancement Examples")
    print("="*60)
    
    # Note: Comment out examples that require data you don't have yet
    
    example_1_generate_training_data()
    # example_2_hotel_box_configuration()
    # # example_3_parameter_optimization()
    # # example_4_sensitivity_analysis()
    # example_5_improved_event_classification()
    # example_6_complete_workflow()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()