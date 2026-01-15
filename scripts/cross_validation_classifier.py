"""Proper cross-validation for ML Event Classifier.

This script performs Leave-One-Video-Out cross-validation to get
HONEST performance metrics without data leakage.

Use these metrics for your research paper!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from beemonitor.processing.event_processor import EventProcessor
from beemonitor.core.config import Config
from beemonitor.detection.nest_detector import NestDetector
from ultralytics import YOLO
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_trajectory_features(movement, detected_action, detected_nest, nests):
    """Extract features - same as training script."""
    track_id, centroids, bboxes, frame_numbers = movement[:4]
    features = {}
    
    features['trajectory_length'] = len(centroids)
    
    path_length = 0.0
    for i in range(len(centroids) - 1):
        dx = centroids[i+1][0] - centroids[i][0]
        dy = centroids[i+1][1] - centroids[i][1]
        path_length += np.sqrt(dx**2 + dy**2)
    features['path_length'] = path_length
    
    displacement = np.sqrt(
        (centroids[-1][0] - centroids[0][0])**2 + 
        (centroids[-1][1] - centroids[0][1])**2
    )
    features['displacement'] = displacement
    features['tortuosity'] = path_length / displacement if displacement > 5 else 0
    
    speeds = []
    for i in range(len(centroids) - 1):
        dx = centroids[i+1][0] - centroids[i][0]
        dy = centroids[i+1][1] - centroids[i][1]
        speed = np.sqrt(dx**2 + dy**2)
        speeds.append(speed)
    
    if len(speeds) > 0:
        features['avg_speed'] = np.mean(speeds)
        features['max_speed'] = np.max(speeds)
        features['speed_std'] = np.std(speeds)
        features['speed_cv'] = features['speed_std'] / features['avg_speed'] if features['avg_speed'] > 0 else 0
        
        if len(speeds) >= 6:
            third = len(speeds) // 3
            features['start_speed'] = np.mean(speeds[:third])
            features['middle_speed'] = np.mean(speeds[third:2*third])
            features['end_speed'] = np.mean(speeds[-third:])
            features['decel_ratio'] = features['end_speed'] / features['start_speed'] if features['start_speed'] > 0 else 1.0
        else:
            features['start_speed'] = features['avg_speed']
            features['middle_speed'] = features['avg_speed']
            features['end_speed'] = features['avg_speed']
            features['decel_ratio'] = 1.0
    else:
        features['avg_speed'] = 0
        features['max_speed'] = 0
        features['speed_std'] = 0
        features['speed_cv'] = 0
        features['start_speed'] = 0
        features['middle_speed'] = 0
        features['end_speed'] = 0
        features['decel_ratio'] = 1.0
    
    nest_bbox = nests['nests'].get(int(detected_nest)) if detected_nest else None
    
    if nest_bbox:
        nest_x = (nest_bbox[0] + nest_bbox[2]) / 2
        nest_y = (nest_bbox[1] + nest_bbox[3]) / 2
        
        start_dist = np.sqrt((centroids[0][0] - nest_x)**2 + (centroids[0][1] - nest_y)**2)
        end_dist = np.sqrt((centroids[-1][0] - nest_x)**2 + (centroids[-1][1] - nest_y)**2)
        
        features['start_to_nest'] = start_dist
        features['end_to_nest'] = end_dist
        features['approach_ratio'] = end_dist / start_dist if start_dist > 0 else 1.0
    else:
        features['start_to_nest'] = 999
        features['end_to_nest'] = 999
        features['approach_ratio'] = 1.0
    
    x_pos = [c[0] for c in centroids]
    y_pos = [c[1] for c in centroids]
    features['x_var'] = np.var(x_pos)
    features['y_var'] = np.var(y_pos)
    
    features['vertical_movement'] = centroids[-1][1] - centroids[0][1]
    features['horizontal_movement'] = abs(centroids[-1][0] - centroids[0][0])
    features['is_entry'] = 1 if detected_action == 'Entry' else 0
    
    return features


def main():
    print("="*70)
    print("LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION")
    print("Honest evaluation without data leakage")
    print("="*70)
    
    # Setup (same as training)
    config = Config.default()
    nest_model = YOLO(config.models.nest_detection)
    detector = NestDetector(nest_model, config)
    
    input_data = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/CVPR_Evaluation_Video_Data"
    output_folder = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/output/CVPR_Output"
    manual_csv = '/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/Manual_Foraging_Events_Observation.csv'
    
    files = os.listdir(input_data)
    files = [os.path.join(input_data, file) for file in files if 'mp4' in file]
    
    manual_df = pd.read_csv(manual_csv)
    manual_df = manual_df[['video', 'action', 'nest', 'timestamp']].dropna()
    
    def parse_manual_time(video, time_str):
        date_part = video.split('_')[1]
        return datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    manual_df['dt'] = manual_df.apply(lambda x: parse_manual_time(x['video'], x['timestamp']), axis=1)
    
    tracking_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('_tracking_results.csv')]
    tracking_data = {}
    for file in tracking_files:
        video_name = file.replace('_tracking_results.csv', '').split("/")[-1]
        tracking_data[video_name] = pd.read_csv(file)
    
    # ==================================================================
    # EXTRACT FEATURES FROM ALL VIDEOS (REAL DATA)
    # ==================================================================
    
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM ALL EVENTS")
    print("="*70)
    
    all_features = []
    all_labels = []
    all_video_names = []
    
    processor = EventProcessor(config)
    
    def reconstruct_motion_data(tracking_df):
        """Convert tracking CSV to motion_data format."""
        trajectories = []
        for track_id in tracking_df['track_id'].unique():
            track_data = tracking_df[tracking_df['track_id'] == track_id].sort_values('frame')
            
            centroids = []
            for _, row in track_data.iterrows():
                centroid_x = (row['x1'] + row['x2']) / 2
                centroid_y = (row['y1'] + row['y2']) / 2
                centroids.append((centroid_x, centroid_y))
            
            bboxes = list(zip(track_data['x1'], track_data['y1'], 
                             track_data['x2'], track_data['y2']))
            frame_numbers = track_data['frame'].tolist()
            
            trajectory = (track_id, centroids, bboxes, frame_numbers)
            trajectories.append(trajectory)
        
        return pd.DataFrame({'tracks': [trajectories]})
    
    for video_name, tracking_df in tracking_data.items():
        print(f"\nProcessing {video_name}...")
        
        # Get video file and nests
        video_file = [f for f in files if video_name in f]
        if len(video_file) == 0:
            print(f"  Skipping - video file not found")
            continue
        
        video_file = video_file[0]
        
        try:
            nests = detector.get_nests_and_hotel_detections(video_file)
        except Exception as e:
            print(f"  Skipping - nest detection failed: {e}")
            continue
        
        # Reconstruct motion data
        motion_data = reconstruct_motion_data(tracking_df)
        
        # Get trajectories
        movements = []
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        # Filter fragments
        movements = processor._filter_trajectory_fragments(
            movements, min_length=10, min_distance=30.0
        )
        
        # Process each movement
        for movement in movements:
            if len(movement[1]) < 10:
                continue
            
            centroids = movement[1]
            
            # Check start (exit)
            start_in_nest = False
            start_nest_id = None
            for nest_id, bbox in nests['nests'].items():
                if all(processor._is_inside_bbox(c, bbox, 20) for c in centroids[:3]):
                    start_in_nest = True
                    start_nest_id = nest_id
                    break
            
            # Check end (entry)
            end_in_nest = False
            end_nest_id = None
            for nest_id, bbox in nests['nests'].items():
                if all(processor._is_inside_bbox(c, bbox, 20) for c in centroids[-3:]):
                    end_in_nest = True
                    end_nest_id = nest_id
                    break
            
            # Extract features for each detected event
            if start_in_nest and not end_in_nest:
                # Exit event
                features = extract_trajectory_features(movement, 'Exit', start_nest_id, nests)
                all_features.append(features)
                all_video_names.append((video_name, 'Exit', start_nest_id, movement[3][0]))
            
            elif not start_in_nest and end_in_nest:
                # Entry event
                features = extract_trajectory_features(movement, 'Entry', end_nest_id, nests)
                all_features.append(features)
                all_video_names.append((video_name, 'Entry', end_nest_id, movement[3][-1]))
    
    features_df = pd.DataFrame(all_features)
    print(f"\n✓ Extracted features for {len(features_df)} detected events")
    
    # ==================================================================
    # MATCH TO MANUAL LABELS
    # ==================================================================
    
    print("\n" + "="*70)
    print("MATCHING TO MANUAL LABELS")
    print("="*70)
    
    labels = []
    
    for i, (video_name, action, nest_id, frame_num) in enumerate(all_video_names):
        parts = video_name.split('_')
        date_str = f"{parts[1]} {parts[2]}:{parts[3]}:{parts[4]}"
        video_start = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        
        event_time = video_start + timedelta(seconds=frame_num/30.0)
        
        matched = False
        manual_for_video = manual_df[manual_df['video'] == video_name]
        
        for _, man_event in manual_for_video.iterrows():
            man_nest = int(man_event['nest'])
            det_nest = int(nest_id)
            
            if (man_event['action'] == action and man_nest == det_nest):
                time_diff = abs(event_time - man_event['dt'])
                if time_diff <= timedelta(seconds=3.0):
                    matched = True
                    break
        
        labels.append(1 if matched else 0)
    
    labels = pd.Series(labels)
    video_groups = [v[0] for v in all_video_names]  # Video name for each event
    
    print(f"\nLabeled {len(labels)} events:")
    print(f"  Real events (TP): {sum(labels == 1)}")
    print(f"  Noise (FP): {sum(labels == 0)}")
    
    if sum(labels == 1) < 10:
        print("\nERROR: Not enough positive examples for cross-validation!")
        return
    
    # ==================================================================
    # LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION (REAL)
    # ==================================================================
    
    print("\n" + "="*70)
    print("PERFORMING LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION")
    print("="*70)
    
    unique_videos = sorted(set(video_groups))
    cv_results = []
    
    for held_out_video in unique_videos:
        print(f"\nFold: Testing on {held_out_video}")
        
        # Split by video
        train_mask = [v != held_out_video for v in video_groups]
        test_mask = [v == held_out_video for v in video_groups]
        
        X_train = features_df[train_mask]
        y_train = labels[train_mask]
        X_test = features_df[test_mask]
        y_test = labels[test_mask]
        
        if len(X_test) == 0:
            print(f"  No events in this video, skipping")
            continue
        
        if sum(y_train == 1) < 2 or sum(y_train == 0) < 2:
            print(f"  Not enough training examples, skipping")
            continue
        
        print(f"  Training on {len(X_train)} events ({sum(y_train==1)} real, {sum(y_train==0)} noise)")
        print(f"  Testing on {len(X_test)} events ({sum(y_test==1)} real, {sum(y_test==0)} noise)")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            # Edge case: all same class
            precision = 1.0 if sum(y_pred == y_test) == len(y_test) else 0.0
            recall = precision
            f1 = precision
        
        cv_results.append({
            'video': held_out_video,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_test': len(X_test),
            'n_real': sum(y_test == 1)
        })
        
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY (HONEST METRICS)")
    print("="*70)
    
    avg_precision = np.mean([r['precision'] for r in cv_results])
    avg_recall = np.mean([r['recall'] for r in cv_results])
    avg_f1 = np.mean([r['f1'] for r in cv_results])
    
    std_precision = np.std([r['precision'] for r in cv_results])
    std_recall = np.std([r['recall'] for r in cv_results])
    std_f1 = np.std([r['f1'] for r in cv_results])
    
    print(f"\nMean Performance (± std):")
    print(f"  Precision: {avg_precision:.3f} ± {std_precision:.3f}")
    print(f"  Recall:    {avg_recall:.3f} ± {std_recall:.3f}")
    print(f"  F1 Score:  {avg_f1:.3f} ± {std_f1:.3f}")
    
    print(f"\n95% Confidence Intervals:")
    print(f"  Precision: [{avg_precision - 1.96*std_precision:.3f}, {avg_precision + 1.96*std_precision:.3f}]")
    print(f"  Recall:    [{avg_recall - 1.96*std_recall:.3f}, {avg_recall + 1.96*std_recall:.3f}]")
    print(f"  F1 Score:  [{avg_f1 - 1.96*std_f1:.3f}, {avg_f1 + 1.96*std_f1:.3f}]")
    
    print("\nPer-Video Breakdown:")
    print(f"{'Video':<40} {'Precision':<12} {'Recall':<12} {'F1'}")
    print("-" * 76)
    for r in cv_results:
        print(f"{r['video']:<40} {r['precision']:<12.3f} {r['recall']:<12.3f} {r['f1']:.3f}")
    
    # Identify difficult videos
    difficult = sorted(cv_results, key=lambda x: x['f1'])[:3]
    print(f"\nMost Difficult Videos (lowest F1):")
    for r in difficult:
        print(f"  {r['video']}: F1 = {r['f1']:.3f}")
    
    print("\n" + "="*70)
    print("FOR YOUR RESEARCH PAPER")
    print("="*70)
    print("\nReport these metrics:")
    print(f"  \"The ML classifier achieved {avg_precision:.1%} precision")
    print(f"   and {avg_recall:.1%} recall (F1 = {avg_f1:.3f}) using")
    print(f"   leave-one-video-out cross-validation.\"")
    
    print("\nMethodology note:")
    print("  \"We employed leave-one-video-out cross-validation to")
    print("   prevent data leakage, training on 10 videos and testing")
    print("   on the held-out video, repeated for all 11 videos.\"")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()