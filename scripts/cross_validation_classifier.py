"""Proper cross-validation for ML Event Classifier.

Leave-One-Video-Out cross-validation for HONEST performance metrics.
Uses current EventProcessor API (ML-first, no trajectory filtering).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from beemonitor.core.config import Config
from beemonitor.detection.nest_detector import NestDetector
from ultralytics import YOLO
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_trajectory_features(centroids, action, nest_bbox):
    """Extract 20 features from trajectory for ML classification.
    
    Mirrors EventProcessor._extract_trajectory_features()
    """
    nest_x = (nest_bbox[0] + nest_bbox[2]) / 2
    nest_y = (nest_bbox[1] + nest_bbox[3]) / 2
    
    # Trajectory shape features
    trajectory_length = len(centroids)
    
    path_length = 0.0
    for i in range(len(centroids) - 1):
        dx = centroids[i+1][0] - centroids[i][0]
        dy = centroids[i+1][1] - centroids[i][1]
        path_length += np.sqrt(dx**2 + dy**2)
    
    displacement = np.sqrt(
        (centroids[-1][0] - centroids[0][0])**2 +
        (centroids[-1][1] - centroids[0][1])**2
    )
    
    tortuosity = path_length / displacement if displacement > 0 else 0
    
    # Speed profile
    speeds = []
    for i in range(len(centroids) - 1):
        dx = centroids[i+1][0] - centroids[i][0]
        dy = centroids[i+1][1] - centroids[i][1]
        speeds.append(np.sqrt(dx**2 + dy**2))
    
    avg_speed = np.mean(speeds) if speeds else 0
    max_speed = np.max(speeds) if speeds else 0
    speed_std = np.std(speeds) if speeds else 0
    speed_cv = speed_std / avg_speed if avg_speed > 0 else 0
    
    third = len(speeds) // 3 if len(speeds) >= 3 else 1
    start_speed = np.mean(speeds[:third]) if speeds else 0
    middle_speed = np.mean(speeds[third:2*third]) if len(speeds) >= 3 else avg_speed
    end_speed = np.mean(speeds[-third:]) if speeds else 0
    decel_ratio = end_speed / start_speed if start_speed > 0 else 1.0
    
    # Nest proximity
    start_to_nest = np.sqrt((centroids[0][0] - nest_x)**2 + (centroids[0][1] - nest_y)**2)
    end_to_nest = np.sqrt((centroids[-1][0] - nest_x)**2 + (centroids[-1][1] - nest_y)**2)
    approach_ratio = end_to_nest / start_to_nest if start_to_nest > 0 else 1.0
    
    # Position variance
    x_var = np.var([c[0] for c in centroids])
    y_var = np.var([c[1] for c in centroids])
    
    # Direction
    vertical_movement = centroids[-1][1] - centroids[0][1]
    horizontal_movement = abs(centroids[-1][0] - centroids[0][0])
    
    # Event type
    is_entry = 1 if action == 'Entry' else 0
    
    return {
        'trajectory_length': trajectory_length,
        'path_length': path_length,
        'displacement': displacement,
        'tortuosity': tortuosity,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'speed_std': speed_std,
        'speed_cv': speed_cv,
        'start_speed': start_speed,
        'middle_speed': middle_speed,
        'end_speed': end_speed,
        'decel_ratio': decel_ratio,
        'start_to_nest': start_to_nest,
        'end_to_nest': end_to_nest,
        'approach_ratio': approach_ratio,
        'x_var': x_var,
        'y_var': y_var,
        'vertical_movement': vertical_movement,
        'horizontal_movement': horizontal_movement,
        'is_entry': is_entry
    }


def is_inside_bbox(point, bbox, padding=40):
    """Check if point is inside bbox with padding (matches EventProcessor)."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - padding <= x <= x2 + padding and 
            y1 - padding <= y <= y2 + padding)


def detect_events_from_tracking(tracking_df, nests):
    """Detect events from tracking data using lenient settings.
    
    Mirrors EventProcessor._detect_all_events():
    - window_size=1 (only need 1 frame inside nest)
    - padding=40 (large detection area)
    """
    events = []
    hole_bboxes = nests['nests']
    padding = 40
    
    for track_id in tracking_df['track_id'].unique():
        track_data = tracking_df[tracking_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) < 2:
            continue
        
        centroids = []
        for _, row in track_data.iterrows():
            cx = (row['x1'] + row['x2']) / 2
            cy = (row['y1'] + row['y2']) / 2
            centroids.append((cx, cy))
        
        frame_numbers = track_data['frame'].tolist()
        
        # Check EXIT (start in nest)
        start_pos = centroids[0]
        for nest_id, bbox in hole_bboxes.items():
            if is_inside_bbox(start_pos, bbox, padding):
                events.append({
                    'action': 'Exit',
                    'nest': nest_id,
                    'frame_number': frame_numbers[0],
                    'track_id': track_id,
                    'centroids': centroids,
                    'nest_bbox': bbox
                })
                break
        
        # Check ENTRY (end in nest)
        end_pos = centroids[-1]
        for nest_id, bbox in hole_bboxes.items():
            if is_inside_bbox(end_pos, bbox, padding):
                events.append({
                    'action': 'Entry',
                    'nest': nest_id,
                    'frame_number': frame_numbers[-1],
                    'track_id': track_id,
                    'centroids': centroids,
                    'nest_bbox': bbox
                })
                break
    
    return events


def main():
    print("="*70)
    print("LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION")
    print("ML Event Classifier Evaluation")
    print("="*70)
    
    # Setup
    config = Config.default()
    nest_model = YOLO(config.models.nest_detection)
    detector = NestDetector(nest_model, config)
    
    input_data = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/CVPR_Evaluation_Video_Data"
    output_folder = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/CVPR_Evaluation_Video_Data_output"
    manual_csv = '/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/Manual_Foraging_Events_Observation.csv'
    
    files = os.listdir(input_data)
    files = [os.path.join(input_data, file) for file in files if 'mp4' in file]
    
    manual_df = pd.read_csv(manual_csv)
    manual_df = manual_df[['video', 'action', 'nest', 'timestamp']].dropna()
    
    def parse_manual_time(video, time_str):
        date_part = video.split('_')[1]
        return datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    manual_df['dt'] = manual_df.apply(lambda x: parse_manual_time(x['video'], x['timestamp']), axis=1)
    
    tracking_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) 
                      if f.endswith('_tracking_results.csv')]
    tracking_data = {}
    for file in tracking_files:
        video_name = file.replace('_tracking_results.csv', '').split("/")[-1]
        tracking_data[video_name] = pd.read_csv(file)
    
    print(f"\nFound {len(tracking_data)} tracking files")
    
    # Extract features
    print("\n" + "="*70)
    print("EXTRACTING FEATURES FROM ALL EVENTS")
    print("="*70)
    
    all_features = []
    all_metadata = []  # (video_name, action, nest_id, frame)
    
    for video_name, tracking_df in tracking_data.items():
        print(f"\nProcessing {video_name}...")
        
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
        
        # Detect events using lenient settings (matches EventProcessor)
        events = detect_events_from_tracking(tracking_df, nests)
        print(f"  Detected {len(events)} candidate events")
        
        for event in events:
            features = extract_trajectory_features(
                event['centroids'],
                event['action'],
                event['nest_bbox']
            )
            all_features.append(features)
            all_metadata.append((
                video_name, 
                event['action'], 
                event['nest'], 
                event['frame_number']
            ))
    
    features_df = pd.DataFrame(all_features)
    print(f"\n✓ Extracted features for {len(features_df)} detected events")
    
    # Match to manual labels
    print("\n" + "="*70)
    print("MATCHING TO MANUAL LABELS")
    print("="*70)
    
    labels = []
    video_groups = []
    
    for i, (video_name, action, nest_id, frame_num) in enumerate(all_metadata):
        video_groups.append(video_name)
        
        parts = video_name.split('_')
        date_str = f"{parts[1]} {parts[2]}:{parts[3]}:{parts[4]}"
        video_start = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        event_time = video_start + timedelta(seconds=frame_num/30.0)
        
        matched = False
        manual_for_video = manual_df[manual_df['video'] == video_name]
        
        for _, man_event in manual_for_video.iterrows():
            man_nest = int(man_event['nest'])
            det_nest = int(nest_id)
            
            if man_event['action'] == action and man_nest == det_nest:
                time_diff = abs(event_time - man_event['dt'])
                if time_diff <= timedelta(seconds=3.0):
                    matched = True
                    break
        
        labels.append(1 if matched else 0)
    
    labels = np.array(labels)
    video_groups = np.array(video_groups)
    
    print(f"\nLabeled {len(labels)} events:")
    print(f"  Real events (TP candidates): {sum(labels == 1)}")
    print(f"  Noise (FP candidates): {sum(labels == 0)}")
    
    if sum(labels == 1) < 10:
        print("\nERROR: Not enough positive examples for cross-validation!")
        return
    
    # LOVO Cross-Validation
    print("\n" + "="*70)
    print("LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION")
    print("="*70)
    
    unique_videos = sorted(set(video_groups))
    cv_results = []
    
    all_y_true = []
    all_y_pred = []
    
    for held_out_video in unique_videos:
        print(f"\nFold: Testing on {held_out_video}")
        
        train_mask = video_groups != held_out_video
        test_mask = video_groups == held_out_video
        
        X_train = features_df[train_mask].values
        y_train = labels[train_mask]
        X_test = features_df[test_mask].values
        y_test = labels[test_mask]
        
        if len(X_test) == 0:
            print(f"  No events in this video, skipping")
            continue
        
        if sum(y_train == 1) < 2 or sum(y_train == 0) < 2:
            print(f"  Not enough training examples, skipping")
            continue
        
        print(f"  Train: {len(X_train)} ({sum(y_train==1)} real, {sum(y_train==0)} noise)")
        print(f"  Test:  {len(X_test)} ({sum(y_test==1)} real, {sum(y_test==0)} noise)")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Collect for aggregate metrics
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Per-fold metrics
        if len(np.unique(y_test)) > 1:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = recall = f1 = float('nan')
        
        cv_results.append({
            'video': held_out_video,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_test': len(X_test),
            'n_real': sum(y_test == 1),
            'tp': sum((y_pred == 1) & (y_test == 1)),
            'fp': sum((y_pred == 1) & (y_test == 0)),
            'fn': sum((y_pred == 0) & (y_test == 1)),
            'tn': sum((y_pred == 0) & (y_test == 0))
        })
        
        print(f"  P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    # Aggregate metrics (pooled predictions)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    pooled_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    pooled_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    pooled_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    
    print(f"\nPOOLED METRICS (aggregated predictions):")
    print(f"  Precision: {pooled_precision:.3f}")
    print(f"  Recall:    {pooled_recall:.3f}")
    print(f"  F1 Score:  {pooled_f1:.3f}")
    
    # Mean per-fold metrics
    valid_results = [r for r in cv_results if not np.isnan(r['f1'])]
    
    avg_precision = np.mean([r['precision'] for r in valid_results])
    avg_recall = np.mean([r['recall'] for r in valid_results])
    avg_f1 = np.mean([r['f1'] for r in valid_results])
    
    std_precision = np.std([r['precision'] for r in valid_results])
    std_recall = np.std([r['recall'] for r in valid_results])
    std_f1 = np.std([r['f1'] for r in valid_results])
    
    print(f"\nMEAN PER-FOLD METRICS (± std):")
    print(f"  Precision: {avg_precision:.3f} ± {std_precision:.3f}")
    print(f"  Recall:    {avg_recall:.3f} ± {std_recall:.3f}")
    print(f"  F1 Score:  {avg_f1:.3f} ± {std_f1:.3f}")
    
    # Confusion matrix totals
    total_tp = sum(r['tp'] for r in cv_results)
    total_fp = sum(r['fp'] for r in cv_results)
    total_fn = sum(r['fn'] for r in cv_results)
    total_tn = sum(r['tn'] for r in cv_results)
    
    print(f"\nAGGREGATE CONFUSION MATRIX:")
    print(f"  TP: {total_tp}  FP: {total_fp}")
    print(f"  FN: {total_fn}  TN: {total_tn}")
    
    print("\nPER-VIDEO BREAKDOWN:")
    print(f"{'Video':<40} {'P':<8} {'R':<8} {'F1':<8} {'TP':<5} {'FP':<5} {'FN'}")
    print("-" * 80)
    for r in cv_results:
        p = f"{r['precision']:.3f}" if not np.isnan(r['precision']) else "N/A"
        rec = f"{r['recall']:.3f}" if not np.isnan(r['recall']) else "N/A"
        f1 = f"{r['f1']:.3f}" if not np.isnan(r['f1']) else "N/A"
        print(f"{r['video']:<40} {p:<8} {rec:<8} {f1:<8} {r['tp']:<5} {r['fp']:<5} {r['fn']}")
    
    # F1 range for paper
    valid_f1s = [r['f1'] for r in valid_results if not np.isnan(r['f1'])]
    min_f1 = min(valid_f1s) if valid_f1s else 0
    max_f1 = max(valid_f1s) if valid_f1s else 0
    
    print("\n" + "="*70)
    print("FOR YOUR RESEARCH PAPER")
    print("="*70)
    print(f"""
METHODOLOGY TEXT (Section 2.6):
"To validate the ML classifier's generalization across different recording 
conditions, we performed leave-one-video-out cross-validation. For each fold, 
the classifier was trained on trajectory features from {len(valid_results)-1} videos and 
tested on the held-out video, repeated for all {len(valid_results)} videos."

RESULTS TEXT (New Section 3.5 - ML Classifier Validation):
"The Random Forest event classifier achieved {pooled_precision:.1%} precision and 
{pooled_recall:.1%} recall (F1 = {pooled_f1:.3f}) using leave-one-video-out 
cross-validation. Mean per-fold performance was {avg_f1:.3f} ± {std_f1:.3f} F1, 
with individual video F1 scores ranging from {min_f1:.3f} to {max_f1:.3f}."

LIMITATION TEXT (Section 4.1):
"The ML event classifier requires training data from annotated videos. 
Cross-validation showed performance variance across videos (F1 range: 
{min_f1:.3f}-{max_f1:.3f}), indicating that certain lighting conditions, 
activity levels, or bee behaviors may challenge the classifier. Additional 
annotation from diverse recording conditions would improve generalization."
""")
    
    # Save results
    results_df = pd.DataFrame(cv_results)
    output_path = '/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/output/ml_classifier_cv_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()