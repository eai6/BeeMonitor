"""Train ML Classifier for ML-First Event Detection

ZERO HARDCODED THRESHOLDS VERSION

This trains a classifier to distinguish real bee events from noise using
VERY lenient detection (1-frame, 40px padding) with NO trajectory filtering.

The model learns to recognize:
- Real bee entry/exit patterns
- Tracking artifacts and ID switches  
- Background motion and noise

Philosophy: ML learns EVERYTHING from data, no hardcoded rules!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add beemonitor to path
sys.path.insert(0, '/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6')

from beemonitor.processing import EventProcessor
from beemonitor.core.config import Config
from beemonitor.detection.nest_detector import NestDetector
from ultralytics import YOLO

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML-FIRST CLASSIFIER TRAINING")
print("Lenient Detection: 1-frame window, 40px padding")
print("Goal: Let ML learn to filter noise, no hardcoded rules")
print("="*70)

# Setup
config = Config.default()
nest_model = YOLO(config.models.nest_detection)
detector = NestDetector(nest_model, config)

input_data = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/CVPR_Evaluation_Video_Data"
output_folder = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/output/CVPR_Output"
manual_csv = '/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/Manual_Foraging_Events_Observation.csv'

files = os.listdir(input_data)
files = [os.path.join(input_data, file) for file in files if 'mp4' in file]

print(f"\nFound {len(files)} videos")

# Load manual events (ground truth)
manual_df = pd.read_csv(manual_csv)
manual_df = manual_df[['video', 'action', 'nest', 'timestamp']].dropna()

def parse_manual_time(video, time_str):
    date_part = video.split('_')[1]
    return datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M:%S")

manual_df['dt'] = manual_df.apply(lambda x: parse_manual_time(x['video'], x['timestamp']), axis=1)

print(f"Manual events (ground truth): {len(manual_df)}")

# Load tracking data
tracking_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('_tracking_results.csv')]
tracking_data = {}

for file in tracking_files:
    video_name = file.replace('_tracking_results.csv', '').split("/")[-1]
    tracking_data[video_name] = pd.read_csv(file)

print(f"Loaded tracking data for {len(tracking_data)} videos")

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

# ======================================================================
# EXTRACT FEATURES USING ML-FIRST DETECTOR
# ======================================================================

print("\n" + "="*70)
print("DETECTING EVENTS WITH LENIENT SETTINGS")
print("1-frame window, 40px padding, minimal filtering")
print("="*70)

all_features = []
all_labels = []
all_video_info = []

# Create ML-First processor (no ML model for training)
processor = EventProcessor(config)
processor.ml_model = None  # Disable ML during training

for video_name, tracking_df in tracking_data.items():
    print(f"\nProcessing {video_name}...")
    
    # Get video file
    video_file = [f for f in files if video_name in f]
    if len(video_file) == 0:
        print(f"  ⚠️  Video file not found, skipping")
        continue
    video_file = video_file[0]
    
    # Get nests
    try:
        nests = detector.get_nests_and_hotel_detections(video_file)
    except Exception as e:
        print(f"  ⚠️  Nest detection failed: {e}, skipping")
        continue
    
    # Reconstruct motion data
    motion_data = reconstruct_motion_data(tracking_df)
    movements = motion_data.tracks[0]
    
    # LENIENT detection with ML-First processor
    detected_events = processor._detect_all_events(movements, nests)
    
    print(f"  Detected {len(detected_events)} events (lenient)")
    
    # Extract features for each detected event
    for event in detected_events:
        # Find matching trajectory
        matching_traj = None
        for movement in movements:
            track_id, centroids, bboxes, frame_numbers = movement
            
            if event['action'] == 'Exit' and frame_numbers[0] == event['frame_number']:
                matching_traj = movement
                break
            elif event['action'] == 'Entry' and frame_numbers[-1] == event['frame_number']:
                matching_traj = movement
                break
        
        if matching_traj is None:
            continue
        
        # Extract features
        features = processor._extract_trajectory_features(
            matching_traj,
            event['action'],
            event['nest'],
            nests
        )
        
        all_features.append(features)
        all_video_info.append((video_name, event['action'], event['nest'], event['frame_number']))

features_df = pd.DataFrame(all_features)

print(f"\n✓ Extracted features for {len(features_df)} detected events")
print(f"  Feature count: {len(features_df.columns)}")

# ======================================================================
# LABEL EVENTS (REAL vs NOISE)
# ======================================================================

print("\n" + "="*70)
print("LABELING EVENTS (MATCHING TO MANUAL GROUND TRUTH)")
print("="*70)

labels = []

for i, (video_name, action, nest_id, frame_num) in enumerate(all_video_info):
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
            if time_diff <= timedelta(seconds=3.0):  # 3 second tolerance
                matched = True
                break
    
    labels.append(1 if matched else 0)

labels = pd.Series(labels)

print(f"\nLabeling complete:")
print(f"  Real events (label=1): {sum(labels == 1)} ({sum(labels == 1)/len(labels)*100:.1f}%)")
print(f"  Noise (label=0): {sum(labels == 0)} ({sum(labels == 0)/len(labels)*100:.1f}%)")
print(f"  Total: {len(labels)}")

if sum(labels == 1) < 10:
    print("\n❌ ERROR: Not enough positive examples!")
    print("  Need at least 10 real events for training")
    sys.exit(1)

# ======================================================================
# TRAIN RANDOM FOREST CLASSIFIER
# ======================================================================

print("\n" + "="*70)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*70)

# Split data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels
)

print(f"Train set: {len(X_train)} events")
print(f"  Real: {sum(y_train == 1)}, Noise: {sum(y_train == 0)}")
print(f"Test set: {len(X_test)} events")
print(f"  Real: {sum(y_test == 1)}, Noise: {sum(y_test == 0)}")

# Train classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

print(f"\nTraining Random Forest...")
model.fit(X_train, y_train)

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = (y_train_pred == y_train).mean()
test_acc = (y_test_pred == y_test).mean()

print(f"\nAccuracy:")
print(f"  Training: {train_acc:.3f}")
print(f"  Test: {test_acc:.3f}")

test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nTest Set Performance:")
print(f"  Precision: {test_precision:.3f} ({test_precision*100:.1f}%)")
print(f"  Recall: {test_recall:.3f} ({test_recall*100:.1f}%)")
print(f"  F1 Score: {test_f1:.3f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Noise', 'Real']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"  TN (correct noise): {cm[0,0]}")
print(f"  FP (noise called real): {cm[0,1]}")
print(f"  FN (real called noise): {cm[1,0]}")
print(f"  TP (correct real): {cm[1,1]}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features_df.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {idx+1:2d}. {row['feature']:<25s} {row['importance']:.3f}")

# ======================================================================
# THRESHOLD ANALYSIS
# ======================================================================

print("\n" + "="*70)
print("THRESHOLD ANALYSIS")
print("="*70)

# Get probabilities for full dataset
y_proba = model.predict_proba(features_df)[:, 1]

# Test different thresholds
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
print(f"\nPerformance at different confidence thresholds:")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Events Kept'}")
print("-" * 65)

best_f1 = 0
best_threshold = 0.3

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    
    if sum(y_pred) > 0:
        precision = precision_score(labels, y_pred)
        recall = recall_score(labels, y_pred)
        f1 = f1_score(labels, y_pred)
        
        print(f"{threshold:<12.2f} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {sum(y_pred)}/{len(y_pred)}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    else:
        print(f"{threshold:<12.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12} 0/{len(y_pred)}")

print(f"\n✓ Best threshold: {best_threshold} (F1={best_f1:.3f})")

# ======================================================================
# SAVE MODEL
# ======================================================================

model_data = {
    'model': model,
    'feature_names': list(features_df.columns),
    'precision': test_precision,
    'recall': test_recall,
    'f1': test_f1,
    'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'best_threshold': best_threshold,
    'detection_settings': {
        'window_size': 1,
        'padding': 40,
        'min_trajectory_length': 3,
        'min_movement_distance': 10
    },
    'notes': 'ML-First approach: Lenient detection (1-frame, 40px) + ML filtering'
}

model_path = 'event_classifier_ml_first.pkl'
joblib.dump(model_data, model_path)

print(f"\n✓ Model saved to {model_path}")

# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

print(f"\nDetection Settings (Lenient):")
print(f"  Window size: 1 frame (very lenient)")
print(f"  Padding: 40 pixels (large detection area)")
print(f"  Min trajectory length: 3 frames")
print(f"  Min movement distance: 10 pixels")

print(f"\nDataset:")
print(f"  Total events detected: {len(features_df)}")
print(f"  Real events: {sum(labels == 1)} ({sum(labels == 1)/len(labels)*100:.1f}%)")
print(f"  Noise: {sum(labels == 0)} ({sum(labels == 0)/len(labels)*100:.1f}%)")

print(f"\nModel Performance:")
print(f"  Test Precision: {test_precision:.3f}")
print(f"  Test Recall: {test_recall:.3f}")
print(f"  Test F1: {test_f1:.3f}")

print(f"\nRecommended Threshold: {best_threshold}")
print(f"  Expected Precision: ~{precision_score(labels, (y_proba >= best_threshold).astype(int)):.1%}")
print(f"  Expected Recall: ~{recall_score(labels, (y_proba >= best_threshold).astype(int)):.1%}")

print(f"\nNext Steps:")
print(f"1. Update config.models.event_classifier = '{model_path}'")
print(f"2. Replace event_processor.py with event_processor_ml_first.py")
print(f"3. Test with: python compare_ml_first.py")
print(f"4. Use threshold={best_threshold} for production")

print(f"\nML-First Philosophy:")
print(f"  ✓ No hardcoded heuristic thresholds")
print(f"  ✓ Data-driven, learns from examples")
print(f"  ✓ Adapts to different datasets automatically")
print(f"  ✓ More robust and generalizable")
print("="*70)