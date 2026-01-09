"""QUICK REFERENCE GUIDE - BeeMonitor Architecture

═══════════════════════════════════════════════════════════════════════════
DETECTION MODULE
═══════════════════════════════════════════════════════════════════════════

Import:
    from beemonitor.detection import BlobDetector, SIFTDetector, YOLODetector

Create:
    blob = BlobDetector(min_area=50)
    sift = SIFTDetector(min_keypoints=3)
    yolo = YOLODetector(model, conf_threshold=0.25)

Use:
    detections = blob.detect(frame)
    # Each detection has: bbox, centroid, confidence, label, source

═══════════════════════════════════════════════════════════════════════════
TRACKING MODULE
═══════════════════════════════════════════════════════════════════════════

Import:
    from beemonitor.tracking import BeeTracking, DetectionMode
    from beemonitor.tracking.mot import BeeTracker

Create MOT:
    mot = BeeTracker(config, tracking_classes=['bee'])

Create Tracker:
    tracker = BeeTracking(
        mot_algorithm=mot,
        yolo_model=yolo_model,
        detection_mode=DetectionMode.FGBG_SIFT_YOLO,
        use_noise_filter=True,
        config=config
    )

Use:
    results = tracker.process_video('video.mp4', roi=(100, 100, 800, 600))

═══════════════════════════════════════════════════════════════════════════
DETECTION MODES (Choose one)
═══════════════════════════════════════════════════════════════════════════

FGBG_ONLY          → Fast, motion only
SIFT_ONLY          → Stationary bees only
FGBG_SIFT          → Moving + stationary
FGBG_YOLO          → Motion + YOLO (RECOMMENDED)
SIFT_YOLO          → Stationary + YOLO
FGBG_SIFT_YOLO     → All three (comprehensive)
YOLO_ONLY          → Maximum accuracy (slow)

═══════════════════════════════════════════════════════════════════════════
MOT ALGORITHMS (Choose one)
═══════════════════════════════════════════════════════════════════════════

BeeTracker           → Custom Kalman + Hungarian
UltralyticsTracker   → ByteTrack / BoT-SORT

═══════════════════════════════════════════════════════════════════════════
TYPICAL WORKFLOW
═══════════════════════════════════════════════════════════════════════════

1. Choose detection mode based on needs:
   - Speed? → FGBG_ONLY or FGBG_YOLO
   - Stationary bees? → FGBG_SIFT or FGBG_SIFT_YOLO
   - Max accuracy? → FGBG_SIFT_YOLO or YOLO_ONLY

2. Choose MOT algorithm:
   - Custom control? → BeeTracker
   - State-of-art? → UltralyticsTracker

3. Create tracking system:
   tracker = BeeTracking(mot, yolo_model, mode, config)

4. Process video:
   results = tracker.process_video(video_path, roi)

5. Results is DataFrame with:
   - frame, track_id, x1, y1, x2, y2, species, confidence

═══════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════

Detection:
    tracker.configure_detection(
        blob_min_area=100,
        sift_min_keypoints=5,
        yolo_conf=0.5
    )

Tracking:
    tracker.configure_tracking(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )

═══════════════════════════════════════════════════════════════════════════
WHEN TO USE WHICH DETECTOR
═══════════════════════════════════════════════════════════════════════════

BlobDetector:
    ✓ Fast processing needed
    ✓ Bees are moving
    ✗ Won't detect stationary bees

SIFTDetector:
    ✓ Need to detect stationary bees
    ✓ Bees have texture/features
    ✗ Slower than blob detection
    ✗ May struggle with fast-moving/blurry bees

YOLODetector:
    ✓ Need species classification
    ✓ Need high accuracy
    ✓ Can detect any bee (moving or stationary)
    ✗ Slowest method

Recommended Combinations:
    - Speed: FGBG_ONLY
    - Balanced: FGBG_YOLO (default)
    - Comprehensive: FGBG_SIFT_YOLO
    - Maximum: YOLO_ONLY

═══════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════

Missing stationary bees?
    → Use FGBG_SIFT or SIFT_ONLY mode

Too many false positives?
    → Enable noise_filter=True
    → Increase confidence thresholds

Too slow?
    → Use FGBG_ONLY or FGBG_YOLO
    → Reduce YOLO frequency in FGBG_YOLO

Tracks jumping between bees?
    → Decrease iou_threshold
    → Increase min_hits
    → Try different MOT algorithm

Not detecting small bees?
    → Decrease blob_min_area
    → Decrease yolo_conf
    → Use higher resolution

═══════════════════════════════════════════════════════════════════════════
"""
