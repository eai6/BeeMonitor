#!/usr/bin/env python3
"""
Filter Diagnostic Script
========================

Run this BEFORE your analysis to verify filters are set up correctly.
This will tell you exactly what's wrong if filters aren't working.
"""

import sys
from pathlib import Path

print("="*70)
print("FILTER DIAGNOSTIC - Checking Your Setup")
print("="*70)

# ============================================================================
# Check 1: Can we import BeeNoiseFilter?
# ============================================================================
print("\n1. Checking CNN filter class...")
try:
    from beemonitor.detection.noise_filter import BeeNoiseFilter
    print("   ✓ BeeNoiseFilter class found")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    print("   → Install beemonitor package properly")
    sys.exit(1)

# ============================================================================
# Check 2: Does CNN model file exist?
# ============================================================================
print("\n2. Checking CNN model file...")

# Try to load config
try:
    from beemonitor.core.config import Config
    config = Config()
    print("   ✓ Config loaded")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    config = None

# Check model path
if config and hasattr(config.models, 'blob_noise_classifier'):
    model_path = config.models.blob_noise_classifier
    print(f"   Config path: {model_path}")
    
    if Path(model_path).exists():
        print(f"   ✓ CNN model file exists: {model_path}")
        
        # Try to load it
        print("\n3. Testing CNN model loading...")
        try:
            cnn = BeeNoiseFilter(model_path=model_path, noise_threshold=0.9)
            print("   ✓ CNN model loaded successfully")
            print(f"   Threshold: {cnn.noise_threshold}")
            print(f"   Device: {cnn.device}")
        except Exception as e:
            print(f"   ✗ CNN loading FAILED: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print(f"   ✗ CNN model file NOT FOUND: {model_path}")
        print("   → Download or train the CNN model")
else:
    print("   ✗ Model path NOT CONFIGURED")
    print("   → Add to config: models.blob_noise_classifier = 'path/to/model.pth'")

# ============================================================================
# Check 3: Does updated video_analyzer have the filter code?
# ============================================================================
print("\n4. Checking video_analyzer.py has filter code...")
try:
    import beemonitor.core.video_analyzer as va
    import inspect
    
    # Check if get_motion_tracking has detection_mode parameter
    sig = inspect.signature(va.BeeMonitor.get_motion_tracking)
    params = list(sig.parameters.keys())
    
    if 'detection_mode' in params:
        print("   ✓ video_analyzer.py has detection_mode parameter")
    else:
        print("   ✗ video_analyzer.py is OLD VERSION")
        print("   → Install updated video_analyzer.py")
        print(f"   Parameters found: {params}")
    
    # Check if source code mentions CNN filter
    source = inspect.getsource(va.BeeMonitor.get_motion_tracking)
    if 'BeeNoiseFilter' in source:
        print("   ✓ video_analyzer.py has CNN filter code")
    else:
        print("   ✗ video_analyzer.py is missing CNN filter code")
        print("   → Install updated video_analyzer.py")
        
except Exception as e:
    print(f"   ✗ Could not check: {e}")

# ============================================================================
# Check 4: Detection mode mapping
# ============================================================================
print("\n5. Checking DetectionMode enum...")
try:
    from beemonitor.tracking.bee_tracking import DetectionMode
    
    modes = [m.name for m in DetectionMode]
    print(f"   Available modes: {modes}")
    
    if 'FGBG_ONLY' in modes:
        print("   ✓ FGBG_ONLY mode exists")
    else:
        print("   ✗ FGBG_ONLY mode missing")
        
except Exception as e:
    print(f"   ✗ Could not check: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

print("\nTo enable filters, you need:")
print("  1. ✓ BeeNoiseFilter class (in beemonitor.detection.noise_filter)")
print("  2. ✓ CNN model file (models/blob_noise_classifier.pth)")
print("  3. ✓ Updated video_analyzer.py (with detection_mode parameter)")
print("  4. ✓ Config pointing to CNN model")
print("  5. ✓ Run with detection_mode='fgbg' or 'fgbg_yolo'")

print("\nIf any checks failed above, fix those first!")
print("="*70)