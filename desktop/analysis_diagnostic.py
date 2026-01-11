#!/usr/bin/env python3
"""
Analysis Diagnostic - Why is tracking_results.csv empty?
=========================================================

This script helps diagnose why full analysis produces no results.
"""

import sys
import os
from pathlib import Path

def check_video_file(video_path):
    """Check if video file exists and is valid."""
    print("\n" + "="*60)
    print("STEP 1: Video File Check")
    print("="*60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"✓ Video file exists: {video_path}")
    print(f"  Size: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB")
    
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Video opens successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    cap.release()
    return True


def test_detection(video_path, params):
    """Test if detection finds anything."""
    print("\n" + "="*60)
    print("STEP 2: Detection Test")
    print("="*60)
    print(f"Parameters: min_area={params['min_area']}, min_solidity={params['min_solidity']}")
    
    try:
        from beemonitor.detection import BlobDetector
        import cv2
        
        # Initialize detector
        blob = BlobDetector(
            min_area=params['min_area'],
            min_solidity=params['min_solidity']
        )
        
        # Initialize background
        print("\nInitializing background...")
        blob.initialize_from_video(video_path, num_frames=100, start_frame=0)
        print("✓ Background initialized")
        
        # Test on multiple frames
        cap = cv2.VideoCapture(video_path)
        test_frames = [200, 500, 1000, 2000, 5000]
        
        detection_counts = []
        
        for frame_idx in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            detections = blob.detect(frame)
            detection_counts.append(len(detections))
            
            print(f"Frame {frame_idx}: {len(detections)} detections")
            
            if len(detections) > 0:
                # Show some details
                for i, det in enumerate(detections[:3]):
                    x1, y1, x2, y2 = det.bbox
                    area = (x2-x1) * (y2-y1)
                    print(f"  Detection {i+1}: area={area:.0f}, bbox=({x1},{y1},{x2},{y2})")
        
        cap.release()
        
        total_detections = sum(detection_counts)
        avg_detections = total_detections / len(detection_counts) if detection_counts else 0
        
        print(f"\nSummary:")
        print(f"  Total detections across test frames: {total_detections}")
        print(f"  Average per frame: {avg_detections:.1f}")
        
        if total_detections == 0:
            print("\n❌ NO DETECTIONS FOUND!")
            print("\nPossible issues:")
            print("  1. Detection parameters too strict")
            print("  2. Background not properly initialized")
            print("  3. No moving objects in video")
            print("\nTry:")
            print("  • Lower min_area (try 50 instead of 120)")
            print("  • Lower min_solidity (try 0.5 instead of 0.7)")
            print("  • Use 'sensitive' preset in GUI")
            return False
        else:
            print(f"\n✓ Detection is working ({avg_detections:.1f} detections/frame average)")
            return True
            
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis(video_path, params, output_folder):
    """Test full analysis."""
    print("\n" + "="*60)
    print("STEP 3: Full Analysis Test")
    print("="*60)
    
    try:
        from beemonitor import BeeMonitor
        from beemonitor.core.config import Config
        
        # Create config
        config = Config.default()
        config.detection.min_area = params['min_area']
        config.detection.min_solidity = params['min_solidity']
        config.detection.max_area = params.get('max_area', 4000)
        
        config.detection.sync_to_tracking(config.tracking)
        
        print(f"Config:")
        print(f"  min_area: {config.detection.min_area}")
        print(f"  min_solidity: {config.detection.min_solidity}")
        print(f"  max_area: {config.detection.max_area}")
        
        # Create monitor
        monitor = BeeMonitor(config=config)
        
        print(f"\nRunning analysis...")
        print(f"  Video: {video_path}")
        print(f"  Output: {output_folder}")
        
        # Check what parameters analyze_video accepts
        import inspect
        sig = inspect.signature(monitor.analyze_video)
        
        kwargs = {'video_path': video_path}
        if 'output_folder' in sig.parameters:
            kwargs['output_folder'] = output_folder
        if 'visualize' in sig.parameters:
            kwargs['visualize'] = False  # Faster without visualization
        
        result = monitor.analyze_video(**kwargs)
        
        print(f"✓ Analysis completed")
        
        # Check result
        if result is None:
            print("❌ Analysis returned None")
            return False
        
        # Check if result has data
        if hasattr(result, '__len__'):
            print(f"  Result length: {len(result)}")
        
        if hasattr(result, 'shape'):
            print(f"  Result shape: {result.shape}")
        
        # Try to save
        csv_path = os.path.join(output_folder, 'tracking_results.csv')
        
        try:
            if hasattr(result, 'to_csv'):
                result.to_csv(csv_path, index=False)
            elif hasattr(result, 'tracks'):
                import pandas as pd
                tracks = result.tracks
                if isinstance(tracks, list):
                    pd.DataFrame(tracks).to_csv(csv_path, index=False)
                elif hasattr(tracks, 'to_csv'):
                    tracks.to_csv(csv_path, index=False)
            else:
                import pandas as pd
                pd.DataFrame(result).to_csv(csv_path, index=False)
            
            # Check if file has content
            if os.path.exists(csv_path):
                size = os.path.getsize(csv_path)
                if size == 0:
                    print(f"❌ CSV file created but is empty!")
                    return False
                else:
                    print(f"✓ CSV file created: {csv_path}")
                    print(f"  Size: {size} bytes")
                    
                    # Read and show info
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    print(f"  Rows: {len(df)}")
                    print(f"  Columns: {list(df.columns)}")
                    
                    if 'track_id' in df.columns:
                        print(f"  Unique tracks: {df['track_id'].nunique()}")
                    
                    return True
            else:
                print(f"❌ CSV file not created")
                return False
                
        except Exception as e:
            print(f"❌ Failed to save CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main diagnostic routine."""
    print("="*60)
    print("BeeMonitor Analysis Diagnostic")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python analysis_diagnostic.py <video_path>")
        print("\nExample:")
        print("  python analysis_diagnostic.py /path/to/bee_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Default parameters (conservative preset)
    params = {
        'min_area': 120,
        'min_solidity': 0.7,
        'max_area': 4000
    }
    
    print(f"\nVideo: {video_path}")
    print(f"Detection params: min_area={params['min_area']}, min_solidity={params['min_solidity']}")
    
    # Output folder
    video_dir = Path(video_path).parent
    video_name = Path(video_path).stem
    output_folder = str(video_dir / f"{video_name}_diagnostic")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Step 1: Check video
    if not check_video_file(video_path):
        print("\n❌ Video check failed - cannot proceed")
        sys.exit(1)
    
    # Step 2: Test detection
    if not test_detection(video_path, params):
        print("\n❌ Detection test failed")
        print("\nRECOMMENDATION:")
        print("  Try with more sensitive parameters:")
        print(f"  python {sys.argv[0]} {video_path}")
        sys.exit(1)
    
    # Step 3: Test analysis
    if not test_analysis(video_path, params, output_folder):
        print("\n❌ Analysis test failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED!")
    print("="*60)
    print("\nYour analysis pipeline is working correctly.")
    print(f"Check the output folder: {output_folder}")
    print("\nIf GUI analysis still produces empty files:")
    print("  1. Check GUI is using same parameters")
    print("  2. Check GUI output folder location")
    print("  3. Look for error messages in GUI")


if __name__ == '__main__':
    main()
