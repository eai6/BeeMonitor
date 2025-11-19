"""
Diagnostic script to troubleshoot visualization issues.
Run this to see why bounding boxes aren't appearing in visualizations.
"""

import json
from pathlib import Path
import cv2

def diagnose_visualization_issue(output_folder: str):
    """
    Check why visualizations might not be showing bounding boxes.
    """
    output_path = Path(output_folder)
    annotations_path = output_path / 'annotations' / 'annotations.json'
    
    print("="*60)
    print("VISUALIZATION DIAGNOSTICS")
    print("="*60)
    
    # Check 1: Does annotations file exist?
    print("\n1. Checking annotations file...")
    if not annotations_path.exists():
        print(f"   ❌ ERROR: Annotations file not found at {annotations_path}")
        print(f"   → Make sure you ran generate_dataset() first!")
        return
    print(f"   ✓ Annotations file found")
    
    # Check 2: Load annotations
    print("\n2. Loading annotations...")
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        print(f"   ✓ Loaded {len(annotations)} annotations")
    except Exception as e:
        print(f"   ❌ ERROR loading annotations: {e}")
        return
    
    if len(annotations) == 0:
        print(f"   ❌ ERROR: No annotations found in file!")
        return
    
    # Check 3: Examine annotation structure
    print("\n3. Examining annotation structure...")
    sample = annotations[0]
    print(f"   Sample annotation keys: {list(sample.keys())}")
    print(f"   Category: {sample.get('category', 'N/A')}")
    print(f"   Frame path: {sample.get('frame_path', 'N/A')}")
    print(f"   Number of bboxes: {len(sample.get('bboxes', []))}")
    
    # Check 4: Count annotations by bbox count
    print("\n4. Analyzing bboxes across all annotations...")
    empty_count = 0
    with_boxes_count = 0
    total_boxes = 0
    
    for ann in annotations:
        num_boxes = len(ann.get('bboxes', []))
        total_boxes += num_boxes
        if num_boxes == 0:
            empty_count += 1
        else:
            with_boxes_count += 1
    
    print(f"   Total annotations: {len(annotations)}")
    print(f"   Annotations WITH bboxes: {with_boxes_count}")
    print(f"   Annotations WITHOUT bboxes: {empty_count}")
    print(f"   Total bboxes across all annotations: {total_boxes}")
    
    if with_boxes_count == 0:
        print(f"\n   ❌ PROBLEM FOUND: No annotations have bounding boxes!")
        print(f"   This could mean:")
        print(f"   - All frames were classified as 'empty'")
        print(f"   - Detection confidence threshold too high")
        print(f"   - Motion validation rejecting all detections")
        print(f"\n   → Check your detection settings and thresholds")
        return
    
    if empty_count == len(annotations):
        print(f"\n   ❌ PROBLEM FOUND: ALL annotations are empty!")
        print(f"   → This is likely the issue - no detections were saved")
        return
    
    # Check 5: Examine bbox structure
    print("\n5. Examining bbox structure...")
    ann_with_boxes = None
    for ann in annotations:
        if len(ann.get('bboxes', [])) > 0:
            ann_with_boxes = ann
            break
    
    if ann_with_boxes:
        sample_bbox = ann_with_boxes['bboxes'][0]
        print(f"   Sample bbox keys: {list(sample_bbox.keys())}")
        print(f"   Sample bbox: {sample_bbox}")
        
        # Check if coordinates are reasonable
        x = sample_bbox.get('x', 0)
        y = sample_bbox.get('y', 0)
        w = sample_bbox.get('width', 0)
        h = sample_bbox.get('height', 0)
        
        print(f"\n   Bbox coordinates: x={x}, y={y}, width={w}, height={h}")
        
        if w == 0 or h == 0:
            print(f"   ⚠️  WARNING: Bbox has zero width or height!")
        if x < 0 or y < 0:
            print(f"   ⚠️  WARNING: Bbox has negative coordinates!")
        
    # Check 6: Verify image files exist
    print("\n6. Checking if frame images exist...")
    missing_frames = 0
    checked_frames = 0
    
    for ann in annotations[:10]:  # Check first 10
        frame_path = Path(ann['frame_path'])
        checked_frames += 1
        if not frame_path.exists():
            missing_frames += 1
            print(f"   ❌ Missing: {frame_path}")
    
    if missing_frames > 0:
        print(f"   ⚠️  WARNING: {missing_frames}/{checked_frames} frame files missing!")
    else:
        print(f"   ✓ All checked frames exist")
    
    # Check 7: Test visualization on one frame
    print("\n7. Testing visualization on a sample frame...")
    
    # Find an annotation with bboxes
    test_ann = None
    for ann in annotations:
        if len(ann.get('bboxes', [])) > 0:
            frame_path = Path(ann['frame_path'])
            if frame_path.exists():
                test_ann = ann
                break
    
    if test_ann is None:
        print(f"   ❌ No suitable annotation found for testing")
        if with_boxes_count > 0:
            print(f"   → Annotations have boxes but frames are missing")
        return
    
    # Try to draw on the frame
    frame_path = Path(test_ann['frame_path'])
    print(f"   Testing with: {frame_path.name}")
    print(f"   Number of bboxes: {len(test_ann['bboxes'])}")
    
    try:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"   ❌ Could not read frame")
            return
        
        print(f"   ✓ Frame loaded: {frame.shape}")
        
        # Try to draw boxes
        for i, bbox in enumerate(test_ann['bboxes']):
            x = int(bbox.get('x', 0))
            y = int(bbox.get('y', 0))
            w = int(bbox.get('width', 0))
            h = int(bbox.get('height', 0))
            
            print(f"   Box {i+1}: x={x}, y={y}, w={w}, h={h}")
            
            # Check if box is within frame
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                print(f"   ⚠️  WARNING: Box {i+1} is outside frame bounds!")
                print(f"      Frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            # Try drawing
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Save test image
        test_output = output_path / 'test_visualization.jpg'
        cv2.imwrite(str(test_output), frame)
        print(f"\n   ✓ Test visualization saved to: {test_output}")
        print(f"   → Check this image to see if boxes appear")
        
    except Exception as e:
        print(f"   ❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if with_boxes_count == 0:
        print("❌ MAIN ISSUE: No annotations contain bounding boxes")
        print("\nPossible causes:")
        print("1. Detection confidence threshold too high")
        print("2. Motion validation rejecting all detections")
        print("3. No objects detected in the videos")
        print("4. YOLO model not detecting anything")
        print("\nSuggestions:")
        print("- Lower min_detection_confidence (try 0.1)")
        print("- Check your YOLO model is working")
        print("- Verify videos contain visible bees/insects")
    elif empty_count > with_boxes_count * 0.5:
        print(f"⚠️  ISSUE: Many annotations are empty ({empty_count}/{len(annotations)})")
        print("\nThis is expected if include_empty_frames=True")
        print("But you should also have annotations with boxes")
    else:
        print("✓ Annotations look good!")
        print(f"  - {with_boxes_count} annotations have bounding boxes")
        print(f"  - {total_boxes} total boxes across all annotations")
        print("\nIf visualizations still don't show boxes:")
        print("1. Check the test_visualization.jpg file")
        print("2. Verify you're looking at the right output folder")
        print("3. Make sure visualization was called AFTER generate_dataset")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_folder = sys.argv[1]
    else:
        output_folder = "/Users/edwardamoah/Documents/GitHub/BeeMonitor/videos/mendels_2025/training_data/training_data"  # Default
        print(f"Using default output folder: {output_folder}")
        print(f"Usage: python diagnose_visualization.py <output_folder>\n")
    
    diagnose_visualization_issue(output_folder)