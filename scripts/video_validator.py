"""
Video validation and diagnostics for bee-monitor.

This module helps diagnose and fix common video issues.
"""

import cv2
from pathlib import Path


def validate_video(video_path: str) -> dict:
    """Validate a video file and return diagnostic information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "exists": False,
        "can_open": False,
        "has_frames": False,
        "frame_count": 0,
        "fps": 0,
        "width": 0,
        "height": 0,
        "codec": "",
        "can_read_frame": False,
        "errors": []
    }
    
    # Check if file exists
    video_file = Path(video_path)
    if not video_file.exists():
        results["errors"].append(f"File does not exist: {video_path}")
        return results
    
    results["exists"] = True
    
    # Try to open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        results["errors"].append("Cannot open video file")
        return results
    
    results["can_open"] = True
    
    # Get video properties
    try:
        results["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
        results["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        results["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        results["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        if results["frame_count"] > 0:
            results["has_frames"] = True
        else:
            results["errors"].append("Video reports 0 frames")
    
    except Exception as e:
        results["errors"].append(f"Error reading video properties: {e}")
    
    # Try to read first frame
    try:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            results["can_read_frame"] = True
        else:
            results["errors"].append("Cannot read first frame")
    except Exception as e:
        results["errors"].append(f"Error reading frame: {e}")
    
    cap.release()
    
    # Overall validation
    results["valid"] = (
        results["exists"] and 
        results["can_open"] and 
        results["has_frames"] and 
        results["can_read_frame"]
    )
    
    return results


def diagnose_video(video_path: str) -> None:
    """Print detailed diagnostics for a video file.
    
    Args:
        video_path: Path to video file
    """
    print(f"🔍 Diagnosing video: {video_path}\n")
    
    results = validate_video(video_path)
    
    print("File Check:")
    print(f"  {'✅' if results['exists'] else '❌'} File exists")
    
    print("\nVideo Properties:")
    print(f"  {'✅' if results['can_open'] else '❌'} Can open video")
    print(f"  {'✅' if results['has_frames'] else '❌'} Has frames: {results['frame_count']}")
    print(f"  Resolution: {results['width']}x{results['height']}")
    print(f"  FPS: {results['fps']}")
    print(f"  Codec: {results['codec']}")
    
    print("\nFrame Reading:")
    print(f"  {'✅' if results['can_read_frame'] else '❌'} Can read frames")
    
    if results["errors"]:
        print("\n⚠️  Errors Found:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    print(f"\n{'🎉 Video is valid!' if results['valid'] else '❌ Video has issues'}")
    
    if not results["valid"]:
        print("\n💡 Suggestions:")
        if not results["exists"]:
            print("  - Check the file path")
        elif not results["can_open"]:
            print("  - Video file may be corrupted")
            print("  - Try re-encoding: ffmpeg -i input.mp4 -c:v libx264 output.mp4")
        elif not results["has_frames"]:
            print("  - Video may be corrupted or incomplete")
        elif not results["can_read_frame"]:
            print("  - Codec may not be supported")
            print("  - Try re-encoding: ffmpeg -i input.mp4 -c:v libx264 output.mp4")


def fix_video_codec(input_path: str, output_path: str = None) -> str:
    """Re-encode video to a compatible format.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video (optional, auto-generated if None)
        
    Returns:
        Path to fixed video
        
    Note:
        Requires ffmpeg to be installed
    """
    import subprocess
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_fixed.mp4")
    
    print(f"🔧 Re-encoding video...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y",  # Overwrite output
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ Video re-encoded successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error re-encoding: {e}")
        print("   Make sure ffmpeg is installed: brew install ffmpeg")
        raise
    except FileNotFoundError:
        print("❌ ffmpeg not found")
        print("   Install with: brew install ffmpeg")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_validator.py <video_path>")
        sys.exit(1)
    
    diagnose_video(sys.argv[1])