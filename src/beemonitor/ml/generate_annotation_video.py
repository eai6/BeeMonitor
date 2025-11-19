"""
Generate Annotation Video with Enhanced Track IDs

This script creates a video with large, prominent track IDs
for easy manual annotation and training data generation.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_annotation_video(
    video_path: str,
    motion_data,
    nests: dict,
    config,
    output_folder: str = "annotation_videos",
    show_trajectories: bool = True
):
    """Generate video with enhanced track IDs for annotation.
    
    Args:
        video_path: Path to input video
        motion_data: DataFrame with tracking results
        nests: Dictionary with nest locations
        config: Config object
        output_folder: Where to save annotation video
        show_trajectories: Whether to show full trajectory paths
        
    Returns:
        Path to generated annotation video
    """
    from beemonitor.output.video_synthesizer import VideoSynthesizer
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("Generating Annotation Video")
    logger.info("=" * 60)
    
    # Create empty events DataFrame (or use actual events if available)
    events = pd.DataFrame(columns=['frame_number', 'nest', 'action'])
    
    # Create synthesizer in ANNOTATION MODE
    logger.info("Creating synthesizer in annotation mode...")
    synthesizer = VideoSynthesizer(config, annotation_mode=True)
    
    # Generate video
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Show trajectories: {show_trajectories}")
    
    output_path = synthesizer.synthesize(
        video_path=video_path,
        events=events,
        motion=motion_data,
        nest_data=nests,
        output_folder=output_folder,
        show_trajectories=show_trajectories
    )
    
    logger.info(f"\n✅ Annotation video created: {output_path}")
    logger.info("\nVideo features:")
    logger.info("  - Large, prominent track IDs with backgrounds")
    logger.info("  - Yellow highlighting for easy reading")
    if show_trajectories:
        logger.info("  - Full trajectory paths (cyan lines)")
        logger.info("  - Centroid points (yellow dots)")
    logger.info("  - Nest locations and IDs")
    logger.info("  - Hotel boundary")
    
    logger.info("\n💡 Use this video to:")
    logger.info("  1. Watch each track and note its ID")
    logger.info("  2. Determine if it's a bee or noise")
    logger.info("  3. Classify behavior (entry/exit/pass)")
    logger.info("  4. Fill in annotations.json")
    
    return output_path


def create_side_by_side_comparison(
    video_path: str,
    motion_data,
    events,
    nests: dict,
    config,
    output_folder: str = "comparison_videos"
):
    """Create both normal and annotation videos for comparison.
    
    Args:
        video_path: Path to input video
        motion_data: DataFrame with tracking results
        events: DataFrame with events
        nests: Dictionary with nest locations
        config: Config object
        output_folder: Where to save videos
        
    Returns:
        Tuple of (normal_path, annotation_path)
    """
    from beemonitor.output.video_synthesizer import VideoSynthesizer
    
    logger.info("=" * 60)
    logger.info("Creating Comparison Videos")
    logger.info("=" * 60)
    
    # Normal mode
    logger.info("\n1. Creating normal visualization...")
    synthesizer_normal = VideoSynthesizer(config, annotation_mode=False)
    normal_path = synthesizer_normal.synthesize(
        video_path=video_path,
        events=events,
        motion=motion_data,
        nest_data=nests,
        output_folder=f"{output_folder}/normal",
        show_trajectories=False
    )
    logger.info(f"✅ Normal video: {normal_path}")
    
    # Annotation mode
    logger.info("\n2. Creating annotation visualization...")
    synthesizer_annotation = VideoSynthesizer(config, annotation_mode=True)
    annotation_path = synthesizer_annotation.synthesize(
        video_path=video_path,
        events=events,
        motion=motion_data,
        nest_data=nests,
        output_folder=f"{output_folder}/annotation",
        show_trajectories=True
    )
    logger.info(f"✅ Annotation video: {annotation_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nNormal:     {normal_path}")
    logger.info(f"Annotation: {annotation_path}")
    
    return normal_path, annotation_path


def extract_frames_for_annotation(
    video_path: str,
    motion_data,
    nests: dict,
    config,
    output_folder: str = "annotation_frames",
    frame_interval: int = 30
):
    """Extract key frames with track IDs for static annotation.
    
    Useful if you prefer to annotate from still images rather than video.
    
    Args:
        video_path: Path to input video
        motion_data: DataFrame with tracking results
        nests: Dictionary with nest locations
        config: Config object
        output_folder: Where to save frames
        frame_interval: Extract every N frames
        
    Returns:
        List of saved frame paths
    """
    import cv2
    from beemonitor.output.video_synthesizer import VideoSynthesizer
    
    logger.info("=" * 60)
    logger.info("Extracting Annotation Frames")
    logger.info("=" * 60)
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    synthesizer = VideoSynthesizer(config, annotation_mode=True)
    
    cap = cv2.VideoCapture(video_path)
    saved_frames = []
    
    for period_idx in range(len(motion_data)):
        try:
            period = motion_data.iloc[period_idx]['frame_number']
            tracks_data = motion_data.iloc[period_idx]['tracks']
            
            # Convert to Track objects
            from beemonitor.output.video_synthesizer import Track
            track_objects = [
                Track(track[0], track[2], track[3])
                for track in tracks_data
            ]
            
            start_frame, end_frame = period
            
            # Extract frames at intervals
            for frame_num in range(start_frame, end_frame + 1, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Resize
                frame = cv2.resize(frame, (config.video.res_width, config.video.res_height))
                
                # Draw annotations
                frame = synthesizer._draw_nest_holes(frame, nests['nests'])
                frame = synthesizer._draw_tracks(frame, track_objects, frame_num)
                if 'hotel' in nests:
                    frame = synthesizer._draw_hotel_boundary(frame, nests['hotel'])
                
                # Save frame
                frame_path = output_path / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(str(frame_path))
                
                logger.info(f"Saved: {frame_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing period {period_idx}: {e}")
            continue
    
    cap.release()
    
    logger.info(f"\n✅ Extracted {len(saved_frames)} frames to: {output_folder}")
    
    return saved_frames


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

def complete_annotation_workflow_example():
    """
    Complete example: analyze video → generate annotation video → create template
    """
    print("\n" + "=" * 70)
    print(" ANNOTATION VIDEO GENERATION - Complete Workflow ")
    print("=" * 70)
    
    print("\n📹 Step 1: Analyze Your Video")
    print("─" * 70)
    print("""
from beemonitor.video_analyzer import analyze_video
from beemonitor.core.config import Config

config = Config.default()
motion_data, nests = analyze_video('your_video.mp4', config)
    """)
    
    print("\n📹 Step 2: Generate Annotation Video")
    print("─" * 70)
    print("""
from generate_annotation_video import generate_annotation_video

annotation_video_path = generate_annotation_video(
    video_path='your_video.mp4',
    motion_data=motion_data,
    nests=nests,
    config=config,
    output_folder='annotation_videos',
    show_trajectories=True  # Shows full paths
)

print(f"Watch this video: {annotation_video_path}")
    """)
    
    print("\n📝 Step 3: Create Annotation Template")
    print("─" * 70)
    print("""
from beemonitor.processing.training_data_helper import create_annotation_template

create_annotation_template(motion_data, 'annotations.json')
    """)
    
    print("\n✍️ Step 4: Watch Video and Label Tracks")
    print("─" * 70)
    print("""
# Watch the annotation video
# For each track ID you see:
#   1. Note the track ID number
#   2. Decide: Is it a bee? (true/false)
#   3. Decide: What behavior? ('entry', 'exit', 'pass', or null)
#   4. Update annotations.json

Example annotation:
{
  "42": {
    "is_bee": true,
    "event_type": "entry",
    "notes": "Clear entry into nest 5, frame ~1240"
  },
  "43": {
    "is_bee": false,
    "event_type": null,
    "notes": "Shadow from tree movement"
  }
}
    """)
    
    print("\n🤖 Step 5: Train ML Classifier")
    print("─" * 70)
    print("""
# See train_classifier_example.py for complete training workflow
    """)
    
    print("\n" + "=" * 70)
    print(" All set! Follow the steps above to create training data ")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # Show the workflow
    complete_annotation_workflow_example()
    
    # Uncomment to use:
    
    # # After analyzing your video:
    # from beemonitor.core.config import Config
    # config = Config.default()
    # 
    # # Generate annotation video
    # annotation_video = generate_annotation_video(
    #     video_path='your_video.mp4',
    #     motion_data=motion_data,
    #     nests=nests,
    #     config=config,
    #     show_trajectories=True
    # )
    # 
    # # Or create comparison
    # normal_vid, annot_vid = create_side_by_side_comparison(
    #     video_path='your_video.mp4',
    #     motion_data=motion_data,
    #     events=events,
    #     nests=nests,
    #     config=config
    # )