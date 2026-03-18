
# """Video synthesis module with resolution-adaptive visualization.

# This module handles the creation of annotated videos showing tracking results,
# nest detections, and entry/exit events. All visual elements (text sizes, line
# thicknesses, padding) automatically scale with video resolution.
# """

# import cv2
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# import pandas as pd
# from pathlib import Path


# class VideoSynthesizer:
#     """Handles synthesis of annotated tracking videos with resolution scaling.
    
#     This class creates visualization videos that show:
#     - Detected nest holes with IDs
#     - Tracked bee trajectories with IDs
#     - Entry/exit events with annotations
    
#     All visual elements automatically scale based on video resolution.
    
#     Attributes:
#         config: Configuration object with resolution and visualization settings
#         res_width: Target video width
#         res_height: Target video height
#         font_scale_base: Base font scale factor
#         thickness_base: Base line thickness
    
#     Example:
#         >>> from config import Config
#         >>> config = Config.default()
#         >>> synthesizer = VideoSynthesizer(config)
#         >>> video_path = synthesizer.synthesize(
#         ...     video_path="input.mp4",
#         ...     events=events_df,
#         ...     motion=motion_df,
#         ...     nest_data=nest_dict,
#         ...     output_folder="/output"
#         ... )
#     """
    
#     # Base values at reference resolution (1280x720)
#     REFERENCE_WIDTH = 1280
#     REFERENCE_HEIGHT = 720
    
#     # Base visualization parameters
#     FONT_SCALE_BASE = 0.75
#     THICKNESS_BASE = 2
#     TEXT_OFFSET_X_BASE = 50
#     TEXT_OFFSET_Y_BASE = 75
#     FRAMES_TO_HOLD_BASE = 30  # Resolution-independent (time-based)
    
#     # Colors (BGR format)
#     COLOR_NEST = (255, 0, 0)  # Blue for nest boxes
#     COLOR_TRACK = (0, 255, 0)  # Green for tracks
#     COLOR_EXIT = (255, 0, 0)  # Blue for exit events
#     COLOR_ENTRY = (0, 255, 0)  # Green for entry events
#     COLOR_HOTEL = (0, 0, 0)  # Black for hotel boundary
#     COLOR_BLOB = (255, 255, 0)  # Cyan for FG blobs
#     COLOR_YOLO = (0, 165, 255)  # Orange for YOLO detections
    
#     # REPLACE WITH:
#     def __init__(self, config):
#         """Initialize video synthesizer with configuration.
        
#         Args:
#             config: Config object containing video and output settings
#             annotation_mode: If True, enhances track IDs for annotation purposes
#         """
#         self.config = config
#         self.res_width = config.video.res_width
#         self.res_height = config.video.res_height
#         self.annotation_mode = config.video.annotation_mode
#         self.debug_mode = getattr(config.video, 'debug_mode', False)  # Debug visualization flag
        
#         # Calculate scale factors
#         self.scale_x = self.res_width / self.REFERENCE_WIDTH
#         self.scale_y = self.res_height / self.REFERENCE_HEIGHT
#         self.scale_avg = (self.scale_x + self.scale_y) / 2
        
#         # Scaled visualization parameters
#         base_font = self.FONT_SCALE_BASE * (1.5 if self.annotation_mode else 1.0)
#         self.font_scale = base_font * self.scale_avg
#         self.thickness = max(1, int(self.THICKNESS_BASE * self.scale_avg))
#         self.text_offset_x = int(self.TEXT_OFFSET_X_BASE * self.scale_x)
#         self.text_offset_y = int(self.TEXT_OFFSET_Y_BASE * self.scale_y)
        
#         # Get scaled nest parameters
#         nest_params = config.get_nest_params()
#         self.nest_padding_x = nest_params['padding_x']
#         self.nest_padding_y = nest_params['padding_y']
    
#     def _create_video_writer(self, output_path: str) -> cv2.VideoWriter:
#         """Create a video writer object.
        
#         Args:
#             output_path: Path for output video file
            
#         Returns:
#             OpenCV VideoWriter object
#         """
#         fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
#         fps = self.config.video.fps
#         size = (self.res_width, self.res_height)
#         return cv2.VideoWriter(output_path, fourcc, fps, size)
    
#     def _draw_nest_holes(self, frame: np.ndarray, nests: Dict[str, Tuple]) -> np.ndarray:
#         """Draw nest hole bounding boxes and IDs on frame.
        
#         Args:
#             frame: Input frame
#             nests: Dictionary mapping nest IDs to (x1, y1, x2, y2) coordinates
            
#         Returns:
#             Frame with nest annotations
#         """
#         for nest_id, coords in nests.items():
#             # Extract ID number
#             id_num = nest_id.split('_')[-1] if '_' in nest_id else nest_id
            
#             # Get coordinates and apply padding
#             x1, y1, x2, y2 = coords
#             x1 = int(x1 - self.nest_padding_x)
#             y1 = int(y1 - self.nest_padding_y)
#             x2 = int(x2 + self.nest_padding_x)
#             y2 = int(y2 + self.nest_padding_y)
            
#             # Draw rectangle
#             cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_NEST, self.thickness)
            
#             # Draw ID text
#             cv2.putText(
#                 frame, str(id_num), (x1, y1),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 self.font_scale,
#                 self.COLOR_NEST,
#                 self.thickness
#             )
        
#         return frame
    
#     def _draw_hotel_boundary(self, frame: np.ndarray, hotel_roi: Tuple) -> np.ndarray:
#         """Draw hotel region of interest boundary.
        
#         Args:
#             frame: Input frame
#             hotel_roi: Hotel bounding box (x1, y1, x2, y2)
            
#         Returns:
#             Frame with hotel boundary
#         """
#         x1, y1, x2, y2 = [int(coord) for coord in hotel_roi]
#         cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_HOTEL, self.thickness)
#         return frame
    
#     def _draw_debug_data(
#         self, 
#         frame: np.ndarray, 
#         debug_blobs: List[Dict], 
#         debug_yolo: List[Dict]
#     ) -> np.ndarray:
#         """Draw debug visualization: FG blobs and YOLO detections.
        
#         Args:
#             frame: Input frame
#             debug_blobs: List of FG blob dictionaries with 'bbox' and 'centroid'
#             debug_yolo: List of YOLO detection dictionaries with 'bbox', 'centroid', 'species'
            
#         Returns:
#             Frame with debug annotations
#         """
#         # Draw FG blobs (cyan)
#         for blob in debug_blobs:
#             x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_BLOB, 1)
#             # Draw centroid
#             cx, cy = [int(c) for c in blob['centroid']]
#             cv2.circle(frame, (cx, cy), 3, self.COLOR_BLOB, -1)
        
#         # Draw YOLO detections (orange)
#         for det in debug_yolo:
#             x1, y1, x2, y2 = [int(c) for c in det['bbox']]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_YOLO, 2)
#             # Draw label
#             label = det.get('species', 'unknown')
#             cv2.putText(
#                 frame, f"DL:{label}", (x1, y1 - 5),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 self.font_scale * 0.7,
#                 self.COLOR_YOLO,
#                 self.thickness
#             )
        
#         return frame
    
#     # REPLACE WITH:
#     def _draw_tracks(self, frame: np.ndarray, tracks: List, frame_num: int) -> np.ndarray:
#         """Draw active tracks on frame.
        
#         Args:
#             frame: Input frame
#             tracks: List of Track objects
#             frame_num: Current frame number
            
#         Returns:
#             Frame with track annotations
#         """
#         for track in tracks:
#             if track.is_in_frame(frame_num):
#                 bbox = track.get_bbox(frame_num)
#                 x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
#                 # Draw track bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_TRACK, self.thickness)
                
#                 # Draw track ID with enhanced visibility for annotation mode
#                 track_id = track.getID()
#                 text = f"ID:{track_id}"
                
#                 if self.annotation_mode:
#                     # Get text size for background box
#                     (text_width, text_height), baseline = cv2.getTextSize(
#                         text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
#                     )
                    
#                     # Position text above bounding box
#                     text_x = x1
#                     text_y = y1 - 10
                    
#                     # Draw background rectangle for better readability
#                     padding = 5
#                     bg_x1 = text_x - padding
#                     bg_y1 = text_y - text_height - padding
#                     bg_x2 = text_x + text_width + padding
#                     bg_y2 = text_y + baseline + padding
                    
#                     # Draw semi-transparent background
#                     overlay = frame.copy()
#                     cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
#                     cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
#                     # Draw text on top
#                     cv2.putText(
#                         frame, text, (text_x, text_y),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         self.font_scale,
#                         (0, 255, 255),  # Yellow for high visibility
#                         self.thickness + 1
#                     )
#                 else:
#                     # Normal mode - simple text
#                     cv2.putText(
#                         frame, text, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         self.font_scale,
#                         self.COLOR_TRACK,
#                         self.thickness
#                     )
        
#         return frame
    
#     def _draw_trajectory_path(
#         self, 
#         frame: np.ndarray, 
#         tracks: List, 
#         frame_num: int,
#         show_centroids: bool = True
#     ) -> np.ndarray:
#         """Draw trajectory paths for tracks (for annotation visualization).
        
#         Args:
#             frame: Input frame
#             tracks: List of Track objects
#             frame_num: Current frame number
#             show_centroids: Whether to draw centroid points
            
#         Returns:
#             Frame with trajectory paths
#         """
#         for track in tracks:
#             if track.is_in_frame(frame_num):
#                 # Get all centroids up to current frame
#                 centroids = track.get_trajectory_up_to(frame_num)
                
#                 if len(centroids) < 2:
#                     continue
                
#                 # Draw path
#                 for i in range(len(centroids) - 1):
#                     pt1 = (int(centroids[i][0]), int(centroids[i][1]))
#                     pt2 = (int(centroids[i+1][0]), int(centroids[i+1][1]))
#                     cv2.line(frame, pt1, pt2, (255, 255, 0), self.thickness)  # Cyan path
                
#                 # Draw centroids if requested
#                 if show_centroids:
#                     for centroid in centroids:
#                         pt = (int(centroid[0]), int(centroid[1]))
#                         cv2.circle(frame, pt, 3, (0, 255, 255), -1)  # Yellow dots
        
#         return frame
    
#     def _draw_event(self, frame: np.ndarray, action: str, nest_id: str) -> np.ndarray:
#         """Draw entry/exit event annotation on frame.
        
#         Args:
#             frame: Input frame
#             action: "Entry" or "Exit"
#             nest_id: Nest hole ID
            
#         Returns:
#             Frame with event annotation
#         """
#         text = f"{action} at nest {nest_id}"
#         color = self.COLOR_ENTRY if action == "Entry" else self.COLOR_EXIT
        
#         cv2.putText(
#             frame,
#             text,
#             (self.text_offset_x, self.text_offset_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             self.font_scale,
#             color,
#             self.thickness
#         )
        
#         return frame
    

#     def synthesize(
#         self,
#         video_path: str,
#         events: pd.DataFrame,
#         motion: pd.DataFrame,
#         nest_data: Dict,
#         output_folder: str,
#         show_trajectories: bool = False
#     ) -> str:
        

#         """Create annotated video showing tracking results.
        
#         Args:
#             video_path: Path to input video file
#             events: DataFrame with columns ['frame_number', 'nest', 'action']
#             motion: DataFrame with columns ['frame_number', 'tracks']
#             nest_data: Dictionary with 'nests' and 'hotel' keys
#             output_folder: Directory for output video
            
#         Returns:
#             Path to created video file
            
#         Example:
#             >>> output_path = synthesizer.synthesize(
#             ...     video_path="input.mp4",
#             ...     events=events_df,
#             ...     motion=motion_df,
#             ...     nest_data={'nests': {...}, 'hotel': (x1, y1, x2, y2)},
#             ...     output_folder="/output"
#             ... )
#         """
#         # Prepare output path
#         output_folder = Path(output_folder)
#         output_folder.mkdir(parents=True, exist_ok=True)
        
#         filename = Path(video_path).stem
#         output_path = str(output_folder / f"synthesized_{filename}.mp4")
        
#         # Open input video
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Could not open video: {video_path}")
        
#         # Create output video writer
#         video_writer = self._create_video_writer(output_path)
        
#         # Extract event information
#         frame_numbers = events['frame_number'].tolist() if not events.empty else []
#         nest_holes = events['nest'].tolist() if not events.empty else []
#         actions = events['action'].tolist() if not events.empty else []
        
#         # Create event lookup
#         event_dict = {}
#         for fn, nest, action in zip(frame_numbers, nest_holes, actions):
#             event_dict[fn] = (nest, action)
        
#         # Process each motion period
#         for period_idx in range(len(motion)):
#             try:
#                 period = motion.iloc[period_idx]['frame_number']
#                 tracks_data = motion.iloc[period_idx]['tracks']
                
#                 # Convert tracks to Track objects
#                 track_objects = [
#                     Track(track[0], track[2], track[3])
#                     for track in tracks_data
#                 ]
                
#                 start_frame, end_frame = period
                
#                 # Process each frame in the period
#                 for frame_num in range(start_frame, end_frame + 1):
#                     # Read frame
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                     ret, frame = cap.read()
                    
#                     if not ret:
#                         continue
                    
#                     # Resize frame
#                     frame = cv2.resize(frame, (self.res_width, self.res_height))
                    
#                     # Draw debug data if enabled (blobs and YOLO detections)
#                     if self.debug_mode:
#                         detections = motion.iloc[period_idx].get('detections', {})
#                         frame_detections = detections.get(frame_num, {})
#                         debug_blobs = frame_detections.get('debug_blobs', [])
#                         debug_yolo = frame_detections.get('debug_yolo', [])
#                         if debug_blobs or debug_yolo:
#                             frame = self._draw_debug_data(frame, debug_blobs, debug_yolo)
                    
#                     # Draw nest holes
#                     frame = self._draw_nest_holes(frame, nest_data['nests'])
                    
#                     # Draw trajectory paths if requested (for annotation mode)
#                     if show_trajectories and self.annotation_mode:
#                         frame = self._draw_trajectory_path(frame, track_objects, frame_num)
                    
#                     # Draw tracks
#                     frame = self._draw_tracks(frame, track_objects, frame_num)





                    
#                     # Draw hotel boundary
#                     if 'hotel' in nest_data:
#                         frame = self._draw_hotel_boundary(frame, nest_data['hotel'])
                    
#                     # Draw event if present
#                     if frame_num in event_dict:
#                         nest_id, action = event_dict[frame_num]
#                         frame = self._draw_event(frame, action, nest_id)
                        
#                         # Hold frame for visibility
#                         hold_frames = int(self.FRAMES_TO_HOLD_BASE)
#                         for _ in range(hold_frames):
#                             video_writer.write(frame)
                    
#                     # Write frame
#                     video_writer.write(frame)
                    
#             except Exception as e:
#                 print(f"Error processing period {period_idx}: {e}")
#                 continue
        
#         # Cleanup
#         cap.release()
#         video_writer.release()
#         cv2.destroyAllWindows()
        
#         return output_path
    
#     def create_nest_visualization(
#         self,
#         video_path: str,
#         nest_data: Dict,
#         output_folder: str,
#         frame_num: int = 0
#     ) -> str:
#         """Create a single frame showing detected nests.
        
#         Args:
#             video_path: Path to input video
#             nest_data: Dictionary with nest detection results
#             output_folder: Directory for output image
#             frame_num: Frame number to visualize (default: 0)
            
#         Returns:
#             Path to saved image
#         """
#         output_folder = Path(output_folder)
#         output_folder.mkdir(parents=True, exist_ok=True)
        
#         filename = Path(video_path).stem
#         output_path = str(output_folder / f"nest_detection_{filename}.png")
        
#         # Read frame
#         cap = cv2.VideoCapture(video_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         cap.release()
        
#         if not ret:
#             raise ValueError(f"Could not read frame {frame_num} from video")
        
#         # Resize
#         frame = cv2.resize(frame, (self.res_width, self.res_height))
        
#         # Draw nests
#         frame = self._draw_nest_holes(frame, nest_data['nests'])
        
#         # Draw hotel boundary
#         if 'hotel' in nest_data:
#             frame = self._draw_hotel_boundary(frame, nest_data['hotel'])
        
#         # Save
#         cv2.imwrite(output_path, frame)
        
#         return output_path


# class Track:
#     """Helper class to represent a track for visualization.
    
#     This class provides a simple interface for querying track information
#     during video synthesis.
#     """
    
#     def __init__(self, track_id: int, trajectory: List[Tuple], frame_numbers: List[int]):
#         """Initialize track.
        
#         Args:
#             track_id: Unique track identifier
#             trajectory: List of (x1, y1, x2, y2) bounding boxes
#             frame_numbers: List of frame numbers corresponding to trajectory
#         """
#         self.track_id = track_id
#         self.trajectory = trajectory
#         self.frame_numbers = frame_numbers
    
#     def is_in_frame(self, frame_num: int) -> bool:
#         """Check if track is present in given frame.
        
#         Args:
#             frame_num: Frame number to check
            
#         Returns:
#             True if track is present in frame
#         """
#         return frame_num in self.frame_numbers
    
#     def get_bbox(self, frame_num: int) -> Tuple[float, float, float, float]:
#         """Get bounding box for given frame.
        
#         Args:
#             frame_num: Frame number
            
#         Returns:
#             Bounding box (x1, y1, x2, y2)
            
#         Raises:
#             ValueError: If frame_num not in track
#         """
#         if frame_num not in self.frame_numbers:
#             raise ValueError(f"Frame {frame_num} not in track {self.track_id}")
        
#         idx = self.frame_numbers.index(frame_num)
#         return self.trajectory[idx]
    
#     def getID(self) -> int:
#         """Get track ID.
        
#         Returns:
#             Track identifier
#         """
#         return self.track_id
    

#     def get_trajectory_up_to(self, frame_num: int) -> List[Tuple[float, float]]:
#         """Get trajectory centroids up to specified frame.
        
#         Args:
#             frame_num: Frame number
            
#         Returns:
#             List of (x, y) centroid coordinates
#         """
#         centroids = []
#         for i, fn in enumerate(self.frame_numbers):
#             if fn > frame_num:
#                 break
#             # Calculate centroid from bbox
#             bbox = self.bboxes[i]
#             cx = (bbox[0] + bbox[2]) / 2
#             cy = (bbox[1] + bbox[3]) / 2
#             centroids.append((cx, cy))
#         return centroids


# # Legacy function for backward compatibility
# def synthesize(
#     video_path: str,
#     events: pd.DataFrame,
#     motion: pd.DataFrame,
#     nests: Dict,
#     output_folder: str,
#     res_height: int = 720,
#     res_width: int = 1280
# ) -> str:
#     """Legacy function for backward compatibility.
    
#     Args:
#         video_path: Path to input video
#         events: Events DataFrame
#         motion: Motion DataFrame
#         nests: Nest data dictionary
#         output_folder: Output directory
#         res_height: Video height
#         res_width: Video width
        
#     Returns:
#         Path to output video
        
#     Note:
#         This function is deprecated. Use VideoSynthesizer class instead.
#     """
#     from dataclasses import dataclass, field
    
#     @dataclass
#     class LegacyVideoConfig:
#         res_width: int = res_width
#         res_height: int = res_height
#         fps: int = 30
    
#     @dataclass
#     class LegacyOutputConfig:
#         video_codec: str = "mp4v"
#         video_fps: int = 30
    
#     @dataclass
#     class LegacyNestConfig:
#         reference_width: int = 1280
#         reference_height: int = 720
#         padding_x_base: int = 5
#         padding_y_base: int = 7
        
#         def padding_x(self, w, h):
#             return int(5 * w / 1280)
        
#         def padding_y(self, w, h):
#             return int(7 * h / 720)
    
#     @dataclass
#     class LegacyConfig:
#         video: LegacyVideoConfig = field(default_factory=LegacyVideoConfig)
#         output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)
#         nest: LegacyNestConfig = field(default_factory=LegacyNestConfig)
        
#         def get_nest_params(self):
#             return {
#                 'padding_x': self.nest.padding_x(self.video.res_width, self.video.res_height),
#                 'padding_y': self.nest.padding_y(self.video.res_width, self.video.res_height),
#             }
    
#     config = LegacyConfig()
#     synthesizer = VideoSynthesizer(config)
#     return synthesizer.synthesize(video_path, events, motion, nests, output_folder)


# # Convenience function
# def synthesize_video(
#     video_path: str,
#     events: pd.DataFrame,
#     motion: pd.DataFrame,
#     nest_data: Dict,
#     config,
#     output_folder: Optional[str] = None
# ) -> str:
#     """Convenience function to synthesize video with config.
    
#     Args:
#         video_path: Path to input video
#         events: Events DataFrame
#         motion: Motion DataFrame
#         nest_data: Nest data dictionary
#         config: Config object
#         output_folder: Optional output directory (uses config default if not provided)
        
#     Returns:
#         Path to output video
        
#     Example:
#         >>> from config import Config
#         >>> config = Config.default()
#         >>> video_path = synthesize_video(
#         ...     "input.mp4", events, motion, nests, config, "/output"
#         ... )
#     """
#     if output_folder is None:
#         output_folder = config.output.base_folder
    
#     synthesizer = VideoSynthesizer(config)
#     return synthesizer.synthesize(video_path, events, motion, nest_data, output_folder)























"""Video synthesis module with resolution-adaptive visualization.

This module handles the creation of annotated videos showing tracking results,
nest detections, and entry/exit events. All visual elements (text sizes, line
thicknesses, padding) automatically scale with video resolution.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path


class VideoSynthesizer:
    """Handles synthesis of annotated tracking videos with resolution scaling.
    
    This class creates visualization videos that show:
    - Detected nest holes with IDs
    - Tracked bee trajectories with IDs
    - Entry/exit events with annotations
    
    All visual elements automatically scale based on video resolution.
    
    Attributes:
        config: Configuration object with resolution and visualization settings
        res_width: Target video width
        res_height: Target video height
        font_scale_base: Base font scale factor
        thickness_base: Base line thickness
    
    Example:
        >>> from config import Config
        >>> config = Config.default()
        >>> synthesizer = VideoSynthesizer(config)
        >>> video_path = synthesizer.synthesize(
        ...     video_path="input.mp4",
        ...     events=events_df,
        ...     motion=motion_df,
        ...     nest_data=nest_dict,
        ...     output_folder="/output"
        ... )
    """
    
    # Base values at reference resolution (1280x720)
    REFERENCE_WIDTH = 1280
    REFERENCE_HEIGHT = 720
    
    # Base visualization parameters
    FONT_SCALE_BASE = 0.75
    THICKNESS_BASE = 2
    TEXT_OFFSET_X_BASE = 50
    TEXT_OFFSET_Y_BASE = 75
    FRAMES_TO_HOLD_BASE = 30  # Resolution-independent (time-based)
    
    # Colors (BGR format)
    COLOR_NEST = (255, 0, 0)  # Blue for nest boxes
    COLOR_TRACK = (0, 255, 0)  # Green for tracks
    COLOR_EXIT = (255, 0, 0)  # Blue for exit events
    COLOR_ENTRY = (0, 255, 0)  # Green for entry events
    COLOR_HOTEL = (0, 0, 0)  # Black for hotel boundary
    COLOR_BLOB = (255, 255, 0)  # Cyan for FG blobs
    COLOR_YOLO = (0, 165, 255)  # Orange for YOLO detections
    
    # REPLACE WITH:
    def __init__(self, config):
        """Initialize video synthesizer with configuration.
        
        Args:
            config: Config object containing video and output settings
            annotation_mode: If True, enhances track IDs for annotation purposes
        """
        self.config = config
        self.res_width = config.video.res_width
        self.res_height = config.video.res_height
        self.annotation_mode = config.video.annotation_mode
        self.debug_mode = getattr(config.video, 'debug_mode', False)  # Debug visualization flag
        
        # Calculate scale factors
        self.scale_x = self.res_width / self.REFERENCE_WIDTH
        self.scale_y = self.res_height / self.REFERENCE_HEIGHT
        self.scale_avg = (self.scale_x + self.scale_y) / 2
        
        # Scaled visualization parameters
        base_font = self.FONT_SCALE_BASE * (1.5 if self.annotation_mode else 1.0)
        self.font_scale = base_font * self.scale_avg
        self.thickness = max(1, int(self.THICKNESS_BASE * self.scale_avg))
        self.text_offset_x = int(self.TEXT_OFFSET_X_BASE * self.scale_x)
        self.text_offset_y = int(self.TEXT_OFFSET_Y_BASE * self.scale_y)
        
        # Get scaled nest parameters
        nest_params = config.get_nest_params()
        self.nest_padding_x = nest_params['padding_x']
        self.nest_padding_y = nest_params['padding_y']
    
    def _create_video_writer(self, output_path: str) -> cv2.VideoWriter:
        """Create a video writer object.
        
        Args:
            output_path: Path for output video file
            
        Returns:
            OpenCV VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
        fps = self.config.video.fps
        size = (self.res_width, self.res_height)
        return cv2.VideoWriter(output_path, fourcc, fps, size)
    
    def _draw_nest_holes(self, frame: np.ndarray, nests: Dict[str, Tuple]) -> np.ndarray:
        """Draw nest hole bounding boxes and IDs on frame.
        
        Args:
            frame: Input frame
            nests: Dictionary mapping nest IDs to (x1, y1, x2, y2) coordinates
            
        Returns:
            Frame with nest annotations
        """
        for nest_id, coords in nests.items():
            # Extract ID number
            id_num = nest_id.split('_')[-1] if '_' in nest_id else nest_id
            
            # Get coordinates and apply padding
            x1, y1, x2, y2 = coords
            x1 = int(x1 - self.nest_padding_x)
            y1 = int(y1 - self.nest_padding_y)
            x2 = int(x2 + self.nest_padding_x)
            y2 = int(y2 + self.nest_padding_y)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_NEST, self.thickness)
            
            # Draw ID text
            cv2.putText(
                frame, str(id_num), (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.COLOR_NEST,
                self.thickness
            )
        
        return frame
    
    def _draw_hotel_boundary(self, frame: np.ndarray, hotel_roi: Tuple) -> np.ndarray:
        """Draw hotel region of interest boundary.
        
        Args:
            frame: Input frame
            hotel_roi: Hotel bounding box (x1, y1, x2, y2)
            
        Returns:
            Frame with hotel boundary
        """
        x1, y1, x2, y2 = [int(coord) for coord in hotel_roi]
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_HOTEL, self.thickness)
        return frame
    
    def _draw_debug_data(
        self, 
        frame: np.ndarray, 
        debug_blobs: List[Dict], 
        debug_yolo: List[Dict]
    ) -> np.ndarray:
        """Draw debug visualization: FG blobs and YOLO detections.
        
        Args:
            frame: Input frame
            debug_blobs: List of FG blob dictionaries with 'bbox' and 'centroid'
            debug_yolo: List of YOLO detection dictionaries with 'bbox', 'centroid', 'species'
            
        Returns:
            Frame with debug annotations
        """
        # Draw FG blobs (cyan)
        for blob in debug_blobs:
            x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_BLOB, 1)
            # Draw centroid
            cx, cy = [int(c) for c in blob['centroid']]
            cv2.circle(frame, (cx, cy), 3, self.COLOR_BLOB, -1)
        
        # Draw YOLO detections (orange)
        for det in debug_yolo:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_YOLO, 2)
            # Draw label
            label = det.get('species', 'unknown')
            cv2.putText(
                frame, f"DL:{label}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale * 0.7,
                self.COLOR_YOLO,
                self.thickness
            )
        
        return frame
    
    # REPLACE WITH:
    def _draw_tracks(self, frame: np.ndarray, tracks: List, frame_num: int) -> np.ndarray:
        """Draw active tracks on frame.
        
        Args:
            frame: Input frame
            tracks: List of Track objects
            frame_num: Current frame number
            
        Returns:
            Frame with track annotations
        """
        for track in tracks:
            if track.is_in_frame(frame_num):
                bbox = track.get_bbox(frame_num)
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Draw track bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_TRACK, self.thickness)
                
                # Draw track ID with enhanced visibility for annotation mode
                track_id = track.getID()
                text = f"ID:{track_id}"
                
                if self.annotation_mode:
                    # Get text size for background box
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
                    )
                    
                    # Position text above bounding box
                    text_x = x1
                    text_y = y1 - 10
                    
                    # Draw background rectangle for better readability
                    padding = 5
                    bg_x1 = text_x - padding
                    bg_y1 = text_y - text_height - padding
                    bg_x2 = text_x + text_width + padding
                    bg_y2 = text_y + baseline + padding
                    
                    # Draw semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    # Draw text on top
                    cv2.putText(
                        frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (0, 255, 255),  # Yellow for high visibility
                        self.thickness + 1
                    )
                else:
                    # Normal mode - simple text
                    cv2.putText(
                        frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        self.COLOR_TRACK,
                        self.thickness
                    )
        
        return frame
    
    def _draw_trajectory_path(
        self, 
        frame: np.ndarray, 
        tracks: List, 
        frame_num: int,
        show_centroids: bool = True
    ) -> np.ndarray:
        """Draw trajectory paths for tracks (for annotation visualization).
        
        Args:
            frame: Input frame
            tracks: List of Track objects
            frame_num: Current frame number
            show_centroids: Whether to draw centroid points
            
        Returns:
            Frame with trajectory paths
        """
        for track in tracks:
            if track.is_in_frame(frame_num):
                # Get all centroids up to current frame
                centroids = track.get_trajectory_up_to(frame_num)
                
                if len(centroids) < 2:
                    continue
                
                # Draw path
                for i in range(len(centroids) - 1):
                    pt1 = (int(centroids[i][0]), int(centroids[i][1]))
                    pt2 = (int(centroids[i+1][0]), int(centroids[i+1][1]))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), self.thickness)  # Cyan path
                
                # Draw centroids if requested
                if show_centroids:
                    for centroid in centroids:
                        pt = (int(centroid[0]), int(centroid[1]))
                        cv2.circle(frame, pt, 3, (0, 255, 255), -1)  # Yellow dots
        
        return frame
    
    def _draw_event(self, frame: np.ndarray, action: str, nest_id: str) -> np.ndarray:
        """Draw entry/exit event annotation on frame.
                
        Args:
            frame: Input frame
            action: "Entry" or "Exit"
            nest_id: Nest hole ID
            
        Returns:
            Frame with event annotation
        """
        text = f"{action} at nest {nest_id}"
        color = self.COLOR_ENTRY if action == "Entry" else self.COLOR_EXIT
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Measure text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, self.font_scale, self.thickness
        )

        # Padding scaled to resolution
        padding = max(4, int(6 * self.scale_avg))

        # Compute background rectangle coordinates (clamped to frame)
        x = int(self.text_offset_x)
        y = int(self.text_offset_y)
        bg_x1 = max(0, x - padding)
        bg_y1 = max(0, y - text_height - padding)
        bg_x2 = min(frame.shape[1], x + text_width + padding)
        bg_y2 = min(frame.shape[0], y + baseline + padding)

        # Draw semi-transparent black background for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw text on top
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            self.font_scale,
            color,
            self.thickness
        )
        return frame
    

    def synthesize(
        self,
        video_path: str,
        events: pd.DataFrame,
        motion: pd.DataFrame,
        nest_data: Dict,
        output_folder: str,
        show_trajectories: bool = False
    ) -> str:
        

        """Create annotated video showing tracking results.
        
        Args:
            video_path: Path to input video file
            events: DataFrame with columns ['frame_number', 'nest', 'action']
            motion: DataFrame with columns ['frame_number', 'tracks']
            nest_data: Dictionary with 'nests' and 'hotel' keys
            output_folder: Directory for output video
            
        Returns:
            Path to created video file
            
        Example:
            >>> output_path = synthesizer.synthesize(
            ...     video_path="input.mp4",
            ...     events=events_df,
            ...     motion=motion_df,
            ...     nest_data={'nests': {...}, 'hotel': (x1, y1, x2, y2)},
            ...     output_folder="/output"
            ... )
        """
        # Prepare output path
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        filename = Path(video_path).stem
        output_path = str(output_folder / f"synthesized_{filename}.mp4")
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Create output video writer
        video_writer = self._create_video_writer(output_path)
        
        # Extract event information
        frame_numbers = events['frame_number'].tolist() if not events.empty else []
        nest_holes = events['nest'].tolist() if not events.empty else []
        actions = events['action'].tolist() if not events.empty else []
        
        # Create event lookup
        event_dict = {}
        for fn, nest, action in zip(frame_numbers, nest_holes, actions):
            event_dict[fn] = (nest, action)
        
        # Process each motion period
        for period_idx in range(len(motion)):
            try:
                period = motion.iloc[period_idx]['frame_number']
                tracks_data = motion.iloc[period_idx]['tracks']
                
                # Convert tracks to Track objects
                track_objects = [
                    Track(track[0], track[2], track[3])
                    for track in tracks_data
                ]
                
                start_frame, end_frame = period
                
                # Process each frame in the period
                for frame_num in range(start_frame, end_frame + 1):
                    # Read frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Resize frame
                    frame = cv2.resize(frame, (self.res_width, self.res_height))
                    
                    # Draw debug data if enabled (blobs and YOLO detections)
                    if self.debug_mode:
                        detections = motion.iloc[period_idx].get('detections', {})
                        frame_detections = detections.get(frame_num, {})
                        debug_blobs = frame_detections.get('debug_blobs', [])
                        debug_yolo = frame_detections.get('debug_yolo', [])
                        if debug_blobs or debug_yolo:
                            frame = self._draw_debug_data(frame, debug_blobs, debug_yolo)
                    
                    # Draw nest holes
                    frame = self._draw_nest_holes(frame, nest_data['nests'])
                    
                    # Draw trajectory paths if requested (for annotation mode)
                    if show_trajectories and self.annotation_mode:
                        frame = self._draw_trajectory_path(frame, track_objects, frame_num)
                    
                    # Draw tracks
                    frame = self._draw_tracks(frame, track_objects, frame_num)

                    
                    # Draw hotel boundary
                    if 'hotel' in nest_data:
                        frame = self._draw_hotel_boundary(frame, nest_data['hotel'])
                    
                    # Draw event if present
                    if frame_num in event_dict:
                        nest_id, action = event_dict[frame_num]
                        frame = self._draw_event(frame, action, nest_id)
                        
                        # Hold frame for visibility
                        hold_frames = int(self.FRAMES_TO_HOLD_BASE)
                        for _ in range(hold_frames):
                            video_writer.write(frame)
                    
                    # Write frame
                    video_writer.write(frame)
                    
            except Exception as e:
                print(f"Error processing period {period_idx}: {e}")
                continue
        
        # Cleanup
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        return output_path
    
    def create_nest_visualization(
        self,
        video_path: str,
        nest_data: Dict,
        output_folder: str,
        frame_num: int = 0
    ) -> str:
        """Create a single frame showing detected nests.
        
        Args:
            video_path: Path to input video
            nest_data: Dictionary with nest detection results
            output_folder: Directory for output image
            frame_num: Frame number to visualize (default: 0)
            
        Returns:
            Path to saved image
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        filename = Path(video_path).stem
        output_path = str(output_folder / f"nest_detection_{filename}.png")
        
        # Read frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_num} from video")
        
        # Resize
        frame = cv2.resize(frame, (self.res_width, self.res_height))
        
        # Draw nests
        frame = self._draw_nest_holes(frame, nest_data['nests'])
        
        # Draw hotel boundary
        if 'hotel' in nest_data:
            frame = self._draw_hotel_boundary(frame, nest_data['hotel'])
        
        # Save
        cv2.imwrite(output_path, frame)
        
        return output_path


class Track:
    """Helper class to represent a track for visualization.
    
    This class provides a simple interface for querying track information
    during video synthesis.
    """
    
    def __init__(self, track_id: int, trajectory: List[Tuple], frame_numbers: List[int]):
        """Initialize track.
        
        Args:
            track_id: Unique track identifier
            trajectory: List of (x1, y1, x2, y2) bounding boxes
            frame_numbers: List of frame numbers corresponding to trajectory
        """
        self.track_id = track_id
        self.trajectory = trajectory
        self.frame_numbers = frame_numbers
    
    def is_in_frame(self, frame_num: int) -> bool:
        """Check if track is present in given frame.
        
        Args:
            frame_num: Frame number to check
            
        Returns:
            True if track is present in frame
        """
        return frame_num in self.frame_numbers
    
    def get_bbox(self, frame_num: int) -> Tuple[float, float, float, float]:
        """Get bounding box for given frame.
        
        Args:
            frame_num: Frame number
            
        Returns:
            Bounding box (x1, y1, x2, y2)
            
        Raises:
            ValueError: If frame_num not in track
        """
        if frame_num not in self.frame_numbers:
            raise ValueError(f"Frame {frame_num} not in track {self.track_id}")
        
        idx = self.frame_numbers.index(frame_num)
        return self.trajectory[idx]
    
    def getID(self) -> int:
        """Get track ID.
        
        Returns:
            Track identifier
        """
        return self.track_id
    

    def get_trajectory_up_to(self, frame_num: int) -> List[Tuple[float, float]]:
        """Get trajectory centroids up to specified frame.
        
        Args:
            frame_num: Frame number
            
        Returns:
            List of (x, y) centroid coordinates
        """
        centroids = []
        for i, fn in enumerate(self.frame_numbers):
            if fn > frame_num:
                break
            # Calculate centroid from bbox
            bbox = self.bboxes[i]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centroids.append((cx, cy))
        return centroids


# Legacy function for backward compatibility
def synthesize(
    video_path: str,
    events: pd.DataFrame,
    motion: pd.DataFrame,
    nests: Dict,
    output_folder: str,
    res_height: int = 720,
    res_width: int = 1280
) -> str:
    """Legacy function for backward compatibility.
    
    Args:
        video_path: Path to input video
        events: Events DataFrame
        motion: Motion DataFrame
        nests: Nest data dictionary
        output_folder: Output directory
        res_height: Video height
        res_width: Video width
        
    Returns:
        Path to output video
        
    Note:
        This function is deprecated. Use VideoSynthesizer class instead.
    """
    from dataclasses import dataclass, field
    
    @dataclass
    class LegacyVideoConfig:
        res_width: int = res_width
        res_height: int = res_height
        fps: int = 30
    
    @dataclass
    class LegacyOutputConfig:
        video_codec: str = "mp4v"
        video_fps: int = 30
    
    @dataclass
    class LegacyNestConfig:
        reference_width: int = 1280
        reference_height: int = 720
        padding_x_base: int = 5
        padding_y_base: int = 7
        
        def padding_x(self, w, h):
            return int(5 * w / 1280)
        
        def padding_y(self, w, h):
            return int(7 * h / 720)
    
    @dataclass
    class LegacyConfig:
        video: LegacyVideoConfig = field(default_factory=LegacyVideoConfig)
        output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)
        nest: LegacyNestConfig = field(default_factory=LegacyNestConfig)
        
        def get_nest_params(self):
            return {
                'padding_x': self.nest.padding_x(self.video.res_width, self.video.res_height),
                'padding_y': self.nest.padding_y(self.video.res_width, self.video.res_height),
            }
    
    config = LegacyConfig()
    synthesizer = VideoSynthesizer(config)
    return synthesizer.synthesize(video_path, events, motion, nests, output_folder)


# Convenience function
def synthesize_video(
    video_path: str,
    events: pd.DataFrame,
    motion: pd.DataFrame,
    nest_data: Dict,
    config,
    output_folder: Optional[str] = None
) -> str:
    """Convenience function to synthesize video with config.
    
    Args:
        video_path: Path to input video
        events: Events DataFrame
        motion: Motion DataFrame
        nest_data: Nest data dictionary
        config: Config object
        output_folder: Optional output directory (uses config default if not provided)
        
    Returns:
        Path to output video
        
    Example:
        >>> from config import Config
        >>> config = Config.default()
        >>> video_path = synthesize_video(
        ...     "input.mp4", events, motion, nests, config, "/output"
        ... )
    """
    if output_folder is None:
        output_folder = config.output.base_folder
    
    synthesizer = VideoSynthesizer(config)
    return synthesizer.synthesize(video_path, events, motion, nest_data, output_folder)