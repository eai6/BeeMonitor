"""Visual Testing Tool for BeeMonitor

Interactive tool to visualize detection and tracking in real-time.

Run: python tests/visual_test.py [video_path]

Controls:
- Press 'q' to quit
- Press 'p' to pause/resume
- Press 's' to step frame-by-frame (when paused)
- Press '1-7' to switch detection modes
- Press 'r' to reset tracker
- Press 'h' to toggle help display
"""

import cv2
import numpy as np
import sys
from typing import Optional

from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker
from beemonitor.core.config import Config


class VisualTester:
    """Interactive visual testing tool."""
    
    def __init__(self, video_path: Optional[str] = None):
        """Initialize visual tester.
        
        Args:
            video_path: Path to video file (None = create test video)
        """
        self.config = Config.default()
        self.video_path = video_path or self._create_test_video()
        
        # Detection modes
        self.modes = [
            DetectionMode.FGBG_ONLY,
            DetectionMode.SIFT_ONLY,
            DetectionMode.FGBG_SIFT,
            DetectionMode.FGBG_YOLO,
            DetectionMode.SIFT_YOLO,
            DetectionMode.FGBG_SIFT_YOLO,
            DetectionMode.YOLO_ONLY,
        ]
        self.current_mode_idx = 0
        
        # State
        self.paused = False
        self.show_help = True
        self.frame_num = 0
        
        # Create initial tracker
        self.tracker = None
        self._create_tracker()
    
    def _create_test_video(self) -> str:
        """Create test video if none provided."""
        print("Creating test video...")
        output_path = '/tmp/visual_test_video.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
        
        # Create moving objects
        for frame_num in range(300):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Moving bee
            x = int(200 + frame_num * 3) % 1280
            y = int(360 + 100 * np.sin(frame_num * 0.05))
            self._draw_bee(frame, (x, y))
            
            # Stationary bee
            self._draw_bee(frame, (900, 400))
            
            # Random moving bees
            for i in range(3):
                x = int(300 + i * 200 + 50 * np.sin(frame_num * 0.1 + i))
                y = int(200 + i * 150 + 50 * np.cos(frame_num * 0.1 + i))
                self._draw_bee(frame, (x, y))
            
            out.write(frame)
        
        out.release()
        print(f"Test video created: {output_path}")
        return output_path
    
    def _draw_bee(self, frame: np.ndarray, pos: tuple):
        """Draw bee-like object."""
        x, y = pos
        cv2.circle(frame, (int(x), int(y)), 25, (255, 255, 255), -1)
        cv2.circle(frame, (int(x), int(y)), 20, (200, 200, 200), -1)
        cv2.circle(frame, (int(x), int(y)), 10, (150, 150, 150), -1)
    
    def _create_tracker(self):
        """Create tracker with current mode."""
        mode = self.modes[self.current_mode_idx]
        
        print(f"Creating tracker with mode: {mode.value}")
        
        mot = BeeTracker(self.config, ['bee'])
        
        self.tracker = BeeTracking(
            mot_algorithm=mot,
            yolo_model=None,  # Set this to your YOLO model if available
            detection_mode=mode,
            use_noise_filter=False,
            config=self.config
        )
        
        self.frame_num = 0
    
    def _switch_mode(self, mode_idx: int):
        """Switch to different detection mode."""
        if 0 <= mode_idx < len(self.modes):
            self.current_mode_idx = mode_idx
            self._create_tracker()
    
    def _draw_detection(self, frame: np.ndarray, det, color: tuple, label: str):
        """Draw single detection."""
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cx, cy = [int(c) for c in det.centroid]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 3, color, -1)
        
        # Label
        label_text = f"{label} ({det.source})"
        cv2.putText(frame, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_track(self, frame: np.ndarray, track_id: int, track):
        """Draw single track."""
        x1, y1, x2, y2 = [int(c) for c in track.bbox]
        cx, cy = [int(c) for c in track.centroid]
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # ID
        cv2.putText(frame, f"ID:{track_id}", (cx - 20, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Trajectory
        if len(track.trajectory) > 1:
            points = [pt[1] for pt in track.trajectory[-20:]]
            for i in range(len(points) - 1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i+1]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Age indicator
        age_text = f"age:{track.age}"
        cv2.putText(frame, age_text, (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    def _draw_info(self, frame: np.ndarray, result: dict):
        """Draw info overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Info text
        y_offset = 25
        line_height = 25
        
        mode = self.modes[self.current_mode_idx]
        stats = self.tracker.get_statistics()
        
        info_lines = [
            f"Mode: {mode.value} (Press 1-7 to switch)",
            f"Frame: {self.frame_num}",
            f"Detections: {len(result['detections'])}",
            f"Active Tracks: {len(result['tracks'])}",
            f"Total Tracks: {stats['total_tracks']}",
        ]
        
        if self.paused:
            info_lines.append("PAUSED (Press 'p' to resume, 's' to step)")
        
        for i, line in enumerate(info_lines):
            y = y_offset + i * line_height
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_help(self, frame: np.ndarray):
        """Draw help overlay."""
        if not self.show_help:
            return
        
        h, w = frame.shape[:2]
        
        # Help box
        help_x = w - 300
        help_y = 10
        help_w = 290
        help_h = 200
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (help_x, help_y), 
                     (help_x + help_w, help_y + help_h),
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Help text
        help_lines = [
            "Controls:",
            "q - Quit",
            "p - Pause/Resume",
            "s - Step (when paused)",
            "r - Reset tracker",
            "h - Toggle this help",
            "1-7 - Switch modes",
        ]
        
        for i, line in enumerate(help_lines):
            y = help_y + 25 + i * 22
            cv2.putText(frame, line, (help_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Run visual testing loop."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return
        
        print("\n" + "="*70)
        print("VISUAL TESTING TOOL")
        print("="*70)
        print("\nControls:")
        print("  q - Quit")
        print("  p - Pause/Resume")
        print("  s - Step frame (when paused)")
        print("  r - Reset tracker")
        print("  h - Toggle help")
        print("  1-7 - Switch detection modes")
        print("\nPress 'h' to toggle help display in video")
        print("="*70 + "\n")
        
        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or read error")
                    break
                
                # Process frame
                result = self.tracker.process_frame(frame, self.frame_num)
                self.frame_num += 1
            else:
                # Keep showing last frame
                pass
            
            # Visualize
            vis_frame = frame.copy()
            
            # Draw detections (green)
            for det in result['detections']:
                self._draw_detection(vis_frame, det, (0, 255, 0), det.label)
            
            # Draw tracks (blue)
            for track_id, track in result['tracks'].items():
                self._draw_track(vis_frame, track_id, track)
            
            # Draw info
            self._draw_info(vis_frame, result)
            self._draw_help(vis_frame)
            
            # Show
            cv2.imshow('BeeMonitor Visual Test', vis_frame)
            
            # Handle keys
            key = cv2.waitKey(30 if not self.paused else 0) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"{'Paused' if self.paused else 'Resumed'}")
            elif key == ord('s') and self.paused:
                # Step one frame
                ret, frame = cap.read()
                if ret:
                    result = self.tracker.process_frame(frame, self.frame_num)
                    self.frame_num += 1
            elif key == ord('r'):
                print("Resetting tracker...")
                self.tracker.reset()
                self.frame_num = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif ord('1') <= key <= ord('7'):
                mode_idx = key - ord('1')
                if mode_idx < len(self.modes):
                    print(f"Switching to mode: {self.modes[mode_idx].value}")
                    self._switch_mode(mode_idx)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.tracker.get_statistics()
        print("\n" + "="*70)
        print("SESSION STATISTICS")
        print("="*70)
        print(f"Mode: {self.modes[self.current_mode_idx].value}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Total tracks: {stats['total_tracks']}")
        print(f"Avg detections/frame: {stats['total_detections'] / max(stats['total_frames'], 1):.2f}")
        print("="*70)


if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if video_path:
        print(f"Using video: {video_path}")
    else:
        print("No video provided, will create test video")
    
    tester = VisualTester(video_path)
    tester.run()
