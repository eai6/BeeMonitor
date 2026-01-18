"""
Analysis Thread - v2.4
=======================

Background thread for running video analysis without blocking GUI.

v2.4 CHANGES:
- Added enable_two_mode parameter support
- Passes two-mode setting to monitor.analyze_video
"""

import os
import inspect
import traceback
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisThread(QThread):
    """Background thread for running video analysis."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)
    
    def __init__(self, monitor, video_path, output_folder, detection_mode='yolo', 
                 nests=None, enable_two_mode=True):
        """Initialize analysis thread.
        
        Args:
            monitor: BeeMonitor instance
            video_path: Path to video file
            output_folder: Output directory for results
            detection_mode: Detection mode (default: 'yolo')
            nests: Optional dict with 'nests' and 'hotel' for manually edited nests
            enable_two_mode: Enable two-mode tracking optimization (default: True)
        """
        super().__init__()
        self.monitor = monitor
        self.video_path = video_path
        self.output_folder = output_folder
        self.detection_mode = 'yolo'  # Always YOLO
        self.nests = nests  # Manually edited nests from GUI
        self.enable_two_mode = enable_two_mode  # NEW v2.4
        
        # Extract hotel ROI for motion detection filtering
        if self.nests and self.nests.get('hotel'):
            hotel_roi = self.nests['hotel']
            # Set ROI in config for motion detector
            if hasattr(self.monitor.config, 'detection'):
                self.monitor.config.detection.roi = hotel_roi
            elif hasattr(self.monitor.config, 'tracking'):
                self.monitor.config.tracking.roi = hotel_roi
            # Also store directly on monitor if it has roi attribute
            if hasattr(self.monitor, 'roi'):
                self.monitor.roi = hotel_roi
        
        # NEW v2.4: Set two-mode in config
        if hasattr(self.monitor.config, 'tracking'):
            self.monitor.config.tracking.enable_two_mode = enable_two_mode
    
    def run(self):
        """Run analysis in background thread."""
        try:
            self.progress.emit("Initializing analysis...")
            self.progress.emit("✓ Detection mode: YOLO-only (v2.4)")
            
            # NEW v2.4: Log two-mode status
            if self.enable_two_mode:
                self.progress.emit("✓ Two-mode optimization ENABLED (5-7x faster)")
                self.progress.emit("  → Motion detection triggers YOLO tracking")
            else:
                self.progress.emit("⚠ Two-mode optimization DISABLED")
                self.progress.emit("  → YOLO runs on every frame (slower)")
            
            if self.nests and self.nests.get('nests'):
                self.progress.emit(f"✓ Using {len(self.nests['nests'])} manually edited nests")
            if self.nests and self.nests.get('hotel'):
                roi = self.nests['hotel']
                self.progress.emit(f"✓ Using edited hotel ROI: ({int(roi[0])},{int(roi[1])}) to ({int(roi[2])},{int(roi[3])})")
            else:
                self.progress.emit("Nest detector will automatically detect hotel ROI...")
            
            sig = inspect.signature(self.monitor.analyze_video)
            
            kwargs = {'video_path': self.video_path}
            
            if 'visualize' in sig.parameters:
                kwargs['visualize'] = False  # Don't generate visualization video
            
            if 'detection_mode' in sig.parameters:
                kwargs['detection_mode'] = 'yolo'
                self.progress.emit("Using detection mode: YOLO-only")
            
            if 'output_folder' in sig.parameters:
                kwargs['output_folder'] = self.output_folder
            
            # Pass manually edited nests if available
            if self.nests and 'nests' in sig.parameters:
                kwargs['nests'] = self.nests
                self.progress.emit("✓ Passing edited nests to event processor")
            
            # Pass hotel ROI if analyze_video accepts it
            if self.nests and self.nests.get('hotel') and 'roi' in sig.parameters:
                kwargs['roi'] = self.nests['hotel']
                self.progress.emit("✓ Passing hotel ROI to motion detection")
            
            # NEW v2.4: Pass two-mode setting if supported
            if 'enable_two_mode' in sig.parameters:
                kwargs['enable_two_mode'] = self.enable_two_mode
                self.progress.emit(f"✓ Two-mode tracking: {'ENABLED' if self.enable_two_mode else 'DISABLED'}")
            
            self.progress.emit("Running analysis (this may take several minutes)...")
            
            if self.enable_two_mode:
                self.progress.emit("Blob detector will scan for motion...")
                self.progress.emit("YOLO will run when motion is detected...")
            else:
                self.progress.emit("YOLO will process every frame...")
            
            result = self.monitor.analyze_video(**kwargs)
            
            # Check if analysis returned a valid result
            if result is None:
                self.error.emit(
                    "Analysis returned no results.\n\n"
                    "Possible causes:\n"
                    "• No bees detected in video\n"
                    "• Video file is corrupted or empty\n"
                    "• Detection thresholds too strict\n\n"
                    "Try adjusting detection parameters or check the video file."
                )
                return
            
            os.makedirs(self.output_folder, exist_ok=True)
            
            self.progress.emit("Saving results...")
            
            try:
                result.to_csv(self.output_folder)
                self.progress.emit("✓ Results saved!")
                
                video_name = os.path.basename(self.video_path).replace('.mp4', '').replace('.avi', '').replace('.mov', '')
                csv_path = os.path.join(self.output_folder, f'{video_name}_tracking_results.csv')
                
                if not os.path.exists(csv_path):
                    self.error.emit(
                        f"Tracking results file not created!\n\n"
                        f"Expected: {csv_path}"
                    )
                    return
                
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    if 'source' in df.columns:
                        self.progress.emit("✓ Detection source tracking enabled")
                        self.progress.emit("  RED = Motion detection (Blob)")
                        self.progress.emit("  BLUE = YOLO tracking")
                except Exception:
                    pass
                
                self.progress.emit("✓ Analysis complete!")
                
            except Exception as e:
                self.error.emit(f"Failed to save results: {e}\n\n{traceback.format_exc()}")
                return
            
            self.finished.emit(result, csv_path)
            
        except Exception as e:
            error_details = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_details)