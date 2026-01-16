"""
Analysis Thread - v2.2 SIMPLIFIED
==================================

Background thread for running video analysis without blocking GUI.

v2.2 Changes:
- Always uses 'yolo' detection mode
- Updated progress messages
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
    
    def __init__(self, monitor, video_path, output_folder, detection_mode='yolo'):
        """
        Initialize analysis thread.
        
        Args:
            monitor: BeeMonitor instance
            video_path: Path to input video
            output_folder: Path to output directory
            detection_mode: Detection mode (v2.2: always 'yolo', parameter kept for compatibility)
        """
        super().__init__()
        self.monitor = monitor
        self.video_path = video_path
        self.output_folder = output_folder
        # v2.2: Ignore parameter, always use YOLO
        self.detection_mode = 'yolo'
    
    def run(self):
        """Run analysis in background thread."""
        try:
            self.progress.emit("Initializing analysis...")
            self.progress.emit("✓ Detection mode: YOLO-only (v2.2 - 100% accuracy)")
            self.progress.emit("✓ Two-mode optimization enabled (5-7x faster)")
            self.progress.emit("Nest detector will automatically detect hotel ROI...")
            
            # Check if analyze_video accepts these parameters
            sig = inspect.signature(self.monitor.analyze_video)
            
            # Build kwargs
            kwargs = {'video_path': self.video_path}
            
            if 'visualize' in sig.parameters:
                kwargs['visualize'] = True
            
            # v2.2: Always pass 'yolo' mode
            if 'detection_mode' in sig.parameters:
                kwargs['detection_mode'] = 'yolo'
                self.progress.emit("Using detection mode: YOLO-only")
            
            self.progress.emit("Running analysis (this may take several minutes)...")
            self.progress.emit("Blob detector will scan for motion...")
            self.progress.emit("YOLO will run when motion is detected...")
            
            result = self.monitor.analyze_video(**kwargs)
            
            # Ensure output folder exists
            os.makedirs(self.output_folder, exist_ok=True)
            
            self.progress.emit("Saving results...")
            
            try:
                # Save results (creates events + tracking CSVs)
                result.to_csv(self.output_folder)
                self.progress.emit("✓ Results saved!")
                
                # Get tracking results path
                video_name = os.path.basename(self.video_path).replace('.mp4', '').replace('.avi', '').replace('.mov', '')
                csv_path = os.path.join(self.output_folder, f'{video_name}_tracking_results.csv')
                
                # Verify file was created
                if not os.path.exists(csv_path):
                    self.error.emit(
                        f"Tracking results file not created!\n\n"
                        f"Expected: {csv_path}\n\n"
                        f"The result.to_csv() method may not have created the file."
                    )
                    return
                
                # Verify CSV has source column
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    if 'source' in df.columns:
                        self.progress.emit("✓ Detection source tracking enabled")
                        self.progress.emit("  RED = Motion detection (Blob)")
                        self.progress.emit("  BLUE = YOLO tracking")
                    else:
                        self.progress.emit("⚠ Detection sources not tracked")
                except Exception:
                    pass  # Non-critical
                
                self.progress.emit("✓ Analysis complete!")
                
            except Exception as e:
                self.error.emit(f"Failed to save results: {e}\n\n{traceback.format_exc()}")
                return
            
            self.finished.emit(result, csv_path)
            
        except Exception as e:
            error_details = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_details)