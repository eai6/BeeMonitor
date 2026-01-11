"""
Analysis Thread
===============

Background thread for running video analysis without blocking GUI.
Uses result.to_csv() method to save both events and tracking results.
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
    
    def __init__(self, monitor, video_path, output_folder, detection_mode='fgbg'):
        """
        Initialize analysis thread.
        
        Args:
            monitor: BeeMonitor instance
            video_path: Path to input video
            output_folder: Path to output directory
            detection_mode: Detection mode ('fgbg', 'fgbg_yolo', 'yolo_only')
        """
        super().__init__()
        self.monitor = monitor
        self.video_path = video_path
        self.output_folder = output_folder
        self.detection_mode = detection_mode
    
    def run(self):
        """Run analysis in background thread."""
        try:
            self.progress.emit("Initializing analysis...")
            self.progress.emit("Nest detector will automatically detect hotel ROI...")
            
            # Check if analyze_video accepts these parameters
            sig = inspect.signature(self.monitor.analyze_video)
            
            # Build kwargs based on what's accepted
            kwargs = {'video_path': self.video_path}
            
            # Don't pass output_folder - we'll save manually using result.to_csv()
            if 'visualize' in sig.parameters:
                kwargs['visualize'] = True
            
            # Pass detection mode if supported
            if 'detection_mode' in sig.parameters:
                kwargs['detection_mode'] = self.detection_mode
                self.progress.emit(f"Using detection mode: {self.detection_mode}")
            
            self.progress.emit("Running analysis (this may take several minutes)...")
            result = self.monitor.analyze_video(**kwargs)
            
            # Ensure output folder exists
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Use the result.to_csv() method which saves BOTH events AND tracking
            self.progress.emit("Saving results...")
            
            try:
                # This will create:
                #   - <video_name>_events.csv
                #   - <video_name>_tracking_results.csv
                result.to_csv(self.output_folder)
                self.progress.emit("✓ Results saved!")
                
                # Return path to tracking results for GUI
                video_name = os.path.basename(self.video_path).replace('.mp4', '').replace('.avi', '').replace('.mov', '')
                csv_path = os.path.join(self.output_folder, f'{video_name}_tracking_results.csv')
                
                # Verify the file was created
                if not os.path.exists(csv_path):
                    self.error.emit(
                        f"Tracking results file not created!\n\n"
                        f"Expected: {csv_path}\n\n"
                        f"The result.to_csv() method may not have created the file."
                    )
                    return
                
                # Verify CSV has source column (for color-coded detection sources)
                try:
                    df = pd.read_csv(csv_path, nrows=1)
                    if 'source' in df.columns:
                        self.progress.emit("✓ Detection source tracking enabled (RED/GREEN/BLUE)")
                    else:
                        self.progress.emit("⚠ Detection sources not tracked (color-coding unavailable)")
                except Exception:
                    pass  # Non-critical verification
                
                self.progress.emit("✓ Analysis complete!")
                
            except Exception as e:
                self.error.emit(f"Failed to save results: {e}\n\n{traceback.format_exc()}")
                return
            
            self.finished.emit(result, csv_path)
            
        except Exception as e:
            error_details = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_details)