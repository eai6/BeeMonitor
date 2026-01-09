"""BeeMonitor Testing Suite

This directory contains comprehensive tests for the BeeMonitor detection and tracking system.

═══════════════════════════════════════════════════════════════════════════
QUICK START
═══════════════════════════════════════════════════════════════════════════

Run all tests:
```bash
python tests/run_tests.py
```

Run specific test suites:
```bash
python tests/run_tests.py --unit           # Unit tests only
python tests/run_tests.py --integration    # Integration tests only
python tests/run_tests.py --benchmark      # Performance benchmarks
python tests/run_tests.py --visual         # Visual testing tool
```

═══════════════════════════════════════════════════════════════════════════
TEST FILES
═══════════════════════════════════════════════════════════════════════════

1. test_detectors.py
   - Unit tests for detection module
   - Tests BlobDetector, SIFTDetector, YOLODetector
   - Validates Detection data class
   - Includes visual comparison test
   
   Run: python tests/test_detectors.py

2. test_tracking.py
   - Integration tests for tracking system
   - Tests BeeTracker MOT algorithm
   - Tests BeeTracking system with different modes
   - Tests track association and prediction
   
   Run: python tests/test_tracking.py

3. benchmark_performance.py
   - Performance benchmarks for all components
   - Compares detection modes (FGBG, SIFT, etc.)
   - Compares MOT algorithms
   - Measures FPS and throughput
   
   Run: python tests/benchmark_performance.py

4. visual_test.py
   - Interactive visual testing tool
   - Real-time detection/tracking visualization
   - Switch modes on-the-fly
   - Useful for debugging and understanding system
   
   Run: python tests/visual_test.py [video_path]
   
   Controls:
   - q: Quit
   - p: Pause/Resume
   - s: Step frame (when paused)
   - r: Reset tracker
   - h: Toggle help
   - 1-7: Switch detection modes

5. run_tests.py
   - Master test runner
   - Runs all test suites
   - Provides summary report
   
   Run: python tests/run_tests.py

═══════════════════════════════════════════════════════════════════════════
TESTING GUIDE
═══════════════════════════════════════════════════════════════════════════

See TESTING_GUIDE.md for comprehensive testing strategies including:
- Unit testing individual components
- Integration testing full pipeline
- Visual testing and debugging
- Performance benchmarking
- Comparison testing
- Stationary bee detection tests
- Noise filter validation

═══════════════════════════════════════════════════════════════════════════
TEST WORKFLOW
═══════════════════════════════════════════════════════════════════════════

Recommended testing workflow:

1. Start with Visual Testing
   ```bash
   python tests/visual_test.py
   ```
   - See what's happening in real-time
   - Try different detection modes (press 1-7)
   - Verify detections look correct

2. Run Unit Tests
   ```bash
   python tests/run_tests.py --unit
   ```
   - Verify individual components work
   - Check detector outputs

3. Run Integration Tests
   ```bash
   python tests/run_tests.py --integration
   ```
   - Test full pipeline
   - Verify modes work together

4. Run Benchmarks
   ```bash
   python tests/run_tests.py --benchmark
   ```
   - Measure performance
   - Compare different modes
   - Choose optimal configuration

═══════════════════════════════════════════════════════════════════════════
TESTING WITH YOUR DATA
═══════════════════════════════════════════════════════════════════════════

To test with your own videos:

1. Visual Testing
   ```bash
   python tests/visual_test.py /path/to/your/video.mp4
   ```

2. Benchmarking
   ```bash
   python tests/benchmark_performance.py /path/to/your/video.mp4
   ```

3. Custom Tests
   ```python
   from beemonitor.tracking import BeeTracking, DetectionMode
   from beemonitor.tracking.mot import BeeTracker
   from beemonitor.core.config import Config
   
   config = Config.default()
   mot = BeeTracker(config, ['bee'])
   
   tracker = BeeTracking(
       mot_algorithm=mot,
       detection_mode=DetectionMode.FGBG_SIFT_YOLO,
       config=config
   )
   
   results = tracker.process_video(
       'your_video.mp4',
       roi=(100, 100, 800, 600)
   )
   ```

═══════════════════════════════════════════════════════════════════════════
UNDERSTANDING TEST RESULTS
═══════════════════════════════════════════════════════════════════════════

Unit Tests:
- ✓ All detectors create valid Detection objects
- ✓ Detectors can be configured and reset
- ✓ Detections have correct attributes

Integration Tests:
- ✓ BeeTracking system processes videos
- ✓ Different modes work correctly
- ✓ Trackers maintain state properly

Benchmarks:
- FPS: Frames per second processed
- Detection/frame: Average detections per frame
- Speed comparison: Relative performance

Visual Tests:
- Green boxes: Detections
- Blue boxes: Tracks with IDs
- Trajectories: Track history

═══════════════════════════════════════════════════════════════════════════
DEBUGGING FAILED TESTS
═══════════════════════════════════════════════════════════════════════════

If tests fail:

1. Check dependencies are installed:
   - opencv-python
   - numpy
   - pandas
   - ultralytics (for YOLO modes)

2. Verify models are available:
   - YOLO models required for YOLO modes
   - CNN classifier for noise filtering

3. Check file paths:
   - Test videos are created in /tmp
   - Ensure write permissions

4. Run visual test to see what's happening:
   ```bash
   python tests/visual_test.py
   ```

5. Check error messages:
   - ImportError: Missing dependencies
   - FileNotFoundError: Missing models/files
   - ValueError: Configuration issues

═══════════════════════════════════════════════════════════════════════════
ADDING NEW TESTS
═══════════════════════════════════════════════════════════════════════════

To add new tests:

1. Create test file in tests/ directory
2. Import unittest and components to test
3. Create TestCase classes
4. Add test methods (must start with test_)
5. Add to run_tests.py if desired

Example:
```python
import unittest
from beemonitor.detection import BlobDetector

class TestMyFeature(unittest.TestCase):
    def test_something(self):
        detector = BlobDetector(min_area=50)
        self.assertIsNotNone(detector)

if __name__ == '__main__':
    unittest.main()
```

═══════════════════════════════════════════════════════════════════════════
CONTINUOUS INTEGRATION
═══════════════════════════════════════════════════════════════════════════

For CI/CD pipelines:

```bash
# Run tests without visual components
python tests/run_tests.py --unit --integration

# Run benchmarks
python tests/run_tests.py --benchmark
```

Exit code 0 = all tests passed
Exit code 1 = some tests failed

═══════════════════════════════════════════════════════════════════════════
"""
