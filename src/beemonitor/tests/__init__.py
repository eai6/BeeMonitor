"""BeeMonitor Test Suite

Comprehensive testing for detection and tracking systems.

Test Modules:
- test_detectors: Unit tests for detection components
- test_tracking: Integration tests for tracking system
- benchmark_performance: Performance benchmarks
- visual_test: Interactive visual testing tool

Quick Start:
    # Run all tests
    python tests/run_tests.py
    
    # Run specific tests
    python tests/run_tests.py --unit
    python tests/run_tests.py --integration
    python tests/run_tests.py --benchmark
    
    # Visual testing
    python tests/visual_test.py [video_path]

See tests/README.md for detailed documentation.
"""

__all__ = [
    'test_detectors',
    'test_tracking',
    'benchmark_performance',
    'visual_test',
]
