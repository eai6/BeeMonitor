"""Test Runner for BeeMonitor

Runs all tests and benchmarks.

Usage:
    python tests/run_tests.py [options]

Options:
    --unit          Run unit tests only
    --integration   Run integration tests only
    --benchmark     Run performance benchmarks
    --visual        Run visual testing tool
    --all           Run everything (default)
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run command and print results."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with code {result.returncode}")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def run_unit_tests():
    """Run unit tests."""
    success = True
    success &= run_command(
        "python tests/test_detectors.py",
        "Unit Tests - Detectors"
    )
    success &= run_command(
        "python tests/test_dl_and_filtering.py",
        "Unit Tests - DL Detection & Noise Filtering"
    )
    return success


def run_integration_tests():
    """Run integration tests."""
    success = True
    success &= run_command(
        "python tests/test_tracking.py",
        "Integration Tests - Tracking"
    )
    success &= run_command(
        "python tests/test_sift_dl_integration.py",
        "Integration Tests - SIFT + DL Workflow"
    )
    return success


def run_benchmarks():
    """Run performance benchmarks."""
    return run_command(
        "python tests/benchmark_performance.py",
        "Performance Benchmarks"
    )


def run_visual_test():
    """Run visual testing tool."""
    print("\n" + "="*70)
    print("VISUAL TESTING TOOL")
    print("="*70)
    print("\nStarting interactive visual testing...")
    print("(This will open a window - press 'q' to quit)")
    
    subprocess.run("python tests/visual_test.py", shell=True)


def main():
    """Main test runner."""
    args = sys.argv[1:]
    
    # Determine what to run
    run_all = not args or '--all' in args
    run_unit = run_all or '--unit' in args
    run_integration = run_all or '--integration' in args
    run_bench = run_all or '--benchmark' in args
    run_visual = '--visual' in args
    
    print("="*70)
    print("BEEMONITOR TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Run tests
    if run_unit:
        results['Unit Tests'] = run_unit_tests()
    
    if run_integration:
        results['Integration Tests'] = run_integration_tests()
    
    if run_bench:
        results['Benchmarks'] = run_benchmarks()
    
    if run_visual:
        run_visual_test()
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{name:<30} {status}")
        
        all_passed = all(results.values())
        
        print("="*70)
        if all_passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*70)
        
        return 0 if all_passed else 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
