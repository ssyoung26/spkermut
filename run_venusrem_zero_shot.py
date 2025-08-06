#!/usr/bin/env python3
"""Script to run VenusREM zero-shot tests and generate predictions"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the VenusREM zero-shot tests"""
    print("=" * 60)
    print("Running VenusREM Zero-Shot Tests")
    print("=" * 60)
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, "tests/test_venusrem_zero_shot.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("[PASSED] All tests passed!")
            return True
        else:
            print("[FAILED] Some tests failed!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to run tests: {e}")
        return False

def run_zero_shot_prediction(dataset="all", overwrite=False):
    """Run VenusREM zero-shot prediction for specified dataset"""
    print("=" * 60)
    print(f"Running VenusREM Zero-Shot Prediction for {dataset}")
    print("=" * 60)
    
    try:
        cmd = [
            sys.executable, "src/data/extract_venusrem_zero_shots.py",
            "--dataset", dataset
        ]
        
        if overwrite:
            cmd.append("--overwrite")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"[PASSED] Zero-shot prediction completed for {dataset}!")
            return True
        else:
            print(f"[FAILED] Zero-shot prediction failed for {dataset}!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to run zero-shot prediction: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VenusREM zero-shot tests and predictions")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--predict", action="store_true", help="Run predictions only")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset to process (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    parser.add_argument("--test-model", action="store_true", help="Test model loading only")
    
    args = parser.parse_args()
    
    if args.test_model:
        print("Testing VenusREM model loading...")
        try:
            result = subprocess.run([
                sys.executable, "src/data/extract_venusrem_zero_shots.py",
                "--test"
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                print("[PASSED] Model loading test passed!")
            else:
                print("[FAILED] Model loading test failed!")
                
        except Exception as e:
            print(f"[ERROR] Failed to test model loading: {e}")
        return
    
    if args.test:
        success = run_tests()
        if not success:
            sys.exit(1)
        return
    
    if args.predict:
        success = run_zero_shot_prediction(args.dataset, args.overwrite)
        if not success:
            sys.exit(1)
        return
    
    # Default: run both tests and predictions
    print("Running tests first...")
    test_success = run_tests()
    
    if test_success:
        print("\nTests passed! Running predictions...")
        pred_success = run_zero_shot_prediction(args.dataset, args.overwrite)
        
        if pred_success:
            print("\n[PASSED] All operations completed successfully!")
        else:
            print("\n[FAILED] Predictions failed!")
            sys.exit(1)
    else:
        print("\n[FAILED] Tests failed! Skipping predictions.")
        sys.exit(1)

if __name__ == "__main__":
    main() 