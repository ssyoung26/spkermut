# This file can be used to perform the benchmark specifically for VenusREM.
# Similar script can be run for ESM2 in the original Kermut paper.

#!/usr/bin/env python3
"""Run Kermut benchmark with VenusREM embeddings"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the benchmark with VenusREM configuration."""
    
    # Check if we're in the right directory
    if not Path("src/experiments/proteingym_benchmark.py").exists():
        print("Error: Please run this script from the kermut root directory")
        sys.exit(1)
    
    # Run the benchmark with VenusREM configuration
    cmd = [
        "python", "src/experiments/proteingym_benchmark.py",
        "--config-name", "proteingym_venusrem"
    ]
    
    print("Running Kermut benchmark with VenusREM embeddings...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("[PASSED] Benchmark completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Benchmark failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
