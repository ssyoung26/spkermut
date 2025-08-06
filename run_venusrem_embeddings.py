# This script can be run to generate embeddings with VenusREM. For example, new embeddings can be generated with a different alpha parameter.
# Change configuration as desired in the configuration files.

#!/usr/bin/env python3
"""
Script to run VenusREM embeddings extraction for ProteinGym datasets.
This script sets up the proper configuration and runs the extraction.
"""

import os
import sys
from pathlib import Path
from kermut.cmdline.preprocess_data.extract_venusrem_embeddings import extract_venusrem_embeddings

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main function to run VenusREM embeddings extraction"""
    
    # Set up environment variables if needed
    os.environ['HYDRA_FULL_ERROR'] = '1'
    
    # The script will use Hydra configuration automatically
    extract_venusrem_embeddings()

if __name__ == "__main__":
    main() 
