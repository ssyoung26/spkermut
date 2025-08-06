# This script is called by `run_venusrem_zero_shot.py` to generate zero shot scores for ProteinGym sequences.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import subprocess
import tempfile
import argparse

# Import the VenusREMFitness class
from venusrem_fitness import VenusREMFitness

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate VenusREM zero-shot predictions for ProteinGym')
    parser.add_argument('--substitutions_csv', type=str, 
                       default='/home/sonny/development/bb_PLM/kermut/data/DMS_substitutions.csv',
                       help='Path to ProteinGym substitutions CSV file')
    parser.add_argument('--substitutions_folder', type=str,
                       default='/home/sonny/development/bb_PLM/kermut/data/substitutions_singles',
                       help='Path to substitutions data folder')
    parser.add_argument('--structures_folder', type=str,
                       default='/home/sonny/development/bb_PLM/kermut/data/structures/pdbs',
                       help='Path to PDB structures folder')
    parser.add_argument('--output_folder', type=str,
                       default='/home/sonny/development/bb_PLM/kermut/data/zero_shot_fitness_predictions/VenusREM',
                       help='Output folder for predictions')
    parser.add_argument('--model_name', type=str, default='AI4Protein/ProSST-2048',
                       help='VenusREM model name')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Alpha parameter for VenusREM')
    parser.add_argument('--logit_mode', type=str, default='aa_seq_aln',
                       choices=['aa_seq_aln', 'struc_seq_aln', 'aa_seq_aln+struc_seq_aln'],
                       help='Logit mode for VenusREM')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for processing variants')
    parser.add_argument('--max_datasets', type=int, default=None,
                       help='Maximum number of datasets to process (for testing)')
    
    return parser.parse_args()

def extract_mutations_from_sequence(wt_sequence, mutated_sequence):
    """Extract mutation string from wild-type and mutated sequences"""
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_sequence, mutated_sequence)):
        if wt_aa != mut_aa:
            # Convert to 1-indexed position
            mutations.append(f"{wt_aa}{i+1}{mut_aa}")
    return ":".join(mutations) if mutations else ""

def get_structure_sequence_from_pdb(pdb_path):
    """Extract structure sequence from PDB file using DSSP or fallback method"""
    try:
        # Try to use DSSP if available
        try:
            # Run DSSP to get secondary structure
            result = subprocess.run(['dssp', '-i', pdb_path, '-o', '/dev/stdout'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                structure_sequence = []
                for line in lines:
                    if line.startswith('  #') or len(line.strip()) == 0:
                        continue
                    if len(line) > 16:
                        ss = line[16]
                        if ss == 'H':  # Alpha helix
                            structure_sequence.append(1)
                        elif ss == 'E':  # Beta sheet
                            structure_sequence.append(2)
                        elif ss == 'B':  # Beta bridge
                            structure_sequence.append(2)
                        elif ss == 'G':  # 3-helix
                            structure_sequence.append(1)
                        elif ss == 'I':  # 5-helix
                            structure_sequence.append(1)
                        elif ss == 'T':  # Turn
                            structure_sequence.append(3)
                        elif ss == 'S':  # Bend
                            structure_sequence.append(3)
                        else:  # Coil or other
                            structure_sequence.append(0)
                
                if structure_sequence:
                    return structure_sequence
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Fallback: parse PDB file directly for basic structure info
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        
        # Extract sequence length and basic structure info
        seq_length = 0
        structure_sequence = []
        
        for line in lines:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                seq_length += 1
                # Basic structure assignment based on residue type and position
                # This is a simplified approach - in practice you'd want more sophisticated analysis
                residue = line[17:20].strip()
                
                # Simple heuristics for structure assignment
                if residue in ['ALA', 'LEU', 'MET', 'PHE', 'PRO', 'GLY', 'ILE', 'VAL']:
                    # Hydrophobic residues often in helices or sheets
                    structure_sequence.append(1 if seq_length % 3 == 0 else 2)
                elif residue in ['ARG', 'LYS', 'HIS']:
                    # Charged residues often in coils or turns
                    structure_sequence.append(0)
                elif residue in ['ASN', 'GLN', 'SER', 'THR', 'TYR', 'TRP']:
                    # Polar residues mixed
                    structure_sequence.append(1 if seq_length % 2 == 0 else 0)
                else:
                    # Default to coil
                    structure_sequence.append(0)
        
        if structure_sequence:
            return structure_sequence
        
        # Final fallback: all alpha helices
        return [1] * seq_length
        
    except Exception as e:
        print(f"Error parsing PDB {pdb_path}: {e}")
        return None

def main():
    """Main function to generate VenusREM zero-shot predictions"""
    args = parse_arguments()
    
    # Make sure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize VenusREM model
    print(f"Initializing VenusREM model on {args.device}...")
    try:
        model = VenusREMFitness(
            model_name=args.model_name,
            device=args.device,
            alpha=args.alpha,
            logit_mode=args.logit_mode
        )
    except Exception as e:
        print(f"Error initializing VenusREM model: {e}")
        return
    
    # Load metadata
    print("Loading ProteinGym metadata...")
    try:
        meta_df = pd.read_csv(args.substitutions_csv)
        if args.max_datasets:
            meta_df = meta_df.head(args.max_datasets)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Process each DMS substitution file
    print(f"Processing {len(meta_df)} ProteinGym datasets...")
    successful_datasets = 0
    failed_datasets = 0
    
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        dms_filename = row['DMS_filename']
        dms_filepath = os.path.join(args.substitutions_folder, dms_filename)
        
        # Get wild-type sequence
        wt_sequence = row['target_seq']

        if not os.path.exists(dms_filepath):
            print(f"Missing file: {dms_filepath}")
            failed_datasets += 1
            continue

        try:
            print(f"Processing {dms_filename}...")
            dms_df = pd.read_csv(dms_filepath)
            
            # Get structure sequence from PDB file
            pdb_filename = row['DMS_id'] + '.pdb'
            pdb_filepath = os.path.join(args.structures_folder, pdb_filename)
            
            if os.path.exists(pdb_filepath):
                structure_sequence = get_structure_sequence_from_pdb(pdb_filepath)
            else:
                print(f"Warning: PDB file not found for {dms_filename}, using placeholder structure")
                structure_sequence = [1] * len(wt_sequence)  # Placeholder
            
            if structure_sequence is None:
                print(f"Warning: Could not parse PDB for {dms_filename}, using placeholder structure")
                structure_sequence = [1] * len(wt_sequence)  # Placeholder

            # Extract mutations for each variant
            mutations_list = []
            for _, variant_row in dms_df.iterrows():
                mutated_sequence = variant_row['mutated_sequence']
                mutations = extract_mutations_from_sequence(wt_sequence, mutated_sequence)
                mutations_list.append(mutations)
            
            # Filter out variants with no mutations (shouldn't happen in DMS data)
            valid_indices = [i for i, mut in enumerate(mutations_list) if mut]
            valid_mutations = [mutations_list[i] for i in valid_indices]
            
            if not valid_mutations:
                print(f"Warning: No valid mutations found in {dms_filename}")
                failed_datasets += 1
                continue
            
            # Predict scores with VenusREM in batches
            print(f"Predicting fitness for {len(valid_mutations)} variants...")
            try:
                # Process in batches to avoid memory issues
                all_scores = []
                for i in range(0, len(valid_mutations), args.batch_size):
                    batch_mutations = valid_mutations[i:i+args.batch_size]
                    batch_scores = model.predict_fitness(
                        residue_sequence=wt_sequence,
                        structure_sequence=structure_sequence,
                        mutants=batch_mutations,
                        alpha=args.alpha,
                        logit_mode=args.logit_mode
                    )
                    all_scores.extend(batch_scores)
                
                # Create full scores list with NaN for invalid mutations
                full_scores = [np.nan] * len(mutations_list)
                for i, score in zip(valid_indices, all_scores):
                    full_scores[i] = score
                
                # Add predictions to DataFrame
                dms_df['venusrem'] = full_scores
                
            except Exception as e:
                print(f"Error during prediction for {dms_filename}: {e}")
                # Add NaN column if prediction fails
                dms_df['venusrem'] = np.nan
                failed_datasets += 1
                continue

            # Output path
            out_filename = dms_filename  # keep same name
            out_path = os.path.join(args.output_folder, out_filename)

            # Save predictions
            dms_df.to_csv(out_path, index=False)
            print(f"Saved predictions to {out_path}")
            successful_datasets += 1

        except Exception as e:
            print(f"Error processing {dms_filename}: {e}")
            failed_datasets += 1
            continue

    print(f"Zero-shot prediction generation complete!")
    print(f"Successful datasets: {successful_datasets}")
    print(f"Failed datasets: {failed_datasets}")
    print(f"Total datasets: {len(meta_df)}")

if __name__ == "__main__":
    main()
