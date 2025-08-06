"""Extract VenusREM zero-shot fitness predictions for ProteinGym datasets"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import venusrem_fitness
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from venusrem_fitness import VenusREMFitness


def _filter_datasets(cfg: DictConfig, zero_shot_dir: Path) -> pd.DataFrame:
    """Filter datasets based on configuration and existing files."""
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    
    # Filter by mode
    if cfg.data.embedding.mode == "singles":
        df_ref = df_ref[df_ref["DMS_id"].str.contains("singles")]
    elif cfg.data.embedding.mode == "multiples":
        df_ref = df_ref[df_ref["DMS_id"].str.contains("multiples")]
    
    # Filter out datasets that already have zero-shot predictions
    existing_files = set()
    if zero_shot_dir.exists():
        existing_files = {f.stem for f in zero_shot_dir.glob("*.csv")}
    
    df_ref = df_ref[~df_ref["DMS_id"].isin(existing_files)]
    
    print(f"Found {len(df_ref)} datasets to process")
    return df_ref


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="proteingym_gpr",
)
def extract_venusrem_zero_shot(cfg: DictConfig) -> None:
    """Extract VenusREM zero-shot fitness predictions."""
    print("Initializing VenusREM zero-shot extraction...")
    
    # Setup paths
    zero_shot_dir = Path("data/zero_shot_fitness_predictions/VenusREM")
    zero_shot_dir.mkdir(parents=True, exist_ok=True)
    
    match cfg.data.embedding.mode:
        case "singles":
            DMS_dir = Path("data/substitutions_singles")
        case "multiples":
            DMS_dir = Path("data/substitutions_multiples")
        case _:
            raise ValueError(f"Invalid mode: {cfg.data.embedding.mode}")
    
    # Filter datasets
    df_ref = _filter_datasets(cfg, zero_shot_dir)
    
    if len(df_ref) == 0:
        print("All datasets already have zero-shot predictions. Exiting.")
        return
    
    # Initialize VenusREM model
    print("Initializing VenusREM model...")
    use_gpu = torch.cuda.is_available() and cfg.use_gpu
    device = "cuda" if use_gpu else "cpu"
    
    # Set PyTorch optimizations for GPU
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        model_name = cfg.data.embedding.model_name
        alpha = cfg.data.embedding.alpha
        logit_mode = cfg.data.embedding.logit_mode
        
        model = VenusREMFitness(
            model_name=model_name,
            device=device,
            alpha=alpha,
            logit_mode=logit_mode
        )
        print(f"VenusREM model loaded on {device}")
        print(f"Model: {model_name}, Alpha: {alpha}, Logit mode: {logit_mode}")
    except Exception as e:
        print(f"Error initializing VenusREM model: {e}")
        return

    for i, DMS_id in tqdm(enumerate(df_ref["DMS_id"])):
        print(f"--- Extracting VenusREM zero-shot for {DMS_id} ({i+1}/{len(df_ref)}) ---")
        
        try:
            df = pd.read_csv(DMS_dir / f"{DMS_id}.csv")
            
            # Get wild-type sequence from reference
            dms_ref = df_ref[df_ref["DMS_id"] == DMS_id].iloc[0]
            wt_sequence = dms_ref["target_seq"]
            
            # Create placeholder structure sequence (all alpha helices)
            structure_sequence = [1] * len(wt_sequence)  # 1 represents alpha helix
            
            mutants = df["mutant"].tolist()
            sequences = df["mutated_sequence"].tolist()
            
            all_scores = []
            
            # Process sequences in batches for better GPU utilization
            batch_size = getattr(cfg.data.embedding, 'toks_per_batch', 64)
            
            for batch_start in range(0, len(sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(sequences))
                batch_sequences = sequences[batch_start:batch_end]
                
                try:
                    # Process batch
                    batch_scores = []
                    for seq in batch_sequences:
                        try:
                            # Get fitness score from VenusREM model
                            score = model.get_fitness_score(
                                residue_sequence=seq,
                                structure_sequence=structure_sequence
                            )
                            batch_scores.append(score)
                        except Exception as e:
                            print(f"Error processing sequence: {e}")
                            batch_scores.append(0.0)  # Default score
                    
                    all_scores.extend(batch_scores)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Fallback to individual processing
                    for seq in batch_sequences:
                        try:
                            score = model.get_fitness_score(
                                residue_sequence=seq,
                                structure_sequence=structure_sequence
                            )
                            all_scores.append(score)
                        except Exception as seq_e:
                            print(f"Error processing sequence: {seq_e}")
                            all_scores.append(0.0)
            
            # Create output DataFrame
            df_zero_shot = pd.DataFrame({
                "mutant": mutants,
                "venusrem_score": all_scores
            })
            
            # Save zero-shot predictions
            output_file = zero_shot_dir / f"{DMS_id}.csv"
            df_zero_shot.to_csv(output_file, index=False)
            
            print(f"Saved VenusREM zero-shot predictions for {DMS_id} to {output_file}")
            print(f"Processed {len(all_scores)} sequences")
            
        except Exception as e:
            print(f"Error processing {DMS_id}: {e}")
            continue


if __name__ == "__main__":
    extract_venusrem_zero_shot() 