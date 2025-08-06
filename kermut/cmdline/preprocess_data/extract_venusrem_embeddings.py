"""Script to extract VenusREM embeddings for ProteinGym DMS assays.
Adapted from extract_esm2_embeddings.py but using VenusREM model."""

from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import h5py
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np

# Import VenusREM model
from venusrem_fitness import VenusREMFitness


def prepare_dataset_data(args):
    """Prepare data for a single dataset (CPU-only work)."""
    DMS_id, df_ref, DMS_dir = args
    
    try:
        df = pd.read_csv(DMS_dir / f"{DMS_id}.csv")
        
        # Get wild-type sequence from reference
        dms_ref = df_ref[df_ref["DMS_id"] == DMS_id].iloc[0]
        wt_sequence = dms_ref["target_seq"]
        
        # Create placeholder structure sequence (all alpha helices)
        structure_sequence = [1] * len(wt_sequence)  # 1 represents alpha helix
        
        mutants = df["mutant"].tolist()
        sequences = df["mutated_sequence"].tolist()
        
        return DMS_id, {
            'mutants': mutants,
            'sequences': sequences,
            'structure_sequence': structure_sequence,
            'wt_sequence': wt_sequence
        }
        
    except Exception as e:
        print(f"Error preparing data for {DMS_id}: {e}")
        return DMS_id, None


def process_dataset_with_gpu(args):
    """Process a single dataset with a specific GPU."""
    DMS_id, data, embedding_dir, gpu_id, model_config = args
    
    try:
        # Set the GPU device for this process
        torch.cuda.set_device(gpu_id)
        
        # Initialize model on this GPU
        model_config['device'] = f'cuda:{gpu_id}'
        model = VenusREMFitness(**model_config)
        
        mutants = data['mutants']
        sequences = data['sequences']
        structure_sequence = data['structure_sequence']
        
        all_labels = []
        all_representations = []
        
        # Process sequences in batches
        batch_size = 64  # Optimized batch size for GPU
        
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            batch_mutants = mutants[batch_start:batch_end]
            
            try:
                batch_hidden = []
                for seq in batch_sequences:
                    hidden = model.get_hidden_representations(
                        residue_sequence=seq,
                        structure_sequence=structure_sequence
                    )
                    batch_hidden.append(hidden)
                
                if batch_hidden:
                    stacked_hidden = torch.stack(batch_hidden, dim=0)
                    batch_embeddings = stacked_hidden.mean(dim=1)
                    batch_representations = batch_embeddings.detach().cpu().numpy()
                else:
                    batch_representations = []
                    
            except Exception as e:
                print(f"Error processing batch for {DMS_id} on GPU {gpu_id}: {e}")
                batch_representations = []
                for seq in batch_sequences:
                    try:
                        hidden = model.get_hidden_representations(
                            residue_sequence=seq,
                            structure_sequence=structure_sequence
                        )
                        embedding = hidden.mean(dim=0).detach().cpu().numpy()
                        batch_representations.append(embedding)
                    except Exception as seq_e:
                        print(f"Error processing sequence for {DMS_id} on GPU {gpu_id}: {seq_e}")
                        hidden_dim = model.model.config.hidden_size
                        embedding = np.zeros(hidden_dim)
                        batch_representations.append(embedding)
            
            all_labels.extend(batch_mutants)
            all_representations.extend(batch_representations)
        
        assert mutants == all_labels, "Labels mismatch"
        
        embeddings_dict = {
            "embeddings": all_representations,
            "mutants": mutants,
        }

        # Store data as HDF5
        output_file = embedding_dir / f"{DMS_id}.h5"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, "w") as h5f:
            for key, value in embeddings_dict.items():
                h5f.create_dataset(key, data=value)
        
        print(f"Saved VenusREM embeddings for {DMS_id} to {output_file} (GPU {gpu_id})")
        print(f"Embedding shape: {len(all_representations)} sequences, {len(all_representations[0])} features")
        
        return DMS_id, True, None
        
    except Exception as e:
        print(f"Error processing {DMS_id} on GPU {gpu_id}: {e}")
        return DMS_id, False, str(e)


def _filter_datasets(cfg: DictConfig, embedding_dir: Path) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    match cfg.dataset:
        case "all":
            if cfg.data.embedding.mode == "multiples":
                df_ref = df_ref[df_ref["includes_multiple_mutants"]]
                df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
        case "benchmark":
            # For benchmark, use all datasets except very large ones
            large_datasets = [
                "POLG_CXB3N_Mattenberger_2021",
                "POLG_DEN26_Suphatrakul_2023",
            ]
            df_ref = df_ref[~df_ref["DMS_id"].isin(large_datasets)]
        case "single":
            if cfg.single.use_id:
                df_ref = df_ref[df_ref["DMS_id"] == cfg.single.id]
            else:
                df_ref = df_ref.iloc[[cfg.single.id]]
        case _:
            raise ValueError(f"Invalid dataset: {cfg.dataset}")

    if not cfg.overwrite:
        existing_results = []
        for DMS_id in df_ref["DMS_id"]:
            output_file = embedding_dir / f"{DMS_id}.h5"
            if output_file.exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="proteingym_gpr",
)
def extract_venusrem_embeddings(cfg: DictConfig) -> None:
    # Override embedding configuration to use VenusREM
    cfg.data.embedding = cfg.data.embedding.venusrem if hasattr(cfg.data.embedding, 'venusrem') else cfg.data.embedding
    
    match cfg.data.embedding.mode:
        case "singles":
            embedding_dir = Path(cfg.data.paths.venusrem_embeddings_singles)
            DMS_dir = Path("data/substitutions_singles")
        case "multiples":
            embedding_dir = Path(cfg.data.paths.venusrem_embeddings_multiples)
            DMS_dir = Path("data/substitutions_multiples")
        case _:
            raise ValueError(f"Invalid mode: {cfg.data.embedding.mode}")

    df_ref = _filter_datasets(cfg, embedding_dir)

    if len(df_ref) == 0:
        print("All embeddings already exist. Exiting.")
        return

    # Initialize VenusREM model
    print("Initializing VenusREM model...")
    use_gpu = torch.cuda.is_available() and cfg.use_gpu
    device = "cuda" if use_gpu else "cpu"
    
    # Check for multiple GPUs
    n_gpus = torch.cuda.device_count() if use_gpu else 0
    print(f"Found {n_gpus} GPU(s)")
    
    # Set PyTorch to use mixed precision for faster computation on GPU
    if use_gpu:
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for faster computation
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        # Use configuration parameters if available, otherwise use defaults
        model_name = getattr(cfg.data.embedding, 'model_name', 'AI4Protein/ProSST-2048')
        alpha = getattr(cfg.data.embedding, 'alpha', 0.7)
        logit_mode = getattr(cfg.data.embedding, 'logit_mode', 'aa_seq_aln')
        
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

    # Prepare arguments for data preparation (CPU-only work)
    prepare_args = [
        (DMS_id, df_ref, DMS_dir)
        for DMS_id in df_ref["DMS_id"]
    ]
    
    # Determine number of processes for data preparation
    n_processes = getattr(cfg.data.embedding, 'n_processes', min(cpu_count() - 1, 8))
    n_processes = min(n_processes, cpu_count() - 1)  # Ensure we don't exceed available CPUs
    print(f"Using {n_processes} processes for data preparation")
    
    # Prepare all dataset data in parallel (CPU-only work)
    print("Preparing dataset data in parallel...")
    try:
        with Pool(processes=n_processes) as pool:
            prepared_data = list(tqdm(
                pool.imap(prepare_dataset_data, prepare_args),
                total=len(prepare_args),
                desc="Preparing data"
            ))
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to single-threaded data preparation...")
        prepared_data = []
        for args in tqdm(prepare_args, desc="Preparing data (single-threaded)"):
            prepared_data.append(prepare_dataset_data(args))
    
    # Filter out failed data preparation
    valid_data = [(DMS_id, data) for DMS_id, data in prepared_data if data is not None]
    print(f"Successfully prepared data for {len(valid_data)} datasets")
    
    # Process datasets with multiple GPUs if available
    if n_gpus > 1:
        print(f"Processing datasets with {n_gpus} GPUs in parallel...")
        
        # Prepare model configuration
        model_config = {
            'model_name': getattr(cfg.data.embedding, 'model_name', 'AI4Protein/ProSST-2048'),
            'alpha': getattr(cfg.data.embedding, 'alpha', 0.7),
            'logit_mode': getattr(cfg.data.embedding, 'logit_mode', 'aa_seq_aln')
        }
        
        # Distribute datasets across GPUs
        gpu_args = []
        for i, (DMS_id, data) in enumerate(valid_data):
            gpu_id = i % n_gpus  # Round-robin distribution
            gpu_args.append((DMS_id, data, embedding_dir, gpu_id, model_config))
        
        # Process with multiprocessing (each process uses a different GPU)
        n_processes = min(n_gpus, 4)  # Limit processes to avoid memory issues
        print(f"Using {n_processes} processes for multi-GPU processing")
        
        try:
            with Pool(processes=n_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_dataset_with_gpu, gpu_args),
                    total=len(gpu_args),
                    desc="Processing with multiple GPUs"
                ))
        except Exception as e:
            print(f"Multi-GPU processing failed: {e}")
            print("Falling back to single-GPU processing...")
            results = []
            for args in tqdm(gpu_args, desc="Processing with single GPU"):
                results.append(process_dataset_with_gpu(args))
        
        # Count results
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
    else:
        # Single GPU or CPU processing
        print("Processing datasets with single GPU/CPU...")
        batch_size = getattr(cfg.data.embedding, 'toks_per_batch', 128)  # Larger batch size for GPU
        
        successful = 0
        failed = 0
        
        for DMS_id, data in tqdm(valid_data, desc="Processing with single GPU/CPU"):
            try:
                mutants = data['mutants']
                sequences = data['sequences']
                structure_sequence = data['structure_sequence']
                
                all_labels = []
                all_representations = []
                
                # Process sequences in larger batches for GPU efficiency
                for batch_start in range(0, len(sequences), batch_size):
                    batch_end = min(batch_start + batch_size, len(sequences))
                    batch_sequences = sequences[batch_start:batch_end]
                    batch_mutants = mutants[batch_start:batch_end]
                    
                    try:
                        # Process entire batch at once for better GPU utilization
                        batch_hidden = []
                        for seq in batch_sequences:
                            # Get hidden representations from VenusREM model
                            hidden = model.get_hidden_representations(
                                residue_sequence=seq,
                                structure_sequence=structure_sequence
                            )
                            batch_hidden.append(hidden)
                        
                        # Stack all hidden representations and compute mean pooling in one operation
                        if batch_hidden:
                            stacked_hidden = torch.stack(batch_hidden, dim=0)  # [batch_size, seq_len, hidden_dim]
                            # Mean pooling across sequence dimension
                            batch_embeddings = stacked_hidden.mean(dim=1)  # [batch_size, hidden_dim]
                            batch_representations = batch_embeddings.detach().cpu().numpy()
                        else:
                            batch_representations = []
                            
                    except Exception as e:
                        print(f"Error processing batch for {DMS_id}: {e}")
                        # Fallback to individual processing
                        batch_representations = []
                        for seq in batch_sequences:
                            try:
                                hidden = model.get_hidden_representations(
                                    residue_sequence=seq,
                                    structure_sequence=structure_sequence
                                )
                                embedding = hidden.mean(dim=0).detach().cpu().numpy()
                                batch_representations.append(embedding)
                            except Exception as seq_e:
                                print(f"Error processing sequence for {DMS_id}: {seq_e}")
                                hidden_dim = model.model.config.hidden_size
                                embedding = np.zeros(hidden_dim)
                                batch_representations.append(embedding)
                    
                    all_labels.extend(batch_mutants)
                    all_representations.extend(batch_representations)
                
                assert mutants == all_labels, "Labels mismatch"
                
                embeddings_dict = {
                    "embeddings": all_representations,
                    "mutants": mutants,
                }

                # Store data as HDF5
                output_file = embedding_dir / f"{DMS_id}.h5"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with h5py.File(output_file, "w") as h5f:
                    for key, value in embeddings_dict.items():
                        h5f.create_dataset(key, data=value)
                
                print(f"Saved VenusREM embeddings for {DMS_id} to {output_file}")
                print(f"Embedding shape: {len(all_representations)} sequences, {len(all_representations[0])} features")
                
                successful += 1
                
            except Exception as e:
                print(f"Error processing {DMS_id}: {e}")
                failed += 1
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    extract_venusrem_embeddings() 