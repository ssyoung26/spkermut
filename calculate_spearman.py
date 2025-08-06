#!/usr/bin/env python3
"""Calculate Spearman correlation coefficients from Kermut benchmark results"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import glob

def calculate_spearman_for_dataset(pred_file: Path) -> dict:
    """Calculate Spearman correlation for a single dataset."""
    try:
        df = pd.read_csv(pred_file)
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = spearmanr(df['y'], df['y_pred'])
        
        # Calculate additional metrics
        mse = np.mean((df['y'] - df['y_pred'])**2)
        mae = np.mean(np.abs(df['y'] - df['y_pred']))
        
        return {
            'dataset': pred_file.stem,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'mse': mse,
            'mae': mae,
            'n_samples': len(df)
        }
    except Exception as e:
        print(f"Error processing {pred_file}: {e}")
        return None

def main():
    """Calculate Spearman correlations for all benchmark results."""
    
    # Find all prediction files
    pred_dir = Path("results/predictions")
    if not pred_dir.exists():
        print("Error: results/predictions directory not found")
        print("Please run the benchmark first")
        return
    
    # Look for VenusREM results
    pred_files = list(pred_dir.glob("**/*VenusREM*.csv"))
    
    if not pred_files:
        print("No VenusREM prediction files found. Looking for any prediction files...")
        pred_files = list(pred_dir.glob("**/*.csv"))
    
    if not pred_files:
        print("No prediction files found in results/predictions/")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Calculate metrics for each dataset
    results = []
    for pred_file in pred_files:
        result = calculate_spearman_for_dataset(pred_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found")
        return
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by Spearman correlation
    df_results = df_results.sort_values('spearman_corr', ascending=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("VENUSREM + KERMUT BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nTotal datasets: {len(df_results)}")
    print(f"Mean Spearman correlation: {df_results['spearman_corr'].mean():.4f}")
    print(f"Median Spearman correlation: {df_results['spearman_corr'].median():.4f}")
    print(f"Std Spearman correlation: {df_results['spearman_corr'].std():.4f}")
    
    print(f"\nMean MSE: {df_results['mse'].mean():.4f}")
    print(f"Mean MAE: {df_results['mae'].mean():.4f}")
    
    # Print top 10 results
    print(f"\nTop 10 datasets by Spearman correlation:")
    print("-" * 80)
    print(f"{'Dataset':<40} {'Spearman':<10} {'MSE':<10} {'MAE':<10} {'N':<5}")
    print("-" * 80)
    
    for _, row in df_results.head(10).iterrows():
        print(f"{row['dataset']:<40} {row['spearman_corr']:<10.4f} {row['mse']:<10.4f} {row['mae']:<10.4f} {row['n_samples']:<5}")
    
    # Print bottom 10 results
    print(f"\nBottom 10 datasets by Spearman correlation:")
    print("-" * 80)
    print(f"{'Dataset':<40} {'Spearman':<10} {'MSE':<10} {'MAE':<10} {'N':<5}")
    print("-" * 80)
    
    for _, row in df_results.tail(10).iterrows():
        print(f"{row['dataset']:<40} {row['spearman_corr']:<10.4f} {row['mse']:<10.4f} {row['mae']:<10.4f} {row['n_samples']:<5}")
    
    # Save results
    output_file = Path("results/venusrem_kermut_spearman_results.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print datasets with negative correlations
    negative_corr = df_results[df_results['spearman_corr'] < 0]
    if len(negative_corr) > 0:
        print(f"\nDatasets with negative Spearman correlation ({len(negative_corr)}):")
        for _, row in negative_corr.iterrows():
            print(f"  {row['dataset']}: {row['spearman_corr']:.4f}")

if __name__ == "__main__":
    main() 