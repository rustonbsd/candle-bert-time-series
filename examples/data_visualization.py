#!/usr/bin/env python3
"""
Data Visualization and Sanity Checks for Crypto Dataset
Visualizes 4e10x rows and 220~ currencies as a 2D heatmap
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Tuple, Optional

def load_dataset(file_path: str) -> pl.DataFrame:
    """Load the processed dataset from parquet file"""
    print(f"Loading dataset from: {file_path}")
    df = pl.read_parquet(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns[:10]}...")  # Show first 10 columns
    return df

def analyze_data_coverage(df: pl.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Analyze data coverage across time and currencies
    Returns coverage matrix and currency names
    """
    print("Analyzing data coverage...")
    
    # Get currency names (column names ending with '_return')
    currency_cols = [col for col in df.columns if col.endswith('_return')]
    print(f"Found {len(currency_cols)} currencies")
    
    # Convert to numpy for faster processing
    data_matrix = df.select(currency_cols).to_numpy()
    
    # Create coverage matrix (1 where data exists, 0 where missing/zero)
    coverage_matrix = np.where(np.isnan(data_matrix) | (data_matrix == 0), 0, 1)
    
    print(f"Coverage matrix shape: {coverage_matrix.shape}")
    print(f"Overall data coverage: {np.mean(coverage_matrix):.2%}")
    
    return coverage_matrix, currency_cols

def create_coverage_heatmap(coverage_matrix: np.ndarray, 
                          currency_names: list,
                          sample_rate: int = 1000,
                          max_currencies: int = 50) -> None:
    """
    Create a 2D heatmap showing data coverage
    """
    print(f"Creating coverage heatmap...")
    
    # Sample data for visualization (too large to plot all points)
    n_rows, n_cols = coverage_matrix.shape
    
    # Sample rows (time dimension)
    if n_rows > sample_rate:
        row_indices = np.linspace(0, n_rows-1, sample_rate, dtype=int)
        sampled_matrix = coverage_matrix[row_indices, :]
        print(f"Sampled {sample_rate} time points from {n_rows} total")
    else:
        sampled_matrix = coverage_matrix
        row_indices = np.arange(n_rows)
    
    # Sample columns (currency dimension) if too many
    if n_cols > max_currencies:
        col_indices = np.linspace(0, n_cols-1, max_currencies, dtype=int)
        sampled_matrix = sampled_matrix[:, col_indices]
        sampled_currencies = [currency_names[i] for i in col_indices]
        print(f"Sampled {max_currencies} currencies from {n_cols} total")
    else:
        sampled_currencies = currency_names
        col_indices = np.arange(n_cols)
    
    # Create the heatmap
    plt.figure(figsize=(16, 12))
    
    # Use a custom colormap where 0 (missing) is black and 1 (present) is bright
    cmap = plt.cm.colors.ListedColormap(['black', 'yellow'])
    
    sns.heatmap(sampled_matrix.T,  # Transpose so currencies are on y-axis
                cmap=cmap,
                cbar_kws={'label': 'Data Present (1) / Missing (0)'},
                xticklabels=False,  # Too many time points to label
                yticklabels=[name.replace('_return', '') for name in sampled_currencies])
    
    plt.title(f'Crypto Data Coverage Heatmap\n'
              f'{sampled_matrix.shape[0]} time points × {sampled_matrix.shape[1]} currencies\n'
              f'Yellow = Data Present, Black = Missing/Zero')
    plt.xlabel('Time (sampled points)')
    plt.ylabel('Cryptocurrency')
    plt.tight_layout()
    
    # Save the plot
    output_path = 'data_coverage_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    plt.show()

def print_data_statistics(df: pl.DataFrame, coverage_matrix: np.ndarray, currency_names: list) -> None:
    """Print comprehensive data statistics"""
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    
    # Basic shape info
    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Total data points: {df.shape[0] * df.shape[1]:,}")
    
    # Coverage statistics
    total_coverage = np.mean(coverage_matrix)
    print(f"Overall data coverage: {total_coverage:.2%}")
    print(f"Missing/zero data points: {(1-total_coverage)*100:.2%}")
    
    # Per-currency statistics
    currency_coverage = np.mean(coverage_matrix, axis=0)
    best_coverage_idx = np.argmax(currency_coverage)
    worst_coverage_idx = np.argmin(currency_coverage)
    
    print(f"\nBest coverage currency: {currency_names[best_coverage_idx]} ({currency_coverage[best_coverage_idx]:.2%})")
    print(f"Worst coverage currency: {currency_names[worst_coverage_idx]} ({currency_coverage[worst_coverage_idx]:.2%})")
    print(f"Average currency coverage: {np.mean(currency_coverage):.2%}")
    
    # Time-based statistics
    time_coverage = np.mean(coverage_matrix, axis=1)
    print(f"\nBest time period coverage: {np.max(time_coverage):.2%}")
    print(f"Worst time period coverage: {np.min(time_coverage):.2%}")
    print(f"Average time period coverage: {np.mean(time_coverage):.2%}")
    
    # Distribution of coverage
    coverage_bins = np.histogram(currency_coverage, bins=10)[0]
    print(f"\nCurrency coverage distribution:")
    for i, count in enumerate(coverage_bins):
        bin_start = i * 0.1
        bin_end = (i + 1) * 0.1
        print(f"  {bin_start:.0%}-{bin_end:.0%}: {count} currencies")

def create_coverage_summary_plots(coverage_matrix: np.ndarray, currency_names: list) -> None:
    """Create summary plots showing coverage patterns"""
    
    # Calculate coverage per currency and per time
    currency_coverage = np.mean(coverage_matrix, axis=0)
    time_coverage = np.mean(coverage_matrix, axis=1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Currency coverage histogram
    ax1.hist(currency_coverage, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Data Coverage Ratio')
    ax1.set_ylabel('Number of Currencies')
    ax1.set_title('Distribution of Data Coverage by Currency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time coverage over time (sampled)
    sample_indices = np.linspace(0, len(time_coverage)-1, min(1000, len(time_coverage)), dtype=int)
    ax2.plot(sample_indices, time_coverage[sample_indices], alpha=0.7, color='orange')
    ax2.set_xlabel('Time Index (sampled)')
    ax2.set_ylabel('Data Coverage Ratio')
    ax2.set_title('Data Coverage Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 20 currencies by coverage
    top_20_indices = np.argsort(currency_coverage)[-20:]
    top_20_names = [currency_names[i].replace('_return', '') for i in top_20_indices]
    top_20_coverage = currency_coverage[top_20_indices]
    
    ax3.barh(range(len(top_20_names)), top_20_coverage, color='lightgreen')
    ax3.set_yticks(range(len(top_20_names)))
    ax3.set_yticklabels(top_20_names, fontsize=8)
    ax3.set_xlabel('Data Coverage Ratio')
    ax3.set_title('Top 20 Currencies by Data Coverage')
    ax3.grid(True, alpha=0.3)
    
    # 4. Bottom 20 currencies by coverage
    bottom_20_indices = np.argsort(currency_coverage)[:20]
    bottom_20_names = [currency_names[i].replace('_return', '') for i in bottom_20_indices]
    bottom_20_coverage = currency_coverage[bottom_20_indices]
    
    ax4.barh(range(len(bottom_20_names)), bottom_20_coverage, color='lightcoral')
    ax4.set_yticks(range(len(bottom_20_names)))
    ax4.set_yticklabels(bottom_20_names, fontsize=8)
    ax4.set_xlabel('Data Coverage Ratio')
    ax4.set_title('Bottom 20 Currencies by Data Coverage')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the summary plots
    output_path = 'data_coverage_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary plots saved to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize crypto dataset coverage')
    parser.add_argument('--data-path', 
                       default='/home/i3/Downloads/processed_dataset.parquet',
                       help='Path to the processed dataset parquet file')
    parser.add_argument('--sample-rate', type=int, default=1000,
                       help='Number of time points to sample for heatmap')
    parser.add_argument('--max-currencies', type=int, default=50,
                       help='Maximum number of currencies to show in heatmap')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.data_path).exists():
        print(f"Error: Dataset file not found at {args.data_path}")
        return
    
    try:
        # Load and analyze data
        df = load_dataset(args.data_path)
        coverage_matrix, currency_names = analyze_data_coverage(df)
        
        # Print statistics
        print_data_statistics(df, coverage_matrix, currency_names)
        
        # Create visualizations
        create_coverage_heatmap(coverage_matrix, currency_names, 
                              args.sample_rate, args.max_currencies)
        create_coverage_summary_plots(coverage_matrix, currency_names)
        
        print("\n✅ Data visualization complete!")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()
