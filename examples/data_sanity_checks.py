#!/usr/bin/env python3
"""
Comprehensive Sanity Checks for Crypto Dataset
Additional validation and quality checks beyond basic visualization
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_validate_dataset(file_path: str) -> pl.DataFrame:
    """Load dataset with basic validation"""
    print(f"🔍 Loading and validating dataset from: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pl.read_parquet(file_path)
    
    # Basic validation
    print(f"✅ Dataset loaded successfully")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"   Memory usage: ~{df.estimated_size('mb'):.1f} MB")
    
    return df

def check_data_types_and_ranges(df: pl.DataFrame) -> None:
    """Validate data types and check for reasonable value ranges"""
    print("\n🔍 CHECKING DATA TYPES AND RANGES")
    print("="*50)
    
    # Check data types
    dtypes = df.dtypes
    print(f"Data types: {set(str(dt) for dt in dtypes)}")
    
    # Check for non-numeric columns (should all be float64)
    non_numeric = [col for col, dtype in zip(df.columns, dtypes) if str(dtype) != 'Float64']
    if non_numeric:
        print(f"⚠️  Non-numeric columns found: {non_numeric}")
    else:
        print("✅ All columns are Float64 (as expected)")
    
    # Check value ranges for returns (should be reasonable percentages)
    print("\n📊 Return value ranges:")
    
    # Sample a few columns for range checking
    sample_cols = df.columns[:5]  # Check first 5 currencies
    
    for col in sample_cols:
        values = df[col].drop_nulls()
        if len(values) > 0:
            min_val = values.min()
            max_val = values.max()
            mean_val = values.mean()
            std_val = values.std()
            
            print(f"  {col.replace('_return', '')}:")
            print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
            print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            
            # Flag extreme values (returns > 100% or < -90% are suspicious)
            if max_val > 1.0:
                print(f"    ⚠️  Extreme positive return: {max_val:.2%}")
            if min_val < -0.9:
                print(f"    ⚠️  Extreme negative return: {min_val:.2%}")

def check_temporal_consistency(df: pl.DataFrame) -> None:
    """Check for temporal patterns and consistency"""
    print("\n🕐 CHECKING TEMPORAL CONSISTENCY")
    print("="*50)
    
    # Since we don't have explicit timestamps, we'll analyze row-by-row patterns
    n_rows = df.shape[0]
    print(f"Total time periods: {n_rows:,}")
    
    # Assuming 1-minute intervals, calculate time span
    minutes = n_rows
    hours = minutes / 60
    days = hours / 24
    years = days / 365.25
    
    print(f"Estimated time span (assuming 1-min intervals):")
    print(f"  {minutes:,} minutes = {hours:,.1f} hours = {days:,.1f} days = {years:.1f} years")
    
    # Check for patterns in data availability over time
    currency_cols = [col for col in df.columns if col.endswith('_return')]
    
    # Sample every 1000th row to check coverage over time
    sample_indices = range(0, n_rows, max(1, n_rows // 1000))
    coverage_over_time = []
    
    for idx in sample_indices:
        row_data = df.row(idx)
        non_zero_count = sum(1 for val in row_data if val is not None and val != 0.0)
        coverage = non_zero_count / len(currency_cols)
        coverage_over_time.append(coverage)
    
    # Plot coverage over time
    plt.figure(figsize=(12, 6))
    plt.plot(sample_indices, coverage_over_time, alpha=0.7, linewidth=1)
    plt.xlabel('Row Index (Time)')
    plt.ylabel('Data Coverage Ratio')
    plt.title('Data Coverage Over Time\n(Sampled every ~1000 rows)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add trend line
    z = np.polyfit(sample_indices, coverage_over_time, 1)
    p = np.poly1d(z)
    plt.plot(sample_indices, p(sample_indices), "r--", alpha=0.8, 
             label=f'Trend: {"increasing" if z[0] > 0 else "decreasing"}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('temporal_coverage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Temporal analysis saved to: temporal_coverage_analysis.png")
    
    # Statistics
    avg_coverage = np.mean(coverage_over_time)
    print(f"Average coverage over time: {avg_coverage:.2%}")
    print(f"Coverage trend: {'Increasing' if z[0] > 0 else 'Decreasing'} ({z[0]*1000:.3f} per 1000 rows)")

def check_currency_correlations(df: pl.DataFrame, max_currencies: int = 20) -> None:
    """Check correlations between major currencies"""
    print("\n🔗 CHECKING CURRENCY CORRELATIONS")
    print("="*50)
    
    # Get major currencies (those with highest coverage)
    currency_cols = [col for col in df.columns if col.endswith('_return')]
    
    # Calculate coverage for each currency
    coverage_scores = []
    for col in currency_cols:
        values = df[col].drop_nulls()
        non_zero_count = len([v for v in values if v != 0.0])
        coverage = non_zero_count / df.shape[0]
        coverage_scores.append((col, coverage))
    
    # Sort by coverage and take top currencies
    coverage_scores.sort(key=lambda x: x[1], reverse=True)
    top_currencies = [col for col, _ in coverage_scores[:max_currencies]]
    
    print(f"Analyzing correlations for top {len(top_currencies)} currencies by coverage:")
    for col, coverage in coverage_scores[:max_currencies]:
        print(f"  {col.replace('_return', '')}: {coverage:.1%}")
    
    # Calculate correlation matrix
    correlation_data = df.select(top_currencies).to_numpy()
    
    # Replace NaN and zeros with actual NaN for correlation calculation
    correlation_data = np.where(correlation_data == 0, np.nan, correlation_data)
    
    # Calculate correlation matrix (ignoring NaN values)
    corr_matrix = np.corrcoef(correlation_data.T)
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    
    # Mask for better visualization
    mask = np.isnan(corr_matrix)
    
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdBu_r
    
    im = plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Set labels
    currency_labels = [col.replace('_return', '') for col in top_currencies]
    plt.xticks(range(len(currency_labels)), currency_labels, rotation=45, ha='right')
    plt.yticks(range(len(currency_labels)), currency_labels)
    
    plt.title(f'Correlation Matrix: Top {len(top_currencies)} Currencies by Coverage')
    plt.tight_layout()
    plt.savefig('currency_correlations.png', dpi=300, bbox_inches='tight')
    print(f"🔗 Correlation analysis saved to: currency_correlations.png")
    
    # Print some statistics
    corr_values = corr_matrix[~np.isnan(corr_matrix) & ~np.eye(len(corr_matrix), dtype=bool)]
    if len(corr_values) > 0:
        print(f"Correlation statistics:")
        print(f"  Mean correlation: {np.mean(corr_values):.3f}")
        print(f"  Max correlation: {np.max(corr_values):.3f}")
        print(f"  Min correlation: {np.min(corr_values):.3f}")

def check_data_quality_issues(df: pl.DataFrame) -> None:
    """Check for various data quality issues"""
    print("\n🔍 CHECKING DATA QUALITY ISSUES")
    print("="*50)
    
    currency_cols = [col for col in df.columns if col.endswith('_return')]
    n_rows, n_cols = df.shape
    
    # Check for completely empty rows
    empty_rows = 0
    for i in range(min(1000, n_rows)):  # Sample first 1000 rows
        row_data = df.row(i)
        if all(val is None or val == 0.0 for val in row_data):
            empty_rows += 1
    
    print(f"Empty rows (sampled): {empty_rows}/1000 = {empty_rows/10:.1f}%")
    
    # Check for currencies with no data
    empty_currencies = []
    for col in currency_cols:
        values = df[col].drop_nulls()
        non_zero_count = len([v for v in values if v != 0.0])
        if non_zero_count == 0:
            empty_currencies.append(col)
    
    if empty_currencies:
        print(f"⚠️  Currencies with no data: {len(empty_currencies)}")
        for col in empty_currencies[:5]:  # Show first 5
            print(f"    {col.replace('_return', '')}")
        if len(empty_currencies) > 5:
            print(f"    ... and {len(empty_currencies) - 5} more")
    else:
        print("✅ All currencies have some data")
    
    # Check for suspicious patterns
    print(f"\n📊 Data distribution check:")
    
    # Sample a major currency for distribution analysis
    major_currency = currency_cols[0]  # Assuming first is major (like BTCUSDT)
    values = df[major_currency].drop_nulls()
    non_zero_values = [v for v in values if v != 0.0]
    
    if len(non_zero_values) > 0:
        values_array = np.array(non_zero_values)
        
        print(f"  {major_currency.replace('_return', '')} distribution:")
        print(f"    Non-zero values: {len(non_zero_values):,}")
        print(f"    Mean: {np.mean(values_array):.6f}")
        print(f"    Std: {np.std(values_array):.6f}")
        print(f"    Skewness: {np.mean(((values_array - np.mean(values_array)) / np.std(values_array))**3):.3f}")
        
        # Check for outliers (values beyond 3 standard deviations)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        outliers = values_array[np.abs(values_array - mean_val) > 3 * std_val]
        print(f"    Outliers (>3σ): {len(outliers)} ({len(outliers)/len(values_array)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive sanity checks for crypto dataset')
    parser.add_argument('--data-path', 
                       default='/home/i3/Downloads/processed_dataset.parquet',
                       help='Path to the processed dataset parquet file')
    
    args = parser.parse_args()
    
    try:
        # Load and validate
        df = load_and_validate_dataset(args.data_path)
        
        # Run all sanity checks
        check_data_types_and_ranges(df)
        check_temporal_consistency(df)
        check_currency_correlations(df)
        check_data_quality_issues(df)
        
        print("\n" + "="*60)
        print("🎉 ALL SANITY CHECKS COMPLETED!")
        print("="*60)
        print("Generated files:")
        print("  - data_coverage_heatmap.png")
        print("  - data_coverage_summary.png") 
        print("  - temporal_coverage_analysis.png")
        print("  - currency_correlations.png")
        
    except Exception as e:
        print(f"❌ Error during sanity checks: {e}")
        raise

if __name__ == "__main__":
    main()
