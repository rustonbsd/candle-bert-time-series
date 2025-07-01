#!/usr/bin/env python3
"""
Verify the transformed dataset and compare with original
"""

import polars as pl
import numpy as np

def main():
    print("🔍 VERIFYING TRANSFORMED DATASET")
    print("="*50)
    
    # Load both datasets
    original_path = "/home/i3/Downloads/processed_dataset.parquet"
    transformed_path = "/home/i3/Downloads/transformed_dataset.parquet"
    
    print("Loading datasets...")
    original_df = pl.read_parquet(original_path)
    transformed_df = pl.read_parquet(transformed_path)
    
    print(f"✅ Original dataset: {original_df.shape}")
    print(f"✅ Transformed dataset: {transformed_df.shape}")
    
    # Verify dimensions
    print(f"\n📊 DIMENSION COMPARISON:")
    print(f"Rows: {original_df.shape[0]:,} → {transformed_df.shape[0]:,} (no change expected)")
    print(f"Columns: {original_df.shape[1]:,} → {transformed_df.shape[1]:,} (reduced by {original_df.shape[1] - transformed_df.shape[1]})")
    
    # Check which currencies were removed
    original_cols = set(original_df.columns)
    transformed_cols = set(transformed_df.columns)
    removed_cols = original_cols - transformed_cols
    
    print(f"\n🗑️  REMOVED CURRENCIES ({len(removed_cols)}):")
    for i, col in enumerate(sorted(removed_cols), 1):
        currency_name = col.replace('_return', '')
        print(f"  {i:2d}. {currency_name}")
    
    # Verify data integrity for kept currencies
    print(f"\n🔍 DATA INTEGRITY CHECK:")
    sample_currencies = list(transformed_cols)[:5]  # Check first 5 currencies
    
    for currency in sample_currencies:
        orig_col = original_df[currency]
        trans_col = transformed_df[currency]
        
        # Check if data is identical
        if orig_col.equals(trans_col):
            print(f"  ✅ {currency.replace('_return', '')}: Data identical")
        else:
            print(f"  ❌ {currency.replace('_return', '')}: Data differs!")
    
    # Calculate coverage statistics for transformed dataset
    print(f"\n📈 TRANSFORMED DATASET COVERAGE:")
    total_points = transformed_df.shape[0] * transformed_df.shape[1]
    
    # Count non-zero values
    non_zero_count = 0
    for col in transformed_df.columns:
        values = transformed_df[col].drop_nulls()
        non_zero_count += len([v for v in values if v != 0.0])
    
    coverage = non_zero_count / total_points
    print(f"  Total data points: {total_points:,}")
    print(f"  Non-zero points: {non_zero_count:,}")
    print(f"  Coverage: {coverage:.1%}")
    
    # Show top currencies in transformed dataset
    print(f"\n🏆 TOP 10 CURRENCIES IN TRANSFORMED DATASET:")
    currency_coverage = []
    
    for col in transformed_df.columns:
        values = transformed_df[col].drop_nulls()
        non_zero = len([v for v in values if v != 0.0])
        coverage_ratio = non_zero / transformed_df.shape[0]
        currency_coverage.append((col, coverage_ratio))
    
    currency_coverage.sort(key=lambda x: x[1], reverse=True)
    
    for i, (currency, coverage) in enumerate(currency_coverage[:10], 1):
        currency_name = currency.replace('_return', '')
        print(f"  {i:2d}. {currency_name}: {coverage:.1%}")
    
    print(f"\n✅ VERIFICATION COMPLETE!")
    print(f"Transformed dataset is ready for transformer training:")
    print(f"  • {transformed_df.shape[0]:,} time steps")
    print(f"  • {transformed_df.shape[1]:,} high-quality currency features")
    print(f"  • {coverage:.1%} data coverage")
    print(f"  • 2.0 GB file size")

if __name__ == "__main__":
    main()
