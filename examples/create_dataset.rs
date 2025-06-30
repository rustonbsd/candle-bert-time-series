// examples/create_dataset.rs

use anyhow::Result;
use polars::prelude::*;
use std::fs;
use std::path::Path;

/// Processes raw 1-minute k-line data into a model-ready dataset of returns.
///
/// # Arguments
/// * `raw_data_dir` - Path to the root directory containing symbol subdirectories (e.g., "./crypto_data_k_lines/1m/").
/// * `output_path` - Path to save the final processed Parquet file.
/// * `symbols` - A slice of symbol strings to process (e.g., &["BTCUSDT", "ETHUSDT"]).
/// * `interval` - The time interval to resample to (e.g., "1h", "15m", "1d").
pub fn create_dataset(
    raw_data_dir: &Path,
    output_path: &Path,
    symbols: &[&str],
    _interval: &str,
) -> Result<()> {
    println!("Starting dataset creation...");
    let mut processed_frames: Vec<LazyFrame> = Vec::new();

    // 1. Process each symbol individually
    for symbol in symbols {
        let symbol_dir = raw_data_dir.join(symbol);
        if !symbol_dir.exists() {
            eprintln!("Warning: Directory not found for symbol {}, skipping.", symbol);
            continue;
        }

        println!("Processing symbol: {}", symbol);

        // Scan all yearly Parquet files in the symbol's directory
        // The new structure has files like BTCUSDT/2023.parquet, BTCUSDT/2024.parquet, etc.
        let raw_df = LazyFrame::scan_parquet(
            symbol_dir.join("*.parquet").to_str().unwrap(),
            Default::default(),
        )?;

        // 2. Process data and calculate returns
        let processed_lf = raw_df
            // Convert Unix ms timestamp to Polars Datetime
            .with_column(
                col("open_time")
                    .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                    .alias("datetime"),
            )
            // Sort by timestamp to ensure proper ordering
            .sort(["datetime"], SortMultipleOptions::default())
            // For now, we'll work with the raw 1-minute data and resample later if needed
            // Calculate percentage return based on close prices
            .with_column(
                ((col("close") - col("close").shift(lit(1))) / col("close").shift(lit(1)))
                    .alias(&format!("{}_return", symbol)),
            )
            // Keep only the timestamp and the new return column
            .select([col("datetime"), col(&format!("{}_return", symbol))]);

        processed_frames.push(processed_lf);
    }

    if processed_frames.is_empty() {
        anyhow::bail!("No data processed. Check symbol directories and names.");
    }

    // 4. Join all individual frames into one large DataFrame
    println!("Joining data for all symbols...");
    let mut final_lf = processed_frames.remove(0);
    for lf in processed_frames {
        final_lf = final_lf.join(
            lf,
            [col("datetime")],
            [col("datetime")],
            JoinArgs::new(JoinType::Full),
        );
    }

    // 5. Clean the joined data
    let final_lf = final_lf
        .sort(["datetime"], SortMultipleOptions::default())
        // Fill any nulls with 0.0 (simple approach instead of forward fill)
        .with_columns([all().fill_null(lit(0.0f64))])
        // Drop the timestamp column as it's not needed for the model's input matrix
        .drop(["datetime"]);


    println!("Collecting final DataFrame...");
    let mut final_df = final_lf.collect()?;

    // 6. Save the final dataset
    println!(
        "Saving processed dataset with shape {:?} to {}",
        final_df.shape(),
        output_path.display()
    );
    let mut file = fs::File::create(output_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut final_df)?;

    println!("✅ Dataset creation complete.");
    Ok(())
}

/// Read crypto pairs from pairlist.txt file
fn read_crypto_pairs_from_file(file_path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(file_path)
        .map_err(|e| anyhow::anyhow!("Failed to read pairlist file {}: {}", file_path, e))?;

    // Parse comma-separated pairs and clean them up
    let pairs: Vec<String> = content
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect();

    println!("Read {} crypto pairs from {}", pairs.len(), file_path);
    Ok(pairs)
}

fn main() -> Result<()> {
    // Configuration
    let raw_data_dir = Path::new("/mnt/storage-box/crypto_data_k_lines/1m");
    let output_path = Path::new("/mnt/storage-box/crypto_data_k_lines/1m/processed_dataset.parquet");
    let pairlist_file = "pairlist.txt";
    let interval = "1h"; // Resample 1-minute data to 1-hour intervals

    println!("🚀 Starting dataset creation process...");
    println!("📂 Raw data directory: {}", raw_data_dir.display());
    println!("💾 Output file: {}", output_path.display());
    println!("⏱️  Resampling interval: {}", interval);

    // Read crypto pairs from file
    let pairs = read_crypto_pairs_from_file(pairlist_file)?;
    let pair_refs: Vec<&str> = pairs.iter().map(|s| s.as_str()).collect();

    println!("📊 Processing {} crypto pairs", pairs.len());

    // Create the dataset
    create_dataset(raw_data_dir, output_path, &pair_refs, interval)?;

    println!("🎉 Dataset creation completed successfully!");
    println!("📈 Output saved to: {}", output_path.display());

    Ok(())
}