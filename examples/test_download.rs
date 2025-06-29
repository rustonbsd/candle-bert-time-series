// Test script to verify Parquet reading functionality

use anyhow::Result;
use polars::prelude::*;
use std::path::PathBuf;

fn main() -> Result<()> {
    let test_dir = PathBuf::from("./crypto_data");
    let parquet_path = test_dir.join("BTCUSDT_agg_trades.parquet");

    if parquet_path.exists() {
        println!("Reading existing Parquet file: {}", parquet_path.display());

        let df = LazyFrame::scan_parquet(&parquet_path, Default::default())?
            .collect()?;

        println!("Parquet file contains {} rows", df.height());
        println!("Columns: {:?}", df.get_column_names());

        if df.height() > 0 {
            println!("First few rows:");
            println!("{}", df.head(Some(5)));

            // Check timestamp range
            let timestamp_stats = df
                .lazy()
                .select([
                    col("timestamp").min().alias("min_timestamp"),
                    col("timestamp").max().alias("max_timestamp"),
                ])
                .collect()?;

            println!("Timestamp range:");
            println!("{}", timestamp_stats);

            // Show file size comparison
            let metadata = std::fs::metadata(&parquet_path)?;
            println!("File size: {:.2} MB", metadata.len() as f64 / 1_048_576.0);
        }
    } else {
        println!("Parquet file not found at: {}", parquet_path.display());
        println!("Run the download_data example first to create the Parquet file.");
    }

    Ok(())
}
