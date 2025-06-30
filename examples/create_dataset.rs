// examples/create_dataset.rs

use anyhow::Result;
use polars::prelude::*;
use std::fs;
use std::path::Path;

/// Processes raw 1-minute k-line data into a model-ready dataset of returns.
/// Uses a more efficient approach that processes symbols in batches to avoid memory issues.
///
/// # Arguments
/// * `raw_data_dir` - Path to the root directory containing symbol subdirectories.
/// * `output_path` - Path to save the final processed Parquet file.
/// * `symbols` - A slice of symbol strings to process.
/// * `batch_size` - Number of symbols to process in each batch.
pub fn create_dataset(
    raw_data_dir: &Path,
    output_path: &Path,
    symbols: &[&str],
    batch_size: usize,
) -> Result<()> {
    println!("Starting dataset creation with batch processing...");
    println!("Total symbols: {}, Batch size: {}", symbols.len(), batch_size);

    // Process symbols in batches to avoid memory issues
    let mut all_return_data: Vec<DataFrame> = Vec::new();
    let mut common_timeline: Option<DataFrame> = None;

    for (batch_idx, symbol_batch) in symbols.chunks(batch_size).enumerate() {
        println!("Processing batch {} ({} symbols)...", batch_idx + 1, symbol_batch.len());

        let mut batch_data: Vec<DataFrame> = Vec::new();

        // Process each symbol in the current batch
        for symbol in symbol_batch {
            let symbol_dir = raw_data_dir.join(symbol);
            if !symbol_dir.exists() {
                eprintln!("Warning: Directory not found for symbol {}, skipping.", symbol);
                continue;
            }

            println!("  Processing symbol: {}", symbol);

            // Load and process the symbol's data
            match process_single_symbol(&symbol_dir, symbol) {
                Ok(df) => {
                    // Extract timeline from first symbol if we don't have one yet
                    if common_timeline.is_none() {
                        common_timeline = Some(df.select(["datetime"])?.clone());
                    }
                    batch_data.push(df);
                }
                Err(e) => {
                    eprintln!("Error processing {}: {}", symbol, e);
                    continue;
                }
            }
        }

        if !batch_data.is_empty() {
            // Combine batch data
            let batch_combined = combine_batch_data(batch_data)?;

            // Save intermediate batch result
            let batch_file = output_path.parent().unwrap().join(format!("batch_{}.parquet", batch_idx + 1));
            let mut file = fs::File::create(&batch_file)?;
            ParquetWriter::new(&mut file)
                .with_compression(ParquetCompression::Snappy)
                .finish(&mut batch_combined.clone())?;
            println!("  💾 Saved batch {} to: {}", batch_idx + 1, batch_file.display());

            all_return_data.push(batch_combined);
            println!("  ✓ Batch {} processed successfully", batch_idx + 1);
        }
    }

    if all_return_data.is_empty() {
        anyhow::bail!("No data processed. Check symbol directories and names.");
    }

    // Combine all batches
    println!("Combining all batches...");
    let final_df = combine_all_batches(all_return_data, common_timeline.unwrap())?;

    // Save the final dataset
    println!(
        "Saving processed dataset with shape {:?} to {}",
        final_df.shape(),
        output_path.display()
    );
    let mut file = fs::File::create(output_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut final_df.clone())?;

    println!("✅ Dataset creation complete.");
    Ok(())
}

/// Process a single symbol and return its data with returns calculated
fn process_single_symbol(symbol_dir: &Path, symbol: &str) -> Result<DataFrame> {
    // Scan all yearly Parquet files in the symbol's directory
    let raw_df = LazyFrame::scan_parquet(
        symbol_dir.join("*.parquet").to_str().unwrap(),
        Default::default(),
    )?;

    // Process data and calculate returns
    let processed_df = raw_df
        // Convert Unix ms timestamp to Polars Datetime
        .with_column(
            col("open_time")
                .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                .alias("datetime"),
        )
        // Sort by timestamp to ensure proper ordering
        .sort(["datetime"], SortMultipleOptions::default())
        // Calculate percentage return based on close prices
        .with_column(
            ((col("close") - col("close").shift(lit(1))) / col("close").shift(lit(1)))
                .alias(&format!("{}_return", symbol)),
        )
        // Keep only the timestamp and the new return column
        .select([col("datetime"), col(&format!("{}_return", symbol))])
        .collect()?;

    Ok(processed_df)
}

/// Combine data from a batch of symbols
fn combine_batch_data(mut batch_data: Vec<DataFrame>) -> Result<DataFrame> {
    if batch_data.is_empty() {
        anyhow::bail!("No data to combine in batch");
    }

    if batch_data.len() == 1 {
        return Ok(batch_data.into_iter().next().unwrap());
    }

    // Start with the first dataframe
    let mut combined = batch_data.remove(0);

    // Join the rest
    for df in batch_data {
        combined = combined.join(
            &df,
            ["datetime"],
            ["datetime"],
            JoinArgs::new(JoinType::Full),
            None,
        )?;
    }

    Ok(combined)
}

/// Combine all batches into the final dataset
fn combine_all_batches(all_batches: Vec<DataFrame>, timeline: DataFrame) -> Result<DataFrame> {
    // Start with just the return columns (no datetime)
    let mut all_return_columns: Vec<polars::prelude::Column> = Vec::new();

    // Extract all return columns from all batches
    for (i, batch_df) in all_batches.into_iter().enumerate() {
        println!("  Extracting columns from batch {} data...", i + 1);

        // Get all columns except datetime
        for column in batch_df.get_columns() {
            if column.name() != "datetime" {
                all_return_columns.push(column.clone());
            }
        }
    }

    // Create final dataframe by combining timeline with all return columns
    println!("  Combining {} return columns with timeline...", all_return_columns.len());
    let mut final_df = timeline;

    // Add all return columns at once
    if !all_return_columns.is_empty() {
        final_df = final_df.hstack(&all_return_columns)?;
    }

    // Fill nulls and clean up
    let final_df = final_df
        .lazy()
        .sort(["datetime"], SortMultipleOptions::default())
        .with_columns([all().fill_null(lit(0.0f64))])
        .drop(["datetime"]) // Remove timestamp for model input
        .collect()?;

    Ok(final_df)
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

/// Load existing batch files if they exist
fn load_existing_batches(output_dir: &Path) -> Result<Vec<DataFrame>> {
    let mut batches = Vec::new();
    let mut batch_num = 1;

    loop {
        let batch_file = output_dir.join(format!("batch_{}.parquet", batch_num));
        if batch_file.exists() {
            println!("📂 Loading existing batch {}: {}", batch_num, batch_file.display());
            let df = LazyFrame::scan_parquet(batch_file.to_str().unwrap(), Default::default())?.collect()?;
            batches.push(df);
            batch_num += 1;
        } else {
            break;
        }
    }

    if !batches.is_empty() {
        println!("✅ Loaded {} existing batch files", batches.len());
    }

    Ok(batches)
}

fn main() -> Result<()> {
    // Configuration
    let raw_data_dir = Path::new("/mnt/storage-box/crypto_data_k_lines/1m");
    let output_path = Path::new("/mnt/storage-box/crypto_data_k_lines/1m/processed_dataset.parquet");
    let pairlist_file = "pairlist.txt";
    let batch_size = 10; // Process 10 symbols at a time to avoid memory issues
    let use_existing_batches = true; // Set to true to use existing batch files

    println!("🚀 Starting dataset creation process...");
    println!("📂 Raw data directory: {}", raw_data_dir.display());
    println!("💾 Output file: {}", output_path.display());
    println!("📦 Batch size: {}", batch_size);

    if use_existing_batches {
        // Try to load existing batch files
        let output_dir = output_path.parent().unwrap();
        let existing_batches = load_existing_batches(output_dir)?;

        if !existing_batches.is_empty() {
            println!("🔄 Using {} existing batch files, combining them...", existing_batches.len());

            // Create a simple timeline from the first batch
            let timeline = existing_batches[0].select(["datetime"])?.clone();

            // Combine all existing batches
            let final_df = combine_all_batches(existing_batches, timeline)?;

            // Save the final dataset
            println!(
                "Saving processed dataset with shape {:?} to {}",
                final_df.shape(),
                output_path.display()
            );
            let mut file = fs::File::create(output_path)?;
            ParquetWriter::new(&mut file)
                .with_compression(ParquetCompression::Snappy)
                .finish(&mut final_df.clone())?;

            println!("✅ Dataset creation complete using existing batches!");
            return Ok(());
        }
    }

    // Read crypto pairs from file
    let pairs = read_crypto_pairs_from_file(pairlist_file)?;
    let pair_refs: Vec<&str> = pairs.iter().map(|s| s.as_str()).collect();

    println!("📊 Processing {} crypto pairs", pairs.len());

    // Create the dataset
    create_dataset(raw_data_dir, output_path, &pair_refs, batch_size)?;

    println!("🎉 Dataset creation completed successfully!");
    println!("📈 Output saved to: {}", output_path.display());

    Ok(())
}