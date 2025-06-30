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
                eprintln!("  ⚠️  Directory not found for symbol {}, skipping.", symbol);
                continue;
            }

            println!("  Processing symbol: {}", symbol);

            // Load and process the symbol's data
            match process_single_symbol(&symbol_dir, symbol) {
                Ok(df) => {
                    // Validate DataFrame structure
                    if df.width() != 2 {
                        eprintln!("  ⚠️  Unexpected DataFrame structure for {}: {} columns (expected 2)",
                                 symbol, df.width());
                        continue;
                    }

                    // Extract timeline from first symbol if we don't have one yet
                    if common_timeline.is_none() {
                        let timeline = df.select(["datetime"])?.clone();
                        println!("  📅 Using {} as timeline reference ({} rows)", symbol, timeline.height());
                        common_timeline = Some(timeline);
                    }
                    batch_data.push(df);
                }
                Err(e) => {
                    eprintln!("  ❌ Error processing {}: {}", symbol, e);
                    continue;
                }
            }
        }

        if !batch_data.is_empty() {
            println!("  Combining {} symbols in batch {}...", batch_data.len(), batch_idx + 1);

            // Show DataFrame sizes for info (but don't skip due to mismatches)
            for (i, df) in batch_data.iter().enumerate() {
                println!("    Symbol {}: {} rows", i + 1, df.height());
            }

            // Combine batch data (now handles mismatched sizes by aligning to common timeline)
            match combine_batch_data(batch_data) {
                Ok(batch_combined) => {
                    // Save intermediate batch result
                    let batch_file = output_path.parent().unwrap().join(format!("batch_{}.parquet", batch_idx + 1));
                    let mut file = fs::File::create(&batch_file)?;
                    ParquetWriter::new(&mut file)
                        .with_compression(ParquetCompression::Snappy)
                        .finish(&mut batch_combined.clone())?;
                    println!("  💾 Saved batch {} to: {}", batch_idx + 1, batch_file.display());

                    all_return_data.push(batch_combined);
                    println!("  ✅ Batch {} processed successfully", batch_idx + 1);
                }
                Err(e) => {
                    eprintln!("  ❌ Failed to combine batch {}: {}", batch_idx + 1, e);
                    continue;
                }
            }
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
    // Check if directory has any parquet files
    let parquet_pattern = symbol_dir.join("*.parquet");
    let parquet_path = parquet_pattern.to_str().unwrap();

    // Scan all yearly Parquet files in the symbol's directory
    let raw_df = LazyFrame::scan_parquet(parquet_path, Default::default())?;

    // Process data and calculate returns with better error handling
    let processed_df = raw_df
        // Convert Unix ms timestamp to Polars Datetime
        .with_column(
            col("open_time")
                .cast(DataType::Datetime(TimeUnit::Milliseconds, None))
                .alias("datetime"),
        )
        // Sort by timestamp to ensure proper ordering
        .sort(["datetime"], SortMultipleOptions::default())
        // Calculate percentage return based on close prices, handling potential nulls
        .with_column(
            ((col("close") - col("close").shift(lit(1))) / col("close").shift(lit(1)))
                .fill_null(lit(0.0f64))  // Fill NaN/null returns with 0
                .alias(&format!("{}_return", symbol)),
        )
        // Keep only the timestamp and the new return column
        .select([col("datetime"), col(&format!("{}_return", symbol))])
        .collect()
        .map_err(|e| anyhow::anyhow!("Failed to process symbol {}: {}", symbol, e))?;

    // Validate the result
    if processed_df.height() == 0 {
        return Err(anyhow::anyhow!("No data found for symbol {}", symbol));
    }

    println!("    ✓ Processed {} rows for {}", processed_df.height(), symbol);
    Ok(processed_df)
}

/// Combine data from a batch of symbols by aligning to a common timeline
fn combine_batch_data(batch_data: Vec<DataFrame>) -> Result<DataFrame> {
    if batch_data.is_empty() {
        anyhow::bail!("No data to combine in batch");
    }

    if batch_data.len() == 1 {
        return Ok(batch_data.into_iter().next().unwrap());
    }

    println!("    Combining {} DataFrames in batch...", batch_data.len());

    // Create a unified timeline by collecting all unique timestamps
    let mut all_timestamps: Vec<i64> = Vec::new();

    for (i, df) in batch_data.iter().enumerate() {
        println!("    DataFrame {} has {} rows", i + 1, df.height());

        // Extract timestamps as i64 (milliseconds)
        let timestamps = df
            .column("datetime")?
            .datetime()?
            .as_datetime_iter()
            .map(|opt_dt| opt_dt.map(|dt| dt.and_utc().timestamp_millis()).unwrap_or(0))
            .collect::<Vec<_>>();

        all_timestamps.extend(timestamps);
    }

    // Remove duplicates and sort to create unified timeline
    all_timestamps.sort_unstable();
    all_timestamps.dedup();

    println!("    Created unified timeline with {} unique timestamps", all_timestamps.len());

    // Convert back to datetime column using Series
    let datetime_series = polars::prelude::Series::new(
        "datetime".into(),
        all_timestamps.clone()
    ).cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;

    let unified_timeline = DataFrame::new(vec![datetime_series.into()])?;

    // Now join each DataFrame to this unified timeline
    let mut result = unified_timeline;

    for (i, df) in batch_data.into_iter().enumerate() {
        println!("    Joining DataFrame {} to unified timeline...", i + 1);

        result = result
            .lazy()
            .join(
                df.lazy(),
                [col("datetime")],
                [col("datetime")],
                JoinArgs::new(JoinType::Left), // Left join to keep all timeline entries
            )
            .collect()?;
    }

    // Fill nulls with 0.0 for missing return values (exclude datetime column)
    // Get all column names and create fill_null expressions for return columns only
    let column_names = result.get_column_names();
    let mut fill_expressions = Vec::new();

    for col_name in column_names {
        if col_name != "datetime" && col_name.ends_with("_return") {
            fill_expressions.push(col(col_name.as_str()).fill_null(lit(0.0f64)));
        }
    }

    let result = if !fill_expressions.is_empty() {
        result
            .lazy()
            .with_columns(fill_expressions)
            .collect()?
    } else {
        result
    };

    println!("    ✓ Combined into DataFrame with {} rows and {} columns",
             result.height(), result.width());

    Ok(result)
}

/// Combine all batches into the final dataset using timeline alignment
fn combine_all_batches(all_batches: Vec<DataFrame>, _timeline: DataFrame) -> Result<DataFrame> {
    if all_batches.is_empty() {
        anyhow::bail!("No batches to combine");
    }

    if all_batches.len() == 1 {
        let mut result = all_batches.into_iter().next().unwrap();
        // Remove datetime column for model input
        result = result.drop("datetime")?;
        return Ok(result);
    }

    println!("  Creating unified timeline from all {} batches...", all_batches.len());

    // Create a unified timeline by collecting all unique timestamps from all batches
    let mut all_timestamps: Vec<i64> = Vec::new();

    for (i, batch_df) in all_batches.iter().enumerate() {
        println!("    Batch {} has {} rows", i + 1, batch_df.height());

        // Extract timestamps as i64 (milliseconds)
        let timestamps = batch_df
            .column("datetime")?
            .datetime()?
            .as_datetime_iter()
            .map(|opt_dt| opt_dt.map(|dt| dt.and_utc().timestamp_millis()).unwrap_or(0))
            .collect::<Vec<_>>();

        all_timestamps.extend(timestamps);
    }

    // Remove duplicates and sort to create unified timeline
    all_timestamps.sort_unstable();
    all_timestamps.dedup();

    println!("    Created unified timeline with {} unique timestamps", all_timestamps.len());

    // Convert back to datetime column using Series
    let datetime_series = polars::prelude::Series::new(
        "datetime".into(),
        all_timestamps.clone()
    ).cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?;

    let mut unified_timeline = DataFrame::new(vec![datetime_series.into()])?;

    // Now join each batch to this unified timeline
    for (i, batch_df) in all_batches.into_iter().enumerate() {
        println!("    Joining batch {} to unified timeline...", i + 1);

        unified_timeline = unified_timeline
            .lazy()
            .join(
                batch_df.lazy(),
                [col("datetime")],
                [col("datetime")],
                JoinArgs::new(JoinType::Left), // Left join to keep all timeline entries
            )
            .collect()?;
    }

    // Fill nulls with 0.0 for missing return values and clean up
    // Get all column names and create fill_null expressions for return columns only
    let column_names = unified_timeline.get_column_names();
    let mut fill_expressions = Vec::new();

    for col_name in column_names {
        if col_name != "datetime" && col_name.ends_with("_return") {
            fill_expressions.push(col(col_name.as_str()).fill_null(lit(0.0f64)));
        }
    }

    let final_df = unified_timeline
        .lazy()
        .sort(["datetime"], SortMultipleOptions::default())
        .with_columns(fill_expressions) // Only fill nulls for return columns
        .drop(["datetime"]) // Remove timestamp for model input
        .collect()?;

    println!("    ✓ Final dataset has {} rows and {} columns",
             final_df.height(), final_df.width());

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

            // Show batch sizes for info
            for (i, batch) in existing_batches.iter().enumerate() {
                println!("  Batch {}: {} rows, {} columns", i + 1, batch.height(), batch.width());
            }

            // Create a dummy timeline (will be replaced by unified timeline in combine_all_batches)
            let dummy_timeline = existing_batches[0].select(["datetime"])?.clone();

            // Combine all existing batches (handles mismatched timelines automatically)
            let final_df = combine_all_batches(existing_batches, dummy_timeline)?;

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