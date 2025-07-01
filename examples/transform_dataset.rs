use anyhow::Result;
use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for dataset transformation
#[derive(Debug)]
struct TransformConfig {
    /// Minimum coverage percentage to keep a currency (e.g., 0.1 for 10%)
    min_coverage_threshold: f64,
    /// Number of years to keep from the end of the dataset
    years_to_keep: Option<u32>,
    /// Keep discontinued currencies if they were discontinued within this many days
    keep_discontinued_within_days: u32,
    /// Input parquet file path
    input_path: PathBuf,
    /// Output parquet file path
    output_path: PathBuf,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            min_coverage_threshold: 0.10, // 10% minimum coverage
            years_to_keep: Some(4), // Keep last 4 years
            keep_discontinued_within_days: 365, // Keep discontinued currencies from last year
            input_path: PathBuf::from("/home/i3/Downloads/processed_dataset.parquet"),
            output_path: PathBuf::from("/home/i3/Downloads/transformed_dataset.parquet"),
        }
    }
}

/// Statistics about a currency's data coverage
#[derive(Debug, Clone)]
struct CurrencyStats {
    name: String,
    total_rows: usize,
    non_zero_rows: usize,
    coverage_ratio: f64,
    first_data_index: Option<usize>,
    last_data_index: Option<usize>,
    is_discontinued: bool,
    days_since_last_data: Option<u32>,
}

impl CurrencyStats {
    fn new(name: String, total_rows: usize) -> Self {
        Self {
            name,
            total_rows,
            non_zero_rows: 0,
            coverage_ratio: 0.0,
            first_data_index: None,
            last_data_index: None,
            is_discontinued: false,
            days_since_last_data: None,
        }
    }

    fn calculate_coverage(&mut self) {
        self.coverage_ratio = if self.total_rows > 0 {
            self.non_zero_rows as f64 / self.total_rows as f64
        } else {
            0.0
        };
    }

    fn calculate_discontinuation(&mut self, total_rows: usize, minutes_per_row: u32) {
        if let Some(last_idx) = self.last_data_index {
            let rows_since_last = total_rows.saturating_sub(last_idx + 1);
            let minutes_since_last = rows_since_last * minutes_per_row as usize;
            self.days_since_last_data = Some((minutes_since_last / (60 * 24)) as u32);
            
            // Consider discontinued if no data in last 30 days (arbitrary threshold)
            self.is_discontinued = self.days_since_last_data.unwrap_or(0) > 30;
        }
    }
}

/// Load and analyze the dataset
fn load_and_analyze_dataset(input_path: &Path) -> Result<(DataFrame, Vec<CurrencyStats>)> {
    println!("🔍 Loading dataset from: {}", input_path.display());
    
    if !input_path.exists() {
        anyhow::bail!("Input file does not exist: {}", input_path.display());
    }

    // Load the dataset
    let df = LazyFrame::scan_parquet(input_path, Default::default())?
        .collect()?;

    println!("✅ Dataset loaded: {} rows × {} columns", df.height(), df.width());

    // Get currency columns (those ending with '_return')
    let currency_columns: Vec<String> = df.get_column_names()
        .iter()
        .filter(|name| name.ends_with("_return"))
        .map(|s| s.to_string())
        .collect();

    println!("📊 Found {} currency columns", currency_columns.len());

    // Analyze each currency
    let mut currency_stats = Vec::new();
    let total_rows = df.height();

    for col_name in &currency_columns {
        let mut stats = CurrencyStats::new(col_name.clone(), total_rows);
        
        // Get the column data and analyze it using a simpler approach
        let column = df.column(col_name)?;

        // Convert to a series and analyze using lazy operations
        let series = column.clone();

        // Use lazy operations to count non-zero, non-null values
        let temp_df = DataFrame::new(vec![series])?;
        let analysis_result = temp_df
            .lazy()
            .select([
                // Count non-zero, non-null values
                col(col_name)
                    .filter(col(col_name).neq(lit(0.0)).and(col(col_name).is_not_null()))
                    .count()
                    .alias("non_zero_count")
            ])
            .collect()?;

        if let Ok(count_value) = analysis_result.column("non_zero_count") {
            if let Ok(count_series) = count_value.u32() {
                if let Some(count) = count_series.get(0) {
                    stats.non_zero_rows = count as usize;
                }
            }
        }

        // Find first and last data indices by iterating through the original column
        if let Ok(float_series) = column.f64() {
            for (idx, opt_val) in float_series.iter().enumerate() {
                if let Some(val) = opt_val {
                    if val != 0.0 && !val.is_nan() {
                        if stats.first_data_index.is_none() {
                            stats.first_data_index = Some(idx);
                        }
                        stats.last_data_index = Some(idx);
                    }
                }
            }
        }

        stats.calculate_coverage();
        stats.calculate_discontinuation(total_rows, 1); // Assuming 1-minute intervals
        
        currency_stats.push(stats);
    }

    // Sort by coverage ratio (descending)
    currency_stats.sort_by(|a, b| b.coverage_ratio.partial_cmp(&a.coverage_ratio).unwrap());

    Ok((df, currency_stats))
}

/// Print analysis results
fn print_analysis_results(stats: &[CurrencyStats], config: &TransformConfig) {
    println!("\n📈 DATASET ANALYSIS RESULTS");
    println!("{}", "=".repeat(60));

    let total_currencies = stats.len();
    let above_threshold = stats.iter()
        .filter(|s| s.coverage_ratio >= config.min_coverage_threshold)
        .count();
    let discontinued = stats.iter()
        .filter(|s| s.is_discontinued)
        .count();
    let discontinued_recent = stats.iter()
        .filter(|s| s.is_discontinued &&
                s.days_since_last_data.unwrap_or(u32::MAX) <= config.keep_discontinued_within_days)
        .count();

    println!("Total currencies: {}", total_currencies);
    println!("Above {}% coverage threshold: {}",
             (config.min_coverage_threshold * 100.0) as u32, above_threshold);
    println!("Discontinued currencies: {}", discontinued);
    println!("Recently discontinued (within {} days): {}",
             config.keep_discontinued_within_days, discontinued_recent);

    println!("\n🏆 TOP 10 CURRENCIES BY COVERAGE:");
    for (i, stat) in stats.iter().take(10).enumerate() {
        let status = if stat.is_discontinued {
            format!("(DISCONTINUED - {} days ago)",
                   stat.days_since_last_data.unwrap_or(0))
        } else {
            "(ACTIVE)".to_string()
        };

        println!("  {}. {}: {:.1}% {}",
                 i + 1,
                 stat.name.replace("_return", ""),
                 stat.coverage_ratio * 100.0,
                 status);
    }

    println!("\n⚠️  BOTTOM 10 CURRENCIES BY COVERAGE:");
    for (i, stat) in stats.iter().rev().take(10).enumerate() {
        let status = if stat.is_discontinued {
            format!("(DISCONTINUED - {} days ago)",
                   stat.days_since_last_data.unwrap_or(0))
        } else {
            "(ACTIVE)".to_string()
        };

        println!("  {}. {}: {:.1}% {}",
                 i + 1,
                 stat.name.replace("_return", ""),
                 stat.coverage_ratio * 100.0,
                 status);
    }
}

/// Filter currencies based on coverage and discontinuation criteria
fn filter_currencies(stats: &[CurrencyStats], config: &TransformConfig) -> Vec<String> {
    let mut kept_currencies = Vec::new();
    let mut removed_low_coverage = 0;
    let mut removed_old_discontinued = 0;
    let mut kept_recent_discontinued = 0;

    for stat in stats {
        let should_keep = if stat.coverage_ratio >= config.min_coverage_threshold {
            // Keep if above coverage threshold
            true
        } else if stat.is_discontinued {
            // For discontinued currencies, only keep if recently discontinued
            if stat.days_since_last_data.unwrap_or(u32::MAX) <= config.keep_discontinued_within_days {
                kept_recent_discontinued += 1;
                true
            } else {
                removed_old_discontinued += 1;
                false
            }
        } else {
            // Remove if below threshold and not discontinued (very sparse active currency)
            removed_low_coverage += 1;
            false
        };

        if should_keep {
            kept_currencies.push(stat.name.clone());
        }
    }

    println!("\n🔧 FILTERING RESULTS:");
    println!("{}", "=".repeat(40));
    println!("Currencies kept: {}", kept_currencies.len());
    println!("Removed (low coverage): {}", removed_low_coverage);
    println!("Removed (old discontinued): {}", removed_old_discontinued);
    println!("Kept (recently discontinued): {}", kept_recent_discontinued);

    kept_currencies
}

/// Apply time-based filtering (keep last N years)
fn apply_time_filtering(df: DataFrame, config: &TransformConfig) -> Result<DataFrame> {
    if let Some(years_to_keep) = config.years_to_keep {
        println!("\n⏰ Applying time filtering: keeping last {} years", years_to_keep);

        let total_rows = df.height();
        let minutes_per_year = 365 * 24 * 60; // Approximate minutes in a year
        let rows_to_keep = (years_to_keep as usize * minutes_per_year).min(total_rows);
        let start_row = total_rows.saturating_sub(rows_to_keep);

        println!("Original rows: {}", total_rows);
        println!("Keeping rows from {} to {} ({} rows)",
                 start_row, total_rows - 1, rows_to_keep);

        // Use slice to keep the last N rows
        let filtered_df = df.slice(start_row as i64, rows_to_keep);

        println!("✅ Time filtering complete: {} rows remaining", filtered_df.height());
        Ok(filtered_df)
    } else {
        println!("⏰ No time filtering applied - keeping all data for transformer training");
        Ok(df)
    }
}

/// Apply currency filtering to the dataset
fn apply_currency_filtering(df: DataFrame, currencies_to_keep: &[String]) -> Result<DataFrame> {
    println!("\n🔧 Applying currency filtering...");

    let original_columns = df.width();

    // Select only the currencies we want to keep
    let filtered_df = df.select(currencies_to_keep)?;

    println!("Original columns: {}", original_columns);
    println!("Filtered columns: {}", filtered_df.width());
    println!("✅ Currency filtering complete");

    Ok(filtered_df)
}

/// Save the transformed dataset
fn save_transformed_dataset(df: &DataFrame, output_path: &Path) -> Result<()> {
    println!("\n💾 Saving transformed dataset to: {}", output_path.display());

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Save as parquet with compression
    let mut file = fs::File::create(output_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df.clone())?;

    println!("✅ Dataset saved successfully");
    println!("   Final shape: {} rows × {} columns", df.height(), df.width());
    println!("   File size: {:.2} MB",
             fs::metadata(output_path)?.len() as f64 / 1_048_576.0);

    Ok(())
}

/// Main transformation function
fn transform_dataset(config: TransformConfig) -> Result<()> {
    println!("🚀 STARTING DATASET TRANSFORMATION");
    println!("{}", "=".repeat(60));
    println!("Configuration:");
    println!("  Min coverage threshold: {:.1}%", config.min_coverage_threshold * 100.0);
    println!("  Years to keep: {:?}", config.years_to_keep);
    println!("  Keep discontinued within: {} days", config.keep_discontinued_within_days);
    println!("  Input: {}", config.input_path.display());
    println!("  Output: {}", config.output_path.display());

    // Step 1: Load and analyze dataset
    let (df, currency_stats) = load_and_analyze_dataset(&config.input_path)?;

    // Step 2: Print analysis results
    print_analysis_results(&currency_stats, &config);

    // Step 3: Filter currencies based on criteria
    let currencies_to_keep = filter_currencies(&currency_stats, &config);

    // Step 4: Apply currency filtering
    let currency_filtered_df = apply_currency_filtering(df, &currencies_to_keep)?;

    // Step 5: Apply time filtering (if configured)
    let final_df = apply_time_filtering(currency_filtered_df, &config)?;

    // Step 6: Save the transformed dataset
    save_transformed_dataset(&final_df, &config.output_path)?;

    println!("\n🎉 TRANSFORMATION COMPLETE!");
    println!("{}", "=".repeat(60));
    println!("Summary:");
    println!("  Original: {} currencies", currency_stats.len());
    println!("  Filtered: {} currencies", currencies_to_keep.len());
    println!("  Reduction: {:.1}%",
             (1.0 - currencies_to_keep.len() as f64 / currency_stats.len() as f64) * 100.0);

    Ok(())
}

fn main() -> Result<()> {
    // You can customize the configuration here
    let mut config = TransformConfig::default();

    // Parse command line arguments for flexibility
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--keep-all-time" => {
                config.years_to_keep = None;
                println!("🕐 Configured to keep all time data for transformer training");
            }
            "--last-3-years" => {
                config.years_to_keep = Some(3);
                println!("🕐 Configured to keep last 3 years");
            }
            "--last-5-years" => {
                config.years_to_keep = Some(5);
                println!("🕐 Configured to keep last 5 years");
            }
            "--help" => {
                println!("Usage: {} [OPTIONS]", args[0]);
                println!("Options:");
                println!("  --keep-all-time    Keep all time data (recommended for transformers)");
                println!("  --last-3-years     Keep only last 3 years");
                println!("  --last-5-years     Keep only last 5 years");
                println!("  --help             Show this help message");
                return Ok(());
            }
            _ => {
                println!("Unknown option: {}. Use --help for usage.", args[1]);
                return Ok(());
            }
        }
    }

    transform_dataset(config)
}
