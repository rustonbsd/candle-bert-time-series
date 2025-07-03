use anyhow::Result;
use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for dataset transformation
#[derive(Debug)]
struct TransformConfig {
    /// Minimum coverage percentage from first data point to keep a currency (e.g., 0.5 for 50%)
    min_coverage_from_start: f64,
    /// Number of years to keep from the end of the dataset
    years_to_keep: Option<u32>,
    /// Only keep currently traded currencies (exclude discontinued ones)
    only_active_currencies: bool,
    /// Percentage of currencies that should have started by the cutoff point (e.g., 0.4 for 40%)
    currency_start_percentile: f64,
    /// Maximum delay allowed for currency start relative to cutoff (e.g., 0.25 for 25%)
    max_start_delay_ratio: f64,
    /// Input parquet file path
    input_path: PathBuf,
    /// Output parquet file path
    output_path: PathBuf,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            min_coverage_from_start: 0.50, // 50% minimum coverage from first data point
            years_to_keep: None, // Keep all time data for transformer training
            only_active_currencies: true, // Only keep currently traded cryptos
            currency_start_percentile: 0.40, // 40% of currencies should have started by cutoff
            max_start_delay_ratio: 0.25, // Allow 25% delay after cutoff point
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
    coverage_from_start: f64, // Coverage from first data point onwards
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
            coverage_from_start: 0.0,
            first_data_index: None,
            last_data_index: None,
            is_discontinued: false,
            days_since_last_data: None,
        }
    }

    fn calculate_coverage(&mut self) {
        // Overall coverage
        self.coverage_ratio = if self.total_rows > 0 {
            self.non_zero_rows as f64 / self.total_rows as f64
        } else {
            0.0
        };

        // Coverage from first data point onwards (this is what we care about)
        if let Some(first_idx) = self.first_data_index {
            let rows_from_start = self.total_rows.saturating_sub(first_idx);
            if rows_from_start > 0 {
                // Count non-zero rows from first data point onwards
                let non_zero_from_start = self.non_zero_rows; // This is already calculated correctly
                self.coverage_from_start = non_zero_from_start as f64 / rows_from_start as f64;
            }
        }
    }

    fn calculate_discontinuation(&mut self, total_rows: usize, minutes_per_row: u32) {
        if let Some(last_idx) = self.last_data_index {
            let rows_since_last = total_rows.saturating_sub(last_idx + 1);
            let minutes_since_last = rows_since_last * minutes_per_row as usize;
            self.days_since_last_data = Some((minutes_since_last / (60 * 24)) as u32);

            // Consider discontinued if no data in last 7 days (more strict threshold)
            self.is_discontinued = self.days_since_last_data.unwrap_or(0) > 7;
        } else {
            // If no data found at all, definitely discontinued
            self.is_discontinued = true;
            self.days_since_last_data = Some(u32::MAX);
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

        // Find first and last data indices and count non-zero values from first data point
        if let Ok(float_series) = column.f64() {
            let mut non_zero_from_start = 0;
            let mut found_first = false;

            for (idx, opt_val) in float_series.iter().enumerate() {
                if let Some(val) = opt_val {
                    if val != 0.0 && !val.is_nan() {
                        if !found_first {
                            stats.first_data_index = Some(idx);
                            found_first = true;
                        }
                        stats.last_data_index = Some(idx);

                        // Count non-zero values from first data point onwards
                        if found_first {
                            non_zero_from_start += 1;
                        }
                    }
                }
            }

            // Update the non_zero_rows to reflect count from first data point
            stats.non_zero_rows = non_zero_from_start;
        }

        stats.calculate_coverage();
        stats.calculate_discontinuation(total_rows, 1); // Assuming 1-minute intervals
        
        currency_stats.push(stats);
    }

    // Sort by coverage from start (descending)
    currency_stats.sort_by(|a, b| b.coverage_from_start.partial_cmp(&a.coverage_from_start).unwrap());

    Ok((df, currency_stats))
}

/// Print analysis results
fn print_analysis_results(stats: &[CurrencyStats], config: &TransformConfig) {
    println!("\n📈 DATASET ANALYSIS RESULTS");
    println!("{}", "=".repeat(60));

    let total_currencies = stats.len();
    let above_threshold = stats.iter()
        .filter(|s| s.coverage_from_start >= config.min_coverage_from_start)
        .count();
    let active_currencies = stats.iter()
        .filter(|s| !s.is_discontinued)
        .count();
    let discontinued = stats.iter()
        .filter(|s| s.is_discontinued)
        .count();

    println!("Total currencies: {}", total_currencies);
    println!("Active currencies: {}", active_currencies);
    println!("Discontinued currencies: {}", discontinued);
    println!("Above {}% coverage from start: {}",
             (config.min_coverage_from_start * 100.0) as u32, above_threshold);

    println!("\n🏆 TOP 10 CURRENCIES BY COVERAGE FROM START:");
    for (i, stat) in stats.iter().take(10).enumerate() {
        let status = if stat.is_discontinued {
            format!("(DISCONTINUED - {} days ago)",
                   stat.days_since_last_data.unwrap_or(0))
        } else {
            "(ACTIVE)".to_string()
        };

        let first_data_info = if let Some(first_idx) = stat.first_data_index {
            format!("starts at row {}", first_idx)
        } else {
            "no data found".to_string()
        };

        println!("  {}. {}: {:.1}% from start ({}) {}",
                 i + 1,
                 stat.name.replace("_return", ""),
                 stat.coverage_from_start * 100.0,
                 first_data_info,
                 status);
    }

    println!("\n⚠️  BOTTOM 10 CURRENCIES BY COVERAGE FROM START:");
    for (i, stat) in stats.iter().rev().take(10).enumerate() {
        let status = if stat.is_discontinued {
            format!("(DISCONTINUED - {} days ago)",
                   stat.days_since_last_data.unwrap_or(0))
        } else {
            "(ACTIVE)".to_string()
        };

        let first_data_info = if let Some(first_idx) = stat.first_data_index {
            format!("starts at row {}", first_idx)
        } else {
            "no data found".to_string()
        };

        println!("  {}. {}: {:.1}% from start ({}) {}",
                 i + 1,
                 stat.name.replace("_return", ""),
                 stat.coverage_from_start * 100.0,
                 first_data_info,
                 status);
    }
}

/// Filter currencies based on coverage from start and active trading status
fn filter_currencies_by_coverage(stats: &[CurrencyStats], config: &TransformConfig) -> Vec<String> {
    let mut kept_currencies = Vec::new();
    let mut removed_low_coverage = 0;
    let mut removed_discontinued = 0;
    let mut removed_no_data = 0;

    for stat in stats {
        let should_keep = if stat.first_data_index.is_none() {
            // Remove currencies with no data at all
            removed_no_data += 1;
            false
        } else if config.only_active_currencies && stat.is_discontinued {
            // Remove discontinued currencies if we only want active ones
            removed_discontinued += 1;
            false
        } else if stat.coverage_from_start >= config.min_coverage_from_start {
            // Keep if above coverage threshold from first data point
            true
        } else {
            // Remove if below coverage threshold from start
            removed_low_coverage += 1;
            false
        };

        if should_keep {
            kept_currencies.push(stat.name.clone());
        }
    }

    println!("\n🔧 COVERAGE FILTERING RESULTS:");
    println!("{}", "=".repeat(40));
    println!("Currencies kept: {}", kept_currencies.len());
    println!("Removed (no data): {}", removed_no_data);
    println!("Removed (discontinued): {}", removed_discontinued);
    println!("Removed (low coverage from start): {}", removed_low_coverage);

    // Show some examples of kept currencies
    println!("\n✅ SAMPLE OF KEPT CURRENCIES:");
    for (i, currency) in kept_currencies.iter().take(10).enumerate() {
        if let Some(stat) = stats.iter().find(|s| &s.name == currency) {
            println!("  {}. {}: {:.1}% coverage from row {}",
                     i + 1,
                     currency.replace("_return", ""),
                     stat.coverage_from_start * 100.0,
                     stat.first_data_index.unwrap_or(0));
        }
    }

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

/// Calculate the row index for August 2020 (assuming 1-minute intervals)
fn calculate_august_2020_row() -> usize {
    // Assuming the dataset starts from some point and uses 1-minute intervals
    // We need to calculate how many minutes from the start of the dataset to August 1, 2020

    // For simplicity, let's assume the dataset starts around 2017 (when crypto trading became more common)
    // From January 1, 2017 to August 1, 2020 is approximately:
    // 2017: 365 days
    // 2018: 365 days
    // 2019: 365 days
    // 2020: 31 (Jan) + 29 (Feb, leap year) + 31 (Mar) + 30 (Apr) + 31 (May) + 30 (Jun) + 31 (Jul) = 213 days
    // Total: 365 + 365 + 365 + 213 = 1308 days

    let days_to_august_2020 = 1308;
    let minutes_per_day = 24 * 60;
    let total_minutes = days_to_august_2020 * minutes_per_day;

    total_minutes
}

/// Set the main starting line to August 2020
fn set_main_starting_line_august_2020() -> usize {
    let august_2020_row = calculate_august_2020_row();
    println!("📅 Setting main starting line to August 2020 (estimated row: {})", august_2020_row);
    august_2020_row
}

/// Filter currencies based on when they started relative to the main starting line
fn filter_currencies_by_start_timing(
    currency_stats: &[CurrencyStats],
    main_start_line: usize,
    tolerance_percent: f64
) -> Vec<String> {
    let tolerance_rows = (main_start_line as f64 * tolerance_percent) as usize;
    let latest_acceptable_start = main_start_line + tolerance_rows;

    println!("\n📍 FILTERING BY START TIMING:");
    println!("Main starting line: row {}", main_start_line);
    println!("Tolerance: {:.0}% ({} rows)", tolerance_percent * 100.0, tolerance_rows);
    println!("Latest acceptable start: row {}", latest_acceptable_start);

    let mut kept_currencies = Vec::new();
    let mut removed_late_start = 0;
    let mut removed_no_data = 0;

    for stat in currency_stats {
        if let Some(first_idx) = stat.first_data_index {
            if first_idx <= latest_acceptable_start {
                kept_currencies.push(stat.name.clone());
            } else {
                removed_late_start += 1;
            }
        } else {
            removed_no_data += 1;
        }
    }

    println!("Currencies kept: {}", kept_currencies.len());
    println!("Removed (late start): {}", removed_late_start);
    println!("Removed (no data): {}", removed_no_data);

    kept_currencies
}

/// Trim dataset to start from the main starting line
fn trim_dataset_from_main_line(df: DataFrame, main_start_line: usize) -> Result<DataFrame> {
    println!("\n✂️  Trimming dataset from main starting line...");

    let original_rows = df.height();
    let trimmed_df = df.slice(main_start_line as i64, original_rows - main_start_line);

    println!("Original rows: {}", original_rows);
    println!("Trimming from row: {}", main_start_line);
    println!("Trimmed rows: {}", trimmed_df.height());
    println!("✅ Dataset trimming complete");

    Ok(trimmed_df)
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
    println!("  Min coverage from start: {:.1}%", config.min_coverage_from_start * 100.0);
    println!("  Years to keep: {:?}", config.years_to_keep);
    println!("  Only active currencies: {}", config.only_active_currencies);
    println!("  Input: {}", config.input_path.display());
    println!("  Output: {}", config.output_path.display());

    // Step 1: Load and analyze dataset
    let (df, currency_stats) = load_and_analyze_dataset(&config.input_path)?;

    // Step 2: Print analysis results
    print_analysis_results(&currency_stats, &config);

    // Step 3: Set the main starting line to August 2020
    let main_start_line = set_main_starting_line_august_2020();

    // Step 4: Filter currencies by start timing (within 25% of main line)
    let timing_filtered_currencies = filter_currencies_by_start_timing(&currency_stats, main_start_line, 0.25);

    // Step 5: Filter by coverage among timing-filtered currencies
    let timing_filtered_stats: Vec<CurrencyStats> = currency_stats
        .iter()
        .filter(|stat| timing_filtered_currencies.contains(&stat.name))
        .cloned()
        .collect();

    let final_currencies_to_keep = filter_currencies_by_coverage(&timing_filtered_stats, &config);

    // Step 6: Apply currency filtering
    let currency_filtered_df = apply_currency_filtering(df, &final_currencies_to_keep)?;

    // Step 7: Trim dataset from main starting line
    let trimmed_df = trim_dataset_from_main_line(currency_filtered_df, main_start_line)?;

    // Step 8: Apply time filtering (if configured)
    let final_df = apply_time_filtering(trimmed_df, &config)?;

    // Step 9: Save the transformed dataset
    save_transformed_dataset(&final_df, &config.output_path)?;

    println!("\n🎉 TRANSFORMATION COMPLETE!");
    println!("{}", "=".repeat(60));
    println!("Summary:");
    println!("  Original: {} currencies", currency_stats.len());
    println!("  After timing filter: {} currencies", timing_filtered_currencies.len());
    println!("  Final kept: {} currencies", final_currencies_to_keep.len());
    println!("  Total reduction: {:.1}%",
             (1.0 - final_currencies_to_keep.len() as f64 / currency_stats.len() as f64) * 100.0);
    println!("  Main starting line: row {}", main_start_line);

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
            "--include-discontinued" => {
                config.only_active_currencies = false;
                println!("📈 Configured to include discontinued currencies");
            }
            "--coverage-30" => {
                config.min_coverage_from_start = 0.30;
                println!("📊 Configured for 30% minimum coverage from start");
            }
            "--coverage-70" => {
                config.min_coverage_from_start = 0.70;
                println!("📊 Configured for 70% minimum coverage from start");
            }
            "--help" => {
                println!("Usage: {} [OPTIONS]", args[0]);
                println!("Options:");
                println!("  --keep-all-time        Keep all time data (recommended for transformers)");
                println!("  --last-3-years         Keep only last 3 years");
                println!("  --last-5-years         Keep only last 5 years");
                println!("  --include-discontinued Include discontinued currencies");
                println!("  --coverage-30          Set minimum coverage to 30%");
                println!("  --coverage-70          Set minimum coverage to 70%");
                println!("  --help                 Show this help message");
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
