use anyhow::Result;
use chrono::{Datelike, Duration, NaiveDate};
use polars::prelude::*;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use walkdir;

const DAILY_BASE_URL: &str = "https://data.binance.vision/data/spot/daily/aggTrades";
const MONTHLY_BASE_URL: &str = "https://data.binance.vision/data/spot/monthly/aggTrades";
const MAX_CONCURRENT_REQUESTS: usize = 20; // Number of parallel downloads
const BATCH_SIZE: usize = 30; // Save data every 30 days to avoid losing progress

/// Check what data already exists by scanning the partitioned directory structure
fn get_latest_date_from_partitioned_data(data_dir: &Path) -> Result<Option<NaiveDate>> {
    if !data_dir.exists() {
        return Ok(None);
    }

    let mut latest_date = None;

    // Scan for year directories
    for year_entry in std::fs::read_dir(data_dir)? {
        let year_entry = year_entry?;
        if !year_entry.file_type()?.is_dir() {
            continue;
        }

        let year_name = year_entry.file_name();
        let year_str = year_name.to_string_lossy();

        // Check if it's a valid year directory (4 digits)
        if year_str.len() != 4 || !year_str.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        // Scan for month directories within this year
        for month_entry in std::fs::read_dir(year_entry.path())? {
            let month_entry = month_entry?;
            if !month_entry.file_type()?.is_dir() {
                continue;
            }

            let month_name = month_entry.file_name();
            let month_str = month_name.to_string_lossy();

            // Check if it's a valid month directory (2 digits)
            if month_str.len() != 2 || !month_str.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            // Scan for parquet files in this month directory
            for file_entry in std::fs::read_dir(month_entry.path())? {
                let file_entry = file_entry?;
                if file_entry.file_type()?.is_file() {
                    let file_name = file_entry.file_name();
                    let file_str = file_name.to_string_lossy();

                    if file_str.ends_with(".parquet") {
                        // Extract date from filename (format: YYYY-MM-DD.parquet or YYYY-MM.parquet)
                        let date_part = file_str.trim_end_matches(".parquet");

                        if let Ok(date) = NaiveDate::parse_from_str(date_part, "%Y-%m-%d") {
                            if latest_date.is_none() || date > latest_date.unwrap() {
                                latest_date = Some(date);
                            }
                        } else if let Ok(month_date) = NaiveDate::parse_from_str(&format!("{}-01", date_part), "%Y-%m-%d") {
                            // For monthly files, find the last day of that month
                            let last_day_of_month = if month_date.month() == 12 {
                                NaiveDate::from_ymd_opt(month_date.year() + 1, 1, 1).unwrap().pred_opt().unwrap()
                            } else {
                                NaiveDate::from_ymd_opt(month_date.year(), month_date.month() + 1, 1).unwrap().pred_opt().unwrap()
                            };

                            if latest_date.is_none() || last_day_of_month > latest_date.unwrap() {
                                latest_date = Some(last_day_of_month);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(latest_date)
}

/// Helper function to add a month's worth of dates to the daily fallback list
fn add_month_to_daily_fallback(
    month_period: &str,
    actual_start_date: &NaiveDate,
    end_date: &NaiveDate,
    daily_dates: &mut Vec<NaiveDate>
) {
    let year: i32 = month_period[0..4].parse().unwrap();
    let month: u32 = month_period[5..7].parse().unwrap();
    let month_start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
    let month_end = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap().pred_opt().unwrap()
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap().pred_opt().unwrap()
    };

    // Add days from this month to daily_dates for fallback
    let mut date = month_start;
    while date <= month_end && date <= *end_date {
        if date >= *actual_start_date {
            daily_dates.push(date);
        }
        date = date.succ_opt().unwrap();
    }
}

/// Main entry point to download all aggregate trades for a given crypto pair.
fn download_agg_trades_for_pair(
    pair: &str,
    start_date: NaiveDate,
    output_dir: &Path,
) -> Result<()> {
    let end_date = chrono::Utc::now().date_naive(); // Download up to today
    let client = reqwest::blocking::Client::new();
    fs::create_dir_all(output_dir)?;

    // Create partitioned directory structure: output_dir/pair/YYYY/MM/
    let pair_data_dir = output_dir.join(pair);
    fs::create_dir_all(&pair_data_dir)?;

    // Check if we already have data and determine the start date
    let actual_start_date = match get_latest_date_from_partitioned_data(&pair_data_dir)? {
        Some(latest_date) => {
            println!("Found existing data up to {}. Resuming from {}",
                     latest_date, latest_date + Duration::days(1));
            latest_date + Duration::days(1)
        }
        None => {
            println!("No existing data found. Starting from {}", start_date);
            start_date
        }
    };

    if actual_start_date > end_date {
        println!("Data is already up to date for {}", pair);
        return Ok(());
    }

    println!(
        "Starting download for {} from {} to {}",
        pair, actual_start_date, end_date
    );

    // Determine which months to download as monthly files vs daily files
    let current_month = end_date.format("%Y-%m").to_string();
    let mut monthly_periods = Vec::new();
    let mut daily_dates = Vec::new();

    // Group dates by month
    let mut current_date = actual_start_date;
    while current_date <= end_date {
        let month_str = current_date.format("%Y-%m").to_string();

        if month_str == current_month {
            // Current month - use daily downloads
            while current_date <= end_date && current_date.format("%Y-%m").to_string() == current_month {
                daily_dates.push(current_date);
                current_date = current_date.succ_opt().unwrap();
            }
        } else {
            // Historical month - try monthly download first
            monthly_periods.push(month_str.clone());
            // Skip to next month
            let next_month = if current_date.month() == 12 {
                NaiveDate::from_ymd_opt(current_date.year() + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(current_date.year(), current_date.month() + 1, 1).unwrap()
            };
            current_date = next_month;
        }
    }

    println!("Monthly periods to download: {}", monthly_periods.len());
    println!("Daily dates to download: {}", daily_dates.len());

    let mut total_saved = 0;

    // Download monthly data first
    for (i, month_period) in monthly_periods.iter().enumerate() {
        println!("Downloading monthly data for {} ({}/{})", month_period, i + 1, monthly_periods.len());

        match download_monthly_data(&client, pair, month_period) {
            Ok(Some(data)) => {
                println!("Downloaded {} rows for month {}", data.height(), month_period);
                save_monthly_parquet(&pair_data_dir, month_period, data)?;
                total_saved += 1;
                println!("✓ Saved monthly data for {}", month_period);
            }
            Ok(None) => {
                println!("⚠ No monthly data available for {}, will try daily downloads", month_period);
                add_month_to_daily_fallback(month_period, &actual_start_date, &end_date, &mut daily_dates);
            }
            Err(e) => {
                println!("⚠ Error downloading monthly data for {}: {}", month_period, e);
                println!("  Falling back to daily downloads for this month");
                add_month_to_daily_fallback(month_period, &actual_start_date, &end_date, &mut daily_dates);
            }
        }
    }

    // Download daily data in batches
    if !daily_dates.is_empty() {
        println!("Processing {} daily dates in batches...", daily_dates.len());

        for (batch_num, date_batch) in daily_dates.chunks(BATCH_SIZE).enumerate() {
            println!("Processing daily batch {} ({} dates)...", batch_num + 1, date_batch.len());

            // Process downloads with threading for this batch
            let successful_data = Arc::new(Mutex::new(Vec::new()));
            let mut handles = vec![];

            for chunk in date_batch.chunks((date_batch.len() + MAX_CONCURRENT_REQUESTS - 1) / MAX_CONCURRENT_REQUESTS) {
                for &date in chunk {
                    let client = client.clone();
                    let pair = pair.to_string();
                    let successful_data = Arc::clone(&successful_data);

                    let handle = thread::spawn(move || {
                        match download_day_data(&client, &pair, date) {
                            Ok(Some(data)) => {
                                successful_data.lock().unwrap().push(data);
                            }
                            Ok(None) => {
                                // No data for this day, that's fine
                            }
                            Err(e) => {
                                eprintln!("Error downloading data for {}: {}", date, e);
                            }
                        }
                    });
                    handles.push(handle);
                }
            }

            // Wait for all downloads to complete
            for handle in handles {
                handle.join().unwrap();
            }

            let batch_data = successful_data.lock().unwrap().clone();

            // Save this batch if we have data
            if !batch_data.is_empty() {
                let batch_count = batch_data.len();
                let total_rows: usize = batch_data.iter().map(|df| df.height()).sum();
                println!("Saving batch with {} DataFrames containing {} total rows",
                         batch_count, total_rows);
                save_daily_parquet_batch(&pair_data_dir, &date_batch, batch_data)?;
                total_saved += batch_count;
                println!("✓ Saved daily batch {} ({} days). Total saved so far: {}",
                         batch_num + 1, batch_count, total_saved);
            } else {
                println!("⚠ No data found for daily batch {}", batch_num + 1);
            }
        }
    }

    println!("✅ Download finished for {}. Total days saved: {}", pair, total_saved);

    // Final summary
    if total_saved > 0 {
        println!("📊 Final summary:");
        println!("   Data directory: {}", pair_data_dir.display());
        println!("   Total periods saved: {}", total_saved);

        // Count total files and estimate total size
        let mut total_files = 0;
        let mut total_size = 0;

        if pair_data_dir.exists() {
            for entry in walkdir::WalkDir::new(&pair_data_dir) {
                let entry = entry?;
                if entry.file_type().is_file() && entry.path().extension().map_or(false, |ext| ext == "parquet") {
                    total_files += 1;
                    total_size += entry.metadata()?.len();
                }
            }
        }

        println!("   Total Parquet files: {}", total_files);
        println!("   Total size: {:.2} MB", total_size as f64 / 1_048_576.0);
        println!("   Average file size: {:.2} MB", (total_size as f64 / total_files as f64) / 1_048_576.0);
    }

    Ok(())
}

/// Downloads the monthly data for a given month (YYYY-MM format) and returns it as a DataFrame.
fn download_monthly_data(
    client: &reqwest::blocking::Client,
    pair: &str,
    month_period: &str, // Format: "2024-01"
) -> Result<Option<DataFrame>> {
    let url = format!(
        "{}/{}/{}-aggTrades-{}.zip",
        MONTHLY_BASE_URL, pair, pair, month_period
    );

    println!("Attempting to download monthly data from: {}", url);

    match download_data_attempt(client, &url, month_period) {
        Ok(Some(df)) => {
            println!("✅ Successfully downloaded monthly data for {} ({} rows)", month_period, df.height());
            Ok(Some(df))
        },
        Ok(None) => {
            println!("ℹ️ No monthly data available for {} (404 response)", month_period);
            Ok(None)
        },
        Err(e) => {
            println!("❌ Failed to download monthly data for {}: {}", month_period, e);
            Err(e)
        },
    }
}

/// Downloads the data for a single day and returns it as a DataFrame.
/// Includes retry logic for transient failures.
fn download_day_data(
    client: &reqwest::blocking::Client,
    pair: &str,
    date: NaiveDate,
) -> Result<Option<DataFrame>> {
    const MAX_RETRIES: usize = 3;
    const RETRY_DELAY_MS: u64 = 1000;

    for attempt in 1..=MAX_RETRIES {
        let date_str = date.format("%Y-%m-%d").to_string();
        let url = format!(
            "{}/{}/{}-aggTrades-{}.zip",
            DAILY_BASE_URL, pair, pair, date_str
        );

        match download_data_attempt(client, &url, &date_str) {
            Ok(Some(df)) => return Ok(Some(df)),
            Ok(None) => return Ok(None), // No data available (404)
            Err(e) if attempt < MAX_RETRIES => {
                eprintln!("Attempt {} failed for {}: {}. Retrying...", attempt, date_str, e);
                std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS * attempt as u64));
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    unreachable!()
}

/// Single download attempt for any time period (day or month)
fn download_data_attempt(
    client: &reqwest::blocking::Client,
    url: &str,
    period_str: &str,
) -> Result<Option<DataFrame>> {
    // Make the HTTP request
    let response = client.get(url).send()
        .map_err(|e| anyhow::anyhow!("Failed to send request to {}: {}", url, e))?;

    // Check if the data exists for this day (Binance returns 404 if not)
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    // Check for other HTTP errors
    if !response.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error {}: {}", response.status(), url));
    }

    // Get the response bytes
    let zip_data = response.bytes()
        .map_err(|e| anyhow::anyhow!("Failed to read response body from {}: {}", url, e))?;

    // Unzip and parse the data
    let mut archive = ::zip::ZipArchive::new(std::io::Cursor::new(zip_data))
        .map_err(|e| anyhow::anyhow!("Failed to read ZIP archive for {}: {}", period_str, e))?;

    if archive.is_empty() {
        return Err(anyhow::anyhow!("Empty zip file for {}", period_str));
    }

    let mut file_in_zip = archive.by_index(0)
        .map_err(|e| anyhow::anyhow!("Failed to extract file from ZIP for {}: {}", period_str, e))?;

    let mut csv_content = String::new();
    file_in_zip.read_to_string(&mut csv_content)
        .map_err(|e| anyhow::anyhow!("Failed to read CSV content for {}: {}", period_str, e))?;

    // Parse CSV content with Polars
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(std::io::Cursor::new(csv_content.as_bytes()))
        .finish()?
        .lazy()
        .with_columns([
            col("column_1").alias("agg_trade_id"),
            col("column_2").alias("price"),
            col("column_3").alias("quantity"),
            col("column_4").alias("first_trade_id"),
            col("column_5").alias("last_trade_id"),
            col("column_6").alias("timestamp"),
            col("column_7").alias("is_buyer_maker"),
            col("column_8").alias("is_best_match"),
        ])
        .select([
            col("agg_trade_id"),
            col("price"),
            col("quantity"),
            col("first_trade_id"),
            col("last_trade_id"),
            col("timestamp"),
            col("is_buyer_maker"),
            col("is_best_match"),
        ])
        .collect()?;

    println!("Successfully downloaded and parsed data for {} ({} rows)",
             period_str, df.height());

    Ok(Some(df))
}

/// Save monthly data to a partitioned Parquet file
fn save_monthly_parquet(data_dir: &Path, month_period: &str, data: DataFrame) -> Result<()> {
    // Parse month period (YYYY-MM)
    let year = &month_period[0..4];
    let month = &month_period[5..7];

    // Create directory structure: data_dir/YYYY/MM/
    let month_dir = data_dir.join(year).join(month);
    fs::create_dir_all(&month_dir)?;

    // Save as YYYY-MM.parquet
    let file_path = month_dir.join(format!("{}.parquet", month_period));

    if file_path.exists() {
        println!("Monthly file already exists: {}", file_path.display());
        return Ok(());
    }

    let sorted_df = data
        .lazy()
        .sort(["timestamp"], SortMultipleOptions::default())
        .collect()?;

    let mut file = File::create(&file_path)?;
    ParquetWriter::new(&mut file)
        .finish(&mut sorted_df.clone())?;

    println!("✅ Saved monthly data: {} ({} rows)", file_path.display(), sorted_df.height());
    Ok(())
}

/// Save daily data batch to partitioned Parquet files
fn save_daily_parquet_batch(data_dir: &Path, dates: &[NaiveDate], dataframes: Vec<DataFrame>) -> Result<()> {
    // Each DataFrame corresponds to a date
    for (date, df) in dates.iter().zip(dataframes.iter()) {
        let year = date.year().to_string();
        let month = format!("{:02}", date.month());

        // Create directory structure: data_dir/YYYY/MM/
        let month_dir = data_dir.join(&year).join(&month);
        fs::create_dir_all(&month_dir)?;

        // Save as YYYY-MM-DD.parquet
        let file_path = month_dir.join(format!("{}.parquet", date.format("%Y-%m-%d")));

        if file_path.exists() {
            println!("Daily file already exists: {}", file_path.display());
            continue;
        }

        let sorted_df = df
            .clone()
            .lazy()
            .sort(["timestamp"], SortMultipleOptions::default())
            .collect()?;

        let mut file = File::create(&file_path)?;
        ParquetWriter::new(&mut file)
            .finish(&mut sorted_df.clone())?;

        println!("✅ Saved daily data: {} ({} rows)", file_path.display(), sorted_df.height());
    }

    Ok(())
}

fn main() -> Result<()> {
    // --- Configuration ---
    let pair_to_download = "BTCUSDT";
    // Binance's BTC data starts on 2017-08-17
    let start_date = NaiveDate::from_ymd_opt(2017, 8, 17).unwrap();
    let output_dir = PathBuf::from("./crypto_data");

    download_agg_trades_for_pair(pair_to_download, start_date, &output_dir)?;

    // You can add more pairs here
    // let pair_to_download_2 = "ETHUSDT";
    // let start_date_2 = NaiveDate::from_ymd_opt(2017, 8, 17).unwrap();
    // download_agg_trades_for_pair(pair_to_download_2, start_date_2, &output_dir)?;

    Ok(())
}