use anyhow::Result;
use chrono::{Datelike, Duration, NaiveDate};
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir;

const DAILY_BASE_URL: &str = "https://data.binance.vision/data/spot/daily/aggTrades";
const MONTHLY_BASE_URL: &str = "https://data.binance.vision/data/spot/monthly/aggTrades";
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


    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600)) // 10 minutes
        .build()?;
    
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

            // Use rayon for parallel processing - much more efficient than manual threading
            let batch_results: Vec<_> = date_batch.par_iter().map(|&date| {
                match download_day_data(&client, pair, date) {
                    Ok(Some(data)) => Some((date, data)),
                    Ok(None) => {
                        // No data for this day, that's fine
                        None
                    }
                    Err(e) => {
                        eprintln!("Error downloading data for {}: {}", date, e);
                        None
                    }
                }
            }).collect();

            // Extract successful downloads
            let batch_data: Vec<_> = batch_results.into_iter()
                .filter_map(|result| result.map(|(_, df)| df))
                .collect();

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
            /*let output = Command::new("/bin/bash")
                .arg("-c")
                .arg("mullvad reconnect")
                .output();

            println!("changing ip and trying again... \n({:?})", output);
            */
            std::thread::sleep(std::time::Duration::from_secs(2));
            download_monthly_data(client, pair, month_period)
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

    // Parse CSV content with Polars - keep it lazy and minimal processing
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(std::io::Cursor::new(csv_content.as_bytes()))
        .finish()?
        .lazy()
        .with_columns([
            col("column_1").alias("agg_trade_id").cast(DataType::UInt64),
            col("column_2").alias("price").cast(DataType::Float64),
            col("column_3").alias("quantity").cast(DataType::Float64),
            col("column_4").alias("first_trade_id").cast(DataType::UInt64),
            col("column_5").alias("last_trade_id").cast(DataType::UInt64),
            col("column_6").alias("timestamp").cast(DataType::Int64),
            col("column_7").alias("is_buyer_maker").cast(DataType::Boolean),
            col("column_8").alias("is_best_match").cast(DataType::Boolean),
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

/// Save monthly data to a partitioned Parquet file (no sorting - data should already be sorted)
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

    // Write directly without sorting - Binance data should already be chronological
    let mut file = File::create(&file_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut data.clone())?;

    println!("✅ Saved monthly data: {} ({} rows)", file_path.display(), data.height());
    Ok(())
}

/// Save daily data batch to partitioned Parquet files (optimized)
fn save_daily_parquet_batch(data_dir: &Path, dates: &[NaiveDate], dataframes: Vec<DataFrame>) -> Result<()> {
    // Process files in parallel using rayon

    let results: Vec<Result<()>> = dates.par_iter().zip(dataframes.par_iter()).map(|(date, df)| {
        let year = date.year().to_string();
        let month = format!("{:02}", date.month());

        // Create directory structure: data_dir/YYYY/MM/
        let month_dir = data_dir.join(&year).join(&month);
        fs::create_dir_all(&month_dir)?;

        // Save as YYYY-MM-DD.parquet
        let file_path = month_dir.join(format!("{}.parquet", date.format("%Y-%m-%d")));

        if file_path.exists() {
            println!("Daily file already exists: {}", file_path.display());
            return Ok(());
        }

        // Write directly without sorting - Binance data should already be chronological
        let mut file = File::create(&file_path)?;
        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Snappy)
            .finish(&mut df.clone())?;

        println!("✅ Saved daily data: {} ({} rows)", file_path.display(), df.height());
        Ok(())
    }).collect();

    // Check for any errors
    for result in results {
        result?;
    }

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

/// Fetch the start date for a crypto pair by scraping Binance's data directory
fn get_start_date_for_pair(client: &reqwest::blocking::Client, pair: &str) -> NaiveDate {
    match fetch_start_date_from_binance(client, pair) {
        Ok(date) => {
            println!("✅ Found start date for {}: {}", pair, date);
            date
        }
        Err(e) => {
            println!("⚠️ Failed to fetch start date for {}: {}. Using fallback date.", pair, e);
            // Fallback to a conservative date
            NaiveDate::from_ymd_opt(2020, 1, 1).unwrap()
        }
    }
}

/// Fetch the earliest available month for a crypto pair from Binance's data directory
fn fetch_start_date_from_binance(client: &reqwest::blocking::Client, pair: &str) -> Result<NaiveDate> {
    let url = format!("https://data.binance.vision/?prefix=data/spot/monthly/aggTrades/{}/", pair);

    println!("🔍 Fetching start date for {} from: {}", pair, url);

    // Make the request
    let response = client.get(&url).send()
        .map_err(|e| anyhow::anyhow!("Failed to fetch data directory for {}: {}", pair, e))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error {} for {}", response.status(), pair));
    }

    let html_content = response.text()
        .map_err(|e| anyhow::anyhow!("Failed to read HTML content for {}: {}", pair, e))?;

    // Parse the HTML to find the earliest month
    parse_earliest_month_from_html(&html_content, pair)
}

/// Parse HTML content to find the earliest available month
fn parse_earliest_month_from_html(html_content: &str, pair: &str) -> Result<NaiveDate> {
    // The Binance data directory uses JavaScript to load content dynamically
    // We need to look for the S3 bucket structure or make a direct API call
    // Let's try the S3 API approach instead

    // Extract the S3 bucket URL from the HTML
    if html_content.contains("s3-ap-northeast-1.amazonaws.com/data.binance.vision") {
        // Use the S3 API to list objects
        return fetch_start_date_from_s3_api(pair);
    }

    Err(anyhow::anyhow!("Could not parse HTML content for {}", pair))
}

/// Fetch start date using S3 API directly
fn fetch_start_date_from_s3_api(pair: &str) -> Result<NaiveDate> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Use S3 list-objects API
    let s3_url = format!(
        "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?list-type=2&prefix=data/spot/monthly/aggTrades/{}/&delimiter=/",
        pair
    );

    println!("🔍 Fetching from S3 API: {}", s3_url);

    let response = client.get(&s3_url).send()
        .map_err(|e| anyhow::anyhow!("Failed to fetch S3 listing for {}: {}", pair, e))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("S3 API error {} for {}", response.status(), pair));
    }

    let xml_content = response.text()
        .map_err(|e| anyhow::anyhow!("Failed to read S3 XML for {}: {}", pair, e))?;

    parse_earliest_month_from_s3_xml(&xml_content, pair)
}

/// Parse S3 XML response to find the earliest month
fn parse_earliest_month_from_s3_xml(xml_content: &str, pair: &str) -> Result<NaiveDate> {
    let mut earliest_month: Option<String> = None;

    // Look for <Key> elements containing month patterns like "BTCUSDT-aggTrades-2017-08.zip"
    for line in xml_content.lines() {
        if line.contains("<Key>") && line.contains(&format!("{}-aggTrades-", pair)) {
            // Extract the key content
            if let Some(start) = line.find("<Key>") {
                if let Some(end) = line.find("</Key>") {
                    let key = &line[start + 5..end];

                    // Extract month pattern: PAIR-aggTrades-YYYY-MM.zip
                    if let Some(month_start) = key.rfind("-aggTrades-") {
                        let month_part = &key[month_start + 11..];
                        if let Some(zip_pos) = month_part.find(".zip") {
                            let month_str = &month_part[..zip_pos]; // Should be YYYY-MM

                            if month_str.len() == 7 && month_str.chars().nth(4) == Some('-') {
                                match &earliest_month {
                                    None => earliest_month = Some(month_str.to_string()),
                                    Some(current_earliest) => {
                                        if month_str < current_earliest.as_str() {
                                            earliest_month = Some(month_str.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    match earliest_month {
        Some(month_str) => {
            // Parse YYYY-MM format and return the first day of that month
            let year: i32 = month_str[0..4].parse()
                .map_err(|_| anyhow::anyhow!("Invalid year in month string: {}", month_str))?;
            let month: u32 = month_str[5..7].parse()
                .map_err(|_| anyhow::anyhow!("Invalid month in month string: {}", month_str))?;

            NaiveDate::from_ymd_opt(year, month, 1)
                .ok_or_else(|| anyhow::anyhow!("Invalid date: {}-{}-01", year, month))
        }
        None => Err(anyhow::anyhow!("No monthly data found for {}", pair))
    }
}

fn main() -> Result<()> {
    // Configure Polars to use all available CPU cores
    let num_cores = num_cpus::get();
    unsafe {
        std::env::set_var("POLARS_MAX_THREADS", num_cores.to_string());
    }

    // Configure rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cores)
        .build_global()
        .unwrap();

    println!("Using {} CPU cores for parallel processing", num_cores);

    // --- Configuration ---
    let output_dir = PathBuf::from("/mnt/storage-box/crypto_data");
    let pairlist_file = "pairlist.txt";
    const BATCH_SIZE: usize = 5; // Process 5 pairs in parallel at a time

    // Read crypto pairs from file
    let pair_names = read_crypto_pairs_from_file(pairlist_file)?;

    println!("Starting batch download for {} crypto pairs in batches of {}",
             pair_names.len(), BATCH_SIZE);

    let mut total_successful = 0;
    let mut total_failed = 0;
    let total_batches = (pair_names.len() + BATCH_SIZE - 1) / BATCH_SIZE;

    // Process pairs in batches of 5
    for (batch_num, batch) in pair_names.chunks(BATCH_SIZE).enumerate() {
        println!("\n🚀 Starting batch {} of {} ({} pairs)",
                 batch_num + 1, total_batches, batch.len());

        // Show which pairs are in this batch
        for pair in batch {
            println!("   - {} (start date will be fetched dynamically)", pair);
        }

        // Download this batch in parallel
        let batch_results: Vec<Result<()>> = batch.par_iter().map(|pair| {
            println!("� Starting download for {}", pair);

            // Create HTTP client for fetching start date
            let start_date_client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new());

            // Fetch start date dynamically
            let start_date = get_start_date_for_pair(&start_date_client, pair);
            println!("📅 Using start date {} for {}", start_date, pair);

            let result = download_agg_trades_for_pair(pair, start_date, &output_dir);
            match &result {
                Ok(_) => println!("✅ Completed download for {}", pair),
                Err(e) => println!("❌ Failed download for {}: {}", pair, e),
            }
            result
        }).collect();

        // Count results for this batch
        let mut batch_successful = 0;
        let mut batch_failed = 0;

        for (_i, result) in batch_results.iter().enumerate() {
            match result {
                Ok(_) => {
                    batch_successful += 1;
                    total_successful += 1;
                }
                Err(_) => {
                    batch_failed += 1;
                    total_failed += 1;
                }
            }
        }

        println!("📊 Batch {} Summary: {} successful, {} failed",
                 batch_num + 1, batch_successful, batch_failed);

        // Small delay between batches to be nice to the server
        if batch_num + 1 < total_batches {
            println!("⏳ Waiting 5 seconds before next batch...");
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
    }

    println!("\n🎉 Final Summary:");
    println!("   Total pairs processed: {}", pair_names.len());
    println!("   Total successful: {}", total_successful);
    println!("   Total failed: {}", total_failed);
    println!("   Success rate: {:.1}%", (total_successful as f64 / pair_names.len() as f64) * 100.0);
    println!("   Data directory: {}", output_dir.display());

    if total_failed > 0 {
        println!("\n⚠️  Some downloads failed. You can re-run the program to retry failed downloads.");
        println!("   Failed pairs will be automatically detected and resumed from where they left off.");
    }

    Ok(())
}