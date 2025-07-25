use polars::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use ::zip::ZipArchive;
use ordered_float::OrderedFloat;
use chrono::{NaiveDate, Utc, Duration};
use rayon::prelude::*;

const BYBIT_BASE_URL: &str = "https://quote-saver.bycsi.com/orderbook/spot";

#[derive(Debug, Clone)]
struct OrderBookLevel {
    price: f64,
    quantity: f64,
}

#[derive(Debug, Clone)]
struct OrderBookSnapshot {
    timestamp: i64,        // nanoseconds
    bids: Vec<OrderBookLevel>,
    asks: Vec<OrderBookLevel>,
}

#[derive(Debug, Clone)]
enum OrderBookEvent {
    Snapshot(OrderBookSnapshot),
    Delta {
        timestamp: i64,
        bids: Vec<(f64, f64)>, // (price, quantity) - 0.0 quantity = deletion
        asks: Vec<(f64, f64)>,
    },
}

fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(url)?;
    let mut file = File::create(path)?;
    let content = response.bytes()?;
    file.write_all(&content)?;
    Ok(())
}

fn extract_zip(zip_path: &Path, extract_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;
    archive.extract(extract_dir)?;
    Ok(())
}

fn parse_orderbook_file(file_path: &Path) -> Result<Vec<OrderBookEvent>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let json: serde_json::Value = serde_json::from_str(&line)?;
        
        let timestamp = json["ts"].as_i64().unwrap_or(0) * 1_000_000; // Convert to nanoseconds
        let event_type = json["type"].as_str().unwrap_or("");
        
        match event_type {
            "snapshot" => {
                let mut bids = Vec::new();
                let mut asks = Vec::new();
                
                if let Some(bid_array) = json["data"]["b"].as_array() {
                    for item in bid_array {
                        if let (Some(price_str), Some(qty_str)) = (
                            item[0].as_str(), 
                            item[1].as_str()
                        ) {
                            if let (Ok(price), Ok(quantity)) = (
                                price_str.parse::<f64>(),
                                qty_str.parse::<f64>()
                            ) {
                                bids.push(OrderBookLevel { price, quantity });
                            }
                        }
                    }
                }
                
                if let Some(ask_array) = json["data"]["a"].as_array() {
                    for item in ask_array {
                        if let (Some(price_str), Some(qty_str)) = (
                            item[0].as_str(), 
                            item[1].as_str()
                        ) {
                            if let (Ok(price), Ok(quantity)) = (
                                price_str.parse::<f64>(),
                                qty_str.parse::<f64>()
                            ) {
                                asks.push(OrderBookLevel { price, quantity });
                            }
                        }
                    }
                }
                
                events.push(OrderBookEvent::Snapshot(OrderBookSnapshot {
                    timestamp,
                    bids,
                    asks,
                }));
            },
            "delta" => {
                let mut bids = Vec::new();
                let mut asks = Vec::new();
                
                if let Some(bid_array) = json["data"]["b"].as_array() {
                    for item in bid_array {
                        if let (Some(price_str), Some(qty_str)) = (
                            item[0].as_str(), 
                            item[1].as_str()
                        ) {
                            if let (Ok(price), Ok(quantity)) = (
                                price_str.parse::<f64>(),
                                qty_str.parse::<f64>()
                            ) {
                                bids.push((price, quantity));
                            }
                        }
                    }
                }
                
                if let Some(ask_array) = json["data"]["a"].as_array() {
                    for item in ask_array {
                        if let (Some(price_str), Some(qty_str)) = (
                            item[0].as_str(), 
                            item[1].as_str()
                        ) {
                            if let (Ok(price), Ok(quantity)) = (
                                price_str.parse::<f64>(),
                                qty_str.parse::<f64>()
                            ) {
                                asks.push((price, quantity));
                            }
                        }
                    }
                }
                
                events.push(OrderBookEvent::Delta {
                    timestamp,
                    bids,
                    asks,
                });
            },
            _ => continue,
        }
    }
    
    Ok(events)
}

fn reconstruct_orderbook_snapshots_fast(events: Vec<OrderBookEvent>) -> Result<Vec<OrderBookSnapshot>, Box<dyn std::error::Error>> {
    println!("Reconstructing orderbook snapshots with time-based sampling...");

    let mut snapshots = Vec::new();
    let mut current_bids: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();
    let mut current_asks: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();

    let total_events = events.len();

    // Time-based sampling: snapshot every N milliseconds (better for ML training)
    let sample_interval_ms = 100; // 100ms intervals = 10 snapshots per second
    let sample_interval_ns = sample_interval_ms * 1_000_000; // Convert to nanoseconds
    let top_levels = 20; // Keep top 20 levels per side (good for ML features)

    let mut last_snapshot_time = 0i64;

    println!("Processing {} events with {}ms time intervals", total_events, sample_interval_ms);

    for (event_idx, event) in events.iter().enumerate() {

        match event {
            OrderBookEvent::Snapshot(snapshot) => {
                // Update current book state
                current_bids.clear();
                current_asks.clear();

                for level in &snapshot.bids {
                    if level.quantity > 0.0 {
                        current_bids.insert(OrderedFloat(level.price), level.quantity);
                    }
                }

                for level in &snapshot.asks {
                    if level.quantity > 0.0 {
                        current_asks.insert(OrderedFloat(level.price), level.quantity);
                    }
                }

                // Always save initial snapshots
                snapshots.push(snapshot.clone());
            },
            OrderBookEvent::Delta { timestamp, bids, asks } => {
                // Apply bid updates
                for (price, quantity) in bids {
                    if *quantity == 0.0 {
                        current_bids.remove(&OrderedFloat(*price));
                    } else {
                        current_bids.insert(OrderedFloat(*price), *quantity);
                    }
                }

                // Apply ask updates
                for (price, quantity) in asks {
                    if *quantity == 0.0 {
                        current_asks.remove(&OrderedFloat(*price));
                    } else {
                        current_asks.insert(OrderedFloat(*price), *quantity);
                    }
                }

                // Time-based sampling: only create snapshot if enough time has passed
                if *timestamp - last_snapshot_time >= sample_interval_ns {
                    // Create new snapshot with limited levels
                    let bid_levels: Vec<OrderBookLevel> = current_bids
                        .iter()
                        .rev() // Highest prices first for bids
                        .take(top_levels)
                        .map(|(&price, &quantity)| OrderBookLevel { price: price.into_inner(), quantity })
                        .collect();

                    let ask_levels: Vec<OrderBookLevel> = current_asks
                        .iter() // Lowest prices first for asks
                        .take(top_levels)
                        .map(|(&price, &quantity)| OrderBookLevel { price: price.into_inner(), quantity })
                        .collect();

                    snapshots.push(OrderBookSnapshot {
                        timestamp: *timestamp,
                        bids: bid_levels,
                        asks: ask_levels,
                    });

                    last_snapshot_time = *timestamp;
                }
            },
        }
    }

    println!("Created {} sampled snapshots from {} events ({}x reduction)",
             snapshots.len(), total_events, total_events / snapshots.len().max(1));

    Ok(snapshots)
}


fn convert_to_hft_format_fast(snapshots: Vec<OrderBookSnapshot>, output_path: &Path, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    // HFT backtest configuration
    let tick_size = get_tick_size(symbol);

    println!("Converting to hftbacktest format with parallel processing...");
    println!("Using tick size {} for symbol {}", tick_size, symbol);

    // Pre-calculate total capacity to avoid reallocations
    let total_levels: usize = snapshots.iter()
        .map(|s| s.bids.len() + s.asks.len())
        .sum();

    println!("Processing {} snapshots with ~{} total levels", snapshots.len(), total_levels);

    // Pre-allocate vectors with known capacity
    let mut ev_flags = Vec::with_capacity(total_levels);
    let mut exch_timestamps = Vec::with_capacity(total_levels);
    let mut local_timestamps = Vec::with_capacity(total_levels);
    let mut prices = Vec::with_capacity(total_levels);
    let mut quantities = Vec::with_capacity(total_levels);
    let mut order_ids = Vec::with_capacity(total_levels);
    let mut ivals = Vec::with_capacity(total_levels);
    let mut fvals = Vec::with_capacity(total_levels);

    let mut order_id_counter = 1u64;

    // Process snapshots in chunks for better progress reporting
    let chunk_size = 10_000;
    let chunks: Vec<_> = snapshots.chunks(chunk_size).collect();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        

        for snapshot in chunk.iter() {
            // Process bids - use extend for better performance
            for bid in &snapshot.bids {
                if bid.quantity > 0.0 {
                    ev_flags.push(0x200000001u64); // LOCAL_BID_DEPTH_EVENT
                    exch_timestamps.push(snapshot.timestamp);
                    local_timestamps.push(snapshot.timestamp);
                    prices.push((bid.price / tick_size).round() as i64);
                    quantities.push(bid.quantity);
                    order_ids.push(order_id_counter);
                    order_id_counter += 1;
                    ivals.push(0i64);
                    fvals.push(0.0f64);
                }
            }

            // Process asks
            for ask in &snapshot.asks {
                if ask.quantity > 0.0 {
                    ev_flags.push(0x400000001u64); // LOCAL_ASK_DEPTH_EVENT
                    exch_timestamps.push(snapshot.timestamp);
                    local_timestamps.push(snapshot.timestamp);
                    prices.push(-((ask.price / tick_size).round() as i64));
                    quantities.push(ask.quantity);
                    order_ids.push(order_id_counter);
                    order_id_counter += 1;
                    ivals.push(0i64);
                    fvals.push(0.0f64);
                }
            }
        }
    }

    println!("Creating DataFrame with {} events...", ev_flags.len());

    // Create DataFrame in hftbacktest Event struct format
    let df = df![
        "ev" => ev_flags,
        "exch_ts" => exch_timestamps,
        "local_ts" => local_timestamps,
        "px" => prices,
        "qty" => quantities,
        "order_id" => order_ids,
        "ival" => ivals,
        "fval" => fvals,
    ]?;

    println!("Writing parquet file...");

    // Save as parquet with compression for better I/O performance
    let file = File::create(output_path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df.clone())?;

    println!("✅ Saved {} hftbacktest events to {:?}", df.height(), output_path);
    Ok(())
}

/// Get tick size for a given symbol (should be configurable or fetched from exchange)
fn get_tick_size(symbol: &str) -> f64 {
    match symbol {
        "BTCUSDC" | "BTCUSDT" => 0.1,
        "ETHUSDC" | "ETHUSDT" => 0.1,
        _ => {
            println!("⚠️  Unknown symbol {}, using default tick size 0.01", symbol);
            0.1
        }
    }
}

/// Check if data already exists for a given symbol and date
fn data_already_exists(symbol: &str, date: &str) -> bool {
    let output_path = format!("/mnt/storage-box/bybit/orderbooks/output/{}_{}_hft.parquet", date, symbol);
    Path::new(&output_path).exists()
}

/// Generate date range from start_date to today
fn generate_date_range(start_date: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")?;
    let today = Utc::now().naive_utc().date();

    let mut dates = Vec::new();
    let mut current = start;

    while current <= today {
        dates.push(current.format("%Y-%m-%d").to_string());
        current = current + Duration::days(1);
    }

    Ok(dates)
}

fn process_orderbook_data(symbol: &str, date: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("{}/{}/{}_{}_ob200.data.zip", BYBIT_BASE_URL, symbol, date, symbol);
    let zip_file_path = format!("/mnt/storage-box/bybit/orderbooks/data/{}_{}_ob200.data.zip", date, symbol);
    let extract_dir = format!("/mnt/storage-box/bybit/orderbooks/data/{}", date);
    let data_file_path = format!("{}/{}_{}_ob200.data", extract_dir, date, symbol);
    let output_path = format!("/mnt/storage-box/bybit/orderbooks/output/{}_{}_hft.parquet", date, symbol);
    
    // Create directories
    std::fs::create_dir_all("/mnt/storage-box/bybit/orderbooks/data")?;
    std::fs::create_dir_all("/mnt/storage-box/bybit/orderbooks/output")?;
    std::fs::create_dir_all(&extract_dir)?;
    
    println!("Downloading {}...", url);
    download_file(&url, Path::new(&zip_file_path))?;
    
    println!("Extracting...");
    extract_zip(Path::new(&zip_file_path), Path::new(&extract_dir))?;
    
    println!("Parsing orderbook data...");
    let events = parse_orderbook_file(Path::new(&data_file_path))?;
    
    let snapshots = reconstruct_orderbook_snapshots_fast(events)?;

    convert_to_hft_format_fast(snapshots, Path::new(&output_path), symbol)?;
    
    println!("Processing complete!");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let start_date = "2025-04-30"; // Start from this date
    let symbols = ["BTCUSDC", "ETHUSDC", "BTCUSDT"];

    println!("🚀 Starting bulk download from {} to today", start_date);
    println!("Symbols: {:?}", symbols);

    // Generate date range
    let dates = generate_date_range(start_date)?;
    println!("📅 Generated {} dates to process", dates.len());

    let mut total_processed = 0;
    let mut total_skipped = 0;
    let mut total_errors = 0;

    // Process each date
    for (date_idx, date) in dates.iter().enumerate() {
        println!("\n📆 Processing date {} ({}/{})...", date, date_idx + 1, dates.len());

        // Process each symbol for this date
        for symbol in &symbols {
            // Check if data already exists
            if data_already_exists(symbol, date) {
                println!("⏭️  Skipping {}-{} (already exists)", symbol, date);
                total_skipped += 1;
                continue;
            }

            println!("📥 Downloading {}-{}...", symbol, date);

            match process_orderbook_data(symbol, date) {
                Ok(()) => {
                    println!("✅ Successfully processed {}-{}", symbol, date);
                    total_processed += 1;
                },
                Err(e) => {
                    println!("❌ Error processing {}-{}: {}", symbol, date, e);
                    total_errors += 1;

                    // Continue with next symbol/date instead of failing completely
                    continue;
                }
            }

            // Small delay to be nice to the server
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    println!("\n🎉 Bulk download complete!");
    println!("📊 Summary:");
    println!("  ✅ Processed: {}", total_processed);
    println!("  ⏭️  Skipped: {}", total_skipped);
    println!("  ❌ Errors: {}", total_errors);
    println!("  📅 Total dates: {}", dates.len());
    println!("  🔢 Total symbols: {}", symbols.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_orderbook_processing() {
        // This would be an integration test with sample data
        // In practice, you'd download a small sample file for testing
    }
}