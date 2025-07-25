use polars::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Write, Read};
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

/// Parse a single JSON line into an OrderBookEvent
fn parse_orderbook_line(line: &str) -> Option<OrderBookEvent> {
    let json: serde_json::Value = serde_json::from_str(line).ok()?;

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

            Some(OrderBookEvent::Snapshot(OrderBookSnapshot {
                timestamp,
                bids,
                asks,
            }))
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

            Some(OrderBookEvent::Delta {
                timestamp,
                bids,
                asks,
            })
        },
        _ => None,
    }
}

/// Parallelized orderbook file parser - uses all CPU cores for faster processing
fn parse_orderbook_file(file_path: &Path) -> Result<Vec<OrderBookEvent>, Box<dyn std::error::Error>> {
    println!("📊 Reading file into memory for parallel processing...");

    // Read entire file into memory for parallel processing
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    println!("📊 Parsing {} lines in parallel...", contents.lines().count());

    // Split into lines and process in parallel
    let lines: Vec<&str> = contents.lines().collect();
    let chunk_size = (lines.len() / rayon::current_num_threads()).max(1000); // At least 1000 lines per chunk

    println!("📊 Using {} threads with ~{} lines per chunk",
             rayon::current_num_threads(), chunk_size);

    // Process chunks in parallel, then flatten results while preserving order
    let events: Vec<OrderBookEvent> = lines
        .par_chunks(chunk_size)
        .map(|chunk| {
            // Process each chunk sequentially to preserve order within chunk
            chunk.iter()
                .filter_map(|line| parse_orderbook_line(line))
                .collect::<Vec<_>>()
        })
        .flatten() // Flatten the chunks back into a single vector
        .collect();

    println!("✅ Parsed {} events using parallel processing", events.len());
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

    for (_event_idx, event) in events.iter().enumerate() {

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



/// Save orderbook snapshots in FinancialBERT training compatible format
///
/// This function:
/// 1. Filters snapshots to 200ms intervals (5 ticks per second)
/// 2. Normalizes prices and quantities using z-score normalization (μ=0, σ=1)
/// 3. Creates feature vectors with configurable orderbook depth (default: 20 levels)
/// 4. Saves as parquet file with timestamp + feature columns
///
/// Output format: timestamp, bid_price_0, bid_qty_0, ask_price_0, ask_qty_0, ..., bid_price_19, bid_qty_19, ask_price_19, ask_qty_19
/// Each row represents one 200ms tick with normalized orderbook features
fn save_financialbert_dataset(snapshots: Vec<OrderBookSnapshot>, output_path: &Path, _symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting to FinancialBERT training format...");

    // Configuration for FinancialBERT training
    let orderbook_depth = 20; // Top 20 levels per side (reasonable depth)
    let tick_interval_ms = 200; // 200ms ticks as requested
    let tick_interval_ns = tick_interval_ms * 1_000_000; // Convert to nanoseconds

    // Filter snapshots to 200ms intervals
    let mut filtered_snapshots = Vec::new();
    let mut last_tick_time = 0i64;

    for snapshot in snapshots {
        if snapshot.timestamp - last_tick_time >= tick_interval_ns {
            filtered_snapshots.push(snapshot);
            last_tick_time = filtered_snapshots.last().unwrap().timestamp;
        }
    }

    println!("Filtered to {} snapshots at {}ms intervals", filtered_snapshots.len(), tick_interval_ms);

    if filtered_snapshots.is_empty() {
        return Err("No snapshots available for FinancialBERT dataset".into());
    }

    // Calculate normalization parameters from all data
    let mut all_prices = Vec::new();
    let mut all_quantities = Vec::new();

    for snapshot in &filtered_snapshots {
        for bid in &snapshot.bids {
            if bid.price > 0.0 && bid.quantity > 0.0 {
                all_prices.push(bid.price);
                all_quantities.push(bid.quantity);
            }
        }
        for ask in &snapshot.asks {
            if ask.price > 0.0 && ask.quantity > 0.0 {
                all_prices.push(ask.price);
                all_quantities.push(ask.quantity);
            }
        }
    }

    // Calculate normalization statistics
    let price_mean = all_prices.iter().sum::<f64>() / all_prices.len() as f64;
    let price_std = {
        let variance = all_prices.iter()
            .map(|&p| (p - price_mean).powi(2))
            .sum::<f64>() / all_prices.len() as f64;
        variance.sqrt()
    };

    let qty_mean = all_quantities.iter().sum::<f64>() / all_quantities.len() as f64;
    let qty_std = {
        let variance = all_quantities.iter()
            .map(|&q| (q - qty_mean).powi(2))
            .sum::<f64>() / all_quantities.len() as f64;
        variance.sqrt()
    };

    println!("Normalization stats - Price: μ={:.2}, σ={:.2} | Qty: μ={:.4}, σ={:.4}",
             price_mean, price_std, qty_mean, qty_std);

    // Create normalized feature vectors in parallel
    let feature_size = orderbook_depth * 4; // bid_price, bid_qty, ask_price, ask_qty per level

    println!("Creating normalized features using {} threads...", rayon::current_num_threads());

    // Process snapshots in parallel to create feature vectors
    let chunk_size = (filtered_snapshots.len() / rayon::current_num_threads()).max(100);
    let results: Vec<_> = filtered_snapshots
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_timestamps = Vec::with_capacity(chunk.len());
            let mut chunk_features = Vec::with_capacity(chunk.len() * feature_size);

            for snapshot in chunk {
                chunk_timestamps.push(snapshot.timestamp);

                // Create normalized feature vector for this snapshot
                let mut features = vec![0.0f32; feature_size];

                // Process bids (highest prices first)
                for (i, bid) in snapshot.bids.iter().take(orderbook_depth).enumerate() {
                    let base_idx = i * 4;
                    if bid.price > 0.0 && bid.quantity > 0.0 {
                        // Normalize price and quantity using z-score normalization
                        features[base_idx] = ((bid.price - price_mean) / price_std) as f32;     // bid_price
                        features[base_idx + 1] = ((bid.quantity - qty_mean) / qty_std) as f32; // bid_qty
                    }
                    // ask_price and ask_qty remain 0.0 for bid levels
                }

                // Process asks (lowest prices first)
                for (i, ask) in snapshot.asks.iter().take(orderbook_depth).enumerate() {
                    let base_idx = i * 4;
                    if ask.price > 0.0 && ask.quantity > 0.0 {
                        // Normalize price and quantity using z-score normalization
                        features[base_idx + 2] = ((ask.price - price_mean) / price_std) as f32; // ask_price
                        features[base_idx + 3] = ((ask.quantity - qty_mean) / qty_std) as f32;  // ask_qty
                    }
                    // bid_price and bid_qty remain 0.0 for ask levels
                }

                chunk_features.extend_from_slice(&features);
            }

            (chunk_timestamps, chunk_features)
        })
        .collect();

    // Combine results from all chunks
    let mut timestamps = Vec::with_capacity(filtered_snapshots.len());
    let mut feature_matrix = Vec::with_capacity(filtered_snapshots.len() * feature_size);

    for (chunk_timestamps, chunk_features) in results {
        timestamps.extend(chunk_timestamps);
        feature_matrix.extend(chunk_features);
    }

    // Create DataFrame in FinancialBERT compatible format
    // Structure: timestamp column + feature columns (bid_price_0, bid_qty_0, ask_price_0, ask_qty_0, ...)
    let mut columns = vec![
        Series::new("timestamp".into(), timestamps)
    ];

    // Add feature columns
    for level in 0..orderbook_depth {
        let bid_price_col: Vec<f32> = (0..filtered_snapshots.len())
            .map(|t| feature_matrix[t * feature_size + level * 4])
            .collect();
        let bid_qty_col: Vec<f32> = (0..filtered_snapshots.len())
            .map(|t| feature_matrix[t * feature_size + level * 4 + 1])
            .collect();
        let ask_price_col: Vec<f32> = (0..filtered_snapshots.len())
            .map(|t| feature_matrix[t * feature_size + level * 4 + 2])
            .collect();
        let ask_qty_col: Vec<f32> = (0..filtered_snapshots.len())
            .map(|t| feature_matrix[t * feature_size + level * 4 + 3])
            .collect();

        columns.push(Series::new(format!("bid_price_{}", level).into(), bid_price_col));
        columns.push(Series::new(format!("bid_qty_{}", level).into(), bid_qty_col));
        columns.push(Series::new(format!("ask_price_{}", level).into(), ask_price_col));
        columns.push(Series::new(format!("ask_qty_{}", level).into(), ask_qty_col));
    }

    let df = DataFrame::new(columns.into_iter().map(|s| s.into()).collect())?;

    println!("Created FinancialBERT dataset: {} timesteps × {} features",
             df.height(), df.width() - 1); // -1 for timestamp column

    // Save as parquet with compression
    let file = File::create(output_path)?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Snappy)
        .finish(&mut df.clone())?;

    println!("✅ Saved FinancialBERT dataset to {:?}", output_path);
    println!("   Normalization: Z-score (μ=0, σ=1)");
    println!("   Tick interval: {}ms", tick_interval_ms);
    println!("   Orderbook depth: {} levels per side", orderbook_depth);

    Ok(())
}

fn convert_to_hft_format_fast(snapshots: Vec<OrderBookSnapshot>, output_path: &Path, symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
    // HFT backtest configuration
    let tick_size = get_tick_size(symbol);

    println!("Converting to hftbacktest format with parallel processing...");
    println!("Using tick size {} for symbol {} with {} threads",
             tick_size, symbol, rayon::current_num_threads());

    // Pre-calculate total capacity to avoid reallocations
    let total_levels: usize = snapshots.iter()
        .map(|s| s.bids.len() + s.asks.len())
        .sum();

    println!("Processing {} snapshots with ~{} total levels", snapshots.len(), total_levels);

    // Process snapshots in parallel chunks
    let chunk_size = (snapshots.len() / rayon::current_num_threads()).max(1000);

    // First, calculate how many events each chunk will produce to assign proper order IDs
    let chunk_event_counts: Vec<usize> = snapshots
        .chunks(chunk_size)
        .map(|chunk| {
            chunk.iter()
                .map(|snapshot| {
                    snapshot.bids.iter().filter(|bid| bid.quantity > 0.0).count() +
                    snapshot.asks.iter().filter(|ask| ask.quantity > 0.0).count()
                })
                .sum()
        })
        .collect();

    // Calculate starting order ID for each chunk
    let mut chunk_start_order_ids = Vec::with_capacity(chunk_event_counts.len());
    let mut current_order_id = 1u64;
    for &count in &chunk_event_counts {
        chunk_start_order_ids.push(current_order_id);
        current_order_id += count as u64;
    }

    // Process chunks in parallel and collect results
    let results: Vec<_> = snapshots
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut ev_flags = Vec::new();
            let mut exch_timestamps = Vec::new();
            let mut local_timestamps = Vec::new();
            let mut prices = Vec::new();
            let mut quantities = Vec::new();
            let mut order_ids = Vec::new();
            let mut ivals = Vec::new();
            let mut fvals = Vec::new();

            // Use pre-calculated starting order ID for this chunk
            let mut order_id_counter = chunk_start_order_ids[chunk_idx];

            for snapshot in chunk.iter() {
                // Process bids
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

            (ev_flags, exch_timestamps, local_timestamps, prices, quantities, order_ids, ivals, fvals)
        })
        .collect();

    println!("Combining {} parallel chunks...", results.len());

    // Combine all results
    let mut ev_flags = Vec::with_capacity(total_levels);
    let mut exch_timestamps = Vec::with_capacity(total_levels);
    let mut local_timestamps = Vec::with_capacity(total_levels);
    let mut prices = Vec::with_capacity(total_levels);
    let mut quantities = Vec::with_capacity(total_levels);
    let mut order_ids = Vec::with_capacity(total_levels);
    let mut ivals = Vec::with_capacity(total_levels);
    let mut fvals = Vec::with_capacity(total_levels);

    for (chunk_ev_flags, chunk_exch_ts, chunk_local_ts, chunk_prices, chunk_qtys, chunk_order_ids, chunk_ivals, chunk_fvals) in results {
        ev_flags.extend(chunk_ev_flags);
        exch_timestamps.extend(chunk_exch_ts);
        local_timestamps.extend(chunk_local_ts);
        prices.extend(chunk_prices);
        quantities.extend(chunk_qtys);
        order_ids.extend(chunk_order_ids);
        ivals.extend(chunk_ivals);
        fvals.extend(chunk_fvals);
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
    let hft_path = format!("/mnt/storage-box/bybit/orderbooks/output/{}_{}_hft.parquet", date, symbol);
    let bert_path = format!("/mnt/storage-box/bybit/orderbooks/output/{}_{}_financialbert.parquet", date, symbol);
    Path::new(&hft_path).exists() && Path::new(&bert_path).exists()
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
    let hft_output_path = format!("/mnt/storage-box/bybit/orderbooks/output/{}_{}_hft.parquet", date, symbol);
    let bert_output_path = format!("/mnt/storage-box/bybit/orderbooks/output/{}_{}_financialbert.parquet", date, symbol);

    // Create directories
    std::fs::create_dir_all("/mnt/storage-box/bybit/orderbooks/data")?;
    std::fs::create_dir_all("/mnt/storage-box/bybit/orderbooks/output")?;
    std::fs::create_dir_all(&extract_dir)?;

    // Check if we need to download and parse (if either output doesn't exist)
    let hft_exists = Path::new(&hft_output_path).exists();
    let bert_exists = Path::new(&bert_output_path).exists();

    if hft_exists && bert_exists {
        println!("✅ Both datasets already exist for {}-{}, skipping", symbol, date);
        return Ok(());
    }

    // Download and parse data if needed
    let snapshots = if !Path::new(&data_file_path).exists() {
        println!("📥 Downloading {}...", url);
        download_file(&url, Path::new(&zip_file_path))?;

        println!("📦 Extracting...");
        extract_zip(Path::new(&zip_file_path), Path::new(&extract_dir))?;

        println!("📊 Parsing orderbook data...");
        let events = parse_orderbook_file(Path::new(&data_file_path))?;
        reconstruct_orderbook_snapshots_fast(events)?
    } else {
        println!("📊 Parsing existing orderbook data...");
        let events = parse_orderbook_file(Path::new(&data_file_path))?;
        reconstruct_orderbook_snapshots_fast(events)?
    };

    // Create HFT format if it doesn't exist
    if !hft_exists {
        println!("🔄 Creating HFT format dataset...");
        convert_to_hft_format_fast(snapshots.clone(), Path::new(&hft_output_path), symbol)?;
    } else {
        println!("⏭️  HFT dataset already exists, skipping");
    }

    // Create FinancialBERT format if it doesn't exist
    if !bert_exists {
        println!("🤖 Creating FinancialBERT training dataset...");
        save_financialbert_dataset(snapshots, Path::new(&bert_output_path), symbol)?;
    } else {
        println!("⏭️  FinancialBERT dataset already exists, skipping");
    }

    println!("✅ Processing complete!");
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
    println!("\n📁 Output formats created:");
    println!("  🔄 HFT backtest format: *_hft.parquet");
    println!("  🤖 FinancialBERT training format: *_financialbert.parquet");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_financialbert_dataset_creation() {
        // Create sample orderbook snapshots
        let snapshots = vec![
            OrderBookSnapshot {
                timestamp: 1000000000, // 1 second
                bids: vec![
                    OrderBookLevel { price: 50000.0, quantity: 1.0 },
                    OrderBookLevel { price: 49999.0, quantity: 2.0 },
                ],
                asks: vec![
                    OrderBookLevel { price: 50001.0, quantity: 1.5 },
                    OrderBookLevel { price: 50002.0, quantity: 2.5 },
                ],
            },
            OrderBookSnapshot {
                timestamp: 1200000000, // 1.2 seconds (200ms later)
                bids: vec![
                    OrderBookLevel { price: 50010.0, quantity: 1.2 },
                    OrderBookLevel { price: 50009.0, quantity: 2.2 },
                ],
                asks: vec![
                    OrderBookLevel { price: 50011.0, quantity: 1.7 },
                    OrderBookLevel { price: 50012.0, quantity: 2.7 },
                ],
            },
        ];

        let test_output = "/tmp/test_financialbert.parquet";

        // Test the function
        let result = save_financialbert_dataset(snapshots, Path::new(test_output), "BTCUSDT");
        assert!(result.is_ok(), "FinancialBERT dataset creation should succeed");

        // Verify file was created
        assert!(Path::new(test_output).exists(), "Output file should exist");

        // Clean up
        let _ = fs::remove_file(test_output);
    }

    #[test]
    fn test_parallel_vs_sequential_order_ids() {
        // Test that parallel processing produces sequential order IDs
        let snapshots = vec![
            OrderBookSnapshot {
                timestamp: 1000000000,
                bids: vec![OrderBookLevel { price: 50000.0, quantity: 1.0 }],
                asks: vec![OrderBookLevel { price: 50001.0, quantity: 1.5 }],
            },
            OrderBookSnapshot {
                timestamp: 1200000000,
                bids: vec![OrderBookLevel { price: 50010.0, quantity: 1.2 }],
                asks: vec![OrderBookLevel { price: 50011.0, quantity: 1.7 }],
            },
        ];

        let test_output = "/tmp/test_hft_parallel.parquet";
        let result = convert_to_hft_format_fast(snapshots, Path::new(test_output), "BTCUSDT");
        assert!(result.is_ok(), "HFT conversion should succeed");

        // Load and verify order IDs are sequential
        if let Ok(df) = LazyFrame::scan_parquet(test_output, Default::default()) {
            if let Ok(df) = df.collect() {
                if let Ok(order_ids) = df.column("order_id") {
                    let ids: Vec<u64> = order_ids.u64().unwrap().into_iter().flatten().collect();
                    // Verify order IDs are sequential starting from 1
                    for (i, &id) in ids.iter().enumerate() {
                        assert_eq!(id, (i + 1) as u64, "Order IDs should be sequential");
                    }
                }
            }
        }

        // Clean up
        let _ = fs::remove_file(test_output);
    }

    #[test]
    fn test_orderbook_processing() {
        // This would be an integration test with sample data
        // In practice, you'd download a small sample file for testing
    }
}