//! Test hftbacktest with downloaded Bybit orderbook data
//!
//! This example tests the hftbacktest functionality using the real orderbook data
//! downloaded from Bybit to ensure everything works correctly before deployment.
//!
//! Usage:
//!   cargo run --example test_hftbacktest_bybit

use polars::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;
use anyhow::{anyhow, Result};
use hftbacktest::prelude::*;

/// Validate that the parquet file has the correct hftbacktest schema
fn validate_hftbacktest_schema(file_path: &str) -> Result<()> {
    println!("🔍 Validating hftbacktest schema...");

    if !Path::new(file_path).exists() {
        return Err(anyhow!("File does not exist: {}", file_path));
    }

    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .limit(1) // Just check schema
        .collect()?;

    let expected_columns = vec![
        ("ev", "UInt64"),
        ("exch_ts", "Int64"),
        ("local_ts", "Int64"),
        ("px", "Int64"),
        ("qty", "Float64"),
        ("order_id", "UInt64"),
        ("ival", "Int64"),
        ("fval", "Float64"),
    ];

    println!("📊 Schema validation:");
    let column_names = df.get_column_names();
    let dtypes = df.dtypes();
    let actual_columns: Vec<_> = column_names.iter().zip(dtypes.iter()).collect();

    for ((expected_name, expected_type), (actual_name, actual_type)) in
        expected_columns.iter().zip(actual_columns.iter()) {

        let type_str = format!("{:?}", actual_type);
        let matches = actual_name.as_str() == *expected_name && type_str.contains(expected_type);
        let status = if matches { "✅" } else { "❌" };

        println!("  {}: {} {} (expected: {} {})",
                 status, actual_name, type_str, expected_name, expected_type);
    }

    Ok(())
}

/// Load the parquet file and inspect its structure
fn inspect_orderbook_data(file_path: &str) -> Result<()> {
    println!("🔍 Inspecting orderbook data structure...");
    
    if !Path::new(file_path).exists() {
        return Err(anyhow!("File does not exist: {}", file_path));
    }

    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .limit(10) // Just look at first 10 rows
        .collect()?;

    println!("📊 Data structure:");
    println!("Columns: {:?}", df.get_column_names());
    println!("Shape: {:?}", df.shape());
    println!("Data types: {:?}", df.dtypes());
    
    println!("\n📋 Sample data:");
    println!("{}", df);

    // Get full statistics
    let full_df = LazyFrame::scan_parquet(file_path, Default::default())?
        .collect()?;
    
    println!("\n📈 Data statistics:");
    println!("Total events: {}", full_df.height());
    
    // Count event types
    if let Ok(event_counts) = full_df
        .clone()
        .lazy()
        .group_by([col("event_type")])
        .agg([len().alias("count")])
        .collect()
    {
        println!("Event type distribution:");
        println!("{}", event_counts);
    }

    // Time range
    if let Ok(time_stats) = full_df
        .clone()
        .lazy()
        .select([
            col("timestamp").min().alias("min_time"),
            col("timestamp").max().alias("max_time"),
        ])
        .collect()
    {
        println!("\nTime range:");
        println!("{}", time_stats);
    }

    // Price range
    if let Ok(price_stats) = full_df
        .clone()
        .lazy()
        .select([
            col("price").min().alias("min_price"),
            col("price").max().alias("max_price"),
            col("price").mean().alias("avg_price"),
        ])
        .collect()
    {
        println!("\nPrice statistics:");
        println!("{}", price_stats);
    }

    Ok(())
}

/// Load hftbacktest-compatible data with parallel processing and batching
fn load_hftbacktest_data_fast(file_path: &str, max_events: Option<usize>) -> Result<Vec<Event>> {
    use rayon::prelude::*;

    println!("📂 Loading hftbacktest-compatible data with parallel processing...");

    let mut lazy_frame = LazyFrame::scan_parquet(file_path, Default::default())?;

    // Limit events for testing if specified
    if let Some(limit) = max_events {
        lazy_frame = lazy_frame.limit(limit as u32);
        println!("⚠️  Limiting to {} events for testing", limit);
    }

    let df = lazy_frame.collect()?;
    println!("Loaded DataFrame with {} events", df.height());

    // Extract columns as slices for vectorized processing
    let ev_flags = df.column("ev")?.u64()?;
    let exch_timestamps = df.column("exch_ts")?.i64()?;
    let local_timestamps = df.column("local_ts")?.i64()?;
    let prices = df.column("px")?.i64()?;
    let quantities = df.column("qty")?.f64()?;
    let order_ids = df.column("order_id")?.u64()?;
    let ivals = df.column("ival")?.i64()?;
    let fvals = df.column("fval")?.f64()?;

    println!("Converting to Event structs with parallel processing...");

    // Use parallel iterator to create events in chunks
    let events: Vec<Event> = (0..df.height())
        .into_par_iter()
        .map(|i| Event {
            ev: ev_flags.get(i).unwrap_or(0),
            exch_ts: exch_timestamps.get(i).unwrap_or(0),
            local_ts: local_timestamps.get(i).unwrap_or(0),
            px: prices.get(i).unwrap_or(0) as f64,
            qty: quantities.get(i).unwrap_or(0.0),
            order_id: order_ids.get(i).unwrap_or(0),
            ival: ivals.get(i).unwrap_or(0),
            fval: fvals.get(i).unwrap_or(0.0),
        })
        .collect();

    println!("✅ Loaded {} events with parallel processing", events.len());
    Ok(events)
}

/// Streaming version that processes events in batches (most memory efficient)
fn process_hftbacktest_streaming(file_path: &str, batch_size: usize, max_batches: Option<usize>) -> Result<()> {
    println!("📂 Processing hftbacktest data in streaming batches...");

    let tick_size = 0.1;
    let lot_size = 0.001;
    let mut hbt = HashMapMarketDepth::new(tick_size, lot_size);
    let mut strategy = SimpleMarketMaker::new();

    let mut total_processed = 0;
    let mut trade_count = 0;
    let mut batch_count = 0;

    // Process in batches to avoid loading everything into memory
    loop {
        if let Some(max) = max_batches {
            if batch_count >= max {
                println!("⚠️  Reached max batches limit: {}", max);
                break;
            }
        }

        let offset = batch_count * batch_size;

        let df = LazyFrame::scan_parquet(file_path, Default::default())?
            .slice(offset as i64, batch_size as u32)
            .collect()?;

        if df.height() == 0 {
            break; // No more data
        }

        println!("Processing batch {} with {} events (offset: {})",
                 batch_count + 1, df.height(), offset);

        // Process this batch
        let ev_flags = df.column("ev")?.u64()?;
        let exch_timestamps = df.column("exch_ts")?.i64()?;
        let local_timestamps = df.column("local_ts")?.i64()?;
        let prices = df.column("px")?.i64()?;
        let quantities = df.column("qty")?.f64()?;
        let order_ids = df.column("order_id")?.u64()?;
        let ivals = df.column("ival")?.i64()?;
        let fvals = df.column("fval")?.f64()?;

        for i in 0..df.height() {
            let event = Event {
                ev: ev_flags.get(i).unwrap_or(0),
                exch_ts: exch_timestamps.get(i).unwrap_or(0),
                local_ts: local_timestamps.get(i).unwrap_or(0),
                px: prices.get(i).unwrap_or(0) as f64,
                qty: quantities.get(i).unwrap_or(0.0),
                order_id: order_ids.get(i).unwrap_or(0),
                ival: ivals.get(i).unwrap_or(0),
                fval: fvals.get(i).unwrap_or(0.0),
            };

            // Process event immediately
            if event.is(LOCAL_BID_DEPTH_EVENT) {
                let price = event.px * tick_size;
                hbt.update_bid_depth(price, event.qty, event.local_ts);

                if let Err(e) = strategy.on_depth_update(&mut hbt, event.local_ts) {
                    println!("Strategy error: {}", e);
                }
            } else if event.is(LOCAL_ASK_DEPTH_EVENT) {
                let price = (-event.px) * tick_size;
                hbt.update_ask_depth(price, event.qty, event.local_ts);

                if let Err(e) = strategy.on_depth_update(&mut hbt, event.local_ts) {
                    println!("Strategy error: {}", e);
                }
            } else if event.is(LOCAL_BUY_TRADE_EVENT) || event.is(LOCAL_SELL_TRADE_EVENT) {
                strategy.on_trade(&event);
                trade_count += 1;
            }
        }

        total_processed += df.height();
        batch_count += 1;

        // Print progress
        let best_bid = hbt.best_bid_tick();
        let best_ask = hbt.best_ask_tick();
        println!("Batch {} complete. Total processed: {}, Best: {}/{}",
                 batch_count, total_processed, best_bid, best_ask);
    }

    println!("\n📊 Streaming Backtest Results:");
    println!("Total events processed: {}", total_processed);
    println!("Total trades executed: {}", trade_count);
    println!("Final position: {}", strategy.position);

    let best_bid = hbt.best_bid_tick();
    let best_ask = hbt.best_ask_tick();
    println!("Final market: bid={}, ask={}", best_bid, best_ask);

    Ok(())
}

/// Simple market making strategy for testing
struct SimpleMarketMaker {
    position: f64,
    spread_ticks: i64,
    order_qty: f64,
    max_position: f64,
}

impl SimpleMarketMaker {
    fn new() -> Self {
        Self {
            position: 0.0,
            spread_ticks: 10, // 10 tick spread
            order_qty: 0.1,   // 0.1 BTC orders
            max_position: 1.0, // Max 1 BTC position
        }
    }

    fn on_depth_update(&mut self, hbt: &mut HashMapMarketDepth, _timestamp: i64) -> Result<()> {
        // Get best bid/ask
        let best_bid_tick = hbt.best_bid_tick();
        let best_ask_tick = hbt.best_ask_tick();

        if best_bid_tick == INVALID_MIN || best_ask_tick == INVALID_MAX {
            return Ok(()); // No valid market
        }

        // For this simple test, we'll just track the market state
        // In a real strategy, you would place orders here

        Ok(())
    }

    fn on_trade(&mut self, trade: &Event) {
        // Update position based on trade
        if trade.px > 0.0 {
            self.position += trade.qty;
        } else {
            self.position -= trade.qty;
        }

        println!("Trade executed: price={}, qty={}, new_position={}",
                 trade.px, trade.qty, self.position);
    }
}

/// Run a simple backtest with the loaded data
fn run_simple_backtest(events: Vec<Event>) -> Result<()> {
    println!("🚀 Running simple backtest...");

    // Create market depth with tick size and lot size (matching download script)
    let tick_size = 0.1;  // Updated to match BTCUSDC tick size
    let lot_size = 0.001;
    let mut hbt = HashMapMarketDepth::new(tick_size, lot_size);
    let mut strategy = SimpleMarketMaker::new();

    let mut trade_count = 0;
    let mut last_print_time = 0i64;
    let print_interval = 60_000_000_000i64; // Print every minute (nanoseconds)

    println!("Processing {} events...", events.len());

    for (i, event) in events.iter().enumerate() {
        // Apply the event to market depth manually
        if event.is(LOCAL_BID_DEPTH_EVENT) {
            // Bid side update - px is already in ticks, convert to price
            let price = event.px * tick_size;
            hbt.update_bid_depth(price, event.qty, event.local_ts);

            // Update strategy on depth changes
            if let Err(e) = strategy.on_depth_update(&mut hbt, event.local_ts) {
                println!("Strategy error: {}", e);
            }
        } else if event.is(LOCAL_ASK_DEPTH_EVENT) {
            // Ask side update - px is negative ticks for asks, convert to positive price
            let price = (-event.px) * tick_size;
            hbt.update_ask_depth(price, event.qty, event.local_ts);

            // Update strategy on depth changes
            if let Err(e) = strategy.on_depth_update(&mut hbt, event.local_ts) {
                println!("Strategy error: {}", e);
            }
        } else if event.is(LOCAL_BUY_TRADE_EVENT) || event.is(LOCAL_SELL_TRADE_EVENT) {
            strategy.on_trade(event);
            trade_count += 1;
        }

        // Print progress periodically
        if event.local_ts - last_print_time > print_interval {
            let best_bid = hbt.best_bid_tick();
            let best_ask = hbt.best_ask_tick();
            println!("Progress: {}/{} events, Time: {}, Best: {}/{}",
                     i + 1, events.len(), event.local_ts, best_bid, best_ask);
            last_print_time = event.local_ts;
        }

        // Break early for testing (process only first 100k events)
        if i >= 100_000 {
            println!("⚠️  Stopping early after 100k events for testing");
            break;
        }
    }

    println!("\n📊 Backtest Results:");
    println!("Total trades executed: {}", trade_count);
    println!("Final position: {}", strategy.position);

    // Get final market state
    let best_bid = hbt.best_bid_tick();
    let best_ask = hbt.best_ask_tick();
    println!("Final market: bid={}, ask={}", best_bid, best_ask);

    Ok(())
}

fn main() -> Result<()> {
    println!("🧪 Testing hftbacktest with Bybit orderbook data");
    println!("================================================");

    let data_file = "/home/i3/Downloads/2025-04-30_BTCUSDC_hft.parquet";

    // Step 1: Validate hftbacktest schema
    validate_hftbacktest_schema(data_file)?;

    println!("\n{}", "=".repeat(50));

    // Step 2: Inspect the data structure
    inspect_orderbook_data(data_file)?;

    println!("\n{}", "=".repeat(50));

    // Step 3: Process data with streaming (much faster and memory efficient)
    println!("🚀 Running streaming backtest (fast & memory efficient)...");
    process_hftbacktest_streaming(data_file, 100_000, Some(50))?; // Process 50 batches of 100k events each

    println!("\n✅ Test completed successfully!");
    println!("The hftbacktest integration is working with your Bybit data.");

    Ok(())
}
