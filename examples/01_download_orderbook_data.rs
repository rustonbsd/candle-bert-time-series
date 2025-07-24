//! Download real orderbook data from Binance
//!
//! This example shows how to download real Level 2 orderbook data
//! that can be used for training and backtesting with hftbacktest.
//!
//! Usage:
//!   cargo run --example 01_download_orderbook_data

use candle_bert_time_series::download::{OrderbookDownloader, days_ago_timestamp, current_timestamp_ms};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Orderbook Data Downloader");
    println!("============================");

    let downloader = OrderbookDownloader::new();
    
    // Configuration
    let symbol = "BTCUSDT";
    let days_back = 1; // Download last 1 day of data
    let interval_ms = 5000; // 5 second intervals
    let depth = 20; // 20 levels of orderbook depth
    
    let end_time = current_timestamp_ms();
    let start_time = days_ago_timestamp(days_back);
    
    println!("📊 Download Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Time Range: {} to {}", start_time, end_time);
    println!("  Interval: {}ms", interval_ms);
    println!("  Depth: {} levels", depth);
    
    // Download orderbook data
    println!("\n📈 Starting download...");
    match downloader.download_orderbook_data(
        symbol,
        start_time,
        end_time,
        interval_ms,
        depth,
    ).await {
        Ok(snapshots) => {
            println!("✅ Downloaded {} orderbook snapshots", snapshots.len());
            
            // Show sample data
            if !snapshots.is_empty() {
                println!("\n📋 Sample snapshot:");
                let sample = &snapshots[0];
                println!("  Timestamp: {}", sample.timestamp);
                println!("  Symbol: {}", sample.symbol);
                println!("  Tick Size: {}", sample.tick_size);
                println!("  Lot Size: {}", sample.lot_size);
                
                if !sample.bid_levels.is_empty() {
                    println!("  Best Bid: ${:.2} (qty: {:.4})", 
                        sample.bid_levels[0].price, 
                        sample.bid_levels[0].quantity);
                }
                
                if !sample.ask_levels.is_empty() {
                    println!("  Best Ask: ${:.2} (qty: {:.4})", 
                        sample.ask_levels[0].price, 
                        sample.ask_levels[0].quantity);
                }
                
                println!("  Bid Levels: {}", sample.bid_levels.len());
                println!("  Ask Levels: {}", sample.ask_levels.len());
            }
            
            // Save to file
            let filename = format!("data/orderbook_{}_{}_days.json", symbol.to_lowercase(), days_back);
            downloader.save_orderbook_data(&snapshots, &filename)?;
            
            println!("\n📊 Data Statistics:");
            println!("  Total Snapshots: {}", snapshots.len());
            println!("  Time Span: {:.2} hours", 
                (end_time - start_time) as f64 / (1000.0 * 60.0 * 60.0));
            println!("  Average Interval: {:.2}s", 
                (end_time - start_time) as f64 / snapshots.len() as f64 / 1000.0);
            
            // Calculate some basic statistics
            let mut total_spread = 0.0;
            let mut valid_spreads = 0;
            
            for snapshot in &snapshots {
                if !snapshot.bid_levels.is_empty() && !snapshot.ask_levels.is_empty() {
                    let spread = snapshot.ask_levels[0].price - snapshot.bid_levels[0].price;
                    total_spread += spread;
                    valid_spreads += 1;
                }
            }
            
            if valid_spreads > 0 {
                let avg_spread = total_spread / valid_spreads as f64;
                println!("  Average Spread: ${:.4}", avg_spread);
            }
            
            println!("\n✨ Next Steps:");
            println!("  1. Run: cargo run --example 02_train_transformer");
            println!("  2. Use the downloaded data for transformer training");
            println!("  3. The data is saved in: {}", filename);
        }
        Err(e) => {
            println!("❌ Download failed: {}", e);
            println!("\n🔧 Troubleshooting:");
            println!("  - Check your internet connection");
            println!("  - Verify Binance API is accessible");
            println!("  - Try reducing the time range or interval");
            println!("  - Check if the symbol '{}' exists on Binance", symbol);
            
            // Create demo data as fallback
            println!("\n🎲 Creating demo data instead...");
            create_demo_data(&downloader)?;
        }
    }

    Ok(())
}

/// Create demo orderbook data for testing when real download fails
fn create_demo_data(downloader: &OrderbookDownloader) -> Result<(), Box<dyn std::error::Error>> {
    use candle_bert_time_series::download::{OrderbookSnapshot, OrderbookLevel};
    
    let mut demo_snapshots = Vec::new();
    let base_price = 50000.0;
    let start_time = current_timestamp_ms() - 3600000; // 1 hour ago
    
    for i in 0..720 { // 720 snapshots = 1 hour at 5s intervals
        let timestamp = start_time + (i * 5000); // 5 second intervals
        let price_offset = (i as f64 * 0.1).sin() * 100.0; // Sine wave price movement
        let mid_price = base_price + price_offset;
        
        // Create bid levels (descending prices)
        let mut bid_levels = Vec::new();
        for j in 0..20 {
            bid_levels.push(OrderbookLevel {
                price: mid_price - 0.5 - (j as f64 * 0.5),
                quantity: 1.0 + (j as f64 * 0.1),
            });
        }
        
        // Create ask levels (ascending prices)
        let mut ask_levels = Vec::new();
        for j in 0..20 {
            ask_levels.push(OrderbookLevel {
                price: mid_price + 0.5 + (j as f64 * 0.5),
                quantity: 1.0 + (j as f64 * 0.1),
            });
        }
        
        demo_snapshots.push(OrderbookSnapshot {
            timestamp: timestamp as i64,
            symbol: "BTCUSDT".to_string(),
            bid_levels,
            ask_levels,
            tick_size: 0.01,
            lot_size: 0.001,
        });
    }
    
    let filename = "data/orderbook_btcusdt_demo.json";
    downloader.save_orderbook_data(&demo_snapshots, filename)?;
    
    println!("✅ Created {} demo snapshots", demo_snapshots.len());
    println!("💾 Saved demo data to: {}", filename);
    println!("🎯 You can now proceed with training using this demo data");
    
    Ok(())
}
