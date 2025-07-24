use polars::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use ::zip::ZipArchive;
use ordered_float::OrderedFloat;

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

fn reconstruct_orderbook_snapshots(events: Vec<OrderBookEvent>) -> Result<Vec<OrderBookSnapshot>, Box<dyn std::error::Error>> {
    let mut snapshots = Vec::new();
    let mut current_bids: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();
    let mut current_asks: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();
    
    for event in events {
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
                
                snapshots.push(snapshot);
            },
            OrderBookEvent::Delta { timestamp, bids, asks } => {
                // Apply bid updates
                for (price, quantity) in bids {
                    if quantity == 0.0 {
                        current_bids.remove(&OrderedFloat(price));
                    } else {
                        current_bids.insert(OrderedFloat(price), quantity);
                    }
                }

                // Apply ask updates
                for (price, quantity) in asks {
                    if quantity == 0.0 {
                        current_asks.remove(&OrderedFloat(price));
                    } else {
                        current_asks.insert(OrderedFloat(price), quantity);
                    }
                }
                
                // Create new snapshot
                let bid_levels: Vec<OrderBookLevel> = current_bids
                    .iter()
                    .map(|(&price, &quantity)| OrderBookLevel { price: price.into_inner(), quantity })
                    .collect();

                let ask_levels: Vec<OrderBookLevel> = current_asks
                    .iter()
                    .map(|(&price, &quantity)| OrderBookLevel { price: price.into_inner(), quantity })
                    .collect();
                
                snapshots.push(OrderBookSnapshot {
                    timestamp,
                    bids: bid_levels,
                    asks: ask_levels,
                });
            },
        }
    }
    
    Ok(snapshots)
}

fn convert_to_hft_format(snapshots: Vec<OrderBookSnapshot>, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Prepare vectors for DataFrame
    let mut timestamps = Vec::new();
    let mut event_types = Vec::new(); // 1 = bid, 2 = ask
    let mut prices = Vec::new();
    let mut quantities = Vec::new();
    
    for snapshot in snapshots {
        // Add bid levels
        for bid in &snapshot.bids {
            timestamps.push(snapshot.timestamp);
            event_types.push(1i32); // BID_EVENT
            prices.push(bid.price);
            quantities.push(bid.quantity);
        }
        
        // Add ask levels
        for ask in &snapshot.asks {
            timestamps.push(snapshot.timestamp);
            event_types.push(2i32); // ASK_EVENT
            prices.push(ask.price);
            quantities.push(ask.quantity);
        }
    }
    
    // Create DataFrame
    let df = df![
        "timestamp" => timestamps,
        "event_type" => event_types,
        "price" => prices,
        "quantity" => quantities,
    ]?;
    
    // Save as parquet (better for hftbacktest than CSV)
    let file = File::create(output_path)?;
    ParquetWriter::new(file).finish(&mut df.clone())?;
    
    println!("Saved {} events to {:?}", df.height(), output_path);
    Ok(())
}

fn process_orderbook_data(symbol: &str, date: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("{}/{}/{}_{}_ob200.data.zip", BYBIT_BASE_URL, symbol, date, symbol);
    let zip_file_path = format!("./data/{}_{}_ob200.data.zip", date, symbol);
    let extract_dir = format!("./data/{}", date);
    let data_file_path = format!("{}/{}_{}_ob200.data", extract_dir, date, symbol);
    let output_path = format!("./output/{}_{}_hft.parquet", date, symbol);
    
    // Create directories
    std::fs::create_dir_all("./data")?;
    std::fs::create_dir_all("./output")?;
    std::fs::create_dir_all(&extract_dir)?;
    
    println!("Downloading {}...", url);
    download_file(&url, Path::new(&zip_file_path))?;
    
    println!("Extracting...");
    extract_zip(Path::new(&zip_file_path), Path::new(&extract_dir))?;
    
    println!("Parsing orderbook data...");
    let events = parse_orderbook_file(Path::new(&data_file_path))?;
    
    println!("Reconstructing full snapshots...");
    let snapshots = reconstruct_orderbook_snapshots(events)?;
    
    println!("Converting to hftbacktest format...");
    convert_to_hft_format(snapshots, Path::new(&output_path))?;
    
    println!("Processing complete!");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let symbol = "BTCUSDC";
    let date = "2025-07-07";
    
    process_orderbook_data(symbol, date)?;
    
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