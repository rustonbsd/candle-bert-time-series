//! Real orderbook data downloader for hftbacktest
//!
//! This module downloads real Level 2 orderbook data from cryptocurrency exchanges
//! and converts it to hftbacktest-compatible format for training and backtesting.

use anyhow::{anyhow, Result};
use reqwest;
use serde::{Deserialize, Serialize};

use std::fs;
use std::path::Path;
use tokio;

/// Binance orderbook depth snapshot
#[derive(Debug, Deserialize, Serialize)]
pub struct BinanceDepthSnapshot {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    pub bids: Vec<[String; 2]>, // [price, quantity]
    pub asks: Vec<[String; 2]>, // [price, quantity]
}

/// Binance kline data for timestamps
#[derive(Debug, Deserialize, Serialize)]
pub struct BinanceKline {
    pub open_time: u64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub close_time: u64,
    pub quote_asset_volume: String,
    pub number_of_trades: u64,
    pub taker_buy_base_asset_volume: String,
    pub taker_buy_quote_asset_volume: String,
    pub ignore: String,
}

/// HFT-compatible orderbook level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// HFT-compatible orderbook snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookSnapshot {
    pub timestamp: i64,
    pub symbol: String,
    pub bid_levels: Vec<OrderbookLevel>,
    pub ask_levels: Vec<OrderbookLevel>,
    pub tick_size: f64,
    pub lot_size: f64,
}

/// Orderbook data downloader
pub struct OrderbookDownloader {
    client: reqwest::Client,
    base_url: String,
}

impl OrderbookDownloader {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.binance.com".to_string(),
        }
    }

    /// Download orderbook snapshots for a symbol over a time period
    pub async fn download_orderbook_data(
        &self,
        symbol: &str,
        start_time: u64,
        end_time: u64,
        interval_ms: u64,
        depth: usize,
    ) -> Result<Vec<OrderbookSnapshot>> {
        println!("📊 Downloading orderbook data for {} from {} to {}", symbol, start_time, end_time);
        
        let mut snapshots = Vec::new();
        let mut current_time = start_time;

        // Get exchange info for tick size and lot size
        let (tick_size, lot_size) = self.get_symbol_info(symbol).await?;
        
        while current_time < end_time {
            // Download depth snapshot
            match self.get_depth_snapshot(symbol, depth).await {
                Ok(depth_data) => {
                    let snapshot = OrderbookSnapshot {
                        timestamp: current_time as i64,
                        symbol: symbol.to_string(),
                        bid_levels: depth_data.bids.iter()
                            .take(depth)
                            .map(|level| OrderbookLevel {
                                price: level[0].parse().unwrap_or(0.0),
                                quantity: level[1].parse().unwrap_or(0.0),
                            })
                            .collect(),
                        ask_levels: depth_data.asks.iter()
                            .take(depth)
                            .map(|level| OrderbookLevel {
                                price: level[0].parse().unwrap_or(0.0),
                                quantity: level[1].parse().unwrap_or(0.0),
                            })
                            .collect(),
                        tick_size,
                        lot_size,
                    };
                    
                    snapshots.push(snapshot);
                    
                    if snapshots.len() % 100 == 0 {
                        println!("  Downloaded {} snapshots...", snapshots.len());
                    }
                }
                Err(e) => {
                    println!("⚠️  Failed to get snapshot at {}: {}", current_time, e);
                }
            }
            
            current_time += interval_ms;
            
            // Rate limiting - Binance allows 1200 requests per minute
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        println!("✅ Downloaded {} orderbook snapshots", snapshots.len());
        Ok(snapshots)
    }

    /// Get current depth snapshot from Binance
    async fn get_depth_snapshot(&self, symbol: &str, limit: usize) -> Result<BinanceDepthSnapshot> {
        let url = format!("{}/api/v3/depth?symbol={}&limit={}", self.base_url, symbol, limit);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("API request failed: {}", response.status()));
        }

        let depth: BinanceDepthSnapshot = response.json().await?;
        Ok(depth)
    }

    /// Get symbol information (tick size, lot size)
    async fn get_symbol_info(&self, symbol: &str) -> Result<(f64, f64)> {
        let url = format!("{}/api/v3/exchangeInfo", self.base_url);
        
        let response = self.client
            .get(&url)
            .send()
            .await?;

        let exchange_info: serde_json::Value = response.json().await?;
        
        if let Some(symbols) = exchange_info["symbols"].as_array() {
            for sym in symbols {
                if sym["symbol"].as_str() == Some(symbol) {
                    let mut tick_size = 0.01;
                    let mut lot_size = 0.001;
                    
                    if let Some(filters) = sym["filters"].as_array() {
                        for filter in filters {
                            if filter["filterType"].as_str() == Some("PRICE_FILTER") {
                                if let Some(ts) = filter["tickSize"].as_str() {
                                    tick_size = ts.parse().unwrap_or(0.01);
                                }
                            }
                            if filter["filterType"].as_str() == Some("LOT_SIZE") {
                                if let Some(ls) = filter["stepSize"].as_str() {
                                    lot_size = ls.parse().unwrap_or(0.001);
                                }
                            }
                        }
                    }
                    
                    return Ok((tick_size, lot_size));
                }
            }
        }
        
        // Default values if not found
        Ok((0.01, 0.001))
    }

    /// Download historical kline data for timestamps
    pub async fn download_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<BinanceKline>> {
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&startTime={}&endTime={}&limit=1000",
            self.base_url, symbol, interval, start_time, end_time
        );
        
        let response = self.client.get(&url).send().await?;
        let klines_data: Vec<serde_json::Value> = response.json().await?;
        
        let mut klines = Vec::new();
        for kline_data in klines_data {
            if let Some(array) = kline_data.as_array() {
                if array.len() >= 12 {
                    let kline = BinanceKline {
                        open_time: array[0].as_u64().unwrap_or(0),
                        open: array[1].as_str().unwrap_or("0").to_string(),
                        high: array[2].as_str().unwrap_or("0").to_string(),
                        low: array[3].as_str().unwrap_or("0").to_string(),
                        close: array[4].as_str().unwrap_or("0").to_string(),
                        volume: array[5].as_str().unwrap_or("0").to_string(),
                        close_time: array[6].as_u64().unwrap_or(0),
                        quote_asset_volume: array[7].as_str().unwrap_or("0").to_string(),
                        number_of_trades: array[8].as_u64().unwrap_or(0),
                        taker_buy_base_asset_volume: array[9].as_str().unwrap_or("0").to_string(),
                        taker_buy_quote_asset_volume: array[10].as_str().unwrap_or("0").to_string(),
                        ignore: array[11].as_str().unwrap_or("0").to_string(),
                    };
                    klines.push(kline);
                }
            }
        }
        
        Ok(klines)
    }

    /// Save orderbook data to file
    pub fn save_orderbook_data(&self, data: &[OrderbookSnapshot], file_path: &str) -> Result<()> {
        println!("💾 Saving {} snapshots to {}", data.len(), file_path);
        
        // Create directory if it doesn't exist
        if let Some(parent) = Path::new(file_path).parent() {
            fs::create_dir_all(parent)?;
        }
        
        let json_data = serde_json::to_string_pretty(data)?;
        fs::write(file_path, json_data)?;
        
        println!("✅ Saved orderbook data to {}", file_path);
        Ok(())
    }

    /// Load orderbook data from file
    pub fn load_orderbook_data(&self, file_path: &str) -> Result<Vec<OrderbookSnapshot>> {
        println!("📂 Loading orderbook data from {}", file_path);
        
        if !Path::new(file_path).exists() {
            return Err(anyhow!("File does not exist: {}", file_path));
        }
        
        let json_data = fs::read_to_string(file_path)?;
        let data: Vec<OrderbookSnapshot> = serde_json::from_str(&json_data)?;
        
        println!("✅ Loaded {} orderbook snapshots", data.len());
        Ok(data)
    }
}

/// Utility function to get current timestamp in milliseconds
pub fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Utility function to convert days ago to timestamp
pub fn days_ago_timestamp(days: u64) -> u64 {
    let now = current_timestamp_ms();
    now - (days * 24 * 60 * 60 * 1000)
}
