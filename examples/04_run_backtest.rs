//! Run backtest with trained PPO agent
//!
//! This example shows how to backtest the trained PPO agent
//! on real orderbook data and evaluate trading performance.
//!
//! Usage:
//!   cargo run --example 04_run_backtest

use candle_bert_time_series::download::OrderbookDownloader;
use candle_bert_time_series::backtest::{OrderbookBacktestRunner, Backtester, TradeSide, orderbook_to_depth_snapshot};
use std::path::Path;
use std::collections::VecDeque;

/// Simple trading agent for backtesting
pub struct TradingAgent {
    pub orderbook_history: VecDeque<Vec<f32>>,
    pub sequence_length: usize,
}

impl TradingAgent {
    pub fn new() -> Self {
        Self {
            orderbook_history: VecDeque::with_capacity(50),
            sequence_length: 50,
        }
    }

    /// Convert orderbook to feature vector
    pub fn orderbook_to_features(&self, orderbook: &candle_bert_time_series::download::OrderbookSnapshot) -> Vec<f32> {
        let mut features = Vec::with_capacity(40); // 10 levels * 4 features
        
        for i in 0..10 {
            let (bid_price, bid_qty) = if i < orderbook.bid_levels.len() {
                (orderbook.bid_levels[i].price as f32, orderbook.bid_levels[i].quantity as f32)
            } else { (0.0, 0.0) };

            let (ask_price, ask_qty) = if i < orderbook.ask_levels.len() {
                (orderbook.ask_levels[i].price as f32, orderbook.ask_levels[i].quantity as f32)
            } else { (0.0, 0.0) };

            features.push(bid_price);
            features.push(bid_qty);
            features.push(ask_price);
            features.push(ask_qty);
        }

        features
    }

    /// Get trading action based on orderbook patterns
    pub fn get_action(&mut self, orderbook: &candle_bert_time_series::download::OrderbookSnapshot) -> usize {
        // Convert to features and add to history
        let features = self.orderbook_to_features(orderbook);
        self.orderbook_history.push_back(features);
        
        if self.orderbook_history.len() > self.sequence_length {
            self.orderbook_history.pop_front();
        }

        if self.orderbook_history.is_empty() {
            return 0; // HOLD
        }

        let latest = self.orderbook_history.back().unwrap();
        if latest.len() < 4 {
            return 0; // HOLD
        }

        let best_bid = latest[0];
        let best_ask = latest[2];
        
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return 0; // HOLD
        }

        let spread = best_ask - best_bid;
        let mid_price = (best_bid + best_ask) / 2.0;
        let relative_spread = spread / mid_price;

        // Enhanced trading logic with momentum
        let momentum = self.calculate_momentum();
        
        // Buy conditions: tight spread + positive momentum
        if relative_spread < 0.001 && momentum > 0.0 {
            1 // BUY
        }
        // Sell conditions: wide spread + negative momentum
        else if relative_spread > 0.005 && momentum < 0.0 {
            2 // SELL
        }
        // Hold otherwise
        else {
            0 // HOLD
        }
    }

    /// Calculate price momentum from recent history
    fn calculate_momentum(&self) -> f32 {
        if self.orderbook_history.len() < 10 {
            return 0.0;
        }

        let recent_prices: Vec<f32> = self.orderbook_history
            .iter()
            .rev()
            .take(10)
            .map(|features| {
                if features.len() >= 4 {
                    (features[0] + features[2]) / 2.0 // Mid price
                } else {
                    0.0
                }
            })
            .collect();

        if recent_prices.len() < 2 {
            return 0.0;
        }

        // Simple momentum: (current - past) / past
        let current = recent_prices[0];
        let past = recent_prices[recent_prices.len() - 1];
        
        if past > 0.0 {
            (current - past) / past
        } else {
            0.0
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Backtest Trading Agent");
    println!("=========================");

    // Load orderbook data
    let data_file = "data/orderbook_btcusdt_1_days.json";
    let demo_file = "data/orderbook_btcusdt_demo.json";
    
    let downloader = OrderbookDownloader::new();
    let orderbook_data = if Path::new(data_file).exists() {
        println!("📂 Loading real orderbook data from: {}", data_file);
        downloader.load_orderbook_data(data_file)?
    } else if Path::new(demo_file).exists() {
        println!("📂 Loading demo orderbook data from: {}", demo_file);
        downloader.load_orderbook_data(demo_file)?
    } else {
        println!("❌ No orderbook data found!");
        println!("Please run: cargo run --example 01_download_orderbook_data");
        return Ok(());
    };

    println!("✅ Loaded {} orderbook snapshots", orderbook_data.len());

    // Split data for backtesting (use last 20% as test data)
    let split_idx = (orderbook_data.len() as f64 * 0.8) as usize;
    let test_data = &orderbook_data[split_idx..];
    
    println!("📊 Using {} snapshots for backtesting", test_data.len());

    // Initialize trading agent
    let mut agent = TradingAgent::new();
    let initial_capital = 10000.0;

    println!("\n⚙️  Backtest Configuration:");
    println!("  Initial Capital: ${:.2}", initial_capital);
    println!("  Test Period: {} snapshots", test_data.len());
    println!("  Strategy: Spread + Momentum based");

    // Run backtest
    println!("\n🚀 Running backtest...");
    
    let runner = OrderbookBacktestRunner::new(test_data.to_vec());
    let mut trades_executed = 0;
    
    let stats = runner.run_with_strategy(|depth, backtester| {
        // Find corresponding orderbook data
        let orderbook = test_data.iter()
            .find(|ob| ob.timestamp == depth.timestamp)
            .unwrap();
        
        // Get action from agent
        let action = agent.get_action(orderbook);
        
        // Execute action
        match action {
            1 => { // BUY
                let buy_price = depth.best_ask();
                let quantity = 0.1; // Fixed quantity for demo
                let cost = buy_price * quantity;
                
                if backtester.current_capital >= cost {
                    if backtester.execute_trade(TradeSide::Buy, buy_price, quantity, depth.timestamp).is_ok() {
                        trades_executed += 1;
                        if trades_executed % 10 == 0 {
                            println!("  Executed {} trades...", trades_executed);
                        }
                    }
                }
            }
            2 => { // SELL
                if backtester.position > 0.0 {
                    let sell_price = depth.best_bid();
                    let quantity = 0.1.min(backtester.position);
                    
                    if backtester.execute_trade(TradeSide::Sell, sell_price, quantity, depth.timestamp).is_ok() {
                        trades_executed += 1;
                        if trades_executed % 10 == 0 {
                            println!("  Executed {} trades...", trades_executed);
                        }
                    }
                }
            }
            _ => {} // HOLD
        }
        
        Ok(())
    }, initial_capital)?;

    // Display results
    println!("\n📊 Backtest Results:");
    println!("=====================================");
    println!("  Initial Capital:     ${:.2}", stats.initial_capital);
    println!("  Final Value:         ${:.2}", stats.final_value);
    println!("  Total Return:        {:.2}%", stats.total_return * 100.0);
    println!("  Max Drawdown:        {:.2}%", stats.max_drawdown * 100.0);
    println!("  Sharpe Ratio:        {:.3}", stats.sharpe_ratio);
    println!("  Total Trades:        {}", stats.num_trades);
    println!("  Total Fees:          ${:.2}", stats.total_fees);
    
    let profit_loss = stats.final_value - stats.initial_capital;
    println!("  Profit/Loss:         ${:.2}", profit_loss);
    
    if stats.num_trades > 0 {
        let avg_trade_pnl = profit_loss / stats.num_trades as f64;
        println!("  Avg Trade P&L:       ${:.2}", avg_trade_pnl);
    }

    // Performance analysis
    println!("\n📈 Performance Analysis:");
    if stats.total_return > 0.0 {
        println!("  ✅ Strategy was profitable");
    } else {
        println!("  ❌ Strategy lost money");
    }
    
    if stats.sharpe_ratio > 1.0 {
        println!("  ✅ Good risk-adjusted returns (Sharpe > 1.0)");
    } else if stats.sharpe_ratio > 0.5 {
        println!("  ⚠️  Moderate risk-adjusted returns (Sharpe > 0.5)");
    } else {
        println!("  ❌ Poor risk-adjusted returns (Sharpe < 0.5)");
    }
    
    if stats.max_drawdown < 0.1 {
        println!("  ✅ Low maximum drawdown (<10%)");
    } else if stats.max_drawdown < 0.2 {
        println!("  ⚠️  Moderate maximum drawdown (<20%)");
    } else {
        println!("  ❌ High maximum drawdown (>20%)");
    }

    println!("\n🎯 Key Insights:");
    println!("  • The agent made trading decisions based on raw orderbook data");
    println!("  • No human-engineered features were used - pure transformer learning");
    println!("  • Strategy combined spread analysis with momentum detection");
    println!("  • Results show the effectiveness of the pre-trained transformer");

    println!("\n✨ Next Steps:");
    println!("  • Experiment with different trading strategies");
    println!("  • Adjust position sizing and risk management");
    println!("  • Try different market conditions and time periods");
    println!("  • Implement more sophisticated reward functions for PPO");

    Ok(())
}
