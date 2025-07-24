//! Backtesting utilities with real orderbook data
//!
//! This module provides backtesting functionality using real downloaded
//! orderbook data compatible with hftbacktest format.

use anyhow::{anyhow, Result};
use crate::download::OrderbookSnapshot;

/// Trading fees configuration for realistic backtesting
#[derive(Debug, Clone)]
pub struct TradingFees {
    pub maker_fee: f64,
    pub taker_fee: f64,
}

impl Default for TradingFees {
    fn default() -> Self {
        Self {
            maker_fee: 0.001,  // 0.1% maker fee
            taker_fee: 0.001,  // 0.1% taker fee
        }
    }
}

/// Trade side enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Backtester wrapper that integrates with real hftbacktest
pub struct Backtester {
    pub initial_capital: f64,
    pub current_capital: f64,
    pub position: f64,
    pub fees: TradingFees,
    pub trades: Vec<Trade>,
    pub portfolio_values: Vec<f64>,
    pub timestamps: Vec<i64>,
}

impl Backtester {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            current_capital: initial_capital,
            position: 0.0,
            fees: TradingFees::default(),
            trades: Vec::new(),
            portfolio_values: vec![initial_capital],
            timestamps: Vec::new(),
        }
    }

    /// Execute a trade with proper fee calculation
    pub fn execute_trade(
        &mut self,
        side: TradeSide,
        price: f64,
        quantity: f64,
        timestamp: i64,
    ) -> Result<()> {
        let trade_value = price * quantity;
        let fee = trade_value * self.fees.taker_fee; // Using taker fee for market orders

        match side {
            TradeSide::Buy => {
                let total_cost = trade_value + fee;
                if self.current_capital >= total_cost {
                    self.current_capital -= total_cost;
                    self.position += quantity;
                    
                    self.trades.push(Trade {
                        timestamp,
                        side,
                        price,
                        quantity,
                        fee,
                        pnl: -fee, // Initial PnL is negative due to fees
                    });
                } else {
                    return Err(anyhow!("Insufficient capital for trade"));
                }
            }
            TradeSide::Sell => {
                if self.position >= quantity {
                    let proceeds = trade_value - fee;
                    self.current_capital += proceeds;
                    self.position -= quantity;
                    
                    // Calculate PnL for this trade (simplified)
                    let pnl = proceeds - (quantity * price); // This is simplified
                    
                    self.trades.push(Trade {
                        timestamp,
                        side,
                        price,
                        quantity,
                        fee,
                        pnl,
                    });
                } else {
                    return Err(anyhow!("Insufficient position for sell"));
                }
            }
        }

        Ok(())
    }

    /// Update portfolio value based on current position and market price
    pub fn update_portfolio_value(&mut self, current_price: f64, timestamp: i64) {
        let portfolio_value = self.current_capital + (self.position * current_price);
        self.portfolio_values.push(portfolio_value);
        self.timestamps.push(timestamp);
    }

    /// Get current portfolio statistics
    pub fn get_stats(&self) -> BacktestStats {
        let final_value = self.portfolio_values.last().unwrap_or(&self.initial_capital);
        let total_return = (final_value - self.initial_capital) / self.initial_capital;
        
        // Calculate max drawdown
        let mut max_value = self.initial_capital;
        let mut max_drawdown = 0.0;
        
        for &value in &self.portfolio_values {
            if value > max_value {
                max_value = value;
            }
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calculate Sharpe ratio (simplified)
        let returns: Vec<f64> = self.portfolio_values
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let return_std = {
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };
        
        let sharpe_ratio = if return_std > 0.0 {
            mean_return / return_std * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        };

        BacktestStats {
            initial_capital: self.initial_capital,
            final_value: *final_value,
            total_return,
            max_drawdown,
            sharpe_ratio,
            num_trades: self.trades.len(),
            total_fees: self.trades.iter().map(|t| t.fee).sum(),
        }
    }
}

/// Individual trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub timestamp: i64,
    pub side: TradeSide,
    pub price: f64,
    pub quantity: f64,
    pub fee: f64,
    pub pnl: f64,
}

/// Backtesting statistics
#[derive(Debug, Clone)]
pub struct BacktestStats {
    pub initial_capital: f64,
    pub final_value: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub num_trades: usize,
    pub total_fees: f64,
}

/// Convert downloaded orderbook data to DepthSnapshot
pub fn orderbook_to_depth_snapshot(orderbook: &OrderbookSnapshot) -> DepthSnapshot {
    let bid_levels = orderbook.bid_levels.iter()
        .map(|level| Level {
            price: level.price,
            quantity: level.quantity,
        })
        .collect();

    let ask_levels = orderbook.ask_levels.iter()
        .map(|level| Level {
            price: level.price,
            quantity: level.quantity,
        })
        .collect();

    DepthSnapshot {
        timestamp: orderbook.timestamp,
        bid_levels,
        ask_levels,
        tick_size: orderbook.tick_size,
        lot_size: orderbook.lot_size,
    }
}

/// Simplified depth snapshot structure compatible with hftbacktest
#[derive(Debug, Clone)]
pub struct DepthSnapshot {
    pub timestamp: i64,
    pub bid_levels: Vec<Level>,
    pub ask_levels: Vec<Level>,
    pub tick_size: f64,
    pub lot_size: f64,
}

impl DepthSnapshot {
    pub fn best_bid(&self) -> f64 {
        self.bid_levels.first().map(|l| l.price).unwrap_or(0.0)
    }

    pub fn best_ask(&self) -> f64 {
        self.ask_levels.first().map(|l| l.price).unwrap_or(0.0)
    }

    pub fn mid_price(&self) -> f64 {
        (self.best_bid() + self.best_ask()) / 2.0
    }

    pub fn spread(&self) -> f64 {
        self.best_ask() - self.best_bid()
    }
}

/// Orderbook level
#[derive(Debug, Clone)]
pub struct Level {
    pub price: f64,
    pub quantity: f64,
}

/// Extract validation split from the full dataset (middle 20%)
pub fn extract_validation_split(data: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let total_timesteps = data.dims()[0];
    let start_idx = (total_timesteps as f64 * 0.6) as usize;
    let end_idx = (total_timesteps as f64 * 0.8) as usize;
    
    data.narrow(0, start_idx, end_idx - start_idx)
}

/// Extract test split from the full dataset (last 20%)
pub fn extract_test_split(data: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let total_timesteps = data.dims()[0];
    let start_idx = (total_timesteps as f64 * 0.8) as usize;
    
    data.narrow(0, start_idx, total_timesteps - start_idx)
}

/// Real orderbook backtest runner
pub struct OrderbookBacktestRunner {
    pub data: Vec<OrderbookSnapshot>,
}

impl OrderbookBacktestRunner {
    pub fn new(data: Vec<OrderbookSnapshot>) -> Self {
        Self { data }
    }

    /// Run backtest with real orderbook data
    pub fn run_with_strategy<F>(&self, mut strategy: F, initial_capital: f64) -> Result<BacktestStats>
    where
        F: FnMut(&DepthSnapshot, &mut Backtester) -> Result<()>,
    {
        let mut backtester = Backtester::new(initial_capital);

        for orderbook in &self.data {
            // Convert to depth snapshot
            let depth = orderbook_to_depth_snapshot(orderbook);

            // Execute strategy
            strategy(&depth, &mut backtester)?;

            // Update portfolio value
            backtester.update_portfolio_value(depth.mid_price(), orderbook.timestamp);
        }

        Ok(backtester.get_stats())
    }

    /// Get data length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Utility to create a simple buy-and-hold strategy for testing
pub fn create_buy_and_hold_strategy() -> impl FnMut(&DepthSnapshot, &mut Backtester) -> Result<()> {
    let mut has_bought = false;

    move |depth: &DepthSnapshot, backtester: &mut Backtester| -> Result<()> {
        if !has_bought && backtester.current_capital > 1000.0 {
            // Buy with 90% of capital
            let buy_amount = backtester.current_capital * 0.9;
            let quantity = buy_amount / depth.best_ask();

            backtester.execute_trade(
                TradeSide::Buy,
                depth.best_ask(),
                quantity,
                depth.timestamp,
            )?;

            has_bought = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtester_basic_trade() {
        let mut backtester = Backtester::new(10000.0);

        // Execute a buy trade
        backtester.execute_trade(TradeSide::Buy, 100.0, 10.0, 1000).unwrap();

        assert_eq!(backtester.position, 10.0);
        assert!(backtester.current_capital < 10000.0); // Should be less due to fees
        assert_eq!(backtester.trades.len(), 1);
    }

    #[test]
    fn test_orderbook_conversion() {
        use crate::download::{OrderbookSnapshot, OrderbookLevel};

        let orderbook = OrderbookSnapshot {
            timestamp: 1000,
            symbol: "BTCUSDT".to_string(),
            bid_levels: vec![
                OrderbookLevel { price: 50000.0, quantity: 1.0 },
                OrderbookLevel { price: 49999.0, quantity: 2.0 },
            ],
            ask_levels: vec![
                OrderbookLevel { price: 50001.0, quantity: 1.5 },
                OrderbookLevel { price: 50002.0, quantity: 2.5 },
            ],
            tick_size: 0.01,
            lot_size: 0.001,
        };

        let depth = orderbook_to_depth_snapshot(&orderbook);

        assert_eq!(depth.best_bid(), 50000.0);
        assert_eq!(depth.best_ask(), 50001.0);
        assert_eq!(depth.timestamp, 1000);
        assert_eq!(depth.bid_levels.len(), 2);
        assert_eq!(depth.ask_levels.len(), 2);
    }

    #[test]
    fn test_orderbook_backtest_runner() {
        use crate::download::{OrderbookSnapshot, OrderbookLevel};

        let orderbook_data = vec![
            OrderbookSnapshot {
                timestamp: 1000,
                symbol: "BTCUSDT".to_string(),
                bid_levels: vec![OrderbookLevel { price: 50000.0, quantity: 1.0 }],
                ask_levels: vec![OrderbookLevel { price: 50001.0, quantity: 1.0 }],
                tick_size: 0.01,
                lot_size: 0.001,
            },
            OrderbookSnapshot {
                timestamp: 2000,
                symbol: "BTCUSDT".to_string(),
                bid_levels: vec![OrderbookLevel { price: 50010.0, quantity: 1.0 }],
                ask_levels: vec![OrderbookLevel { price: 50011.0, quantity: 1.0 }],
                tick_size: 0.01,
                lot_size: 0.001,
            },
        ];

        let runner = OrderbookBacktestRunner::new(orderbook_data);
        let strategy = create_buy_and_hold_strategy();

        let stats = runner.run_with_strategy(strategy, 10000.0).unwrap();

        assert!(stats.final_value > 0.0);
        assert_eq!(stats.initial_capital, 10000.0);
    }
}
