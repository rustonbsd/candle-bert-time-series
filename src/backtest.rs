use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// Binance trading fees configuration
#[derive(Debug, Clone)]
pub struct TradingFees {
    /// Maker fee (when providing liquidity)
    pub maker_fee: f64,
    /// Taker fee (when taking liquidity)
    pub taker_fee: f64,
}

impl Default for TradingFees {
    fn default() -> Self {
        Self {
            maker_fee: 0.001,  // 0.1% Binance maker fee
            taker_fee: 0.001,  // 0.1% Binance taker fee
        }
    }
}

/// Position information for a single asset
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_time: usize,
    pub current_value: f64,
}

/// Trade execution record
#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: usize,
    pub fee: f64,
    pub pnl: Option<f64>, // Only for closing trades
}

#[derive(Debug, Clone, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Portfolio state at a given time
#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub timestamp: usize,
    pub cash: f64,
    pub positions: HashMap<String, Position>,
    pub total_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// Backtest performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub total_fees: f64,
    pub final_portfolio_value: f64,
}

/// Main backtesting engine
pub struct Backtester {
    pub initial_capital: f64,
    pub fees: TradingFees,
    pub portfolio_history: Vec<PortfolioState>,
    pub trades: Vec<Trade>,
    pub symbol_names: Vec<String>,
    pub returns_data: Tensor, // [timesteps, num_assets] - percentage returns
    pub current_prices: Vec<f64>, // Reconstructed absolute prices for position tracking
}

impl Backtester {
    /// Create a new backtester instance
    pub fn new(
        initial_capital: f64,
        returns_data: Tensor,
        symbol_names: Vec<String>,
        fees: Option<TradingFees>,
    ) -> Result<Self> {
        let fees = fees.unwrap_or_default();
        let num_assets = symbol_names.len();

        // Initialize current prices to 100.0 for all assets (arbitrary base price)
        let current_prices = vec![100.0; num_assets];

        // Initialize portfolio with cash only
        let initial_portfolio = PortfolioState {
            timestamp: 0,
            cash: initial_capital,
            positions: HashMap::new(),
            total_value: initial_capital,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        };

        Ok(Self {
            initial_capital,
            fees,
            portfolio_history: vec![initial_portfolio],
            trades: Vec::new(),
            symbol_names,
            returns_data,
            current_prices,
        })
    }

    /// Execute a trade (buy or sell)
    pub fn execute_trade(
        &mut self,
        symbol: &str,
        side: TradeSide,
        quantity: f64,
        timestamp: usize,
    ) -> Result<()> {
        let symbol_idx = self.symbol_names.iter()
            .position(|s| s == symbol)
            .ok_or_else(|| candle_core::Error::Msg(format!("Symbol {} not found", symbol)))?;

        let current_price = self.current_prices[symbol_idx];
        let trade_value = quantity * current_price;
        let fee = trade_value * self.fees.taker_fee;

        let mut current_portfolio = self.portfolio_history.last().unwrap().clone();
        current_portfolio.timestamp = timestamp;

        match side {
            TradeSide::Buy => {
                let total_cost = trade_value + fee;
                if current_portfolio.cash < total_cost {
                    return Err(candle_core::Error::Msg("Insufficient cash for trade".to_string()));
                }

                current_portfolio.cash -= total_cost;

                // Add or update position
                let position = current_portfolio.positions.entry(symbol.to_string())
                    .or_insert(Position {
                        symbol: symbol.to_string(),
                        quantity: 0.0,
                        entry_price: current_price,
                        entry_time: timestamp,
                        current_value: 0.0,
                    });

                // Update average entry price for additional purchases
                let total_quantity = position.quantity + quantity;
                position.entry_price = (position.entry_price * position.quantity + current_price * quantity) / total_quantity;
                position.quantity = total_quantity;
                position.current_value = position.quantity * current_price;

                self.trades.push(Trade {
                    symbol: symbol.to_string(),
                    side,
                    quantity,
                    price: current_price,
                    timestamp,
                    fee,
                    pnl: None,
                });
            },
            TradeSide::Sell => {
                let position = current_portfolio.positions.get_mut(symbol)
                    .ok_or_else(|| candle_core::Error::Msg(format!("No position in {} to sell", symbol)))?;

                if position.quantity < quantity {
                    return Err(candle_core::Error::Msg("Insufficient position size for sale".to_string()));
                }

                let sale_proceeds = trade_value - fee;
                let pnl = (current_price - position.entry_price) * quantity;

                current_portfolio.cash += sale_proceeds;
                current_portfolio.realized_pnl += pnl;

                position.quantity -= quantity;
                position.current_value = position.quantity * current_price;

                // Remove position if fully closed
                if position.quantity <= 1e-8 {
                    current_portfolio.positions.remove(symbol);
                }

                self.trades.push(Trade {
                    symbol: symbol.to_string(),
                    side,
                    quantity,
                    price: current_price,
                    timestamp,
                    fee,
                    pnl: Some(pnl),
                });
            }
        }

        self.update_portfolio_value(&mut current_portfolio);
        self.portfolio_history.push(current_portfolio);
        Ok(())
    }

    /// Update portfolio value and unrealized PnL
    fn update_portfolio_value(&self, portfolio: &mut PortfolioState) {
        let mut total_position_value = 0.0;
        let mut unrealized_pnl = 0.0;

        for position in portfolio.positions.values_mut() {
            let symbol_idx = self.symbol_names.iter()
                .position(|s| s == &position.symbol)
                .unwrap();
            let current_price = self.current_prices[symbol_idx];

            position.current_value = position.quantity * current_price;
            total_position_value += position.current_value;

            let position_pnl = (current_price - position.entry_price) * position.quantity;
            unrealized_pnl += position_pnl;
        }

        portfolio.total_value = portfolio.cash + total_position_value;
        portfolio.unrealized_pnl = unrealized_pnl;
    }

    /// Step forward one time period, updating prices based on returns
    pub fn step_forward(&mut self, timestamp: usize) -> Result<()> {
        if timestamp == 0 {
            return Ok(());
        }

        // Update prices based on returns data
        let returns_row = self.returns_data.get(timestamp - 1)?; // Get returns for this period
        let returns_vec: Vec<f32> = returns_row.to_vec1()?;

        for (i, return_rate) in returns_vec.iter().enumerate() {
            // Apply return: new_price = old_price * (1 + return_rate)
            self.current_prices[i] *= 1.0 + (*return_rate as f64);
        }

        // Update current portfolio state
        if let Some(mut current_portfolio) = self.portfolio_history.last().cloned() {
            current_portfolio.timestamp = timestamp;
            self.update_portfolio_value(&mut current_portfolio);
            self.portfolio_history.push(current_portfolio);
        }

        Ok(())
    }

    /// Calculate comprehensive performance metrics
    pub fn calculate_metrics(&self) -> Result<PerformanceMetrics> {
        if self.portfolio_history.len() < 2 {
            return Err(candle_core::Error::Msg("Insufficient data for metrics calculation".to_string()));
        }

        let final_value = self.portfolio_history.last().unwrap().total_value;
        let total_return = (final_value - self.initial_capital) / self.initial_capital;

        // Calculate returns series for Sharpe ratio and drawdown
        let mut returns = Vec::new();
        let mut portfolio_values = Vec::new();

        for portfolio in &self.portfolio_history {
            portfolio_values.push(portfolio.total_value);
        }

        for i in 1..portfolio_values.len() {
            let period_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1];
            returns.push(period_return);
        }

        // Annualized return (assuming minute data: 525,600 minutes per year)
        let periods_per_year = 525_600.0;
        let total_periods = returns.len() as f64;
        let annualized_return = (1.0 + total_return).powf(periods_per_year / total_periods) - 1.0;

        // Sharpe ratio
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let return_std = {
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };
        let sharpe_ratio = if return_std > 0.0 {
            (mean_return * periods_per_year.sqrt()) / (return_std * periods_per_year.sqrt())
        } else {
            0.0
        };

        // Maximum drawdown
        let (max_drawdown, max_dd_duration) = self.calculate_max_drawdown(&portfolio_values);

        // Trading statistics
        let total_trades = self.trades.len();
        let total_fees: f64 = self.trades.iter().map(|t| t.fee).sum();

        let profitable_trades = self.trades.iter()
            .filter(|t| t.pnl.map_or(false, |pnl| pnl > 0.0))
            .count();
        let win_rate = if total_trades > 0 {
            profitable_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = self.trades.iter()
            .filter_map(|t| t.pnl)
            .filter(|&pnl| pnl > 0.0)
            .sum();
        let gross_loss: f64 = self.trades.iter()
            .filter_map(|t| t.pnl)
            .filter(|&pnl| pnl < 0.0)
            .sum();
        let profit_factor = if gross_loss.abs() > 0.0 {
            gross_profit / gross_loss.abs()
        } else {
            f64::INFINITY
        };

        Ok(PerformanceMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            max_drawdown_duration: max_dd_duration,
            win_rate,
            profit_factor,
            total_trades,
            total_fees,
            final_portfolio_value: final_value,
        })
    }

    /// Calculate maximum drawdown and its duration
    fn calculate_max_drawdown(&self, portfolio_values: &[f64]) -> (f64, usize) {
        let mut max_drawdown = 0.0;
        let mut max_dd_duration = 0;
        let mut peak = portfolio_values[0];
        let mut dd_start = 0;

        for (i, &value) in portfolio_values.iter().enumerate() {
            if value > peak {
                peak = value;
                dd_start = i;
            } else {
                let drawdown = (peak - value) / peak;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                    max_dd_duration = i - dd_start;
                }
            }
        }

        (max_drawdown, max_dd_duration)
    }

    /// Get current portfolio state
    pub fn get_current_portfolio(&self) -> &PortfolioState {
        self.portfolio_history.last().unwrap()
    }

    /// Get all portfolio history
    pub fn get_portfolio_history(&self) -> &[PortfolioState] {
        &self.portfolio_history
    }

    /// Get all trades
    pub fn get_trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get current price for a symbol
    pub fn get_current_price(&self, symbol: &str) -> Option<f64> {
        self.symbol_names.iter()
            .position(|s| s == symbol)
            .map(|idx| self.current_prices[idx])
    }

    /// Get all current prices
    pub fn get_current_prices(&self) -> &[f64] {
        &self.current_prices
    }

    /// Get symbol names
    pub fn get_symbol_names(&self) -> &[String] {
        &self.symbol_names
    }
}

/// Helper function to extract test split data to prevent data leakage
/// Uses the same split logic as the training loop: train (70%), validation (15%), test (15%)
pub fn extract_test_split(full_data: &Tensor) -> Result<Tensor> {
    let total_timesteps = full_data.dims()[0];

    // Apply the SAME split logic as in main.rs training loop
    let val_split = (total_timesteps as f32 * 0.85) as usize;

    // Return only the test split (last 15% of data)
    let test_data = full_data.narrow(0, val_split, total_timesteps - val_split)?;

    Ok(test_data)
}

/// Helper function to get data split information
pub fn get_data_split_info(total_timesteps: usize) -> (usize, usize, usize) {
    let train_split = (total_timesteps as f32 * 0.7) as usize;
    let val_split = (total_timesteps as f32 * 0.85) as usize;
    let test_size = total_timesteps - val_split;

    (train_split, val_split - train_split, test_size)
}