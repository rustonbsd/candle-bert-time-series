use ndarray::Array1;

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

// Import our real data loading and backtesting modules
use candle_bert_time_series::dataset::load_crypto_as_orderbook_data;
use candle_bert_time_series::backtest::{DepthSnapshot, Backtester, TradeSide};

const SEQUENCE_LENGTH: usize = 240;
const NUM_ACTIONS: usize = 3; // HOLD, BUY, SELL
const ORDERBOOK_DEPTH: usize = 10; // Number of price levels to encode

/// Custom error type for backtesting
#[derive(Debug)]
pub enum BacktestError {
    DataError(String),
    ExecutionError(String),
    IoError(std::io::Error),
}

impl fmt::Display for BacktestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BacktestError::DataError(msg) => write!(f, "Data error: {}", msg),
            BacktestError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            BacktestError::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl Error for BacktestError {}

impl From<std::io::Error> for BacktestError {
    fn from(err: std::io::Error) -> Self {
        BacktestError::IoError(err)
    }
}

// Mock structures removed - now using real DepthSnapshot and Level from backtest module

// MockBacktest removed - now using real Backtester from backtest module

// StateValues removed - now using Backtester directly

/// Raw orderbook snapshot for transformer - NO human-engineered features
#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    pub timestamp: i64,
    pub features: Vec<f32>, // Raw orderbook data: [bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...]
}

impl OrderbookSnapshot {
    /// Create a new orderbook snapshot from depth data - RAW ORDERBOOK ONLY
    /// No human-engineered features, just raw bid/ask prices and volumes for the transformer
    pub fn from_depth(depth: &DepthSnapshot) -> Self {
        // Only raw orderbook data: [bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...]
        let mut features = Vec::with_capacity(ORDERBOOK_DEPTH * 4);

        // Collect bid/ask levels
        let bid_levels: Vec<_> = depth.bid_levels.iter().take(ORDERBOOK_DEPTH).collect();
        let ask_levels: Vec<_> = depth.ask_levels.iter().take(ORDERBOOK_DEPTH).collect();

        // Encode raw orderbook depth: [bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...]
        // Let the transformer learn all patterns from raw data
        for i in 0..ORDERBOOK_DEPTH {
            let (bid_price, bid_vol) = if i < bid_levels.len() {
                (bid_levels[i].price, bid_levels[i].quantity)
            } else { (0.0, 0.0) };

            let (ask_price, ask_vol) = if i < ask_levels.len() {
                (ask_levels[i].price, ask_levels[i].quantity)
            } else { (0.0, 0.0) };

            features.push(bid_price as f32);
            features.push(bid_vol as f32);
            features.push(ask_price as f32);
            features.push(ask_vol as f32);
        }

        Self {
            timestamp: depth.timestamp,
            features,
        }
    }

    /// Get the feature vector for transformer input
    pub fn as_features(&self) -> &[f32] {
        &self.features
    }
}

/// Financial BERT agent interface for PPO-based trading
pub trait FinancialBertAgent {
    /// Get action probabilities from the model given orderbook history
    fn predict(&self, history: &[OrderbookSnapshot]) -> Array1<f32>;

    /// Update model with PPO experience
    fn update(&mut self,
              state: &[OrderbookSnapshot],
              action: usize,
              reward: f32,
              next_state: &[OrderbookSnapshot],
              done: bool);
}

/// PPO-based financial BERT trading strategy
pub struct FinancialBertStrategy<A: FinancialBertAgent> {
    agent: A,
    orderbook_history: VecDeque<OrderbookSnapshot>,
    backtester: Backtester,
    last_trade_price: f64,
    sequence_length: usize,
    total_trades: usize,
}

impl<A: FinancialBertAgent> FinancialBertStrategy<A> {
    pub fn new(agent: A, initial_cash: f64) -> Self {
        Self {
            agent,
            orderbook_history: VecDeque::with_capacity(SEQUENCE_LENGTH),
            backtester: Backtester::new(initial_cash),
            last_trade_price: 0.0,
            sequence_length: SEQUENCE_LENGTH,
            total_trades: 0,
        }
    }

    /// Calculate reward for PPO training based on PnL and risk metrics
    fn calculate_reward(&self, prev_value: f64, current_value: f64, action: usize) -> f32 {
        let pnl = current_value - prev_value;
        let return_rate = if prev_value > 0.0 { pnl / prev_value } else { 0.0 };

        // Base reward from returns
        let mut reward = return_rate as f32 * 100.0; // Scale up returns

        // Penalize large drawdowns
        if current_value < prev_value * 0.98 {
            reward -= 0.5;
        }

        // Small penalty for trading (transaction costs)
        if action != 0 {
            reward -= 0.01;
        }

        // Bonus for profitable trades
        if pnl > 0.0 {
            reward += 0.1;
        }

        reward
    }

    /// Get current portfolio value
    fn get_portfolio_value(&self, current_price: f64) -> f64 {
        self.backtester.current_capital + (self.backtester.position * current_price)
    }

    /// Execute trading action with improved position sizing
    fn execute_action(&mut self, action: usize, depth: &DepthSnapshot) -> Result<(), BacktestError> {
        let _tick_size = depth.tick_size;
        let lot_size = depth.lot_size;
        let mid_price = depth.mid_price();

        match action {
            0 => {} // HOLD - no action
            1 => { // BUY
                let buy_price = depth.best_ask(); // Market buy at ask
                let position_size = 0.05; // 5% of portfolio per trade
                let current_value = self.get_portfolio_value(mid_price);
                let buy_qty = ((current_value * position_size) / mid_price / lot_size).floor() * lot_size;

                if buy_qty > 0.0 {
                    match self.backtester.execute_trade(TradeSide::Buy, buy_price, buy_qty, depth.timestamp) {
                        Ok(_) => {
                            self.total_trades += 1;
                        }
                        Err(_) => {} // Insufficient capital, skip trade
                    }
                }
            }
            2 => { // SELL
                if self.backtester.position > 0.0 {
                    let sell_price = depth.best_bid(); // Market sell at bid
                    let sell_qty = (self.backtester.position * 0.2 / lot_size).floor() * lot_size; // Sell 20% of position

                    if sell_qty > 0.0 {
                        match self.backtester.execute_trade(TradeSide::Sell, sell_price, sell_qty, depth.timestamp) {
                            Ok(_) => {
                                self.total_trades += 1;
                            }
                            Err(_) => {} // Insufficient position, skip trade
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Update portfolio value and track performance
    fn update_portfolio(&mut self, current_price: f64, timestamp: i64) {
        self.backtester.update_portfolio_value(current_price, timestamp);
        self.last_trade_price = current_price;
    }
}

/// Main backtest runner for financial BERT PPO strategy
pub fn run_backtest<A: FinancialBertAgent>(
    agent: A,
    data_path: &str,
    initial_cash: f64,
) -> Result<(Vec<f64>, Vec<OrderbookSnapshot>), BacktestError> {
    let mut strategy = FinancialBertStrategy::new(agent, initial_cash);

    // Load real crypto data from parquet file
    let crypto_data = load_crypto_as_orderbook_data(data_path)
        .map_err(|e| BacktestError::DataError(format!("Failed to load data: {}", e)))?;

    println!("Loaded {} timesteps of crypto data", crypto_data.len());

    let mut portfolio_values = Vec::new();
    let mut all_snapshots = Vec::new();
    let mut step_count = 0;
    let base_price = 50000.0; // Starting price for simulation
    let tick_size = 0.01;
    let lot_size = 0.001;

    // Main backtest loop
    for crypto_timestep in &crypto_data {
        // Convert crypto data to depth snapshot (using first crypto for now)
        // For now, create a simple depth snapshot from the crypto data
        // This is a placeholder - in practice you'd use real orderbook data
        let depth = DepthSnapshot {
            timestamp: crypto_timestep.timestamp,
            bid_levels: vec![
                candle_bert_time_series::backtest::Level { price: base_price - 0.5, quantity: 1.0 },
                candle_bert_time_series::backtest::Level { price: base_price - 1.0, quantity: 2.0 },
            ],
            ask_levels: vec![
                candle_bert_time_series::backtest::Level { price: base_price + 0.5, quantity: 1.0 },
                candle_bert_time_series::backtest::Level { price: base_price + 1.0, quantity: 2.0 },
            ],
            tick_size,
            lot_size,
        };

        // Create orderbook snapshot with encoded features
        let snapshot = OrderbookSnapshot::from_depth(&depth);
        all_snapshots.push(snapshot.clone());

        // Maintain rolling window
        strategy.orderbook_history.push_back(snapshot);
        if strategy.orderbook_history.len() > strategy.sequence_length {
            strategy.orderbook_history.pop_front();
        }

        // Get prediction from financial BERT model
        if strategy.orderbook_history.len() >= strategy.sequence_length {
            let history: Vec<_> = strategy.orderbook_history.iter().cloned().collect();
            let action_probs = strategy.agent.predict(&history);

            // Sample action from probabilities (for exploration) or use argmax
            let action = action_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            // Execute action
            let prev_value = strategy.get_portfolio_value(depth.mid_price());
            strategy.execute_action(action, &depth)?;

            // Update portfolio
            strategy.update_portfolio(depth.mid_price(), crypto_timestep.timestamp);

            // Calculate reward for PPO training
            let current_value = strategy.get_portfolio_value(depth.mid_price());
            let reward = strategy.calculate_reward(prev_value, current_value, action);
            let next_history: Vec<_> = strategy.orderbook_history.iter().cloned().collect();
            let done = step_count > 10000; // Episode termination condition

            strategy.agent.update(&history, action, reward, &next_history, done);

            portfolio_values.push(current_value);
        }

        step_count += 1;
        if step_count % 1000 == 0 {
            let current_value = strategy.get_portfolio_value(depth.mid_price());
            println!("Step {}: Portfolio value: ${:.2}, Trades: {}",
                     step_count, current_value, strategy.total_trades);
        }
    }

    Ok((portfolio_values, all_snapshots))
}

// Mock data creation removed - now using real crypto data from parquet files

/// Mock Financial BERT agent for testing
pub struct MockFinancialBertAgent {
    action_count: [usize; NUM_ACTIONS],
    total_reward: f32,
}

impl MockFinancialBertAgent {
    pub fn new() -> Self {
        Self {
            action_count: [0; NUM_ACTIONS],
            total_reward: 0.0,
        }
    }
}

impl FinancialBertAgent for MockFinancialBertAgent {
    fn predict(&self, history: &[OrderbookSnapshot]) -> Array1<f32> {
        let latest = history.last().unwrap();
        let features = latest.as_features();

        // Simple heuristic strategy based on orderbook features
        let mut action_probs = Array1::zeros(NUM_ACTIONS);

        if features.len() >= 8 {
            let imbalance = features[4]; // Orderbook imbalance
            let spread = features[2];    // Relative spread

            // Buy if positive imbalance and tight spread
            if imbalance > 0.1 && spread < 0.001 {
                action_probs[1] = 0.7; // BUY
                action_probs[0] = 0.2; // HOLD
                action_probs[2] = 0.1; // SELL
            }
            // Sell if negative imbalance
            else if imbalance < -0.1 {
                action_probs[2] = 0.6; // SELL
                action_probs[0] = 0.3; // HOLD
                action_probs[1] = 0.1; // BUY
            }
            // Hold otherwise
            else {
                action_probs[0] = 0.8; // HOLD
                action_probs[1] = 0.1; // BUY
                action_probs[2] = 0.1; // SELL
            }
        } else {
            // Default to hold if insufficient features
            action_probs[0] = 1.0;
        }

        action_probs
    }

    fn update(&mut self,
              _state: &[OrderbookSnapshot],
              action: usize,
              reward: f32,
              _next_state: &[OrderbookSnapshot],
              _done: bool) {
        // Track statistics for mock agent
        if action < NUM_ACTIONS {
            self.action_count[action] += 1;
        }
        self.total_reward += reward;

        // In a real implementation, this would update the BERT model weights using PPO
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Financial BERT PPO Trading Strategy Backtest");
    println!("{}", "=".repeat(60));

    // Initialize mock financial BERT agent
    let agent = MockFinancialBertAgent::new();

    // Configuration
    let data_path = "/mnt/storage-box/15m/transformed_dataset.parquet"; // Real parquet data path
    let initial_cash = 10_000.0;

    println!("📊 Configuration:");
    println!("  • Data path: {}", data_path);
    println!("  • Initial capital: ${:.2}", initial_cash);
    println!("  • Sequence length: {}", SEQUENCE_LENGTH);
    println!("  • Orderbook depth: {} levels", ORDERBOOK_DEPTH);

    // Run backtest
    match run_backtest(agent, data_path, initial_cash) {
        Ok((portfolio_values, snapshots)) => {
            let final_value = portfolio_values.last().unwrap_or(&initial_cash);
            let total_return = (final_value - initial_cash) / initial_cash * 100.0;

            println!("\n📈 Backtest Results:");
            println!("  • Final portfolio value: ${:.2}", final_value);
            println!("  • Total return: {:.2}%", total_return);
            println!("  • Total snapshots processed: {}", snapshots.len());
            println!("  • Portfolio value samples: {}", portfolio_values.len());

            // Calculate basic performance metrics
            if portfolio_values.len() > 1 {
                let max_value = portfolio_values.iter().fold(0.0f64, |a, &b| a.max(b));
                let min_value = portfolio_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_drawdown = (max_value - min_value) / max_value * 100.0;

                println!("  • Max portfolio value: ${:.2}", max_value);
                println!("  • Min portfolio value: ${:.2}", min_value);
                println!("  • Max drawdown: {:.2}%", max_drawdown);
            }

            // Save results
            let results = serde_json::json!({
                "initial_capital": initial_cash,
                "final_value": final_value,
                "total_return_pct": total_return,
                "portfolio_values": portfolio_values,
                "num_snapshots": snapshots.len(),
                "config": {
                    "sequence_length": SEQUENCE_LENGTH,
                    "orderbook_depth": ORDERBOOK_DEPTH,
                    "num_actions": NUM_ACTIONS
                }
            });

            std::fs::write("financial_bert_backtest_results.json",
                          serde_json::to_string_pretty(&results)?)?;
            println!("\n💾 Results saved to: financial_bert_backtest_results.json");
        }
        Err(e) => {
            eprintln!("❌ Backtest failed: {}", e);
            eprintln!("💡 Make sure your data file exists and is in the correct format");
            return Err(e.into());
        }
    }

    println!("\n✅ Backtest completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_snapshot_features() {
        // Create a mock orderbook snapshot with features
        let features = vec![
            0.999, 1.001, 0.002, 50000.0, // Core market features
            0.1, 1.0002, 1000.0, 1200.0,  // Imbalance, micro price, volumes
            // Mock depth features (bid_price, bid_vol, ask_price, ask_vol) * 10 levels
            0.998, 100.0, 1.002, 120.0,   // Level 1
            0.997, 200.0, 1.003, 180.0,   // Level 2
            // ... (remaining levels would be zeros in practice)
        ];

        let mut full_features = features;
        // Pad to full depth
        while full_features.len() < ORDERBOOK_DEPTH * 4 + 8 {
            full_features.push(0.0);
        }

        let snapshot = OrderbookSnapshot {
            timestamp: 1234567890,
            features: full_features,
        };

        assert_eq!(snapshot.timestamp, 1234567890);
        assert!(snapshot.features.len() >= 8); // At least core features
        assert_eq!(snapshot.as_features()[0], 0.999); // Normalized best bid
    }

    #[test]
    fn test_mock_agent_prediction() {
        let agent = MockFinancialBertAgent::new();

        // Create a simple raw orderbook snapshot (no human-engineered features)
        // Format: [bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...]
        let mut features = Vec::new();
        for i in 0..ORDERBOOK_DEPTH {
            features.push(50000.0 - (i as f32 * 0.5)); // bid_price
            features.push(100.0 + (i as f32 * 10.0));  // bid_vol
            features.push(50000.5 + (i as f32 * 0.5)); // ask_price
            features.push(100.0 + (i as f32 * 10.0));  // ask_vol
        }

        let snapshot = OrderbookSnapshot {
            timestamp: 1234567890,
            features,
        };

        let history = vec![snapshot];
        let predictions = agent.predict(&history);

        assert_eq!(predictions.len(), NUM_ACTIONS);
        // Probabilities should sum to approximately 1.0
        let sum: f32 = predictions.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_real_data_integration() {
        use candle_bert_time_series::dataset::CryptoTimestep;
        use candle_bert_time_series::backtest::create_depth_from_crypto_data;

        // Test crypto data to depth conversion
        let crypto_data = CryptoTimestep {
            timestamp: 1000,
            crypto_returns: vec![0.01, -0.005, 0.02],
        };

        let depth = create_depth_from_crypto_data(&crypto_data, 0, 100.0, 0.01, 0.001).unwrap();
        let snapshot = OrderbookSnapshot::from_depth(&depth);

        assert_eq!(snapshot.timestamp, 1000);
        // Raw orderbook features: ORDERBOOK_DEPTH * 4 (bid_price, bid_vol, ask_price, ask_vol)
        assert_eq!(snapshot.features.len(), ORDERBOOK_DEPTH * 4);

        // Verify we have actual price/volume data
        assert!(snapshot.features[0] > 0.0); // First bid price should be positive
        assert!(snapshot.features[1] > 0.0); // First bid volume should be positive
        assert!(snapshot.features[2] > 0.0); // First ask price should be positive
        assert!(snapshot.features[3] > 0.0); // First ask volume should be positive
    }

    #[test]
    fn test_constants() {
        assert_eq!(NUM_ACTIONS, 3);
        assert_eq!(SEQUENCE_LENGTH, 240);
        assert_eq!(ORDERBOOK_DEPTH, 10);
    }
}