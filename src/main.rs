use hftbacktest::prelude::*;
use ndarray::Array1;
use rand;

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

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

/// Mock orderbook level for demonstration
#[derive(Debug, Clone)]
pub struct Level {
    pub px: f64,  // price
    pub qty: f64, // quantity
}

/// Mock depth structure for demonstration (replace with actual hftbacktest Depth)
#[derive(Debug, Clone)]
pub struct MockDepth {
    pub timestamp: i64,
    pub bid_levels: Vec<Level>,
    pub ask_levels: Vec<Level>,
    pub tick_size: f64,
    pub lot_size: f64,
}

impl MockDepth {
    pub fn best_bid(&self) -> f64 {
        self.bid_levels.first().map(|l| l.px).unwrap_or(0.0)
    }

    pub fn best_ask(&self) -> f64 {
        self.ask_levels.first().map(|l| l.px).unwrap_or(0.0)
    }

    pub fn bid_levels(&self) -> impl Iterator<Item = &Level> {
        self.bid_levels.iter()
    }

    pub fn ask_levels(&self) -> impl Iterator<Item = &Level> {
        self.ask_levels.iter()
    }
}

/// Mock backtest structure for demonstration
#[derive(Debug)]
pub struct MockBacktest {
    pub current_step: usize,
    pub depth_data: Vec<MockDepth>,
    pub positions: f64,
    pub cash: f64,
}

impl MockBacktest {
    pub fn new(depth_data: Vec<MockDepth>, initial_cash: f64) -> Self {
        Self {
            current_step: 0,
            depth_data,
            positions: 0.0,
            cash: initial_cash,
        }
    }

    pub fn elapse(&mut self, _asset_no: usize) -> Result<(), BacktestError> {
        if self.current_step >= self.depth_data.len() {
            return Err(BacktestError::ExecutionError("No more data".to_string()));
        }
        self.current_step += 1;
        Ok(())
    }

    pub fn depth(&self, _asset_no: usize) -> &MockDepth {
        &self.depth_data[self.current_step.min(self.depth_data.len() - 1)]
    }

    pub fn clear_inactive_orders(&mut self, _asset_no: usize) {
        // Mock implementation
    }

    pub fn submit_buy_order(&mut self, _asset_no: usize, _order_id: u64, price: f64, qty: f64, _tif: TimeInForce, _order_type: OrdType, _wait: bool) -> Result<(), BacktestError> {
        let cost = price * qty;
        if self.cash >= cost {
            self.cash -= cost;
            self.positions += qty;
        }
        Ok(())
    }

    pub fn submit_sell_order(&mut self, _asset_no: usize, _order_id: u64, price: f64, qty: f64, _tif: TimeInForce, _order_type: OrdType, _wait: bool) -> Result<(), BacktestError> {
        if self.positions >= qty {
            self.positions -= qty;
            self.cash += price * qty;
        }
        Ok(())
    }

    pub fn state_values(&self, _asset_no: usize) -> StateValues {
        StateValues {
            position: self.positions,
            balance: self.cash,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateValues {
    pub position: f64,
    pub balance: f64,
}

/// Enhanced orderbook snapshot with proper depth encoding for financial BERT
#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    pub timestamp: i64,
    pub features: Vec<f32>, // Flattened feature vector for transformer input
}

impl OrderbookSnapshot {
    /// Create a new orderbook snapshot from depth data
    pub fn from_depth(depth: &MockDepth) -> Self {
        let mut features = Vec::with_capacity(ORDERBOOK_DEPTH * 4 + 8);

        let best_bid = depth.best_bid();
        let best_ask = depth.best_ask();
        let mid_price = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;

        // Core market features
        features.push((best_bid / mid_price) as f32); // Normalized best bid
        features.push((best_ask / mid_price) as f32); // Normalized best ask
        features.push((spread / mid_price) as f32);   // Relative spread
        features.push(mid_price as f32);              // Mid price (absolute)

        // Collect bid/ask levels with volumes
        let bid_levels: Vec<_> = depth.bid_levels().take(ORDERBOOK_DEPTH).collect();
        let ask_levels: Vec<_> = depth.ask_levels().take(ORDERBOOK_DEPTH).collect();

        let total_bid_vol: f64 = bid_levels.iter().map(|l| l.qty).sum();
        let total_ask_vol: f64 = ask_levels.iter().map(|l| l.qty).sum();

        // Orderbook imbalance and micro price
        let imbalance = if total_bid_vol + total_ask_vol > 0.0 {
            (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        } else { 0.0 };

        let micro_price = if total_bid_vol + total_ask_vol > 0.0 {
            (best_bid * total_ask_vol + best_ask * total_bid_vol) / (total_bid_vol + total_ask_vol)
        } else { mid_price };

        features.push(imbalance as f32);
        features.push((micro_price / mid_price) as f32); // Normalized micro price
        features.push(total_bid_vol as f32);
        features.push(total_ask_vol as f32);

        // Encode orderbook depth: [bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...]
        for i in 0..ORDERBOOK_DEPTH {
            let (bid_price, bid_vol) = if i < bid_levels.len() {
                (bid_levels[i].px / mid_price, bid_levels[i].qty)
            } else { (0.0, 0.0) };

            let (ask_price, ask_vol) = if i < ask_levels.len() {
                (ask_levels[i].px / mid_price, ask_levels[i].qty)
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
    position: f64,
    cash: f64,
    asset_value: f64,
    last_trade_price: f64,
    sequence_length: usize,
    total_trades: usize,
    total_pnl: f64,
}

impl<A: FinancialBertAgent> FinancialBertStrategy<A> {
    pub fn new(agent: A, initial_cash: f64) -> Self {
        Self {
            agent,
            orderbook_history: VecDeque::with_capacity(SEQUENCE_LENGTH),
            position: 0.0,
            cash: initial_cash,
            asset_value: initial_cash,
            last_trade_price: 0.0,
            sequence_length: SEQUENCE_LENGTH,
            total_trades: 0,
            total_pnl: 0.0,
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

    /// Execute trading action with improved position sizing
    fn execute_action(&mut self, hbt: &mut MockBacktest, action: usize, depth: &MockDepth) -> Result<(), BacktestError> {
        let asset_no = 0;
        let _tick_size = depth.tick_size;
        let lot_size = depth.lot_size;
        let mid_price = (depth.best_bid() + depth.best_ask()) / 2.0;

        hbt.clear_inactive_orders(asset_no);

        match action {
            0 => {} // HOLD - no action
            1 => { // BUY
                let buy_price = depth.best_ask(); // Market buy at ask
                let position_size = 0.05; // 5% of portfolio per trade
                let buy_qty = ((self.cash * position_size) / mid_price / lot_size).floor() * lot_size;

                if buy_qty > 0.0 && self.cash >= buy_qty * buy_price {
                    let order_id = (depth.timestamp % 1_000_000) as u64;
                    hbt.submit_buy_order(
                        asset_no,
                        order_id,
                        buy_price,
                        buy_qty,
                        TimeInForce::IOC, // Immediate or Cancel for market orders
                        OrdType::Limit,
                        false,
                    )?;
                    self.total_trades += 1;
                }
            }
            2 => { // SELL
                if self.position > 0.0 {
                    let sell_price = depth.best_bid(); // Market sell at bid
                    let sell_qty = (self.position * 0.2 / lot_size).floor() * lot_size; // Sell 20% of position

                    if sell_qty > 0.0 {
                        let order_id = (depth.timestamp % 1_000_000) as u64 + 1;
                        hbt.submit_sell_order(
                            asset_no,
                            order_id,
                            sell_price,
                            sell_qty,
                            TimeInForce::IOC,
                            OrdType::Limit,
                            false,
                        )?;
                        self.total_trades += 1;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Update portfolio value and track performance
    fn update_portfolio(&mut self, hbt: &mut MockBacktest) {
        let asset_no = 0;
        let depth = hbt.depth(asset_no);
        let mid_price = (depth.best_bid() + depth.best_ask()) / 2.0;

        // Update position from filled orders
        let state_values = hbt.state_values(asset_no);
        self.position = state_values.position;
        self.cash = state_values.balance;

        self.asset_value = self.cash + self.position * mid_price;
        self.last_trade_price = mid_price;

        // Track total PnL
        self.total_pnl = self.asset_value - 10_000.0; // Assuming 10k initial capital
    }
}

/// Main backtest runner for financial BERT PPO strategy
pub fn run_backtest<A: FinancialBertAgent>(
    agent: A,
    _data_path: &str,
    initial_cash: f64,
) -> Result<(Vec<f64>, Vec<OrderbookSnapshot>), BacktestError> {
    let mut strategy = FinancialBertStrategy::new(agent, initial_cash);

    // Create mock depth data for demonstration
    // In a real implementation, you would load this from your HFT data files
    let mock_depth_data = create_mock_depth_data(1000); // 1000 time steps

    // Create backtest with mock data
    let mut hbt = MockBacktest::new(mock_depth_data, initial_cash);

    let mut portfolio_values = Vec::new();
    let mut all_snapshots = Vec::new();
    let mut step_count = 0;

    // Main backtest loop
    while hbt.elapse(0).is_ok() {
        let asset_no = 0;
        let depth = hbt.depth(asset_no).clone(); // Clone to avoid borrowing issues

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
            let prev_value = strategy.asset_value;
            strategy.execute_action(&mut hbt, action, &depth)?;

            // Update portfolio
            strategy.update_portfolio(&mut hbt);

            // Calculate reward for PPO training
            let reward = strategy.calculate_reward(prev_value, strategy.asset_value, action);
            let next_history: Vec<_> = strategy.orderbook_history.iter().cloned().collect();
            let done = step_count > 10000; // Episode termination condition

            strategy.agent.update(&history, action, reward, &next_history, done);

            portfolio_values.push(strategy.asset_value);
        }

        step_count += 1;
        if step_count % 1000 == 0 {
            println!("Step {}: Portfolio value: ${:.2}, Trades: {}",
                     step_count, strategy.asset_value, strategy.total_trades);
        }
    }
    
    Ok((portfolio_values, all_snapshots))
}

/// Create mock depth data for demonstration
fn create_mock_depth_data(num_steps: usize) -> Vec<MockDepth> {
    let mut data = Vec::with_capacity(num_steps);
    let mut base_price = 50000.0;

    for i in 0..num_steps {
        // Simulate price movement
        base_price += (i as f64 * 0.1).sin() * 10.0 + rand::random::<f64>() * 20.0 - 10.0;

        // Create bid levels
        let mut bid_levels = Vec::new();
        for j in 0..ORDERBOOK_DEPTH {
            bid_levels.push(Level {
                px: base_price - (j as f64 + 1.0) * 0.5,
                qty: 100.0 + rand::random::<f64>() * 200.0,
            });
        }

        // Create ask levels
        let mut ask_levels = Vec::new();
        for j in 0..ORDERBOOK_DEPTH {
            ask_levels.push(Level {
                px: base_price + (j as f64 + 1.0) * 0.5,
                qty: 100.0 + rand::random::<f64>() * 200.0,
            });
        }

        data.push(MockDepth {
            timestamp: i as i64,
            bid_levels,
            ask_levels,
            tick_size: 0.01,
            lot_size: 0.001,
        });
    }

    data
}

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
    let data_path = "data/BTCUSDT_2024-01-01.npz"; // Update with your HFT data path
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

        // Create a simple orderbook snapshot
        let features = vec![0.999, 1.001, 0.001, 50000.0, 0.05, 1.0001, 1000.0, 1100.0];
        let mut full_features = features;
        while full_features.len() < ORDERBOOK_DEPTH * 4 + 8 {
            full_features.push(0.0);
        }

        let snapshot = OrderbookSnapshot {
            timestamp: 1234567890,
            features: full_features,
        };

        let history = vec![snapshot];
        let predictions = agent.predict(&history);

        assert_eq!(predictions.len(), NUM_ACTIONS);
        // Probabilities should sum to approximately 1.0
        let sum: f32 = predictions.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_constants() {
        assert_eq!(NUM_ACTIONS, 3);
        assert_eq!(SEQUENCE_LENGTH, 240);
        assert_eq!(ORDERBOOK_DEPTH, 10);
    }
}