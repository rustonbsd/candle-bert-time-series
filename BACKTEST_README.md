# Cryptocurrency Backtesting Library

A comprehensive backtesting library for cryptocurrency trading strategies using historical minute-level return data. This library handles Binance trading fees, portfolio management, and calculates key performance metrics including Sharpe ratio and Maximum Drawdown.

## Features

- **Realistic Trading Simulation**: Includes Binance maker/taker fees (0.1% default)
- **Portfolio Management**: Tracks cash, positions, and portfolio value over time
- **Leverage/Margin Support**: Optional negative cash allowance to see full P&L potential
- **Performance Metrics**: Calculates comprehensive metrics including:
  - Total and annualized returns
  - Sharpe ratio
  - Maximum drawdown and duration
  - Win rate and profit factor
  - Leverage and margin usage metrics
  - Trade statistics
- **Flexible Data Format**: Works with percentage return data (no absolute prices needed)
- **Memory Efficient**: Processes large datasets efficiently

## Data Format

The library expects cryptocurrency data in the format produced by the dataset creation tools:
- **Input**: Tensor of shape `[timesteps, num_assets]` containing percentage returns
- **Returns**: Each value represents the percentage change from the previous period
- **Missing Data**: Handled as 0.0 returns (no change)

## ⚠️ **CRITICAL: Preventing Data Leakage**

**ALWAYS use only the test split for backtesting!** Using training or validation data will give misleadingly optimistic results.

The library provides helper functions to ensure proper data splitting:
```rust
use candle_bert_time_series::backtest::{extract_test_split, get_data_split_info};

// Load full dataset
let (full_data, num_assets) = load_and_prepare_data("data.parquet", &device)?;

// Extract ONLY test split (last 15% of data)
let test_data = extract_test_split(&full_data)?;

// Use test_data for backtesting - never full_data!
let backtester = Backtester::new(capital, test_data, symbols, fees)?;
```

**Data Split Logic (matches training loop):**
- Train: 70% (first 70% of time series)
- Validation: 15% (next 15% of time series)
- Test: 15% (last 15% of time series) ← **Use this for backtesting**

## Quick Start

### Basic Usage

```rust
use candle_bert_time_series::backtest::{Backtester, TradingFees, TradeSide};
use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_core::Device;

// Load your data
let device = Device::Cpu;
let (returns_data, num_assets) = load_and_prepare_data("data.parquet", &device)?;

// Create symbol names
let symbol_names: Vec<String> = (0..num_assets)
    .map(|i| format!("ASSET_{}", i))
    .collect();

// Initialize backtester with leverage support
let mut backtester = Backtester::new_with_leverage(
    100_000.0,  // $100k initial capital
    returns_data,
    symbol_names,
    Some(TradingFees::default()), // Binance fees
    true, // Allow negative cash (leverage/margin)
)?;

// Run backtest
for timestamp in 1..total_timesteps {
    // Step forward (updates prices based on returns)
    backtester.step_forward(timestamp)?;
    
    // Execute your strategy
    backtester.execute_trade(
        "ASSET_0",
        TradeSide::Buy,
        100.0, // quantity
        timestamp,
    )?;
}

// Get performance metrics
let metrics = backtester.calculate_metrics()?;
println!("Total Return: {:.2}%", metrics.total_return * 100.0);
println!("Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
```

## Leverage and Margin Trading

The backtesting library supports leverage/margin trading to show the full profit and loss potential of strategies without being limited by cash constraints.

### Enabling Leverage

```rust
// Enable leverage (allow negative cash)
let mut backtester = Backtester::new_with_leverage(
    initial_capital,
    returns_data,
    symbol_names,
    Some(fees),
    true, // Allow negative cash
)?;

// Disable leverage (traditional cash-only trading)
let mut backtester = Backtester::new_with_leverage(
    initial_capital,
    returns_data,
    symbol_names,
    Some(fees),
    false, // Require positive cash
)?;

// Default constructor enables leverage
let mut backtester = Backtester::new(/* ... */)?; // leverage = true
```

### Leverage Metrics

When leverage is enabled, additional metrics are tracked:

- **Max Leverage**: Highest leverage ratio achieved (position value / portfolio value)
- **Max Margin Used**: Maximum amount of borrowed money (negative cash)
- **Min Cash Balance**: Lowest cash balance reached (can be negative)
- **Current Leverage**: Real-time leverage ratio in portfolio state

### Benefits of Leverage Mode

1. **See Full Strategy Potential**: No artificial stops due to cash constraints
2. **Realistic P&L**: Shows what you would win or lose with sufficient capital
3. **Risk Assessment**: Understand maximum leverage requirements
4. **Strategy Comparison**: Compare strategies without capital limitations

### Example Output with Leverage

```
📊 LEVERAGE & MARGIN METRICS
Max Leverage Used: 3.45x
Max Margin Used: $45,230.50
Min Cash Balance: -$45,230.50
  ⚠️  Strategy went into margin (negative cash)

💼 FINAL PORTFOLIO
Cash: -$12,450.30
  ⚠️  NEGATIVE CASH (Margin used: $12,450.30)
Total value: $87,549.70
Current leverage: 1.14x
```

### Strategy Implementation

See `examples/simple_strategy.rs` for a complete momentum strategy example:

```rust
// Example strategy structure
struct SimpleMomentumStrategy {
    lookback_period: usize,
    position_size: f64,
    rebalance_frequency: usize,
}

impl SimpleMomentumStrategy {
    fn execute_step(&self, backtester: &mut Backtester, timestamp: usize) -> Result<()> {
        // 1. Calculate signals based on historical returns
        // 2. Close existing positions
        // 3. Open new positions based on signals
        // 4. Apply position sizing and risk management
    }
}
```

## API Reference

### Core Types

#### `Backtester`
Main backtesting engine that manages portfolio state and executes trades.

**Key Methods:**
- `new()`: Initialize with capital, data, and fees
- `step_forward()`: Advance time and update prices
- `execute_trade()`: Execute buy/sell orders
- `calculate_metrics()`: Get performance statistics
- `get_current_portfolio()`: Access current portfolio state

#### `TradingFees`
Configuration for trading costs.
```rust
TradingFees {
    maker_fee: 0.001,  // 0.1% for providing liquidity
    taker_fee: 0.001,  // 0.1% for taking liquidity
}
```

#### `PerformanceMetrics`
Comprehensive performance statistics:
- `total_return`: Overall return percentage
- `annualized_return`: Yearly return percentage
- `sharpe_ratio`: Risk-adjusted return metric
- `max_drawdown`: Largest peak-to-trough decline
- `win_rate`: Percentage of profitable trades
- `profit_factor`: Gross profit / gross loss ratio

### Trading Operations

#### Execute Trades
```rust
backtester.execute_trade(
    "BTCUSDT",           // symbol
    TradeSide::Buy,      // Buy or Sell
    1.5,                 // quantity
    timestamp,           // current time
)?;
```

#### Portfolio Access
```rust
let portfolio = backtester.get_current_portfolio();
println!("Cash: ${:.2}", portfolio.cash);
println!("Total Value: ${:.2}", portfolio.total_value);
println!("Positions: {}", portfolio.positions.len());
```

## Performance Considerations

### Memory Usage
- The library stores full portfolio history for metrics calculation
- For very long backtests, consider periodic checkpointing
- Returns data is kept in GPU/CPU memory as Candle tensors

### Speed Optimization
- Use batch operations when possible
- Minimize frequent portfolio queries
- Consider reducing rebalancing frequency for faster execution

### Accuracy Notes
- Prices are reconstructed from returns starting at 100.0 for all assets
- This doesn't affect relative performance calculations
- Absolute price levels are not meaningful, only returns matter

## Example Results

Running the simple momentum strategy on cryptocurrency data:

```
📊 PERFORMANCE RESULTS
====================================
Initial Capital: $100,000.00
Final Portfolio Value: $125,430.50
Total Return: 25.43%
Annualized Return: 12.15%
Sharpe Ratio: 1.245
Maximum Drawdown: 8.32%
Max Drawdown Duration: 1,440 periods
Win Rate: 58.3%
Profit Factor: 1.67
Total Trades: 2,847
Total Fees: $1,234.56
```

## Running the Example

```bash
# Compile and run the momentum strategy example
cargo run --example simple_strategy

# Or build and run separately
cargo build --example simple_strategy
./target/debug/examples/simple_strategy
```

## Integration with ML Models

The backtesting library is designed to work seamlessly with the BERT-based time series models:

1. **Model Predictions**: Use model outputs as trading signals
2. **Feature Engineering**: Incorporate model confidence scores
3. **Risk Management**: Combine predictions with portfolio constraints
4. **Performance Attribution**: Analyze model performance across different market conditions

## Contributing

When adding new features:
1. Maintain compatibility with the existing data format
2. Add comprehensive tests for new functionality
3. Update performance metrics calculations if needed
4. Document new parameters and their effects

## License

This backtesting library is part of the candle-bert-time-series project.
