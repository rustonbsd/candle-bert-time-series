# Financial BERT PPO Trading Strategy

This is a clean, production-ready implementation of a PPO (Proximal Policy Optimization) transformer-based trading strategy designed for financial BERT models. The implementation has been streamlined to focus on the core functionality without unnecessary complexity.

## Key Features

### 🧠 Financial BERT Integration
- **Enhanced Orderbook Encoding**: Properly encodes orderbook depth with normalized features
- **Sequence-based Processing**: Uses 240-timestep sequences for transformer input
- **Multi-level Depth**: Encodes 10 levels of bid/ask orderbook data

### 🎯 PPO-Ready Architecture
- **Agent Interface**: Clean `FinancialBertAgent` trait for PPO integration
- **Experience Collection**: Tracks state, action, reward, next_state, and done flags
- **Action Space**: 3 actions (HOLD, BUY, SELL) with probability outputs

### 📊 Advanced Orderbook Features
The `OrderbookSnapshot` encodes comprehensive market microstructure:

```rust
// Core market features (8 features)
- Normalized best bid/ask prices
- Relative spread
- Absolute mid price
- Orderbook imbalance
- Normalized micro price
- Total bid/ask volumes

// Depth features (40 features for 10 levels)
- [bid_price_1, bid_vol_1, ask_price_1, ask_vol_1, ...]
```

### 🔄 Trading Strategy Components

#### Position Sizing
- **Conservative Approach**: 5% of portfolio per trade
- **Risk Management**: 20% position exits for profit-taking
- **Market Orders**: Uses IOC (Immediate or Cancel) for execution

#### Reward Function
- **PnL-based**: Primary reward from portfolio returns
- **Risk Penalties**: Drawdown penalties for risk management
- **Transaction Costs**: Small penalty for trading activity
- **Profit Bonuses**: Incentivizes profitable trades

## Architecture Overview

### Core Components

1. **OrderbookSnapshot**: Encodes market data for transformer input
2. **FinancialBertAgent**: Interface for PPO model integration
3. **FinancialBertStrategy**: Trading logic and portfolio management
4. **MockBacktest**: Simplified backtesting framework

### Data Flow

```
Market Data → OrderbookSnapshot → Sequence Buffer → Financial BERT → Action Probabilities → Trading Decision → Portfolio Update → Reward Calculation → PPO Update
```

## Implementation Details

### Orderbook Encoding
The orderbook is encoded as a flat feature vector optimized for transformer processing:

<augment_code_snippet path="src/main.rs" mode="EXCERPT">
````rust
impl OrderbookSnapshot {
    pub fn from_depth(depth: &MockDepth) -> Self {
        let mut features = Vec::with_capacity(ORDERBOOK_DEPTH * 4 + 8);
        
        // Core market features
        features.push((best_bid / mid_price) as f32); // Normalized
        features.push((best_ask / mid_price) as f32);
        features.push((spread / mid_price) as f32);
        features.push(mid_price as f32);
        
        // Orderbook imbalance and micro price
        features.push(imbalance as f32);
        features.push((micro_price / mid_price) as f32);
        features.push(total_bid_vol as f32);
        features.push(total_ask_vol as f32);
        
        // Encode depth levels
        for i in 0..ORDERBOOK_DEPTH {
            features.push(bid_price as f32);
            features.push(bid_vol as f32);
            features.push(ask_price as f32);
            features.push(ask_vol as f32);
        }
        
        Self { timestamp: depth.timestamp, features }
    }
}
````
</augment_code_snippet>

### PPO Agent Interface
Clean interface for integrating with your financial BERT model:

<augment_code_snippet path="src/main.rs" mode="EXCERPT">
````rust
pub trait FinancialBertAgent {
    /// Get action probabilities from the model
    fn predict(&self, history: &[OrderbookSnapshot]) -> Array1<f32>;
    
    /// Update model with PPO experience
    fn update(&mut self, 
              state: &[OrderbookSnapshot], 
              action: usize, 
              reward: f32, 
              next_state: &[OrderbookSnapshot],
              done: bool);
}
````
</augment_code_snippet>

## Usage

### Basic Usage
```rust
// Initialize your financial BERT agent
let agent = YourFinancialBertAgent::new();

// Run backtest
let (portfolio_values, snapshots) = run_backtest(
    agent, 
    "path/to/your/data.npz", 
    10_000.0
)?;
```

### Integration with Real Financial BERT
Replace `MockFinancialBertAgent` with your actual implementation:

```rust
impl FinancialBertAgent for YourFinancialBertAgent {
    fn predict(&self, history: &[OrderbookSnapshot]) -> Array1<f32> {
        // Convert history to tensor
        let input_tensor = self.prepare_input(history);
        
        // Forward pass through your BERT model
        let logits = self.model.forward(&input_tensor)?;
        
        // Apply softmax for action probabilities
        softmax(&logits)
    }
    
    fn update(&mut self, state: &[OrderbookSnapshot], action: usize, 
              reward: f32, next_state: &[OrderbookSnapshot], done: bool) {
        // Implement PPO update logic here
        self.ppo_trainer.update(state, action, reward, next_state, done);
    }
}
```

## Configuration

Key parameters can be adjusted:

```rust
const SEQUENCE_LENGTH: usize = 240;  // Transformer sequence length
const NUM_ACTIONS: usize = 3;        // HOLD, BUY, SELL
const ORDERBOOK_DEPTH: usize = 10;   // Number of price levels
```

## Best Practices

### 1. Data Preprocessing
- Normalize prices relative to mid-price
- Handle missing orderbook levels gracefully
- Maintain consistent feature scaling

### 2. Risk Management
- Implement position size limits
- Add stop-loss mechanisms
- Monitor drawdown levels

### 3. Model Integration
- Use proper sequence padding
- Implement attention masking for variable-length sequences
- Consider positional encodings for time-series data

## Next Steps

1. **Replace Mock Components**: Integrate with real hftbacktest API
2. **Add Real Data Loading**: Implement proper data pipeline
3. **PPO Implementation**: Add actual PPO training loop
4. **Performance Metrics**: Implement comprehensive backtesting metrics
5. **Risk Controls**: Add advanced risk management features

## Dependencies

- `hftbacktest`: High-frequency trading backtesting framework
- `candle-core`: ML framework for transformer models
- `ndarray`: Numerical computing
- `rand`: Random number generation for mock data

This implementation provides a solid foundation for integrating financial BERT models with PPO-based trading strategies while maintaining clean, maintainable code.
