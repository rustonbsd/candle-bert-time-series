# FinancialBERT for Orderbook Data

This implementation trains a BERT-style transformer on normalized orderbook data from cryptocurrency exchanges. The model learns to predict masked orderbook levels using self-attention mechanisms.

## Overview

The system is designed to work with high-frequency orderbook data (200ms intervals) and uses specialized masking strategies that respect the structure of financial data.

## Architecture

- **Model**: FinancialBERT transformer with configurable layers and attention heads
- **Input**: Normalized orderbook features (bid/ask prices and quantities)
- **Masking**: Orderbook-level masking (masks entire bid/ask pairs)
- **Training**: Self-supervised learning with MSE loss on masked positions

## Data Format

The system expects parquet files with the following structure:

```
timestamp (optional) | bid_price_0 | bid_qty_0 | ask_price_0 | ask_qty_0 | ... | bid_price_19 | bid_qty_19 | ask_price_19 | ask_qty_19
```

- **80 features total**: 20 orderbook levels × 4 features per level
- **Normalized data**: Z-score normalization (μ=0, σ=1)
- **200ms intervals**: 5 ticks per second
- **20 levels**: Top 20 bid/ask levels from the orderbook

## Configuration

Key parameters in `main_orderbook_financialbert.rs`:

```rust
const SEQUENCE_LENGTH: usize = 240;     // 48 seconds at 200ms intervals
const MODEL_DIMS: usize = 128;          // Hidden size
const NUM_LAYERS: usize = 8;            // Transformer layers
const NUM_HEADS: usize = 8;             // Attention heads
const BATCH_SIZE: usize = 256;          // Batch size
const LEARNING_RATE: f64 = 3e-6;        // Learning rate
const ORDERBOOK_MASK_PROB: f32 = 0.15;  // Masking probability
```

## Usage

### 1. Prepare Data

Ensure your orderbook data is in the expected format. You can use the Bybit orderbook downloader:

```bash
cargo run --example download_bybit_orderbook
```

### 2. Update Data Paths

Edit the `DATA_PATHS` constant in `main_orderbook_financialbert.rs`:

```rust
const DATA_PATHS: &[&str] = &[
    "/path/to/your/orderbook_data_day1.parquet",
    "/path/to/your/orderbook_data_day2.parquet",
    // Add more files...
];
```

### 3. Run Training

```bash
# Full training
cargo run --bin main_orderbook_financialbert

# Quick test (2 epochs only)
# Set TEST_MODE = true in the code first
cargo run --bin main_orderbook_financialbert
```

### 4. Monitor Progress

The system will:
- Validate data files before training
- Print data statistics and structure information
- Save checkpoints after each epoch to `training_saves_orderbook/`
- Display training and validation losses

## Masking Strategies

### 1. Orderbook-Level Masking (Default)
Masks entire orderbook levels (bid_price, bid_qty, ask_price, ask_qty together):
- Preserves orderbook structure
- More realistic for financial data
- Better representation learning

### 2. Random Masking
Randomly masks individual features:
- Standard BERT-style masking
- Available but not used by default

### 3. Temporal Masking
Masks the most recent timesteps:
- Useful for forecasting tasks
- Available but not used by default

## Model Architecture

```
Input: [batch_size, sequence_length, 80]
  ↓
FinancialEmbeddings (Linear projection + positional encoding)
  ↓
BertEncoder (8 layers × 8 attention heads)
  ↓
RegressionHead (Transform + Linear decoder)
  ↓
Output: [batch_size, sequence_length, 80]
```

## Files

- `main_orderbook_financialbert.rs`: Main training script
- `orderbook_dataset.rs`: Data loading and preprocessing utilities
- `src/financial_bert.rs`: FinancialBERT model implementation
- `src/batcher.rs`: Batching utilities for sequence data

## Future Enhancements

This implementation provides a solid foundation for:

1. **PPO Integration**: Add reinforcement learning for trading strategies
2. **Multi-Asset Training**: Train on multiple cryptocurrency pairs
3. **Real-time Inference**: Deploy for live trading decisions
4. **Advanced Masking**: Implement market-aware masking strategies
5. **Transfer Learning**: Pre-train on one asset, fine-tune on others

## Notes

- The model uses CUDA if available, falls back to CPU
- Checkpoints are saved after each epoch for resuming training
- Data validation helps catch format issues early
- The system handles missing files gracefully
- Memory usage scales with batch size and sequence length

## Troubleshooting

1. **Out of Memory**: Reduce `BATCH_SIZE` or `SEQUENCE_LENGTH`
2. **Data Format Errors**: Use the inspection utilities to check your data
3. **Slow Training**: Ensure CUDA is available and working
4. **Poor Performance**: Check data normalization and masking strategy

For more details, see the inline documentation in the source files.
