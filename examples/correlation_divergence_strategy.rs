use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::{extract_validation_split, extract_test_split, Backtester, TradingFees, TradeSide};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::{HashMap, VecDeque};
use std::io::Write;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

/// Z-Score Reversion Strategy
///
/// This strategy:
/// 1. Trades a single specified cryptocurrency index
/// 2. Masks the target crypto and predicts its values using other cryptos
/// 3. Calculates Z-score of (Predicted - Actual) divergence
/// 4. Uses Z-score thresholds for entry/exit signals:
///    - Long Entry: Z-score > +2.0 (undervalued, expect reversion up)
///    - Take Profit: Z-score returns to 0 (reversion complete)
///    - Stop Loss: Z-score < -2.0 (signal failed, cut losses)
///    - Negative Correlation: Invert all rules
/// 5. No short positions allowed

const SEQUENCE_LENGTH: usize = 240;
const MODEL_DIMS: usize = 384;
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const ZSCORE_WINDOW: usize = 100; // Rolling window for Z-score calculation

/// Trading position state
#[derive(Debug, Clone)]
enum PositionState {
    NoPosition,
    Long { entry_price: f64, entry_zscore: f64 },
}

/// Z-Score Reversion Strategy Implementation
struct ZScoreReversionStrategy {
    model: FinancialTransformerForMaskedRegression,
    device: Device,
    num_assets: usize,
    target_crypto_idx: usize,
    correlation: f64,
    is_positive_correlation: bool,
    divergence_history: VecDeque<f64>,
    position_state: PositionState,
    position_size: f64,
    long_entry_threshold: f64,
    take_profit_threshold: f64,
    stop_loss_threshold: f64,
}

impl ZScoreReversionStrategy {
    fn new(
        model: FinancialTransformerForMaskedRegression,
        device: Device,
        num_assets: usize,
        target_crypto_idx: usize,
        position_size: f64,
    ) -> Self {
        Self {
            model,
            device,
            num_assets,
            target_crypto_idx,
            correlation: 0.0,
            is_positive_correlation: true,
            divergence_history: VecDeque::with_capacity(ZSCORE_WINDOW),
            position_state: PositionState::NoPosition,
            position_size,
            long_entry_threshold: 2.0,
            take_profit_threshold: 0.0,
            stop_loss_threshold: -2.0,
        }
    }

    /// Create masked input for the target cryptocurrency
    fn create_masked_input(&self, data: &Tensor, timestamp: usize) -> Result<Tensor> {
        if timestamp < SEQUENCE_LENGTH {
            return Err(candle_core::Error::Msg("Not enough history for sequence".to_string()));
        }

        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let mut masked_sequence = input_sequence.clone();

        // Zero out the target cryptocurrency for the entire sequence
        let zeros = Tensor::zeros((SEQUENCE_LENGTH, 1), DType::F32, &self.device)?;

        // Get the parts of the sequence
        let before_cols = if self.target_crypto_idx > 0 {
            Some(masked_sequence.narrow(1, 0, self.target_crypto_idx)?)
        } else {
            None
        };
        let after_cols = if self.target_crypto_idx + 1 < masked_sequence.dims()[1] {
            Some(masked_sequence.narrow(1, self.target_crypto_idx + 1,
                                      masked_sequence.dims()[1] - self.target_crypto_idx - 1)?)
        } else {
            None
        };

        // Reconstruct the sequence with zeros in the target column
        masked_sequence = match (before_cols, after_cols) {
            (Some(before), Some(after)) => Tensor::cat(&[&before, &zeros, &after], 1)?,
            (Some(before), None) => Tensor::cat(&[&before, &zeros], 1)?,
            (None, Some(after)) => Tensor::cat(&[&zeros, &after], 1)?,
            (None, None) => zeros,
        };

        Ok(masked_sequence)
    }

    /// Get prediction for the target crypto
    fn get_prediction(&self, data: &Tensor, timestamp: usize) -> Result<f64> {
        let masked_input = self.create_masked_input(data, timestamp)?;
        let input_batch = masked_input.unsqueeze(0)?; // Add batch dimension

        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;

        // Extract prediction for the last timestep of the target crypto
        let last_timestep_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;

        Ok(predictions_vec[self.target_crypto_idx] as f64)
    }

    /// Calculate correlation between predictions and actuals for the target crypto
    fn calculate_correlation(&self, val_data: &Tensor) -> Result<f64> {
        let val_timesteps = val_data.dims()[0];
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();

        // Sample every 10th timestep to reduce computation
        let step_size = 10;
        let start_idx = SEQUENCE_LENGTH;
        let end_idx = val_timesteps.min(start_idx + 1000); // Limit to 1000 samples

        for timestamp in (start_idx..end_idx).step_by(step_size) {
            if let Ok(predicted) = self.get_prediction(val_data, timestamp) {
                if let Ok(actual_row) = val_data.get(timestamp - 1) {
                    if let Ok(actual_vec) = actual_row.to_vec1::<f32>() {
                        let actual = actual_vec[self.target_crypto_idx] as f64;
                        predictions.push(predicted);
                        actuals.push(actual);
                    }
                }
            }
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        // Calculate Pearson correlation
        let n = predictions.len() as f64;
        let sum_pred: f64 = predictions.iter().sum();
        let sum_actual: f64 = actuals.iter().sum();
        let sum_pred_actual: f64 = predictions.iter().zip(actuals.iter()).map(|(p, a)| p * a).sum();
        let sum_pred2: f64 = predictions.iter().map(|p| p * p).sum();
        let sum_actual2: f64 = actuals.iter().map(|a| a * a).sum();

        let numerator = n * sum_pred_actual - sum_pred * sum_actual;
        let denominator = ((n * sum_pred2 - sum_pred * sum_pred) * (n * sum_actual2 - sum_actual * sum_actual)).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate current divergence (Predicted - Actual)
    fn calculate_divergence(&self, data: &Tensor, timestamp: usize) -> Result<f64> {
        // Get prediction for the target crypto
        let predicted = self.get_prediction(data, timestamp)?;

        // Get actual value for the target crypto
        let actual_row = data.get(timestamp - 1)?;
        let actual_vec: Vec<f32> = actual_row.to_vec1()?;
        let actual = actual_vec[self.target_crypto_idx] as f64;

        Ok(predicted - actual)
    }

    /// Update divergence history and calculate Z-score
    fn update_and_calculate_zscore(&mut self, divergence: f64) -> f64 {
        // Add new divergence to history
        self.divergence_history.push_back(divergence);

        // Keep only the last ZSCORE_WINDOW values
        if self.divergence_history.len() > ZSCORE_WINDOW {
            self.divergence_history.pop_front();
        }

        // Need at least 30 samples to calculate meaningful Z-score
        if self.divergence_history.len() < 30 {
            return 0.0;
        }

        // Calculate mean and standard deviation
        let mean: f64 = self.divergence_history.iter().sum::<f64>() / self.divergence_history.len() as f64;
        let variance: f64 = self.divergence_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.divergence_history.len() as f64;
        let std_dev = variance.sqrt();

        // Avoid division by zero
        if std_dev == 0.0 {
            return 0.0;
        }

        // Calculate Z-score
        (divergence - mean) / std_dev
    }

    /// Analyze target crypto correlation on validation set
    fn analyze_target_crypto(&mut self, val_data: &Tensor) -> Result<()> {
        println!("\n🔍 ANALYZING TARGET CRYPTO_{} ON VALIDATION SET", self.target_crypto_idx);
        println!("======================================================================");

        let start_time = std::time::Instant::now();
        print!("  Calculating correlation for CRYPTO_{}... ", self.target_crypto_idx);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        self.correlation = self.calculate_correlation(val_data)?;
        self.is_positive_correlation = self.correlation > 0.0;

        let total_time = start_time.elapsed().as_secs();
        println!("Correlation: {:.6} ({}) - completed in {}s",
                 self.correlation,
                 if self.is_positive_correlation { "Positive" } else { "Negative" },
                 total_time);

        // Adjust thresholds based on correlation sign
        if !self.is_positive_correlation {
            println!("  🔄 Negative correlation detected - inverting all trading rules");
            // For negative correlation, invert the thresholds
            let temp = self.long_entry_threshold;
            self.long_entry_threshold = -self.stop_loss_threshold;
            self.stop_loss_threshold = -temp;
        }

        println!("\n📊 TRADING RULES FOR CRYPTO_{}:", self.target_crypto_idx);
        println!("  - Correlation: {:.4} ({})", self.correlation,
                 if self.is_positive_correlation { "Positive" } else { "Negative" });
        println!("  - Long Entry: Z-score > {:.1}", self.long_entry_threshold);
        println!("  - Take Profit: Z-score returns to {:.1}", self.take_profit_threshold);
        println!("  - Stop Loss: Z-score < {:.1}", self.stop_loss_threshold);

        Ok(())
    }

    /// Generate trading signals based on Z-score thresholds
    fn generate_signal(&mut self, data: &Tensor, timestamp: usize, current_price: f64) -> Result<Option<(TradeSide, f64)>> {
        if timestamp < SEQUENCE_LENGTH {
            return Ok(None);
        }

        // Calculate current divergence and Z-score
        let divergence = self.calculate_divergence(data, timestamp)?;
        let zscore = self.update_and_calculate_zscore(divergence);

        // Skip if Z-score calculation is not ready
        if self.divergence_history.len() < 30 {
            return Ok(None);
        }

        match &self.position_state {
            PositionState::NoPosition => {
                // Check for long entry signal
                if zscore > self.long_entry_threshold {
                    self.position_state = PositionState::Long {
                        entry_price: current_price,
                        entry_zscore: zscore,
                    };
                    return Ok(Some((TradeSide::Buy, self.position_size)));
                }
            },
            PositionState::Long { entry_price: _, entry_zscore: _ } => {
                // Check for take profit (Z-score returns to 0)
                if (zscore - self.take_profit_threshold).abs() < 0.5 {
                    self.position_state = PositionState::NoPosition;
                    return Ok(Some((TradeSide::Sell, 1.0))); // Sell entire position
                }
                // Check for stop loss
                else if zscore < self.stop_loss_threshold {
                    self.position_state = PositionState::NoPosition;
                    return Ok(Some((TradeSide::Sell, 1.0))); // Sell entire position
                }
            }
        }

        Ok(None)
    }

}

fn main() -> Result<()> {
    println!("📊 Z-SCORE REVERSION BACKTEST STRATEGY");
    println!("======================================================================");
    println!("This strategy:");
    println!("1. Trades a single specified cryptocurrency index");
    println!("2. Masks the target crypto and predicts its values using other cryptos");
    println!("3. Calculates Z-score of (Predicted - Actual) divergence");
    println!("4. Uses Z-score thresholds for entry/exit signals:");
    println!("   - Long Entry: Z-score > +2.0 (undervalued, expect reversion up)");
    println!("   - Take Profit: Z-score returns to 0 (reversion complete)");
    println!("   - Stop Loss: Z-score < -2.0 (signal failed, cut losses)");
    println!("   - Negative Correlation: Invert all rules");
    println!("5. No short positions allowed");
    println!("======================================================================");

    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model_large_r2_ep1.safetensors";
    let initial_capital = 100.0;
    let target_crypto_idx = 5; // Specify which crypto to trade (change this as needed)
    let position_size = 0.1; // 10% position size

    // Load data
    println!("\nLoading cryptocurrency data...");
    let (full_data_sequence, num_time_series) = load_and_prepare_data(data_path, &device)?;

    // Validate target crypto index
    if target_crypto_idx >= num_time_series {
        return Err(candle_core::Error::Msg(format!(
            "Target crypto index {} is out of range (max: {})",
            target_crypto_idx, num_time_series - 1
        )));
    }

    // Extract validation and test splits
    let val_data = extract_validation_split(&full_data_sequence)?;
    let test_data = extract_test_split(&full_data_sequence)?;

    let val_timesteps = val_data.dims()[0];
    let test_timesteps = test_data.dims()[0];

    println!("Data loaded: {} assets", num_time_series);
    println!("Target crypto: CRYPTO_{}", target_crypto_idx);
    println!("Validation set: {} timesteps", val_timesteps);
    println!("Test set: {} timesteps", test_timesteps);

    // Load trained model
    println!("\n🤖 Loading trained model...");
    let config = Config {
        num_time_series,
        hidden_size: MODEL_DIMS,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        intermediate_size: MODEL_DIMS * 4,
        hidden_act: financial_bert::HiddenAct::Gelu,
        hidden_dropout_prob: 0.1,
        max_position_embeddings: SEQUENCE_LENGTH,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        position_embedding_type: financial_bert::PositionEmbeddingType::Absolute,
        use_cache: false,
        model_type: Some("financial_transformer".to_string()),
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FinancialTransformerForMaskedRegression::load(vb, &config)?;
    varmap.load(model_path)?;
    println!("✅ Model loaded from: {}", model_path);

    // Initialize strategy
    let mut strategy = ZScoreReversionStrategy::new(
        model,
        device.clone(),
        num_time_series,
        target_crypto_idx,
        position_size,
    );

    // Analyze target crypto correlation on validation set
    strategy.analyze_target_crypto(&val_data)?;

    // Initialize backtester for test set
    println!("\n💰 RUNNING BACKTEST ON TEST SET");
    println!("======================================================================");

    let symbol_names: Vec<String> = (0..num_time_series)
        .map(|i| format!("CRYPTO_{}", i))
        .collect();

    let fees = TradingFees::default();
    let mut backtester = Backtester::new(
        initial_capital,
        test_data.clone(),
        symbol_names.clone(),
        Some(fees),
    )?;

    println!("🚀 Running Z-score reversion backtest...");
    println!("  - Target crypto: CRYPTO_{}", target_crypto_idx);
    println!("  - Correlation: {:.4} ({})", strategy.correlation,
             if strategy.is_positive_correlation { "Positive" } else { "Negative" });
    println!("  - Position size: {:.1}%", position_size * 100.0);
    println!("  - Test period: {} timesteps", test_timesteps);

    let mut total_trades = 0;
    let mut buy_trades = 0;
    let mut sell_trades = 0;
    let mut last_portfolio_value = initial_capital;
    let target_symbol = format!("CRYPTO_{}", target_crypto_idx);

    println!("\n🚀 Starting backtest execution...");
    println!("Initial portfolio value: ${:.2}", initial_capital);

    // Run backtest
    for timestamp in SEQUENCE_LENGTH..test_timesteps {
        // Step forward to update prices
        backtester.step_forward(timestamp)?;

        let current_portfolio_value = backtester.portfolio_history.last().unwrap().total_value;
        let current_price = backtester.current_prices[target_crypto_idx];

        // Generate trading signal for target crypto
        if let Ok(Some((side, size))) = strategy.generate_signal(&test_data, timestamp, current_price) {
            // Calculate position size in shares
            let position_value = match side {
                TradeSide::Buy => current_portfolio_value * size,
                TradeSide::Sell => {
                    // For sell, size represents the fraction of position to sell
                    let current_portfolio = backtester.get_current_portfolio();
                    if let Some(position) = current_portfolio.positions.get(&target_symbol) {
                        position.quantity * size
                    } else {
                        0.0
                    }
                }
            };

            let shares = match side {
                TradeSide::Buy => position_value / current_price,
                TradeSide::Sell => position_value,
            };

            // Execute trade
            if shares > 0.0 {
                if let Ok(_) = backtester.execute_trade(&target_symbol, side.clone(), shares, timestamp) {
                    total_trades += 1;

                    // Get current divergence and Z-score for display
                    let divergence = strategy.calculate_divergence(&test_data, timestamp).unwrap_or(0.0);
                    let zscore = if strategy.divergence_history.len() >= 30 {
                        let mean: f64 = strategy.divergence_history.iter().sum::<f64>() / strategy.divergence_history.len() as f64;
                        let variance: f64 = strategy.divergence_history.iter()
                            .map(|x| (x - mean).powi(2))
                            .sum::<f64>() / strategy.divergence_history.len() as f64;
                        let std_dev = variance.sqrt();
                        if std_dev > 0.0 { (divergence - mean) / std_dev } else { 0.0 }
                    } else { 0.0 };

                    println!("  T{}: {} {} shares of {} (${:.4}/share, Div: {:.4}, Z: {:.2})",
                             timestamp,
                             match side { TradeSide::Buy => "BUY", TradeSide::Sell => "SELL" },
                             shares.round() as i32,
                             target_symbol,
                             current_price,
                             divergence,
                             zscore);

                    match side {
                        TradeSide::Buy => buy_trades += 1,
                        TradeSide::Sell => sell_trades += 1,
                    }
                }
            }
        }

        // Print detailed progress every 500 timesteps
        if timestamp % 500 == 0 {
            let return_pct = (current_portfolio_value - initial_capital) / initial_capital * 100.0;
            let value_change = current_portfolio_value - last_portfolio_value;
            let current_portfolio = backtester.get_current_portfolio();
            let position_info = if let Some(position) = current_portfolio.positions.get(&target_symbol) {
                format!("{:.1} shares (${:.2})", position.quantity, position.current_value)
            } else {
                "No position".to_string()
            };

            println!("  📊 T{}: Portfolio ${:.2} ({:+.2}%, {:+.2} change), Position: {}, {} trades (B:{} S:{})",
                     timestamp, current_portfolio_value, return_pct, value_change,
                     position_info, total_trades, buy_trades, sell_trades);

            // Show position state
            match &strategy.position_state {
                PositionState::NoPosition => println!("    Position State: No Position"),
                PositionState::Long { entry_price, entry_zscore } => {
                    let unrealized_pnl = if let Some(position) = current_portfolio.positions.get(&target_symbol) {
                        (current_price - entry_price) * position.quantity
                    } else { 0.0 };
                    println!("    Position State: Long (Entry: ${:.4}, Z: {:.2}, PnL: ${:.2})",
                             entry_price, entry_zscore, unrealized_pnl);
                }
            }

            last_portfolio_value = current_portfolio_value;
        }
    }

    // Calculate final performance metrics
    let metrics = backtester.calculate_metrics()?;

    println!("\n📈 Z-SCORE REVERSION STRATEGY RESULTS");
    println!("======================================================================");
    println!("💰 Financial Performance:");
    println!("  - Initial capital: ${:.2}", initial_capital);
    println!("  - Final value: ${:.2}", metrics.final_portfolio_value);
    println!("  - Total return: {:.2}%", metrics.total_return * 100.0);
    println!("  - Sharpe ratio: {:.3}", metrics.sharpe_ratio);
    println!("  - Max drawdown: {:.2}%", metrics.max_drawdown * 100.0);

    println!("\n📊 Trading Statistics:");
    println!("  - Total trades: {} (Buy: {}, Sell: {})", total_trades, buy_trades, sell_trades);
    println!("  - Total fees paid: ${:.2}", metrics.total_fees);

    println!("\n🎯 Target Crypto Analysis:");
    println!("  - Target: CRYPTO_{}", target_crypto_idx);
    println!("  - Model correlation: {:.4} ({})", strategy.correlation,
             if strategy.is_positive_correlation { "Positive" } else { "Negative" });
    println!("  - Z-score window: {} samples", ZSCORE_WINDOW);
    println!("  - Long entry threshold: {:.1}", strategy.long_entry_threshold);
    println!("  - Take profit threshold: {:.1}", strategy.take_profit_threshold);
    println!("  - Stop loss threshold: {:.1}", strategy.stop_loss_threshold);

    // Show final position state
    match &strategy.position_state {
        PositionState::NoPosition => println!("  - Final position: No Position"),
        PositionState::Long { entry_price, entry_zscore } => {
            println!("  - Final position: Long (Entry: ${:.4}, Entry Z-score: {:.2})",
                     entry_price, entry_zscore);
        }
    }

    println!("\n✅ Z-score reversion backtest complete!");

    Ok(())
}
