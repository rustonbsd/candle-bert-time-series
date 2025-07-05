use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::{extract_validation_split, extract_test_split, Backtester, TradingFees, TradeSide};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::VecDeque;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression, PositionEmbeddingType};

const SEQUENCE_LENGTH: usize = 240;
const MODEL_DIMS: usize = 384;
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const ZSCORE_WINDOW: usize = 1000;

/// Enhanced divergence calculation methods
#[derive(Debug, Clone)]
pub enum DivergenceMethod {
    /// Original: avg(prediction_path) - actual_price_at_end
    PathAverage,
    /// Path-based: MAE between predicted and actual paths over masked window
    PathMAE,
    /// End-to-end: predict only next timestep (t+1)
    NextTimestep,
}

/// Position state with enhanced tracking
#[derive(Debug, Clone)]
enum PositionState {
    NoPosition,
    Long { 
        entry_price: f64, 
        entry_zscore: f64, 
        entry_timestamp: usize,
        quantity: f64,
        trailing_stop_price: Option<f64>,
        partial_profit_taken: bool,
    },
}

/// Enhanced Z-Score Reversion Strategy with improved divergence signals and risk management
struct EnhancedDivergenceStrategy {
    model: FinancialTransformerForMaskedRegression,
    device: Device,
    num_assets: usize,
    target_crypto_idx: usize,
    correlation: f64,
    is_positive_correlation: bool,
    position_state: PositionState,
    position_size: f64,
    divergence_history: VecDeque<f64>,
    
    // Enhanced parameters
    divergence_method: DivergenceMethod,
    long_entry_threshold: f64,
    take_profit_threshold: f64,
    stop_loss_threshold: f64,
    price_stop_loss_pct: f64,      // Price-based stop loss percentage
    partial_profit_pct: f64,       // Percentage to take as partial profit
    trailing_stop_pct: f64,        // Trailing stop percentage
    min_holding_period: usize,     // Minimum holding period in timesteps
}

impl EnhancedDivergenceStrategy {
    fn new(
        model: FinancialTransformerForMaskedRegression,
        device: Device,
        num_assets: usize,
        target_crypto_idx: usize,
        position_size: f64,
        divergence_method: DivergenceMethod,
    ) -> Self {
        Self {
            model,
            device,
            num_assets,
            target_crypto_idx,
            correlation: 0.0,
            is_positive_correlation: true,
            position_state: PositionState::NoPosition,
            position_size,
            divergence_history: VecDeque::new(),
            divergence_method,
            
            // Conservative thresholds
            long_entry_threshold: 2.0,     // Higher threshold for entry
            take_profit_threshold: -1.5,   // Exit when Z-score returns to -1
            stop_loss_threshold: -4.0,     // Stop loss threshold
            price_stop_loss_pct: 0.05,     // 2% price-based stop loss
            partial_profit_pct: 0.2,       // Take 50% profit at Z=0
            trailing_stop_pct: 0.01,       // 1% trailing stop
            min_holding_period: 10,        // Minimum 20 timesteps
        }
    }

    /// Create masked input for prediction
    fn create_masked_input(&self, data: &Tensor, timestamp: usize) -> Result<Tensor> {
        if timestamp < SEQUENCE_LENGTH {
            return Err(candle_core::Error::Msg("Not enough history for sequence".to_string()));
        }

        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let mut masked_sequence = input_sequence.clone();

        match self.divergence_method {
            DivergenceMethod::PathAverage | DivergenceMethod::PathMAE => {
                // Mask the last 75% of the sequence for the target crypto (same as original)
                let quarter_point = SEQUENCE_LENGTH / 4;

                // Create zeros for the masked portion
                let zeros = Tensor::zeros((SEQUENCE_LENGTH - quarter_point, 1), DType::F32, &self.device)?;

                // Get the parts of the sequence
                let first_quarter = masked_sequence.narrow(0, 0, quarter_point)?;
                let last_three_quarters = masked_sequence.narrow(0, quarter_point, SEQUENCE_LENGTH - quarter_point)?;

                // Mask only the target crypto in the last 75%
                let before_cols = if self.target_crypto_idx > 0 {
                    Some(last_three_quarters.narrow(1, 0, self.target_crypto_idx)?)
                } else {
                    None
                };

                let after_cols = if self.target_crypto_idx < self.num_assets - 1 {
                    Some(last_three_quarters.narrow(1, self.target_crypto_idx + 1, self.num_assets - self.target_crypto_idx - 1)?)
                } else {
                    None
                };

                // Reconstruct the last 75% with zeros in the target column
                let masked_last_part = match (before_cols, after_cols) {
                    (Some(before), Some(after)) => Tensor::cat(&[&before, &zeros, &after], 1)?,
                    (Some(before), None) => Tensor::cat(&[&before, &zeros], 1)?,
                    (None, Some(after)) => Tensor::cat(&[&zeros, &after], 1)?,
                    (None, None) => zeros,
                };

                // Combine first quarter (real) with masked last 75%
                masked_sequence = Tensor::cat(&[&first_quarter, &masked_last_part], 0)?;
            },
            DivergenceMethod::NextTimestep => {
                // Only mask the last timestep for next-step prediction
                let zeros = Tensor::zeros((1, 1), DType::F32, &self.device)?;

                // Get all but the last timestep
                let all_but_last = masked_sequence.narrow(0, 0, SEQUENCE_LENGTH - 1)?;
                let last_timestep = masked_sequence.narrow(0, SEQUENCE_LENGTH - 1, 1)?;

                // Mask only the target crypto in the last timestep
                let before_cols = if self.target_crypto_idx > 0 {
                    Some(last_timestep.narrow(1, 0, self.target_crypto_idx)?)
                } else {
                    None
                };

                let after_cols = if self.target_crypto_idx < self.num_assets - 1 {
                    Some(last_timestep.narrow(1, self.target_crypto_idx + 1, self.num_assets - self.target_crypto_idx - 1)?)
                } else {
                    None
                };

                // Reconstruct the last timestep with zero in the target column
                let masked_last_timestep = match (before_cols, after_cols) {
                    (Some(before), Some(after)) => Tensor::cat(&[&before, &zeros, &after], 1)?,
                    (Some(before), None) => Tensor::cat(&[&before, &zeros], 1)?,
                    (None, Some(after)) => Tensor::cat(&[&zeros, &after], 1)?,
                    (None, None) => zeros,
                };

                // Combine all but last with masked last timestep
                masked_sequence = Tensor::cat(&[&all_but_last, &masked_last_timestep], 0)?;
            }
        }

        Ok(masked_sequence)
    }

    /// Enhanced prediction method supporting different divergence calculations
    fn get_prediction(&self, data: &Tensor, timestamp: usize) -> Result<f64> {
        let masked_input = self.create_masked_input(data, timestamp)?;
        let input_batch = masked_input.unsqueeze(0)?;
        
        let predictions = self.model.forward(&input_batch)?;
        
        match self.divergence_method {
            DivergenceMethod::PathAverage => {
                // Original method: average of last 75% predictions
                let quarter_point = SEQUENCE_LENGTH / 4;
                let mut crypto_predictions = Vec::new();
                
                for t in quarter_point..SEQUENCE_LENGTH {
                    let timestep_predictions = predictions.get(0)?.get(t)?;
                    let predictions_vec: Vec<f32> = timestep_predictions.to_vec1()?;
                    crypto_predictions.push(predictions_vec[self.target_crypto_idx] as f64);
                }
                
                Ok(crypto_predictions.iter().sum::<f64>() / crypto_predictions.len() as f64)
            },
            DivergenceMethod::NextTimestep => {
                // End-to-end: predict only next timestep
                let last_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;
                let predictions_vec: Vec<f32> = last_predictions.to_vec1()?;
                Ok(predictions_vec[self.target_crypto_idx] as f64)
            },
            DivergenceMethod::PathMAE => {
                // This will be handled in calculate_divergence for MAE calculation
                // For now, return the average like PathAverage
                let quarter_point = SEQUENCE_LENGTH / 4;
                let mut crypto_predictions = Vec::new();
                
                for t in quarter_point..SEQUENCE_LENGTH {
                    let timestep_predictions = predictions.get(0)?.get(t)?;
                    let predictions_vec: Vec<f32> = timestep_predictions.to_vec1()?;
                    crypto_predictions.push(predictions_vec[self.target_crypto_idx] as f64);
                }
                
                Ok(crypto_predictions.iter().sum::<f64>() / crypto_predictions.len() as f64)
            }
        }
    }

    /// Enhanced divergence calculation with multiple methods
    fn calculate_divergence(&self, data: &Tensor, timestamp: usize) -> Result<f64> {
        match self.divergence_method {
            DivergenceMethod::PathAverage => {
                // Original method
                let predicted = self.get_prediction(data, timestamp)?;
                let actual_row = data.get(timestamp - 1)?;
                let actual_vec: Vec<f32> = actual_row.to_vec1()?;
                let actual = actual_vec[self.target_crypto_idx] as f64;
                Ok(predicted - actual)
            },
            DivergenceMethod::NextTimestep => {
                // End-to-end prediction
                let predicted = self.get_prediction(data, timestamp)?;
                let actual_row = data.get(timestamp - 1)?;
                let actual_vec: Vec<f32> = actual_row.to_vec1()?;
                let actual = actual_vec[self.target_crypto_idx] as f64;
                Ok(predicted - actual)
            },
            DivergenceMethod::PathMAE => {
                // Path-based MAE calculation
                self.calculate_path_mae(data, timestamp)
            }
        }
    }

    /// Calculate Mean Absolute Error between predicted and actual paths
    fn calculate_path_mae(&self, data: &Tensor, timestamp: usize) -> Result<f64> {
        let masked_input = self.create_masked_input(data, timestamp)?;
        let input_batch = masked_input.unsqueeze(0)?;
        let predictions = self.model.forward(&input_batch)?;

        let quarter_point = SEQUENCE_LENGTH / 4;
        let start_idx = timestamp.saturating_sub(SEQUENCE_LENGTH);

        let mut mae_sum = 0.0;
        let mut count = 0;

        // Calculate MAE over the masked window (last 75%)
        for t in quarter_point..SEQUENCE_LENGTH {
            let actual_timestamp = start_idx + t;
            if actual_timestamp < data.dims()[0] {
                let predicted_timestep = predictions.get(0)?.get(t)?;
                let predicted_vec: Vec<f32> = predicted_timestep.to_vec1()?;
                let predicted_value = predicted_vec[self.target_crypto_idx] as f64;

                let actual_row = data.get(actual_timestamp)?;
                let actual_vec: Vec<f32> = actual_row.to_vec1()?;
                let actual_value = actual_vec[self.target_crypto_idx] as f64;

                mae_sum += (predicted_value - actual_value).abs();
                count += 1;
            }
        }

        if count > 0 {
            Ok(mae_sum / count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Update divergence history and calculate Z-score
    fn update_and_calculate_zscore(&mut self, divergence: f64) -> f64 {
        self.divergence_history.push_back(divergence);

        if self.divergence_history.len() > ZSCORE_WINDOW {
            self.divergence_history.pop_front();
        }

        if self.divergence_history.len() < 100 {
            return 0.0;
        }

        let mean: f64 = self.divergence_history.iter().sum::<f64>() / self.divergence_history.len() as f64;
        let variance: f64 = self.divergence_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.divergence_history.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        (divergence - mean) / std_dev
    }

    /// Calculate correlation between predictions and actuals
    fn calculate_correlation(&self, val_data: &Tensor) -> Result<f64> {
        let val_timesteps = val_data.dims()[0];
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();

        let step_size = 10;
        let start_idx = SEQUENCE_LENGTH;
        let end_idx = val_timesteps.min(start_idx + 1000);

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

    /// Analyze target crypto correlation on validation set
    fn analyze_target_crypto(&mut self, val_data: &Tensor) -> Result<()> {
        println!("\n🔍 ANALYZING TARGET CRYPTO_{} ON VALIDATION SET", self.target_crypto_idx);
        println!("======================================================================");
        println!("Divergence Method: {:?}", self.divergence_method);

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
            let temp = self.long_entry_threshold;
            self.long_entry_threshold = -self.stop_loss_threshold;
            self.stop_loss_threshold = -temp;
        }

        println!("\n📊 ENHANCED TRADING RULES FOR CRYPTO_{}:", self.target_crypto_idx);
        println!("  - Correlation: {:.4} ({})", self.correlation,
                 if self.is_positive_correlation { "Positive" } else { "Negative" });
        println!("  - Long Entry: Z-score > {:.1}", self.long_entry_threshold);
        println!("  - Take Profit: Z-score returns to {:.1}", self.take_profit_threshold);
        println!("  - Stop Loss: Z-score < {:.1}", self.stop_loss_threshold);
        println!("  - Price Stop Loss: {:.1}% below entry", self.price_stop_loss_pct * 100.0);
        println!("  - Partial Profit: {:.0}% at Z=-0.5", self.partial_profit_pct * 100.0);
        println!("  - Trailing Stop: {:.1}%", self.trailing_stop_pct * 100.0);
        println!("  - Min Holding: {} timesteps", self.min_holding_period);

        Ok(())
    }

    /// Enhanced trading signal generation with improved risk management
    fn generate_signal(&mut self, data: &Tensor, timestamp: usize, current_price: f64) -> Result<Option<(TradeSide, f64)>> {
        if timestamp < SEQUENCE_LENGTH {
            return Ok(None);
        }

        let divergence = self.calculate_divergence(data, timestamp)?;
        let zscore = self.update_and_calculate_zscore(divergence);

        if self.divergence_history.len() < 100 {
            return Ok(None);
        }

        // Clone the position state to avoid borrowing issues
        let current_position = self.position_state.clone();

        match current_position {
            PositionState::NoPosition => {
                // Check for long entry signal
                if zscore > self.long_entry_threshold {
                    self.position_state = PositionState::Long {
                        entry_price: current_price,
                        entry_zscore: zscore,
                        entry_timestamp: timestamp,
                        quantity: self.position_size,
                        trailing_stop_price: None,
                        partial_profit_taken: false,
                    };
                    return Ok(Some((TradeSide::Buy, self.position_size)));
                }
            },
            PositionState::Long {
                entry_price,
                entry_zscore: _,
                entry_timestamp,
                quantity,
                trailing_stop_price,
                partial_profit_taken,
            } => {
                // Minimum holding period check
                if timestamp - entry_timestamp < self.min_holding_period {
                    return Ok(None);
                }

                // Price-based stop loss (2% below entry)
                let price_stop = entry_price * (1.0 - self.price_stop_loss_pct);
                if current_price <= price_stop {
                    println!("  💥 Price stop loss triggered at {:.6} (entry: {:.6})", current_price, entry_price);
                    self.position_state = PositionState::NoPosition;
                    return Ok(Some((TradeSide::Sell, quantity)));
                }

                // Update trailing stop
                let new_trailing_stop = if trailing_stop_price.is_none() || current_price > trailing_stop_price.unwrap_or(0.0) {
                    Some(current_price * (1.0 - self.trailing_stop_pct))
                } else {
                    trailing_stop_price
                };

                // Check trailing stop
                if let Some(trailing_price) = new_trailing_stop {
                    if current_price <= trailing_price {
                        println!("  📉 Trailing stop triggered at {:.6} (trailing: {:.6})", current_price, trailing_price);
                        self.position_state = PositionState::NoPosition;
                        return Ok(Some((TradeSide::Sell, quantity)));
                    }
                }

                // Partial profit taking at Z-score return towards -1
                if !partial_profit_taken && zscore < -0.5 {
                    let partial_quantity = quantity * self.partial_profit_pct;
                    let remaining_quantity = quantity - partial_quantity;

                    // Update position state with partial profit taken
                    self.position_state = PositionState::Long {
                        entry_price,
                        entry_zscore: zscore,
                        entry_timestamp,
                        quantity: remaining_quantity,
                        trailing_stop_price: new_trailing_stop,
                        partial_profit_taken: true,
                    };

                    println!("  💰 Taking partial profit: {:.2} units at Z={:.2}", partial_quantity, zscore);
                    return Ok(Some((TradeSide::Sell, partial_quantity)));
                } else {
                    // Update position state with new trailing stop
                    self.position_state = PositionState::Long {
                        entry_price,
                        entry_zscore: zscore,
                        entry_timestamp,
                        quantity,
                        trailing_stop_price: new_trailing_stop,
                        partial_profit_taken,
                    };
                }

                // Full exit conditions
                // 1. Z-score reversion complete (reached target of -1.0)
                if zscore <= self.take_profit_threshold {
                    println!("  ✅ Z-score reversion complete at {:.2}", zscore);
                    self.position_state = PositionState::NoPosition;
                    return Ok(Some((TradeSide::Sell, quantity)));
                }
                // 2. Z-score stop loss
                else if zscore < self.stop_loss_threshold {
                    println!("  🛑 Z-score stop loss triggered at {:.2}", zscore);
                    self.position_state = PositionState::NoPosition;
                    return Ok(Some((TradeSide::Sell, quantity)));
                }
            }
        }

        Ok(None)
    }
}

fn run_strategy_test(
    method: DivergenceMethod,
    model: FinancialTransformerForMaskedRegression,
    device: Device,
    val_data: &Tensor,
    test_data: &Tensor,
    num_time_series: usize,
    target_crypto_idx: usize,
    initial_capital: f64,
    position_size: f64,
) -> Result<()> {
    println!("\n🚀 TESTING DIVERGENCE METHOD: {:?}", method);
    println!("======================================================================");

    // Initialize strategy
    let mut strategy = EnhancedDivergenceStrategy::new(
        model,
        device.clone(),
        num_time_series,
        target_crypto_idx,
        position_size,
        method.clone(),
    );

    // Analyze target crypto correlation on validation set
    strategy.analyze_target_crypto(val_data)?;

    // Initialize backtester for test set
    println!("\n💰 RUNNING BACKTEST ON TEST SET");
    println!("======================================================================");

    let symbol_names: Vec<String> = (0..num_time_series)
        .map(|i| format!("CRYPTO_{}", i))
        .collect();

    let fees = TradingFees::default();
    let mut backtester = Backtester::new_with_leverage(
        initial_capital,
        test_data.clone(),
        symbol_names,
        Some(fees),
        true,
    )?;

    let test_timesteps = test_data.dims()[0];
    let target_symbol = format!("CRYPTO_{}", target_crypto_idx);

    // Run backtest with detailed logging
    let mut trade_count = 0;
    let mut last_portfolio_value = initial_capital;

    for timestamp in 1..test_timesteps {
        backtester.step_forward(timestamp)?;

        let current_price = backtester.current_prices[target_crypto_idx];
        let current_portfolio = backtester.portfolio_history.last().unwrap();

        // Log portfolio updates every 1000 timesteps or when there's a significant change
        if timestamp % 1000 == 0 || (current_portfolio.total_value - last_portfolio_value).abs() > 0.1 {
            let change = current_portfolio.total_value - last_portfolio_value;
            let change_pct = if last_portfolio_value > 0.0 { change / last_portfolio_value * 100.0 } else { 0.0 };

            let position_info = if current_portfolio.positions.contains_key(&target_symbol) {
                let pos = &current_portfolio.positions[&target_symbol];
                format!("Long {:.3} shares (${:.2})", pos.quantity, pos.current_value)
            } else {
                "No position".to_string()
            };

            println!("  📊 T{}: Portfolio ${:.2} ({:+.2}%, {:+.2} change), Position: {}, {} trades",
                     timestamp, current_portfolio.total_value,
                     (current_portfolio.total_value - initial_capital) / initial_capital * 100.0,
                     change, position_info, trade_count);

            last_portfolio_value = current_portfolio.total_value;
        }

        if let Some((side, quantity)) = strategy.generate_signal(test_data, timestamp, current_price)? {
            match backtester.execute_trade(&target_symbol, side.clone(), quantity, timestamp) {
                Ok(()) => {
                    trade_count += 1;
                    let side_str = match side {
                        TradeSide::Buy => "BUY",
                        TradeSide::Sell => "SELL",
                    };

                    // Get the latest trade for PnL info
                    if let Some(latest_trade) = backtester.trades.last() {
                        let pnl_str = if let Some(pnl) = latest_trade.pnl {
                            format!(" (PnL: ${:.2})", pnl)
                        } else {
                            "".to_string()
                        };

                        println!("  🔄 T{}: {} {:.3} shares of {} @ ${:.4}{}",
                                 timestamp, side_str, quantity, target_symbol, current_price, pnl_str);
                    }
                },
                Err(e) => {
                    println!("  ⚠️  T{}: Trade execution failed: {}", timestamp, e);
                }
            }
        }
    }

    // Calculate and display results
    let metrics = backtester.calculate_metrics()?;

    println!("\n📈 BACKTEST RESULTS - {:?}", method);
    println!("======================================================================");
    println!("💰 Financial Performance:");
    println!("  - Initial Capital: ${:.2}", initial_capital);
    println!("  - Final Value: ${:.2}", metrics.final_portfolio_value);
    println!("  - Total Return: {:.2}%", metrics.total_return * 100.0);
    println!("  - Annualized Return: {:.2}%", metrics.annualized_return * 100.0);
    println!("  - Sharpe Ratio: {:.4}", metrics.sharpe_ratio);
    println!("  - Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("  - Total Fees Paid: ${:.2}", metrics.total_fees);

    println!("\n📊 Trading Statistics:");
    println!("  - Total Trades: {}", backtester.trades.len());

    if !backtester.trades.is_empty() {
        let winning_trades = backtester.trades.iter()
            .filter(|t| t.pnl.unwrap_or(0.0) > 0.0)
            .count();
        let losing_trades = backtester.trades.iter()
            .filter(|t| t.pnl.unwrap_or(0.0) < 0.0)
            .count();
        let win_rate = winning_trades as f64 / backtester.trades.len() as f64 * 100.0;

        println!("  - Win Rate: {:.1}% ({} wins, {} losses)", win_rate, winning_trades, losing_trades);
        println!("  - Profit Factor: {:.2}", metrics.profit_factor);

        // Show recent trades
        println!("\n🔍 Recent Trades (last 10):");
        let recent_trades = backtester.trades.iter().rev().take(10).collect::<Vec<_>>();
        for (i, trade) in recent_trades.iter().rev().enumerate() {
            let side_str = match trade.side {
                TradeSide::Buy => "BUY ",
                TradeSide::Sell => "SELL",
            };
            let pnl_str = if let Some(pnl) = trade.pnl {
                format!(" | PnL: ${:+.2}", pnl)
            } else {
                "".to_string()
            };
            println!("    {}. T{}: {} {:.3} @ ${:.4} | Fee: ${:.2}{}",
                     recent_trades.len() - i, trade.timestamp, side_str,
                     trade.quantity, trade.price, trade.fee, pnl_str);
        }
    }

    // Show final position state
    let final_portfolio = backtester.portfolio_history.last().unwrap();
    println!("\n🎯 Final State:");
    if final_portfolio.positions.contains_key(&format!("CRYPTO_{}", target_crypto_idx)) {
        let pos = &final_portfolio.positions[&format!("CRYPTO_{}", target_crypto_idx)];
        println!("  - Final Position: Long {:.3} shares (${:.2})", pos.quantity, pos.current_value);
    } else {
        println!("  - Final Position: No position");
    }
    println!("  - Cash Balance: ${:.2}", final_portfolio.cash);
    println!("  - Unrealized PnL: ${:.2}", final_portfolio.unrealized_pnl);
    println!("  - Realized PnL: ${:.2}", final_portfolio.realized_pnl);

    Ok(())
}

fn main() -> Result<()> {
    println!("🚀 ENHANCED DIVERGENCE STRATEGY COMPARISON");
    println!("======================================================================");
    println!("This enhanced strategy tests three divergence calculation methods:");
    println!("1. Path Average: avg(prediction_path) - actual_price_at_end");
    println!("2. Path MAE: Mean Absolute Error between predicted and actual paths");
    println!("3. Next Timestep: End-to-end prediction of t+1 only");
    println!();
    println!("Enhanced Risk Management Features:");
    println!("- Price-based stop loss (2% below entry)");
    println!("- Partial profit taking (50% at Z-score -0.5)");
    println!("- Full exit at Z-score -1.0 (stronger reversion signal)");
    println!("- Trailing stops (1% below highest price)");
    println!("- Minimum holding periods");
    println!("======================================================================");

    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model_large_r3_ep2+1.safetensors";
    let initial_capital = 100.0;
    let target_crypto_idx = 87; // CRYPTO_58 from previous successful tests
    let position_size = 0.5; // 50% position size

    // Load and prepare data
    println!("\n📊 Loading dataset from: {}", data_path);
    let (data, num_time_series) = load_and_prepare_data(data_path, &device)?;
    println!("✅ Dataset loaded: {} timesteps, {} time series", data.dims()[0], num_time_series);

    // Split data
    let val_data = extract_validation_split(&data)?;
    let test_data = extract_test_split(&data)?;
    println!("✅ Data splits - Val: {} timesteps, Test: {} timesteps",
             val_data.dims()[0], test_data.dims()[0]);

    // Load model
    println!("\n🤖 Loading model from: {}", model_path);
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
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: true,
        model_type: Some("bert".to_string()),
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FinancialTransformerForMaskedRegression::load(vb, &config)?;
    varmap.load(model_path)?;
    println!("✅ Model loaded from: {}", model_path);

    // Test method based on command line argument or default to PathAverage
    let args: Vec<String> = std::env::args().collect();
    let method = if args.len() > 1 {
        match args[1].as_str() {
            "path_average" => DivergenceMethod::PathAverage,
            "path_mae" => DivergenceMethod::PathMAE,
            "next_timestep" => DivergenceMethod::NextTimestep,
            _ => {
                println!("Usage: {} [path_average|path_mae|next_timestep]", args[0]);
                println!("Defaulting to path_average method");
                DivergenceMethod::PathAverage
            }
        }
    } else {
        println!("No method specified, defaulting to path_average");
        println!("Usage: {} [path_average|path_mae|next_timestep]", args[0]);
        DivergenceMethod::PathAverage
    };

    println!("\n🎯 Testing {:?} Method", method);
    run_strategy_test(
        method,
        model,
        device,
        &val_data,
        &test_data,
        num_time_series,
        target_crypto_idx,
        initial_capital,
        position_size,
    )?;

    Ok(())
}
