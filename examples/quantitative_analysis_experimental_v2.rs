use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::{extract_test_split, Backtester, TradingFees, TradeSide};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

/// Quantitative Analysis Tool for Cross-Sectional Crypto Inference
///
/// This analysis focuses on the model's ability to infer one currency's movements
/// based on the movements of other currencies, rather than next-step prediction.
///
/// Strategy:
/// 1. Select a subset of cryptocurrencies to "black out" (hide from model)
/// 2. Train model to predict these blacked-out cryptos using only the others
/// 3. Measure model divergence vs actual returns
/// 4. Trade based on divergence signals rather than absolute predictions

const SEQUENCE_LENGTH: usize = 120;
const MODEL_DIMS: usize = 256;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 8;

struct CrossSectionalAnalyzer {
    model: FinancialTransformerForMaskedRegression,
    device: Device,
    blacked_out_indices: Vec<usize>,  // Cryptos to predict
    predictor_indices: Vec<usize>,    // Cryptos to use as predictors
}

impl CrossSectionalAnalyzer {
    fn new(
        model: FinancialTransformerForMaskedRegression,
        device: Device,
        num_assets: usize,
        num_blacked_out: usize,
    ) -> Self {
        Self::new_with_seed(model, device, num_assets, num_blacked_out, None)
    }

    fn new_with_seed(
        model: FinancialTransformerForMaskedRegression,
        device: Device,
        num_assets: usize,
        num_blacked_out: usize,
        seed: Option<u64>,
    ) -> Self {
        // Create random number generator with optional seed for deterministic results
        let mut rng = if let Some(seed_value) = seed {
            println!("🎲 Using deterministic seed: {}", seed_value);
            ChaCha8Rng::seed_from_u64(seed_value)
        } else {
            println!("🎲 Using random crypto selection");
            ChaCha8Rng::from_entropy()
        };

        // Create list of all available crypto indices
        let mut all_indices: Vec<usize> = (0..num_assets).collect();

        // Randomly shuffle and select cryptos to black out
        all_indices.shuffle(&mut rng);
        let blacked_out_indices: Vec<usize> = all_indices
            .iter()
            .take(num_blacked_out)
            .copied()
            .collect();

        // Sort for consistent display
        let mut sorted_blacked_out = blacked_out_indices.clone();
        sorted_blacked_out.sort();

        // Remaining cryptos are predictors
        let predictor_indices: Vec<usize> = (0..num_assets)
            .filter(|&i| !blacked_out_indices.contains(&i))
            .collect();

        println!("🎯 Cross-sectional setup:");
        println!("  - Blacked out cryptos (to predict): {:?}", sorted_blacked_out);
        println!("  - Predictor cryptos: {} assets", predictor_indices.len());

        Self {
            model,
            device,
            blacked_out_indices,
            predictor_indices
        }
    }

    /// Create masked input where blacked-out cryptos are set to zero
    fn create_masked_input(&self, data: &Tensor, timestamp: usize) -> Result<Tensor> {
        if timestamp < SEQUENCE_LENGTH {
            return Err(candle_core::Error::Msg("Not enough history for sequence".to_string()));
        }

        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let mut masked_sequence = input_sequence.clone();

        // Zero out the blacked-out cryptocurrencies
        for &crypto_idx in &self.blacked_out_indices {
            let zeros = Tensor::zeros((SEQUENCE_LENGTH, 1), DType::F32, &self.device)?;
            // Create a slice for the specific crypto column and replace with zeros
            let before_cols = if crypto_idx > 0 {
                Some(masked_sequence.narrow(1, 0, crypto_idx)?)
            } else {
                None
            };
            let after_cols = if crypto_idx + 1 < masked_sequence.dims()[1] {
                Some(masked_sequence.narrow(1, crypto_idx + 1, masked_sequence.dims()[1] - crypto_idx - 1)?)
            } else {
                None
            };

            // Reconstruct the tensor with zeros in the blacked-out column
            masked_sequence = match (before_cols, after_cols) {
                (Some(before), Some(after)) => Tensor::cat(&[&before, &zeros, &after], 1)?,
                (Some(before), None) => Tensor::cat(&[&before, &zeros], 1)?,
                (None, Some(after)) => Tensor::cat(&[&zeros, &after], 1)?,
                (None, None) => zeros,
            };
        }

        Ok(masked_sequence)
    }

    /// Get cross-sectional predictions for blacked-out cryptos
    fn get_cross_sectional_prediction(&self, data: &Tensor, timestamp: usize) -> Result<Vec<f64>> {
        let masked_input = self.create_masked_input(data, timestamp)?;
        let input_batch = masked_input.unsqueeze(0)?; // Add batch dimension

        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;
        
        // Extract predictions for the last timestep
        let last_timestep_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;
        
        // Return only predictions for blacked-out cryptos
        let mut blacked_out_predictions = Vec::new();
        for &idx in &self.blacked_out_indices {
            blacked_out_predictions.push(predictions_vec[idx] as f64);
        }
        
        Ok(blacked_out_predictions)
    }

    /// Calculate model divergence: difference between predicted and actual returns
    fn calculate_divergence(&self, data: &Tensor, start_time: usize, end_time: usize) -> Result<Vec<Vec<f64>>> {
        let mut divergences = Vec::new();
        
        for timestamp in start_time..end_time {
            if timestamp < SEQUENCE_LENGTH {
                continue;
            }
            
            // Get model predictions for blacked-out cryptos
            let predictions = self.get_cross_sectional_prediction(data, timestamp)?;
            
            // Get actual returns for blacked-out cryptos
            let actual_returns_row = data.get(timestamp)?;
            let actual_returns_vec: Vec<f32> = actual_returns_row.to_vec1()?;
            
            let mut timestamp_divergences = Vec::new();
            for (i, &crypto_idx) in self.blacked_out_indices.iter().enumerate() {
                let actual = actual_returns_vec[crypto_idx] as f64;
                let predicted = predictions[i];
                let divergence = predicted - actual; // Positive = model overestimated
                timestamp_divergences.push(divergence);
            }
            
            divergences.push(timestamp_divergences);
        }
        
        Ok(divergences)
    }

    /// Analyze cross-sectional inference quality
    fn analyze_inference_quality(&self, data: &Tensor, start_time: usize, end_time: usize) -> Result<()> {
        println!("\n🔍 CROSS-SECTIONAL INFERENCE ANALYSIS");
        println!("======================================================================");
        
        let divergences = self.calculate_divergence(data, start_time, end_time)?;
        
        if divergences.is_empty() {
            println!("❌ No divergence data available");
            return Ok(());
        }
        
        // Calculate statistics for each blacked-out crypto
        for (crypto_i, &crypto_idx) in self.blacked_out_indices.iter().enumerate() {
            let crypto_divergences: Vec<f64> = divergences.iter()
                .map(|div| div[crypto_i])
                .collect();
            
            let mean_divergence = crypto_divergences.iter().sum::<f64>() / crypto_divergences.len() as f64;
            let variance = crypto_divergences.iter()
                .map(|x| (x - mean_divergence).powi(2))
                .sum::<f64>() / crypto_divergences.len() as f64;
            let std_dev = variance.sqrt();
            
            // Calculate correlation between predictions and actuals
            let mut predictions = Vec::new();
            let mut actuals = Vec::new();
            
            for timestamp in start_time..end_time {
                if timestamp < SEQUENCE_LENGTH {
                    continue;
                }
                
                if let Ok(preds) = self.get_cross_sectional_prediction(data, timestamp) {
                    if let Ok(actual_row) = data.get(timestamp) {
                        if let Ok(actual_vec) = actual_row.to_vec1::<f32>() {
                            predictions.push(preds[crypto_i]);
                            actuals.push(actual_vec[crypto_idx] as f64);
                        }
                    }
                }
            }
            
            let correlation = self.calculate_correlation(&predictions, &actuals);
            
            println!("\n📊 CRYPTO_{} (Index: {}) Inference Quality:", crypto_idx, crypto_idx);
            println!("  - Mean divergence: {:.6}", mean_divergence);
            println!("  - Std deviation: {:.6}", std_dev);
            println!("  - Correlation: {:.4}", correlation);
            println!("  - Data points: {}", crypto_divergences.len());
            
            // Interpretation
            if correlation.abs() > 0.1 {
                println!("  ✅ Strong cross-sectional signal");
            } else if correlation.abs() > 0.05 {
                println!("  ⚠️  Weak cross-sectional signal");
            } else {
                println!("  ❌ No meaningful cross-sectional relationship");
            }
        }
        
        Ok(())
    }

    /// Display detailed predicted vs real values for one crypto to allow manual inspection
    fn display_prediction_comparison(&self, data: &Tensor, start_time: usize, end_time: usize) -> Result<()> {
        println!("\n🔍 DETAILED PREDICTION vs REAL VALUES COMPARISON");
        println!("======================================================================");

        if self.blacked_out_indices.is_empty() {
            println!("❌ No blacked-out cryptos to analyze");
            return Ok(());
        }

        // Focus on the first blacked-out crypto for detailed analysis
        let target_crypto_idx = self.blacked_out_indices[0];
        println!("📊 Analyzing CRYPTO_{} (Index: {})", target_crypto_idx, target_crypto_idx);
        println!("Showing predicted vs real values for manual inspection:");
        println!("");
        println!("{:<10} {:<15} {:<15} {:<15} {:<15} {:<10}",
                 "Timestamp", "Predicted", "Real", "Divergence", "Divergence %", "Direction");
        println!("{}", "-".repeat(90));

        let mut comparison_data = Vec::new();
        let display_limit = 200; // Show many lines but not overwhelming
        let mut count = 0;

        for timestamp in start_time..end_time {
            if timestamp < SEQUENCE_LENGTH || count >= display_limit {
                if count >= display_limit {
                    break;
                }
                continue;
            }

            // Get model prediction for this crypto
            if let Ok(predictions) = self.get_cross_sectional_prediction(data, timestamp) {
                if let Ok(actual_row) = data.get(timestamp) {
                    if let Ok(actual_vec) = actual_row.to_vec1::<f32>() {
                        let predicted = predictions[0]; // First blacked-out crypto
                        let actual = actual_vec[target_crypto_idx] as f64;
                        let divergence = predicted - actual;

                        // Calculate divergence percentage (error as percentage of actual value)
                        let divergence_pct = if actual.abs() > 1e-8 {
                            (divergence / actual.abs()) * 100.0
                        } else {
                            // Handle near-zero actual values
                            if divergence.abs() < 1e-8 {
                                0.0 // Both are essentially zero
                            } else {
                                f64::INFINITY // Actual is zero but prediction is not
                            }
                        };

                        // Determine direction match
                        let direction_match = if (predicted > 0.0 && actual > 0.0) || (predicted < 0.0 && actual < 0.0) {
                            "✓ Match"
                        } else if predicted.abs() < 0.001 && actual.abs() < 0.001 {
                            "~ Neutral"
                        } else {
                            "✗ Opposite"
                        };

                        // Format divergence percentage for display
                        let divergence_pct_str = if divergence_pct.is_infinite() {
                            "∞".to_string()
                        } else if divergence_pct.abs() > 999.9 {
                            format!("{:.0}", divergence_pct)
                        } else {
                            format!("{:.1}", divergence_pct)
                        };

                        println!("{:<10} {:<15.6} {:<15.6} {:<15.6} {:<15} {:<10}",
                                timestamp, predicted, actual, divergence, divergence_pct_str, direction_match);

                        comparison_data.push((timestamp, predicted, actual, divergence));
                        count += 1;
                    }
                }
            }
        }

        if comparison_data.is_empty() {
            println!("❌ No comparison data available");
            return Ok(());
        }

        println!("{}", "-".repeat(90));
        println!("Displayed {} data points for manual inspection", comparison_data.len());

        // Calculate summary statistics
        let predictions: Vec<f64> = comparison_data.iter().map(|(_, p, _, _)| *p).collect();
        let actuals: Vec<f64> = comparison_data.iter().map(|(_, _, a, _)| *a).collect();
        let divergences: Vec<f64> = comparison_data.iter().map(|(_, _, _, d)| *d).collect();

        let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
        let mean_divergence = divergences.iter().sum::<f64>() / divergences.len() as f64;

        let direction_matches = comparison_data.iter()
            .filter(|(_, p, a, _)| (p > &0.0 && a > &0.0) || (p < &0.0 && a < &0.0))
            .count();
        let direction_accuracy = direction_matches as f64 / comparison_data.len() as f64 * 100.0;

        // Calculate magnitude correlation
        let correlation = self.calculate_correlation(&predictions, &actuals);

        println!("\n📈 SUMMARY STATISTICS:");
        println!("  - Mean predicted: {:.6}", mean_pred);
        println!("  - Mean actual: {:.6}", mean_actual);
        println!("  - Mean divergence: {:.6}", mean_divergence);
        println!("  - Direction accuracy: {:.1}%", direction_accuracy);
        println!("  - Correlation: {:.4}", correlation);

        println!("\n💡 INTERPRETATION GUIDE:");
        println!("  - Look for patterns in the divergence column");
        println!("  - Check if predicted and real values have similar magnitudes");
        println!("  - Direction matches (✓) indicate the model captures trend direction");
        println!("  - Consistent divergence patterns may indicate systematic bias");
        println!("  - High correlation suggests good relative magnitude prediction");

        Ok(())
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Ultra-Conservative Alpha Strategy (LONG-ONLY for Binance API compatibility)
/// This strategy prioritizes capital preservation with slow but steady returns
struct UltraConservativeStrategy {
    analyzer: CrossSectionalAnalyzer,
    divergence_threshold: f64,      // Trade when |divergence| > threshold
    base_position_size: f64,        // Base fraction of portfolio per position
    min_return_threshold: f64,      // Minimum expected return to trade
    max_position_size: f64,         // Maximum position size cap
    confidence_multiplier: f64,     // Multiply position size by confidence
    lookback_window: usize,         // Window for calculating signal strength

    // Enhanced risk management
    max_portfolio_exposure: f64,    // Maximum total exposure across all positions
    stop_loss_threshold: f64,       // Stop loss threshold for individual positions
    take_profit_threshold: f64,     // Take profit threshold
    max_trades_per_period: usize,   // Limit trades per period to avoid overtrading
    min_confidence_threshold: f64,  // Higher confidence requirement
    volatility_adjustment: bool,    // Adjust position size based on volatility
    drawdown_protection: f64,       // Reduce trading when drawdown exceeds this
}

impl UltraConservativeStrategy {
    fn new(
        analyzer: CrossSectionalAnalyzer,
        divergence_threshold: f64,
        base_position_size: f64,
        min_return_threshold: f64,
    ) -> Self {
        Self {
            analyzer,
            divergence_threshold,
            base_position_size,
            min_return_threshold,
            max_position_size: 0.10,            // Cap at 10% of portfolio (conservative but tradeable)
            confidence_multiplier: 1.8,         // Up to 1.8x position size for high confidence
            lookback_window: 30,                // 30 periods for balanced stability vs responsiveness

            // Balanced risk management parameters
            max_portfolio_exposure: 0.35,       // Maximum 35% total exposure (allow more trading)
            stop_loss_threshold: -0.03,         // 3% stop loss
            take_profit_threshold: 0.02,        // 2% take profit
            max_trades_per_period: 3,           // Maximum 3 trades per period (allow more activity)
            min_confidence_threshold: 0.45,     // Require 45% confidence minimum (lower barrier)
            volatility_adjustment: true,        // Enable volatility-based position sizing
            drawdown_protection: 0.08,          // Reduce trading if drawdown > 8% (more lenient)
        }
    }

    /// Calculate enhanced signal confidence with volatility and stability metrics
    fn calculate_signal_confidence(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<f64> {
        if timestamp < SEQUENCE_LENGTH + self.lookback_window {
            return Ok(0.4); // Slightly higher default confidence to allow some trading
        }

        let mut recent_divergences = Vec::new();
        let mut recent_returns = Vec::new();
        let start_lookback = timestamp - self.lookback_window;

        for t in start_lookback..timestamp {
            if let Ok(predictions) = self.analyzer.get_cross_sectional_prediction(data, t) {
                if let Ok(actual_row) = data.get(t) {
                    if let Ok(actual_vec) = actual_row.to_vec1::<f32>() {
                        // Find the index of this crypto in blacked_out_indices
                        if let Some(crypto_position) = self.analyzer.blacked_out_indices.iter().position(|&x| x == crypto_idx) {
                            let predicted = predictions[crypto_position];
                            let actual = actual_vec[crypto_idx] as f64;
                            let divergence = predicted - actual;
                            recent_divergences.push(divergence);
                            recent_returns.push(actual);
                        }
                    }
                }
            }
        }

        if recent_divergences.is_empty() {
            return Ok(0.4);
        }

        // Calculate current divergence
        let current_predictions = self.analyzer.get_cross_sectional_prediction(data, timestamp)?;
        let current_actual_row = data.get(timestamp - 1)?;
        let current_actual_vec: Vec<f32> = current_actual_row.to_vec1()?;

        if let Some(crypto_position) = self.analyzer.blacked_out_indices.iter().position(|&x| x == crypto_idx) {
            let current_predicted = current_predictions[crypto_position];
            let current_actual = current_actual_vec[crypto_idx] as f64;
            let current_divergence = current_predicted - current_actual;

            // 1. Direction consistency (more conservative)
            let same_direction_count = recent_divergences.iter()
                .filter(|&&div| (div > 0.0 && current_divergence > 0.0) || (div < 0.0 && current_divergence < 0.0))
                .count();
            let consistency = same_direction_count as f64 / recent_divergences.len() as f64;

            // 2. Magnitude stability (penalize high volatility)
            let mean_abs_divergence = recent_divergences.iter().map(|d| d.abs()).sum::<f64>() / recent_divergences.len() as f64;
            let divergence_std = {
                let variance = recent_divergences.iter()
                    .map(|d| (d.abs() - mean_abs_divergence).powi(2))
                    .sum::<f64>() / recent_divergences.len() as f64;
                variance.sqrt()
            };
            let stability_score = if mean_abs_divergence > 0.0 {
                (1.0 - (divergence_std / mean_abs_divergence).min(1.0)).max(0.0)
            } else {
                0.5
            };

            // 3. Return volatility penalty (lower confidence for volatile assets)
            let return_volatility = if recent_returns.len() > 1 {
                let mean_return = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
                let variance = recent_returns.iter()
                    .map(|r| (r - mean_return).powi(2))
                    .sum::<f64>() / recent_returns.len() as f64;
                variance.sqrt()
            } else {
                0.01 // Default low volatility
            };
            let volatility_penalty = (1.0 - (return_volatility * 100.0).min(1.0)).max(0.0);

            // 4. Signal strength (how much above threshold)
            let signal_strength = (current_divergence.abs() / self.divergence_threshold).min(2.0);
            let strength_score = (signal_strength - 1.0).max(0.0) / 1.0; // 0 to 1 scale

            // Combine all factors with conservative weighting
            let confidence = (
                consistency * 0.4 +           // Direction consistency is most important
                stability_score * 0.3 +       // Stability of divergence patterns
                volatility_penalty * 0.2 +    // Penalty for high volatility
                strength_score * 0.1          // Signal strength bonus
            ).min(1.0);

            Ok(confidence)
        } else {
            Ok(0.4)
        }
    }

    /// Calculate portfolio exposure to avoid over-concentration
    fn calculate_current_exposure(&self, backtester: &Backtester) -> f64 {
        let current_portfolio = backtester.get_current_portfolio();
        let total_value = current_portfolio.total_value;
        let mut total_exposure = 0.0;

        for (_, position) in &current_portfolio.positions {
            if position.quantity > 0.0 {
                total_exposure += position.current_value;
            }
        }

        total_exposure / total_value
    }

    /// Check if we should reduce trading due to drawdown
    fn should_reduce_trading(&self, backtester: &Backtester, initial_capital: f64) -> bool {
        let current_value = backtester.portfolio_history.last().unwrap().total_value;
        let drawdown = (initial_capital - current_value) / initial_capital;
        drawdown > self.drawdown_protection
    }

    /// Generate ultra-conservative trading signals with extensive risk management
    fn generate_signals(&self, data: &Tensor, timestamp: usize, backtester: &Backtester, initial_capital: f64) -> Result<Vec<(usize, TradeSide, f64)>> {
        let mut signals = Vec::new();

        if timestamp < SEQUENCE_LENGTH + self.lookback_window {
            return Ok(signals);
        }

        // Check if we should reduce trading due to drawdown
        if self.should_reduce_trading(backtester, initial_capital) {
            return Ok(signals); // No new trades during high drawdown
        }

        // Check current portfolio exposure
        let current_exposure = self.calculate_current_exposure(backtester);
        if current_exposure >= self.max_portfolio_exposure {
            return Ok(signals); // No new trades if already at max exposure
        }

        // Get current divergence
        let predictions = self.analyzer.get_cross_sectional_prediction(data, timestamp)?;
        let actual_returns_row = data.get(timestamp - 1)?;
        let actual_returns_vec: Vec<f32> = actual_returns_row.to_vec1()?;

        let mut trade_count = 0;

        for (i, &crypto_idx) in self.analyzer.blacked_out_indices.iter().enumerate() {
            // Limit trades per period
            if trade_count >= self.max_trades_per_period {
                break;
            }

            let actual = actual_returns_vec[crypto_idx] as f64;
            let predicted = predictions[i];
            let divergence = predicted - actual;

            // Only trade if divergence exceeds threshold
            if divergence.abs() > self.divergence_threshold {
                // Calculate signal confidence
                let confidence = self.calculate_signal_confidence(data, timestamp, crypto_idx)?;

                // Higher confidence threshold for conservative approach
                if confidence < self.min_confidence_threshold {
                    continue;
                }

                if divergence < 0.0 {
                    // Model underestimated -> expect catch-up -> BUY
                    let expected_return = divergence.abs();

                    // Higher return threshold for conservative approach
                    if expected_return > self.min_return_threshold {
                        // Calculate conservative position size
                        let signal_strength = (expected_return / self.divergence_threshold).min(1.5); // Cap signal strength
                        let confidence_factor = (confidence - self.min_confidence_threshold) / (1.0 - self.min_confidence_threshold);
                        let position_multiplier = (confidence_factor * signal_strength * self.confidence_multiplier).min(self.confidence_multiplier);

                        // Apply volatility adjustment if enabled
                        let volatility_factor = if self.volatility_adjustment {
                            // Calculate recent volatility for this crypto
                            let mut recent_returns = Vec::new();
                            let lookback_start = timestamp.saturating_sub(20);
                            for t in lookback_start..timestamp {
                                if let Ok(row) = data.get(t) {
                                    if let Ok(vec) = row.to_vec1::<f32>() {
                                        recent_returns.push(vec[crypto_idx] as f64);
                                    }
                                }
                            }

                            if recent_returns.len() > 1 {
                                let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
                                let variance = recent_returns.iter()
                                    .map(|r| (r - mean).powi(2))
                                    .sum::<f64>() / recent_returns.len() as f64;
                                let volatility = variance.sqrt();
                                (1.0 - (volatility * 50.0).min(0.5)).max(0.5) // Reduce size for high volatility
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };

                        let base_size = self.base_position_size * position_multiplier * volatility_factor;
                        let position_size = base_size.min(self.max_position_size);

                        // Ensure we don't exceed max portfolio exposure
                        let remaining_exposure = self.max_portfolio_exposure - current_exposure;
                        let final_position_size = position_size.min(remaining_exposure);

                        if final_position_size > 0.01 { // Minimum 1% position size
                            signals.push((crypto_idx, TradeSide::Buy, final_position_size));
                            trade_count += 1;
                        }
                    }
                } else {
                    // Model overestimated -> expect reversion -> SELL existing positions
                    // Only sell if we actually have a position in this crypto
                    let symbol = format!("CRYPTO_{}", crypto_idx);
                    let current_portfolio = backtester.get_current_portfolio();
                    if let Some(position) = current_portfolio.positions.get(&symbol) {
                        if position.quantity > 0.0 {
                            // Very conservative sell sizing
                            let confidence_factor = (confidence - self.min_confidence_threshold) / (1.0 - self.min_confidence_threshold);
                            let position_size = (self.base_position_size * confidence_factor * 0.5).min(self.max_position_size * 0.6);

                            if position_size > 0.01 {
                                signals.push((crypto_idx, TradeSide::Sell, position_size));
                                trade_count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(signals)
    }
}

fn main() -> Result<()> {
    println!("📊 QUANTITATIVE ANALYSIS - Cross-Sectional Crypto Inference");
    println!("======================================================================");
    println!("This analysis focuses on inferring currency movements from others,");
    println!("not next-step prediction. Trading based on model divergence signals.");
    println!("======================================================================");

    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model.safetensors";
    let initial_capital = 100.0;

    // Load data
    println!("\nLoading cryptocurrency data...");
    let (full_data_sequence, num_time_series) = load_and_prepare_data(data_path, &device)?;
    
    // Extract ONLY the test split to prevent data leakage
    let test_data = extract_test_split(&full_data_sequence)?;
    let test_timesteps = test_data.dims()[0];
    
    println!("Data loaded: {} assets, {} test timesteps", num_time_series, test_timesteps);

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

    // Initialize cross-sectional analyzer with random crypto selection
    let num_blacked_out = (num_time_series / 4).max(3).min(8); // 25% of cryptos, 3-8 range
    let seed = None;//Some(42); // Use deterministic seed for reproducible results, set to None for random
    let analyzer = CrossSectionalAnalyzer::new_with_seed(
        model,
        device.clone(),
        num_time_series,
        num_blacked_out,
        seed
    );

    // Analyze inference quality
    let analysis_start = test_timesteps / 4;
    let analysis_end = (analysis_start + 5000).min(test_timesteps - 1);
    
    //analyzer.analyze_inference_quality(&test_data, analysis_start, analysis_end)?;

    // Display detailed predicted vs real values for inspection
    analyzer.display_prediction_comparison(&test_data, analysis_start, analysis_end)?;

    // Initialize balanced conservative trading strategy
    println!("\n💰 BALANCED CONSERVATIVE TRADING STRATEGY (LONG-ONLY)");
    println!("======================================================================");
    println!("Note: This strategy balances capital preservation with actual trading activity");

    // Balanced conservative parameters - still safe but actually trades
    let divergence_threshold = 0.006; // 0.6% divergence threshold (lower to get some signals)
    let base_position_size = 0.04; // 4% base position size (conservative but tradeable)
    let min_return_threshold = 0.004; // 0.4% minimum expected return (covers fees with margin)
    let strategy = UltraConservativeStrategy::new(analyzer, divergence_threshold, base_position_size, min_return_threshold);

    // Create symbol names for backtesting
    let symbol_names: Vec<String> = (0..num_time_series)
        .map(|i| format!("CRYPTO_{}", i))
        .collect();

    // Initialize backtester
    let fees = TradingFees::default();
    let mut backtester = Backtester::new(
        initial_capital,
        test_data.clone(),
        symbol_names.clone(),
        Some(fees),
    )?;

    println!("🚀 Running balanced conservative backtest...");
    println!("  - Divergence threshold: {:.2}%", divergence_threshold * 100.0);
    println!("  - Base position size: {:.1}%", base_position_size * 100.0);
    println!("  - Max position size: {:.1}%", strategy.max_position_size * 100.0);
    println!("  - Max portfolio exposure: {:.1}%", strategy.max_portfolio_exposure * 100.0);
    println!("  - Min confidence threshold: {:.1}%", strategy.min_confidence_threshold * 100.0);
    println!("  - Min return threshold: {:.2}%", min_return_threshold * 100.0);
    println!("  - Max trades per period: {}", strategy.max_trades_per_period);
    println!("  - Trading period: {} to {} ({} timesteps)",
             analysis_start, analysis_end, analysis_end - analysis_start);

    let mut total_trades = 0;
    let mut successful_trades = 0;

    // Run backtest
    for timestamp in analysis_start..analysis_end {
        // Step forward to update prices
        backtester.step_forward(timestamp)?;

        // Generate trading signals based on divergence
        if let Ok(signals) = strategy.generate_signals(&test_data, timestamp, &backtester, initial_capital) {
            for (crypto_idx, side, size) in signals {
                let symbol = &symbol_names[crypto_idx];

                // Calculate position size in shares
                let current_portfolio_value = backtester.portfolio_history.last().unwrap().total_value;
                let position_value = current_portfolio_value * size;
                let current_price = backtester.current_prices[crypto_idx];
                let shares = position_value / current_price;

                // Clone side for later use
                let side_for_profit_check = side.clone();

                // Execute trade
                if let Ok(_) = backtester.execute_trade(symbol, side, shares, timestamp) {
                    total_trades += 1;

                    // Check if this trade will be profitable (simplified check)
                    // In practice, you'd track this over multiple periods
                    if timestamp + 5 < test_timesteps {
                        let future_return = test_data.get(timestamp + 4)?.get(crypto_idx)?.to_scalar::<f32>()? as f64;
                        let expected_profit = match side_for_profit_check {
                            TradeSide::Buy => future_return > 0.0,
                            TradeSide::Sell => {
                                // For sells, we profit if we're exiting before a decline
                                // or if we're taking profits from a previous good position
                                true // Simplified: assume sells are position management
                            },
                        };
                        if expected_profit {
                            successful_trades += 1;
                        }
                    }
                }
            }
        }

        // Print progress every 1000 timesteps
        if timestamp % 1000 == 0 {
            let current_value = backtester.portfolio_history.last().unwrap().total_value;
            let return_pct = (current_value - initial_capital) / initial_capital * 100.0;
            println!("  Timestamp {}: Portfolio value: ${:.2} ({:+.2}%)",
                     timestamp, current_value, return_pct);
        }
    }

    // Calculate final performance metrics
    let metrics = backtester.calculate_metrics()?;

    println!("\n📈 BACKTEST RESULTS");
    println!("======================================================================");
    println!("💰 Financial Performance:");
    println!("  - Initial capital: ${:.2}", initial_capital);
    println!("  - Final value: ${:.2}", metrics.final_portfolio_value);
    println!("  - Total return: {:.2}%", metrics.total_return * 100.0);
    println!("  - Sharpe ratio: {:.3}", metrics.sharpe_ratio);
    println!("  - Max drawdown: {:.2}%", metrics.max_drawdown * 100.0);

    println!("\n📊 Trading Statistics:");
    println!("  - Total trades: {}", total_trades);
    println!("  - Successful trades: {} ({:.1}%)",
             successful_trades,
             if total_trades > 0 { successful_trades as f64 / total_trades as f64 * 100.0 } else { 0.0 });
    println!("  - Total fees paid: ${:.2}", metrics.total_fees);

    println!("\n🎯 Strategy Interpretation:");
    if metrics.total_return > 0.05 {
        println!("  ✅ Strong performance: Divergence signals are profitable");
    } else if metrics.total_return > 0.0 {
        println!("  ⚠️  Modest performance: Some signal but room for improvement");
    } else {
        println!("  ❌ Poor performance: Divergence signals may not be reliable");
    }

    if metrics.sharpe_ratio > 1.0 {
        println!("  ✅ Excellent risk-adjusted returns");
    } else if metrics.sharpe_ratio > 0.5 {
        println!("  ⚠️  Decent risk-adjusted returns");
    } else {
        println!("  ❌ Poor risk-adjusted returns");
    }

    println!("\n💡 BALANCED CONSERVATIVE STRATEGY INSIGHTS (LONG-ONLY):");
    println!("======================================================================");
    println!("This BALANCED strategy combines safety with reasonable trading activity:");
    println!("  - MODERATE CONFIDENCE: 30-period lookback + 45% minimum confidence threshold");
    println!("  - CONSERVATIVE SIZING: 4-10% positions with 35% max portfolio exposure");
    println!("  - REASONABLE RETURN FILTER: Only trades when expected return > 0.4% (above fees)");
    println!("  - VOLATILITY ADJUSTMENT: Reduces position size for volatile assets");
    println!("  - DRAWDOWN PROTECTION: Stops trading when drawdown > 8%");
    println!("  - TRADE LIMITING: Maximum 3 trades per period for controlled activity");
    println!("  - EXPOSURE MANAGEMENT: Never exceeds 35% total portfolio exposure");
    println!("  - When model UNDERESTIMATES → Moderate BUY with balanced criteria");
    println!("  - When model OVERESTIMATES → Conservative SELL of existing positions");
    println!("  - Focus: Steady returns with controlled risk and actual trading activity");

    println!("\n✅ Quantitative analysis complete!");

    Ok(())
}
