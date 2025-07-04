use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::{extract_test_split, Backtester, TradingFees, TradeSide};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

/// Selective Crypto Analysis - Individual Masking and Correlation-Based Selection
///
/// This strategy:
/// 1. Masks each cryptocurrency individually to test model performance
/// 2. Calculates correlation and divergence metrics for each crypto
/// 3. Selects only the best-performing cryptos for trading
/// 4. Uses 2-hour (120-period) lookback window for analysis
/// 5. Exploits the most reliable divergence patterns

const SEQUENCE_LENGTH: usize = 120;
const MODEL_DIMS: usize = 256;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 8;
const LOOKBACK_WINDOW: usize = 120; // 2-hour lookback for correlation analysis

/// Performance metrics for individual crypto analysis
#[derive(Debug, Clone)]
struct CryptoPerformance {
    crypto_idx: usize,
    correlation: f64,
    mean_divergence: f64,
    divergence_std: f64,
    direction_accuracy: f64,
    signal_strength: f64,
    tradeable_score: f64,
}

/// Selective crypto analyzer that evaluates each crypto individually
struct SelectiveCryptoAnalyzer {
    model: FinancialTransformerForMaskedRegression,
    device: Device,
    num_assets: usize,
    crypto_performances: Vec<CryptoPerformance>,
    selected_cryptos: Vec<usize>,
}

impl SelectiveCryptoAnalyzer {
    fn new(model: FinancialTransformerForMaskedRegression, device: Device, num_assets: usize) -> Self {
        Self {
            model,
            device,
            num_assets,
            crypto_performances: Vec::new(),
            selected_cryptos: Vec::new(),
        }
    }

    /// Create masked input for a single cryptocurrency
    fn create_single_crypto_mask(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<Tensor> {
        if timestamp < SEQUENCE_LENGTH {
            return Err(candle_core::Error::Msg("Not enough history for sequence".to_string()));
        }

        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let mut masked_sequence = input_sequence.clone();

        // Zero out only the target cryptocurrency
        let zeros = Tensor::zeros((SEQUENCE_LENGTH, 1), DType::F32, &self.device)?;
        
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

        // Reconstruct the tensor with zeros in the target column
        masked_sequence = match (before_cols, after_cols) {
            (Some(before), Some(after)) => Tensor::cat(&[&before, &zeros, &after], 1)?,
            (Some(before), None) => Tensor::cat(&[&before, &zeros], 1)?,
            (None, Some(after)) => Tensor::cat(&[&zeros, &after], 1)?,
            (None, None) => zeros,
        };

        Ok(masked_sequence)
    }

    /// Get prediction for a single masked cryptocurrency
    fn get_single_crypto_prediction(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<f64> {
        let masked_input = self.create_single_crypto_mask(data, timestamp, crypto_idx)?;
        let input_batch = masked_input.unsqueeze(0)?; // Add batch dimension

        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;
        
        // Extract prediction for the last timestep and target crypto
        let last_timestep_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;
        
        Ok(predictions_vec[crypto_idx] as f64)
    }

    /// Analyze performance for a single cryptocurrency
    fn analyze_single_crypto(&self, data: &Tensor, crypto_idx: usize, start_time: usize, end_time: usize) -> Result<CryptoPerformance> {
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        let mut divergences = Vec::new();

        println!("  Analyzing CRYPTO_{} ({}/{})", crypto_idx, crypto_idx + 1, self.num_assets);

        for timestamp in start_time..end_time {
            if timestamp < SEQUENCE_LENGTH {
                continue;
            }

            // Get model prediction for this crypto
            if let Ok(predicted) = self.get_single_crypto_prediction(data, timestamp, crypto_idx) {
                if let Ok(actual_row) = data.get(timestamp) {
                    if let Ok(actual_vec) = actual_row.to_vec1::<f32>() {
                        let actual = actual_vec[crypto_idx] as f64;
                        let divergence = predicted - actual;

                        predictions.push(predicted);
                        actuals.push(actual);
                        divergences.push(divergence);
                    }
                }
            }
        }

        if predictions.is_empty() {
            return Ok(CryptoPerformance {
                crypto_idx,
                correlation: 0.0,
                mean_divergence: 0.0,
                divergence_std: 0.0,
                direction_accuracy: 0.0,
                signal_strength: 0.0,
                tradeable_score: 0.0,
            });
        }

        // Calculate correlation
        let correlation = self.calculate_correlation(&predictions, &actuals);

        // Calculate divergence statistics
        let mean_divergence = divergences.iter().sum::<f64>() / divergences.len() as f64;
        let divergence_variance = divergences.iter()
            .map(|d| (d - mean_divergence).powi(2))
            .sum::<f64>() / divergences.len() as f64;
        let divergence_std = divergence_variance.sqrt();

        // Calculate direction accuracy
        let direction_matches = predictions.iter().zip(actuals.iter())
            .filter(|&(&p, &a)| (p > 0.0 && a > 0.0) || (p < 0.0 && a < 0.0))
            .count();
        let direction_accuracy = direction_matches as f64 / predictions.len() as f64;

        // Calculate signal strength (how often divergence exceeds meaningful thresholds)
        let significant_divergences = divergences.iter()
            .filter(|d| d.abs() > 0.005) // 0.5% threshold
            .count();
        let signal_strength = significant_divergences as f64 / divergences.len() as f64;

        // Calculate overall tradeable score
        let tradeable_score = self.calculate_tradeable_score(correlation, direction_accuracy, signal_strength, divergence_std);

        Ok(CryptoPerformance {
            crypto_idx,
            correlation,
            mean_divergence,
            divergence_std,
            direction_accuracy,
            signal_strength,
            tradeable_score,
        })
    }

    /// Calculate a composite tradeable score for ranking cryptos
    fn calculate_tradeable_score(&self, correlation: f64, direction_accuracy: f64, signal_strength: f64, divergence_std: f64) -> f64 {
        // Weights for different factors
        let correlation_weight = 0.4;      // High correlation is good
        let direction_weight = 0.3;        // Good direction accuracy is important
        let signal_weight = 0.2;           // Strong signals are valuable
        let stability_weight = 0.1;        // Lower volatility is better

        let correlation_score = correlation.abs().min(1.0);
        let direction_score = direction_accuracy;
        let signal_score = signal_strength;
        let stability_score = (1.0 - (divergence_std * 100.0).min(1.0)).max(0.0); // Penalize high volatility

        correlation_weight * correlation_score +
        direction_weight * direction_score +
        signal_weight * signal_score +
        stability_weight * stability_score
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



    /// Analyze all cryptocurrencies and select the best ones for trading
    fn analyze_all_cryptos(&mut self, data: &Tensor, start_time: usize, end_time: usize) -> Result<()> {
        println!("\n🔍 INDIVIDUAL CRYPTO PERFORMANCE ANALYSIS");
        println!("======================================================================");
        println!("Analyzing all {} cryptocurrencies individually...", self.num_assets);

        self.crypto_performances.clear();

        // Analyze each crypto individually - no exclusions, focus on best model predictions
        for crypto_idx in 0..self.num_assets {
            let performance = self.analyze_single_crypto(data, crypto_idx, start_time, end_time)?;
            self.crypto_performances.push(performance);
        }

        // Sort by tradeable score (best first)
        self.crypto_performances.sort_by(|a, b| b.tradeable_score.partial_cmp(&a.tradeable_score).unwrap());

        // Display top performers
        println!("\n📊 TOP PERFORMING CRYPTOCURRENCIES:");
        println!("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}", 
                 "Crypto", "Correlation", "Direction%", "Signal%", "Div.Std", "Score");
        println!("{}", "-".repeat(80));

        for (_rank, perf) in self.crypto_performances.iter().take(10).enumerate() {
            println!("{:<8} {:<12.4} {:<12.1} {:<12.1} {:<12.4} {:<12.4}",
                     format!("CRYPTO_{}", perf.crypto_idx),
                     perf.correlation,
                     perf.direction_accuracy * 100.0,
                     perf.signal_strength * 100.0,
                     perf.divergence_std,
                     perf.tradeable_score);
        }

        // Focus on best model predictions - select top performers by score
        println!("🎯 Selecting cryptos with best model prediction performance...");

        // Simply take the top performers by tradeable score (best model predictions)
        let best_performers: Vec<&CryptoPerformance> = self.crypto_performances.iter()
            .filter(|perf| perf.tradeable_score > 0.0) // Any positive score
            .collect();

        // Select top performers (maximum 5 for focused trading)
        let max_selected = 5;
        self.selected_cryptos = best_performers.iter()
            .take(max_selected)
            .map(|perf| perf.crypto_idx)
            .collect();

        println!("\n🎯 SELECTED CRYPTOCURRENCIES FOR TRADING:");
        println!("Selected {} out of {} analyzed cryptocurrencies based on best model predictions:",
                 self.selected_cryptos.len(), self.crypto_performances.len());
        println!("Selection criteria:");
        println!("  - Focus on highest tradeable scores (best model prediction performance)");
        println!("  - No exclusions - all cryptos considered");
        println!("  - Top 5 performers selected for trading");
        println!("");

        if self.selected_cryptos.is_empty() {
            println!("❌ No cryptocurrencies had positive scores!");
        } else {
            println!("🎯 BEST MODEL PREDICTION PERFORMERS:");
            for &crypto_idx in &self.selected_cryptos {
                let perf = &self.crypto_performances.iter().find(|p| p.crypto_idx == crypto_idx).unwrap();
                println!("  - CRYPTO_{}: Score {:.3}, Correlation {:.3}, Direction {:.1}%, Signals {:.1}%",
                         crypto_idx, perf.tradeable_score, perf.correlation,
                         perf.direction_accuracy * 100.0, perf.signal_strength * 100.0);
            }
        }

        Ok(())
    }

    /// Get current divergence for a selected crypto
    fn get_current_divergence(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<f64> {
        let predicted = self.get_single_crypto_prediction(data, timestamp, crypto_idx)?;
        let actual_row = data.get(timestamp - 1)?;
        let actual_vec: Vec<f32> = actual_row.to_vec1()?;
        let actual = actual_vec[crypto_idx] as f64;
        Ok(predicted - actual)
    }

    /// Check if a crypto is in our selected list
    fn is_selected_crypto(&self, crypto_idx: usize) -> bool {
        self.selected_cryptos.contains(&crypto_idx)
    }

    /// Get performance metrics for a crypto
    fn get_crypto_performance(&self, crypto_idx: usize) -> Option<&CryptoPerformance> {
        self.crypto_performances.iter().find(|p| p.crypto_idx == crypto_idx)
    }
}

/// Selective trading strategy using original cross-sectional approach on selected cryptos
struct SelectiveTradingStrategy {
    analyzer: SelectiveCryptoAnalyzer,
    divergence_threshold: f64,
    base_position_size: f64,
    min_return_threshold: f64,
    max_position_size: f64,
    max_portfolio_exposure: f64,
    confidence_lookback: usize,
}

impl SelectiveTradingStrategy {
    fn new(
        analyzer: SelectiveCryptoAnalyzer,
        divergence_threshold: f64,
        base_position_size: f64,
        min_return_threshold: f64,
    ) -> Self {
        Self {
            analyzer,
            divergence_threshold,
            base_position_size,
            min_return_threshold,
            max_position_size: 0.10,        // 10% max position
            max_portfolio_exposure: 0.8,    // 80% max exposure (very aggressive)
            confidence_lookback: 10,        // 10-period lookback (shorter)
        }
    }

    /// Get individual prediction for a single selected crypto (mask only that crypto)
    fn get_individual_prediction(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<f64> {
        // Use the analyzer's existing single crypto masking method
        self.analyzer.get_single_crypto_prediction(data, timestamp, crypto_idx)
    }

    /// Calculate individual divergence for a selected crypto
    fn calculate_individual_divergence(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<f64> {
        let predicted = self.get_individual_prediction(data, timestamp, crypto_idx)?;
        let actual_returns_row = data.get(timestamp - 1)?;
        let actual_returns_vec: Vec<f32> = actual_returns_row.to_vec1()?;
        let actual = actual_returns_vec[crypto_idx] as f64;
        Ok(predicted - actual)
    }

    /// Calculate trading confidence - simplified for more aggressive trading
    fn calculate_trading_confidence(&self, data: &Tensor, timestamp: usize, crypto_idx: usize) -> Result<f64> {
        if timestamp < SEQUENCE_LENGTH + self.confidence_lookback {
            return Ok(0.8); // High default confidence
        }

        // Get crypto performance metrics
        let perf = self.analyzer.get_crypto_performance(crypto_idx).unwrap();

        // Use crypto's selection quality as primary confidence
        // Add small boost for any selected crypto
        let confidence = (perf.tradeable_score + 0.3).min(1.0);
        Ok(confidence)
    }

    /// Generate trading signals using individual masking approach on selected cryptos
    fn generate_signals(&self, data: &Tensor, timestamp: usize, backtester: &Backtester) -> Result<Vec<(usize, TradeSide, f64)>> {
        let mut signals = Vec::new();

        if timestamp < SEQUENCE_LENGTH + self.confidence_lookback {
            return Ok(signals);
        }

        // Check current portfolio exposure
        let current_exposure = self.calculate_current_exposure(backtester);
        if current_exposure >= self.max_portfolio_exposure {
            return Ok(signals);
        }

        // Generate signals for each selected crypto using individual masking
        for &crypto_idx in &self.analyzer.selected_cryptos {
            // Get individual divergence (mask only this crypto)
            let divergence = self.calculate_individual_divergence(data, timestamp, crypto_idx)?;

            // Only trade if divergence exceeds threshold
            if divergence.abs() > self.divergence_threshold {
                // Calculate trading confidence
                let confidence = self.calculate_trading_confidence(data, timestamp, crypto_idx)?;

                // Very low confidence threshold to ensure trading
                if confidence < 0.2 {
                    continue;
                }

                if divergence < 0.0 {
                    // Model underestimated -> expect catch-up -> BUY
                    let expected_return = divergence.abs();

                    if expected_return > self.min_return_threshold {
                        // Get crypto performance for position sizing
                        let perf = self.analyzer.get_crypto_performance(crypto_idx).unwrap();

                        // Position sizing based on signal strength and quality
                        let signal_strength = (expected_return / self.divergence_threshold).min(2.0);
                        let quality_multiplier = perf.tradeable_score;
                        let position_multiplier = confidence * signal_strength * quality_multiplier;

                        let position_size = (self.base_position_size * position_multiplier).min(self.max_position_size);

                        // Ensure we don't exceed max exposure
                        let remaining_exposure = self.max_portfolio_exposure - current_exposure;
                        let final_position_size = position_size.min(remaining_exposure);

                        if final_position_size > 0.02 { // Minimum 2% position
                            signals.push((crypto_idx, TradeSide::Buy, final_position_size));
                        }
                    }
                } else {
                    // Model overestimated -> expect reversion -> SELL existing positions
                    let symbol = format!("CRYPTO_{}", crypto_idx);
                    let current_portfolio = backtester.get_current_portfolio();
                    if let Some(position) = current_portfolio.positions.get(&symbol) {
                        if position.quantity > 0.0 {
                            // Conservative sell sizing
                            let position_size = (self.base_position_size * confidence * 0.8).min(self.max_position_size * 0.8);
                            if position_size > 0.02 {
                                signals.push((crypto_idx, TradeSide::Sell, position_size));
                            }
                        }
                    }
                }
            }
        }

        Ok(signals)
    }

    /// Calculate current portfolio exposure
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
}

fn main() -> Result<()> {
    println!("📊 SELECTIVE CRYPTO QUANTITATIVE ANALYSIS");
    println!("======================================================================");
    println!("This analysis masks each crypto individually to find the best performers,");
    println!("then trades only the most reliable divergence patterns.");
    println!("======================================================================");

    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model_10.safetensors";
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

    // Initialize selective crypto analyzer
    let mut analyzer = SelectiveCryptoAnalyzer::new(model, device.clone(), num_time_series);

    // Use small but sufficient analysis period and maximum trading period
    let analysis_start = 0;
    let analysis_end = 1000; // 1000 timesteps for meaningful analysis (still small)
    let trading_start = analysis_end;
    let trading_end = test_timesteps - 1; // Rest for trading

    println!("Test set split:");
    println!("  - Analysis period: {} to {} ({} timesteps)", analysis_start, analysis_end, analysis_end - analysis_start);
    println!("  - Trading period: {} to {} ({} timesteps)", trading_start, trading_end, trading_end - trading_start);

    analyzer.analyze_all_cryptos(&test_data, analysis_start, analysis_end)?;

    if analyzer.selected_cryptos.is_empty() {
        println!("❌ No cryptocurrencies met the selection criteria!");
        return Ok(());
    }

    // Initialize selective trading strategy
    println!("\n💰 SELECTIVE TRADING STRATEGY (LONG-ONLY)");
    println!("======================================================================");
    println!("Trading only the {} best-performing cryptocurrencies", analyzer.selected_cryptos.len());

    // Much more aggressive parameters to ensure trading activity
    let divergence_threshold = 0.001; // 0.1% divergence threshold (very low)
    let base_position_size = 0.05; // 5% base position size
    let min_return_threshold = 0.0001; // 0.01% minimum expected return (very low)
    let strategy = SelectiveTradingStrategy::new(analyzer, divergence_threshold, base_position_size, min_return_threshold);

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

    println!("🚀 Running aggressive selective crypto backtest...");
    println!("  - Selected cryptos: {}", strategy.analyzer.selected_cryptos.len());
    println!("  - Divergence threshold: {:.3}% (very low)", divergence_threshold * 100.0);
    println!("  - Base position size: {:.1}%", base_position_size * 100.0);
    println!("  - Max position size: {:.1}%", strategy.max_position_size * 100.0);
    println!("  - Max portfolio exposure: {:.1}%", strategy.max_portfolio_exposure * 100.0);
    println!("  - Min return threshold: {:.3}% (very low)", min_return_threshold * 100.0);
    println!("  - Trading period: {} to {} ({} timesteps)",
             trading_start, trading_end, trading_end - trading_start);

    let mut total_trades = 0;
    let mut successful_trades = 0;
    let mut trades_by_crypto: HashMap<usize, usize> = HashMap::new();
    let mut buy_trades = 0;
    let mut sell_trades = 0;
    let mut last_progress_time = trading_start;

    // Run backtest on second half of test data
    for timestamp in trading_start..trading_end {
        // Step forward to update prices
        backtester.step_forward(timestamp)?;

        // Generate trading signals
        if let Ok(signals) = strategy.generate_signals(&test_data, timestamp, &backtester) {
            for (crypto_idx, side, size) in signals {
                let symbol = &symbol_names[crypto_idx];

                // Calculate position size in shares
                let current_portfolio_value = backtester.portfolio_history.last().unwrap().total_value;
                let position_value = current_portfolio_value * size;
                let current_price = backtester.current_prices[crypto_idx];
                let shares = position_value / current_price;

                // Execute trade
                if let Ok(_) = backtester.execute_trade(symbol, side.clone(), shares, timestamp) {
                    total_trades += 1;
                    *trades_by_crypto.entry(crypto_idx).or_insert(0) += 1;

                    // Track trade types
                    match side {
                        TradeSide::Buy => buy_trades += 1,
                        TradeSide::Sell => sell_trades += 1,
                    }

                    // Check if this trade will be profitable (simplified check)
                    if timestamp + 5 < trading_end {
                        let future_return = test_data.get(timestamp + 4)?.get(crypto_idx)?.to_scalar::<f32>()? as f64;
                        let expected_profit = match side {
                            TradeSide::Buy => future_return > 0.0,
                            TradeSide::Sell => true, // Assume sells are position management
                        };
                        if expected_profit {
                            successful_trades += 1;
                        }
                    }
                }
            }
        }

        // Print progress every 1000 timesteps with more trading info
        if timestamp % 1000 == 0 {
            let current_value = backtester.portfolio_history.last().unwrap().total_value;
            let return_pct = (current_value - initial_capital) / initial_capital * 100.0;
            let trades_in_period = total_trades - (if last_progress_time == trading_start { 0 } else {
                trades_by_crypto.values().sum::<usize>() - total_trades
            });
            let current_exposure = strategy.calculate_current_exposure(&backtester);

            println!("  Timestamp {}: Portfolio ${:.2} ({:+.2}%), Trades: {} (B:{} S:{}), Exposure: {:.1}%",
                     timestamp, current_value, return_pct, total_trades, buy_trades, sell_trades, current_exposure * 100.0);
            last_progress_time = timestamp;
        }
    }

    // Calculate final performance metrics
    let metrics = backtester.calculate_metrics()?;

    println!("\n📈 SELECTIVE STRATEGY RESULTS");
    println!("======================================================================");
    println!("💰 Financial Performance:");
    println!("  - Initial capital: ${:.2}", initial_capital);
    println!("  - Final value: ${:.2}", metrics.final_portfolio_value);
    println!("  - Total return: {:.2}%", metrics.total_return * 100.0);
    println!("  - Sharpe ratio: {:.3}", metrics.sharpe_ratio);
    println!("  - Max drawdown: {:.2}%", metrics.max_drawdown * 100.0);

    println!("\n📊 Trading Statistics:");
    println!("  - Total trades: {} (Buy: {}, Sell: {})", total_trades, buy_trades, sell_trades);
    println!("  - Successful trades: {} ({:.1}%)",
             successful_trades,
             if total_trades > 0 { successful_trades as f64 / total_trades as f64 * 100.0 } else { 0.0 });
    println!("  - Total fees paid: ${:.2}", metrics.total_fees);
    println!("  - Average trades per 1000 timesteps: {:.1}",
             if trading_end > trading_start { total_trades as f64 / ((trading_end - trading_start) as f64 / 1000.0) } else { 0.0 });

    println!("\n🎯 Trades by Selected Crypto:");
    for &crypto_idx in &strategy.analyzer.selected_cryptos {
        let trade_count = trades_by_crypto.get(&crypto_idx).unwrap_or(&0);
        let perf = strategy.analyzer.get_crypto_performance(crypto_idx).unwrap();
        println!("  - CRYPTO_{}: {} trades (Score: {:.3})", crypto_idx, trade_count, perf.tradeable_score);
    }

    println!("\n💡 BEST MODEL PREDICTIONS STRATEGY INSIGHTS:");
    println!("======================================================================");
    println!("This strategy focuses purely on the best model prediction performance:");
    println!("  - NO EXCLUSIONS: All cryptos considered based on model performance");
    println!("  - BEST PREDICTIONS: Selects top 5 cryptos by tradeable score");
    println!("  - INDIVIDUAL MASKING: Each crypto masked separately (preserves context)");
    println!("  - AGGRESSIVE TRADING: Very low thresholds to maximize trading activity");
    println!("  - SMALL ANALYSIS WINDOW: 1000 timesteps for analysis, rest for trading");
    println!("  - PERFORMANCE FOCUS: Trades where model predictions are most reliable");

    println!("\n✅ Selective quantitative analysis complete!");

    Ok(())
}
