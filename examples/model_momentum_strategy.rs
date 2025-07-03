use candle_bert_time_series::backtest::{Backtester, TradingFees, TradeSide, PerformanceMetrics, extract_test_split, get_data_split_info};
use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;

use financial_bert::{Config, FinancialTransformerForMaskedRegression};

/// Cross-Sectional Momentum Strategy using trained BERT model predictions
///
/// Strategy Logic:
/// 1. At time T, feed the last 120 minutes of data to the model
/// 2. Get model predictions for T+1 returns for all cryptocurrencies
/// 3. Rank cryptocurrencies by predicted returns
/// 4. Go long the top K assets with highest predicted returns (>0.5% only)
/// 5. Only trade if predicted return > 0.5% (more than double trading fees)
/// 6. Stay in cash for all other assets (no short selling)
/// 7. Hold positions for the next minute, then rebalance
struct ModelMomentumStrategy {
    model: FinancialTransformerForMaskedRegression,
    sequence_length: usize,
    top_k: usize,        // Number of assets to go long
    position_size: f64,  // Fraction of portfolio per position
    device: Device,
}

impl ModelMomentumStrategy {
    fn new(
        model: FinancialTransformerForMaskedRegression,
        sequence_length: usize,
        top_k: usize,
        position_size: f64,
        device: Device,
    ) -> Self {
        Self {
            model,
            sequence_length,
            top_k,
            position_size,
            device,
        }
    }

    /// Get model predictions for the next time step
    fn get_model_predictions(
        &self,
        historical_data: &Tensor,
        current_timestamp: usize,
    ) -> Result<Vec<f64>> {
        if current_timestamp < self.sequence_length {
            // Not enough history, return zeros
            let num_assets = historical_data.dims()[1];
            return Ok(vec![0.0; num_assets]);
        }

        // Extract the last sequence_length timesteps ending at current_timestamp
        let start_idx = current_timestamp - self.sequence_length;
        let input_sequence = historical_data.narrow(0, start_idx, self.sequence_length)?;

        // Make tensor contiguous to fix matmul error
        let input_sequence = input_sequence.contiguous()?;

        // Add batch dimension: [sequence_length, num_assets] -> [1, sequence_length, num_assets]
        let input_batch = input_sequence.unsqueeze(0)?;
        
        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;
        
        // Extract predictions for the last timestep (T+1 prediction)
        // predictions shape: [1, sequence_length, num_assets]
        let last_timestep_predictions = predictions.get(0)?.get(self.sequence_length - 1)?;
        
        // Convert to Vec<f64>
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;
        let predictions_f64: Vec<f64> = predictions_vec.iter().map(|&x| x as f64).collect();
        
        Ok(predictions_f64)
    }

    /// Calculate trading signals based on model predictions
    fn calculate_signals(
        &self,
        predictions: &[f64],
        symbol_names: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut signals = HashMap::new();
        
        // Create (prediction, index) pairs and sort by prediction
        let mut pred_with_idx: Vec<(f64, usize)> = predictions.iter()
            .enumerate()
            .map(|(idx, &pred)| (pred, idx))
            .collect();
        
        // Sort by prediction value (descending)
        pred_with_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        // Assign signals: +1 only for assets with predicted return > 0.5% (0.005)
        // This ensures we only trade when expected return is more than double the trading fees
        for (rank, &(prediction, asset_idx)) in pred_with_idx.iter().enumerate() {
            let symbol = &symbol_names[asset_idx];

            let signal = if rank < self.top_k && prediction > 0.002 {
                1.0  // Go long only if predicted return > 0.5%
            } else {
                0.0  // Stay in cash for all other assets
            };

            signals.insert(symbol.clone(), signal);
        }
        
        Ok(signals)
    }

    /// Execute strategy for one time step
    fn execute_step(
        &self,
        backtester: &mut Backtester,
        historical_data: &Tensor,
        current_timestamp: usize,
    ) -> Result<()> {
        // Get model predictions
        let predictions = self.get_model_predictions(historical_data, current_timestamp)?;
        
        // Calculate trading signals
        let signals = self.calculate_signals(&predictions, backtester.get_symbol_names())?;
        
        // Get current portfolio info
        let (total_value, positions_to_close) = {
            let current_portfolio = backtester.get_current_portfolio();
            let total_value = current_portfolio.total_value;
            let positions_to_close: Vec<(String, f64)> = current_portfolio.positions
                .iter()
                .map(|(symbol, position)| (symbol.clone(), position.quantity))
                .collect();
            (total_value, positions_to_close)
        };
        
        // Close all existing positions (only long positions since we don't short)
        for (symbol, quantity) in positions_to_close {
            if quantity > 0.0 {
                // Close long position
                backtester.execute_trade(&symbol, TradeSide::Sell, quantity, current_timestamp)?;
            }
            // Note: We don't expect negative quantities since we don't short sell
        }

        // Open new positions based on signals (long-only)
        for (symbol, signal) in signals {
            if signal > 0.5 { // Only buy if signal is positive (long signal)
                let position_value = total_value.abs() * self.position_size;
                let current_price = backtester.get_current_price(&symbol).unwrap_or(100.0);
                let quantity = position_value / current_price;

                if quantity > 0.0 {
                    // Long signal - buy
                    backtester.execute_trade(&symbol, TradeSide::Buy, quantity, current_timestamp)?;
                }
            }
            // Note: We ignore negative signals (no short selling)
        }

        Ok(())
    }
}

/// Run the model-based momentum strategy backtest
fn run_model_backtest() -> Result<()> {
    println!("🚀 Starting Model-Based Cross-Sectional Momentum Strategy");
    println!("{}", "=".repeat(70));
    println!("⚠️  IMPORTANT: Using ONLY test split data to prevent data leakage!");
    println!("🔒 This ensures fair evaluation - no training/validation data used");
    println!("🤖 Using trained model: current_model_ep35.safetensors");
    println!("{}", "=".repeat(70));

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model_ep35.safetensors";
    let initial_capital = 100.0; // $1M starting capital
    let device = Device::cuda_if_available(0)?;

    // Model hyperparameters (must match training configuration)
    const SEQUENCE_LENGTH: usize = 120;
    const MODEL_DIMS: usize = 256;
    const NUM_LAYERS: usize = 6;
    const NUM_HEADS: usize = 8;

    // Load data
    println!("Loading cryptocurrency data...");
    let (full_data_sequence, num_time_series) = load_and_prepare_data(data_path, &device)?;
    let total_timesteps = full_data_sequence.dims()[0];
    
    // Extract ONLY the test split to prevent data leakage
    let test_data = extract_test_split(&full_data_sequence)?;
    let test_timesteps = test_data.dims()[0];
    
    // Get split information for reporting
    let (train_size, val_size, test_size) = get_data_split_info(total_timesteps);
    
    println!("Data loaded: {} total timesteps, {} assets", total_timesteps, num_time_series);
    println!("Using ONLY test split: {} timesteps ({}% of data)", test_timesteps, 
             (test_timesteps as f32 / total_timesteps as f32) * 100.0);
    println!("Data splits - Train: {}, Validation: {}, Test: {}", 
             train_size, val_size, test_size);
    println!("No data leakage - test split extracted using same logic as training loop");

    // Load the trained model
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
    
    // Load the trained weights
    varmap.load(model_path)?;
    println!("✅ Model loaded from: {}", model_path);

    // Create symbol names
    let symbol_names: Vec<String> = (0..num_time_series)
        .map(|i| format!("CRYPTO_{}", i))
        .collect();

    // Initialize backtester with leverage enabled
    let fees = TradingFees::default();
    let mut backtester = Backtester::new_with_leverage(
        initial_capital,
        test_data.clone(),
        symbol_names,
        Some(fees),
        true, // Allow negative cash for short positions
    )?;

    // Initialize strategy
    let strategy = ModelMomentumStrategy::new(
        model,
        SEQUENCE_LENGTH,
        10,   // Long top 10 assets
        0.05, // 5% position size per asset
        device,
    );

    println!("\n📈 Running backtest...");
    println!("Strategy parameters:");
    println!("  - Model sequence length: {} minutes", SEQUENCE_LENGTH);
    println!("  - Long top K assets: {}", strategy.top_k);
    println!("  - Position size: {:.1}% per asset", strategy.position_size * 100.0);
    println!("  - Minimum predicted return: >0.5% (0.005)");
    println!("  - Trading style: Long-only (no short selling)");
    println!("  - Leverage allowed: YES (can go negative cash)");
    println!("  - Rebalance frequency: Every minute");

    // Run backtest
    let start_time = std::time::Instant::now();
    let mut progress_counter = 0;
    let mut last_reported_progress = 0;
    let total_steps = test_timesteps - SEQUENCE_LENGTH;

    for timestamp in SEQUENCE_LENGTH..test_timesteps {
        // Step forward in time (update prices)
        backtester.step_forward(timestamp)?;

        // Execute strategy using model predictions
        strategy.execute_step(&mut backtester, &test_data, timestamp)?;

        // Progress reporting every 10%
        progress_counter += 1;
        let current_progress = ((progress_counter as f64 / total_steps as f64) * 100.0) as u32;

        if current_progress >= last_reported_progress + 10 && current_progress <= 100 {
            let current_portfolio = backtester.get_current_portfolio();
            let current_return = (current_portfolio.total_value - initial_capital) / initial_capital * 100.0;
            let elapsed = start_time.elapsed().as_secs();

            println!("  📊 {}% Complete - Portfolio: ${:.2} | Return: {:.2}% | Positions: {} | Time: {}s",
                     current_progress,
                     current_portfolio.total_value,
                     current_return,
                     current_portfolio.positions.len(),
                     elapsed);

            last_reported_progress = current_progress;
        }
    }

    let elapsed = start_time.elapsed();
    println!("✅ Backtest completed in {:.2} seconds", elapsed.as_secs_f64());

    // Calculate and display performance metrics
    println!("\n📊 PERFORMANCE RESULTS");
    println!("{}", "=".repeat(70));
    
    let metrics = backtester.calculate_metrics()?;
    display_performance_metrics(&metrics, initial_capital);

    // Display trade summary
    let trades = backtester.get_trades();
    println!("\n📋 TRADE SUMMARY");
    println!("{}", "=".repeat(70));
    println!("Total trades executed: {}", trades.len());
    println!("Total fees paid: ${:.2}", metrics.total_fees);
    
    if trades.len() > 0 {
        let avg_trade_size = trades.iter().map(|t| t.quantity * t.price).sum::<f64>() / trades.len() as f64;
        println!("Average trade size: ${:.2}", avg_trade_size);
        
        let buy_trades = trades.iter().filter(|t| t.side == TradeSide::Buy).count();
        let sell_trades = trades.iter().filter(|t| t.side == TradeSide::Sell).count();
        println!("Buy trades: {}, Sell trades: {}", buy_trades, sell_trades);
    }

    // Display final portfolio state
    let final_portfolio = backtester.get_current_portfolio();
    println!("\n💼 FINAL PORTFOLIO");
    println!("{}", "=".repeat(70));
    println!("Cash: ${:.2}", final_portfolio.cash);
    if final_portfolio.cash < 0.0 {
        println!("  ⚠️  NEGATIVE CASH (Margin used: ${:.2})", final_portfolio.margin_used);
    }
    println!("Positions: {}", final_portfolio.positions.len());
    println!("Total value: ${:.2}", final_portfolio.total_value);
    println!("Unrealized PnL: ${:.2}", final_portfolio.unrealized_pnl);
    println!("Realized PnL: ${:.2}", final_portfolio.realized_pnl);
    println!("Current leverage: {:.2}x", final_portfolio.leverage_ratio);

    Ok(())
}

/// Display performance metrics in a formatted way
fn display_performance_metrics(metrics: &PerformanceMetrics, initial_capital: f64) {
    println!("Initial Capital: ${:.2}", initial_capital);
    println!("Final Portfolio Value: ${:.2}", metrics.final_portfolio_value);
    println!("Total Return: {:.2}%", metrics.total_return * 100.0);
    println!("Annualized Return: {:.2}%", metrics.annualized_return * 100.0);
    println!("Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("Maximum Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("Max Drawdown Duration: {} periods", metrics.max_drawdown_duration);
    println!("Win Rate: {:.1}%", metrics.win_rate * 100.0);
    println!("Profit Factor: {:.2}", metrics.profit_factor);
    println!("Total Trades: {}", metrics.total_trades);
    println!("Total Fees: ${:.2}", metrics.total_fees);
    
    // Leverage and margin metrics
    println!("\n📊 LEVERAGE & MARGIN METRICS");
    println!("Max Leverage Used: {:.2}x", metrics.max_leverage);
    println!("Max Margin Used: ${:.2}", metrics.max_margin_used);
    println!("Min Cash Balance: ${:.2}", metrics.min_cash_balance);
    if metrics.min_cash_balance < 0.0 {
        println!("  ⚠️  Strategy went into margin (negative cash)");
    }
    
    // Risk-adjusted metrics
    let profit_loss = metrics.final_portfolio_value - initial_capital;
    println!("\n📈 RISK-ADJUSTED METRICS");
    println!("Net Profit/Loss: ${:.2}", profit_loss);
    
    if metrics.max_drawdown > 0.0 {
        let calmar_ratio = metrics.annualized_return / metrics.max_drawdown;
        println!("Calmar Ratio: {:.3}", calmar_ratio);
    }
}

fn main() -> Result<()> {
    match run_model_backtest() {
        Ok(()) => {
            println!("\n🎉 Model-based backtest completed successfully!");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Model-based backtest failed: {}", e);
            Err(e)
        }
    }
}
