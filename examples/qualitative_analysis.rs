use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::extract_test_split;
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

/// Qualitative Analysis Tool for BERT Model Predictions - The "Sanity Check"
///
/// Before you even think about profit and loss, you must understand what your model is predicting.
/// Does it make sense?
///
/// This tool performs three key analyses:
/// 1. Pick a Crypto and Plot: Choose well-known cryptos (SOL, AVAX, etc.) and compare
///    model predictions vs actual returns over time
/// 2. Look for Correlation, Not Equality: Check directional correctness - when actual
///    return spikes positive, was the model's prediction also positive?
/// 3. Test a Historical Event: Find major market events and see if the model "saw"
///    the instability coming

const SEQUENCE_LENGTH: usize = 120;
const MODEL_DIMS: usize = 256;
const NUM_LAYERS: usize = 6;
const NUM_HEADS: usize = 8;

struct QualitativeAnalyzer {
    model: FinancialTransformerForMaskedRegression,
    device: Device,
}

impl QualitativeAnalyzer {
    fn new(model: FinancialTransformerForMaskedRegression, device: Device) -> Self {
        Self { model, device }
    }

    /// Get model prediction for a single timestep
    /// Feed the model sequence t-120 to t-1, ask it to predict returns at time t
    fn get_single_prediction(
        &self,
        data: &Tensor,
        timestamp: usize,
    ) -> Result<Vec<f64>> {
        if timestamp < SEQUENCE_LENGTH {
            let num_assets = data.dims()[1];
            return Ok(vec![0.0; num_assets]);
        }

        // Extract the last sequence_length timesteps ending at timestamp-1
        // (we want to predict timestamp, so we use data up to timestamp-1)
        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let input_sequence = input_sequence.contiguous()?;

        // Add batch dimension: [sequence_length, num_assets] -> [1, sequence_length, num_assets]
        let input_batch = input_sequence.unsqueeze(0)?;

        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;

        // Extract predictions for the last timestep (T+1 prediction)
        // predictions shape: [1, sequence_length, num_assets]
        let last_timestep_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;

        // Convert to Vec<f64>
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;
        let predictions_f64: Vec<f64> = predictions_vec.iter().map(|&x| x as f64).collect();

        Ok(predictions_f64)
    }

    /// Get model predictions for a specific time range
    fn get_predictions_for_range(
        &self,
        data: &Tensor,
        start_timestamp: usize,
        end_timestamp: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let mut all_predictions = Vec::new();

        for timestamp in start_timestamp..end_timestamp {
            let predictions = self.get_single_prediction(data, timestamp)?;
            all_predictions.push(predictions);
        }

        Ok(all_predictions)
    }

    /// Analyze a specific cryptocurrency over a time period
    /// This is the core "Pick a Crypto and Plot" analysis
    fn analyze_crypto(
        &self,
        data: &Tensor,
        crypto_name: &str,
        crypto_idx: usize,
        start_timestamp: usize,
        end_timestamp: usize,
        symbol_names: &[String],
    ) -> Result<()> {
        println!("\n🔍 ANALYZING {} (Index: {})", crypto_name, crypto_idx);
        println!("======================================================================");
        println!("📊 Performing 'Pick a Crypto and Plot' analysis...");
        println!("   - Feeding model sequences t-120 to t-1");
        println!("   - Asking it to predict returns at time t");
        println!("   - Comparing with actual returns");

        let predictions = self.get_predictions_for_range(data, start_timestamp, end_timestamp)?;

        // Extract actual returns for this crypto
        let mut actual_returns = Vec::new();
        let mut predicted_returns = Vec::new();
        let mut timestamps = Vec::new();

        for (i, timestamp) in (start_timestamp..end_timestamp).enumerate() {
            if timestamp < SEQUENCE_LENGTH || i >= predictions.len() {
                continue;
            }

            // Get actual return at this timestamp
            let actual_return: f64 = data.get(timestamp)?.get(crypto_idx)?.to_scalar::<f32>()? as f64;
            let predicted_return = predictions[i][crypto_idx];

            actual_returns.push(actual_return);
            predicted_returns.push(predicted_return);
            timestamps.push(timestamp);
        }

        // Calculate correlation
        let correlation = self.calculate_correlation(&predicted_returns, &actual_returns);

        // Calculate directional accuracy
        let directional_accuracy = self.calculate_directional_accuracy(&predicted_returns, &actual_returns);

        // Find significant events
        let significant_events = self.find_significant_events(&actual_returns, &predicted_returns, &timestamps);

        println!("\n📊 ANALYSIS RESULTS:");
        println!("  - Total data points: {}", actual_returns.len());
        println!("  - Correlation: {:.4}", correlation);
        println!("  - Directional accuracy: {:.2}%", directional_accuracy * 100.0);
        println!("  - Significant events found: {}", significant_events.len());

        // Interpretation guidance
        println!("\n💡 INTERPRETATION:");
        if correlation > 0.1 {
            println!("  ✅ Correlation > 0.1: Model shows some predictive signal");
        } else if correlation > 0.05 {
            println!("  ⚠️  Correlation > 0.05: Weak but potentially useful signal");
        } else {
            println!("  ❌ Correlation ≤ 0.05: Very weak or no predictive signal");
        }

        if directional_accuracy > 0.55 {
            println!("  ✅ Directional accuracy > 55%: Better than random");
        } else if directional_accuracy > 0.50 {
            println!("  ⚠️  Directional accuracy > 50%: Slightly better than random");
        } else {
            println!("  ❌ Directional accuracy ≤ 50%: No better than random guessing");
        }

        // Print sample predictions vs reality
        println!("\n📈 SAMPLE PREDICTIONS vs REALITY (First 20 points):");
        println!("  Timestamp | Predicted | Actual   | Direction Match | Magnitude");
        println!("  ----------|-----------|----------|----------------|----------");

        for i in 0..20.min(predicted_returns.len()) {
            let pred = predicted_returns[i];
            let actual = actual_returns[i];
            let direction_match = (pred > 0.0) == (actual > 0.0);
            let match_symbol = if direction_match { "✓" } else { "✗" };
            let magnitude = if actual.abs() > 0.01 { "HIGH" } else { "LOW" };

            println!("  {:9} | {:8.4} | {:8.4} | {:14} | {}",
                     timestamps[i],
                     pred,
                     actual,
                     match_symbol,
                     magnitude);
        }

        // Print significant events
        if !significant_events.is_empty() {
            println!("\n🚨 SIGNIFICANT EVENTS (|actual return| > 1%):");
            println!("  Timestamp | Predicted | Actual   | Event Type | Model Response | Magnitude");
            println!("  ----------|-----------|----------|------------|---------------|----------");

            for event in significant_events.iter().take(10) {
                let event_type = if event.actual_return > 0.01 { "PUMP" } else { "CRASH" };
                let model_response = if (event.predicted_return > 0.0) == (event.actual_return > 0.0) {
                    "CORRECT"
                } else {
                    "WRONG"
                };
                let magnitude = if event.actual_return.abs() > 0.05 { "EXTREME" }
                               else if event.actual_return.abs() > 0.02 { "HIGH" }
                               else { "MEDIUM" };

                println!("  {:9} | {:8.4} | {:8.4} | {:8} | {:13} | {}",
                         event.timestamp,
                         event.predicted_return,
                         event.actual_return,
                         event_type,
                         model_response,
                         magnitude);
            }
        }

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

    /// Calculate directional accuracy (what % of time model gets direction right)
    fn calculate_directional_accuracy(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        if predicted.len() != actual.len() || predicted.is_empty() {
            return 0.0;
        }
        
        let correct_directions = predicted.iter()
            .zip(actual.iter())
            .filter(|(pred, act)| (pred > &&0.0) == (act > &&0.0))
            .count();
        
        correct_directions as f64 / predicted.len() as f64
    }

    /// Find significant market events (large price movements)
    fn find_significant_events(&self, actual: &[f64], predicted: &[f64], timestamps: &[usize]) -> Vec<SignificantEvent> {
        let mut events = Vec::new();

        for (i, (&actual_return, &predicted_return)) in actual.iter().zip(predicted.iter()).enumerate() {
            if actual_return.abs() > 0.01 { // More than 1% movement (lowered threshold)
                events.push(SignificantEvent {
                    timestamp: timestamps[i],
                    actual_return,
                    predicted_return,
                });
            }
        }

        // Sort by magnitude of actual return
        events.sort_by(|a, b| b.actual_return.abs().partial_cmp(&a.actual_return.abs()).unwrap());

        events
    }

    /// Test a specific historical event
    /// Find a major market event and see if the model "saw" the instability coming
    fn test_historical_event(
        &self,
        data: &Tensor,
        crypto_idx: usize,
        event_timestamp: usize,
        crypto_name: &str,
        lookback_window: usize,
    ) -> Result<()> {
        println!("\n🔍 HISTORICAL EVENT ANALYSIS for {}", crypto_name);
        println!("======================================================================");
        println!("📅 Event timestamp: {}", event_timestamp);
        println!("🔙 Looking back {} timesteps before the event", lookback_window);

        if event_timestamp < SEQUENCE_LENGTH + lookback_window {
            println!("❌ Not enough historical data for this event");
            return Ok(());
        }

        // Get the actual return at the event
        let event_return: f64 = data.get(event_timestamp)?.get(crypto_idx)?.to_scalar::<f32>()? as f64;
        println!("📊 Actual return at event: {:.4} ({:.2}%)", event_return, event_return * 100.0);

        // Get model predictions leading up to the event
        let start_analysis = event_timestamp - lookback_window;
        let mut predictions_leading_up = Vec::new();
        let mut actual_returns_leading_up = Vec::new();

        for t in start_analysis..event_timestamp {
            if t >= SEQUENCE_LENGTH {
                let pred = self.get_single_prediction(data, t)?;
                let actual: f64 = data.get(t)?.get(crypto_idx)?.to_scalar::<f32>()? as f64;

                predictions_leading_up.push(pred[crypto_idx]);
                actual_returns_leading_up.push(actual);
            }
        }

        // Get the model's prediction for the event itself
        let event_prediction = self.get_single_prediction(data, event_timestamp)?[crypto_idx];

        println!("\n🤖 Model's prediction for the event: {:.4} ({:.2}%)",
                 event_prediction, event_prediction * 100.0);

        // Check if model predicted the right direction
        let direction_correct = (event_prediction > 0.0) == (event_return > 0.0);
        let direction_symbol = if direction_correct { "✅" } else { "❌" };

        println!("🎯 Direction prediction: {} {}",
                 direction_symbol,
                 if direction_correct { "CORRECT" } else { "WRONG" });

        // Analyze the buildup
        println!("\n📈 BUILDUP ANALYSIS (last {} predictions before event):", lookback_window.min(10));
        println!("  Timestep | Predicted | Actual   | Trend");
        println!("  ---------|-----------|----------|------");

        let show_count = lookback_window.min(10);
        let start_show = predictions_leading_up.len().saturating_sub(show_count);

        for i in start_show..predictions_leading_up.len() {
            let pred = predictions_leading_up[i];
            let actual = actual_returns_leading_up[i];
            let trend = if pred.abs() > 0.005 {
                if pred > 0.0 { "UP" } else { "DOWN" }
            } else {
                "FLAT"
            };

            println!("  {:8} | {:8.4} | {:8.4} | {}",
                     start_analysis + i,
                     pred,
                     actual,
                     trend);
        }

        // Check if model showed increasing volatility/instability
        let recent_predictions = &predictions_leading_up[predictions_leading_up.len().saturating_sub(5)..];
        let avg_volatility: f64 = recent_predictions.iter().map(|x| x.abs()).sum::<f64>() / recent_predictions.len() as f64;

        println!("\n📊 INSTABILITY DETECTION:");
        println!("  Average prediction magnitude (last 5 steps): {:.4}", avg_volatility);

        if avg_volatility > 0.01 {
            println!("  ✅ Model detected increased volatility before event");
        } else {
            println!("  ❌ Model did not detect increased volatility");
        }

        Ok(())
    }
}

#[derive(Debug)]
struct SignificantEvent {
    timestamp: usize,
    actual_return: f64,
    predicted_return: f64,
}

fn main() -> Result<()> {
    println!("🔍 QUALITATIVE ANALYSIS - BERT Model Sanity Check");
    println!("======================================================================");
    println!("Before you even think about profit and loss, you must understand");
    println!("what your model is predicting. Does it make sense?");
    println!("======================================================================");
    println!("📋 This analysis performs three key checks:");
    println!("   1. Pick a Crypto and Plot: Compare predictions vs actual returns");
    println!("   2. Look for Correlation, Not Equality: Check directional correctness");
    println!("   3. Test a Historical Event: See if model 'saw' instability coming");

    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("\nUsing device: {:?}", device);

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model_ep35.safetensors";

    // Load data
    println!("\nLoading cryptocurrency data...");
    let (full_data_sequence, num_time_series) = load_and_prepare_data(data_path, &device)?;
    let total_timesteps = full_data_sequence.dims()[0];

    // Extract ONLY the test split to prevent data leakage
    let test_data = extract_test_split(&full_data_sequence)?;
    let test_timesteps = test_data.dims()[0];

    println!("Data loaded: {} total timesteps, {} assets", total_timesteps, num_time_series);
    println!("Using ONLY test split: {} timesteps", test_timesteps);

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

    // Load the trained weights
    varmap.load(model_path)?;
    println!("✅ Model loaded from: {}", model_path);

    // Initialize analyzer
    let analyzer = QualitativeAnalyzer::new(model, device);

    // Create symbol names (using generic names since we don't have actual crypto names)
    let symbol_names: Vec<String> = (0..num_time_series)
        .map(|i| format!("CRYPTO_{}", i))
        .collect();

    // Analyze specific cryptocurrencies (using indices for well-known cryptos)
    let cryptos_to_analyze = vec![
        ("SOL-like", 0),    // First crypto in dataset
        ("AVAX-like", 1),   // Second crypto in dataset
        ("BTC-like", 2),    // Third crypto in dataset
        ("ETH-like", 3),    // Fourth crypto in dataset
    ];

    // Find one month of data for analysis (approximately 43,200 minutes)
    let analysis_start = test_timesteps / 4; // Start at 25% of test data
    let analysis_end = (analysis_start + 10080).min(test_timesteps - 1); // One week of data

    println!("\n📅 STEP 1: Pick a Crypto and Plot Analysis");
    println!("Analyzing period: timestamps {} to {} ({} data points)",
             analysis_start, analysis_end, analysis_end - analysis_start);

    for (crypto_display_name, crypto_idx) in &cryptos_to_analyze {
        if *crypto_idx < num_time_series {
            analyzer.analyze_crypto(
                &test_data,
                crypto_display_name,
                *crypto_idx,
                analysis_start,
                analysis_end,
                &symbol_names,
            )?;
        } else {
            println!("\n⚠️  {} (Index: {}) not available in dataset", crypto_display_name, crypto_idx);
        }
    }

    // Historical event analysis
    println!("\n📅 STEP 3: Historical Event Analysis");
    println!("======================================================================");
    println!("Looking for major market events and testing model's response...");

    // Find some significant events in the test data
    let mut significant_events = Vec::new();
    for t in (analysis_start + SEQUENCE_LENGTH)..(analysis_end - 100) {
        for (crypto_name, crypto_idx) in &cryptos_to_analyze {
            if *crypto_idx < num_time_series {
                let return_val: f64 = test_data.get(t)?.get(*crypto_idx)?.to_scalar::<f32>()? as f64;
                if return_val.abs() > 0.03 { // 3% movement
                    significant_events.push((t, *crypto_idx, crypto_name, return_val));
                }
            }
        }
    }

    // Sort by magnitude and take top events
    significant_events.sort_by(|a, b| b.3.abs().partial_cmp(&a.3.abs()).unwrap());

    println!("Found {} significant events (>3% movement)", significant_events.len());

    // Analyze top 3 events
    for (i, (event_timestamp, crypto_idx, crypto_name, return_val)) in significant_events.iter().take(3).enumerate() {
        println!("\n--- Event {} ---", i + 1);
        println!("Event: {:.2}% movement in {}", return_val * 100.0, crypto_name);

        analyzer.test_historical_event(
            &test_data,
            *crypto_idx,
            *event_timestamp,
            crypto_name,
            20, // Look back 20 timesteps
        )?;
    }

    println!("\n✅ Qualitative analysis complete!");
    println!("\n💡 FINAL INTERPRETATION GUIDE:");
    println!("======================================================================");
    println!("✅ GOOD SIGNS:");
    println!("  - Correlation > 0.1: Model shows predictive signal");
    println!("  - Directional accuracy > 55%: Better than random");
    println!("  - Model correctly predicts direction of major events");
    println!("  - Model shows increased volatility before major moves");
    println!("\n❌ WARNING SIGNS:");
    println!("  - Correlation < 0.05: Very weak predictive power");
    println!("  - Directional accuracy < 50%: Worse than random");
    println!("  - Model consistently wrong on major events");
    println!("  - Predictions don't make intuitive sense");
    println!("\n🎯 NEXT STEPS:");
    println!("  - If results look good: Proceed to backtesting");
    println!("  - If results are poor: Retrain model or adjust features");
    println!("  - Always validate on multiple time periods and assets");

    Ok(())
}
