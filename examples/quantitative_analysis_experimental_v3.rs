use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::extract_test_split;
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
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


const SEQUENCE_LENGTH: usize = 240;
const MODEL_DIMS: usize = 384;
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;

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

    /// Create masked input where ONLY the specified crypto is masked (for individual correlation analysis)
    fn create_single_masked_input(&self, data: &Tensor, timestamp: usize, target_crypto_idx: usize) -> Result<Tensor> {
        if timestamp < SEQUENCE_LENGTH {
            return Err(candle_core::Error::Msg("Not enough history for sequence".to_string()));
        }

        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let mut masked_sequence = input_sequence.clone();

        // Zero out ONLY the target cryptocurrency
        let zeros = Tensor::zeros((SEQUENCE_LENGTH, 1), DType::F32, &self.device)?;

        // Create a slice for the specific crypto column and replace with zeros
        let before_cols = if target_crypto_idx > 0 {
            Some(masked_sequence.narrow(1, 0, target_crypto_idx)?)
        } else {
            None
        };
        let after_cols = if target_crypto_idx + 1 < masked_sequence.dims()[1] {
            Some(masked_sequence.narrow(1, target_crypto_idx + 1, masked_sequence.dims()[1] - target_crypto_idx - 1)?)
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

    /// Get prediction for a SINGLE crypto (masking only that crypto, keeping all others with real data)
    fn get_single_crypto_prediction(&self, data: &Tensor, timestamp: usize, target_crypto_idx: usize) -> Result<f64> {
        let masked_input = self.create_single_masked_input(data, timestamp, target_crypto_idx)?;
        let input_batch = masked_input.unsqueeze(0)?; // Add batch dimension

        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;

        // Extract predictions for the last timestep
        let last_timestep_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;

        // Return prediction for the target crypto
        Ok(predictions_vec[target_crypto_idx] as f64)
    }

    /// Get predictions for ALL cryptos (not just blacked-out ones) - DEPRECATED: Uses multi-masking
    fn get_all_predictions(&self, data: &Tensor, timestamp: usize) -> Result<Vec<f64>> {
        let masked_input = self.create_masked_input(data, timestamp)?;
        let input_batch = masked_input.unsqueeze(0)?; // Add batch dimension

        // Get model predictions
        let predictions = self.model.forward(&input_batch)?;

        // Extract predictions for the last timestep
        let last_timestep_predictions = predictions.get(0)?.get(SEQUENCE_LENGTH - 1)?;
        let predictions_vec: Vec<f32> = last_timestep_predictions.to_vec1()?;

        // Return predictions for all cryptos
        Ok(predictions_vec.iter().map(|&x| x as f64).collect())
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

    /// Analyze cross-sectional inference quality for ALL cryptos (individual masking)
    fn analyze_inference_quality(&self, data: &Tensor, start_time: usize, end_time: usize) -> Result<()> {
        println!("\n🔍 CORRELATION ANALYSIS FOR ALL CRYPTOS (INDIVIDUAL MASKING)");
        println!("======================================================================");
        println!("Note: Each crypto is masked individually while all others use real data");

        let num_cryptos = data.dims()[1]; // Get total number of cryptos

        // Calculate evenly spaced sample points instead of going timestep by timestep
        let total_range = end_time - start_time;
        let num_samples = 500; // Number of evenly spaced samples
        let step_size = (total_range as f64 / num_samples as f64).max(1.0) as usize;

        let sample_timestamps: Vec<usize> = (0..num_samples)
            .map(|i| start_time + (i * step_size))
            .filter(|&t| t < end_time && t >= SEQUENCE_LENGTH)
            .collect();

        println!("Using {} evenly spaced samples from {} to {} (step size: {})",
                 sample_timestamps.len(), start_time, end_time, step_size);

        let mut all_correlations = Vec::new();

        // Calculate correlation for each crypto (masking only that crypto)
        for crypto_idx in 0..num_cryptos {
            let mut predictions = Vec::new();
            let mut actuals = Vec::new();

            for &timestamp in &sample_timestamps {
                // Get prediction for THIS crypto only (masking only this crypto)
                if let Ok(prediction) = self.get_single_crypto_prediction(data, timestamp, crypto_idx) {
                    if let Ok(actual_row) = data.get(timestamp) {
                        if let Ok(actual_vec) = actual_row.to_vec1::<f32>() {
                            predictions.push(prediction);
                            actuals.push(actual_vec[crypto_idx] as f64);
                        }
                    }
                }
            }

            if predictions.is_empty() {
                println!("❌ No data available for CRYPTO_{}", crypto_idx);
                continue;
            }

            let correlation = self.calculate_correlation(&predictions, &actuals);
            all_correlations.push(correlation);

            // Calculate basic statistics
            let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
            let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
            let mean_divergence = predictions.iter().zip(actuals.iter())
                .map(|(p, a)| p - a)
                .sum::<f64>() / predictions.len() as f64;

            println!("\n📊 CRYPTO_{} (Index: {}) - INDIVIDUALLY MASKED:", crypto_idx, crypto_idx);
            println!("  - Correlation: {:.4}", correlation);
            println!("  - Mean predicted: {:.6}", mean_pred);
            println!("  - Mean actual: {:.6}", mean_actual);
            println!("  - Mean divergence: {:.6}", mean_divergence);
            println!("  - Data points: {}", predictions.len());

            // Interpretation
            if correlation.abs() > 0.1 {
                println!("  ✅ Strong cross-sectional predictive signal");
            } else if correlation.abs() > 0.05 {
                println!("  ⚠️  Weak cross-sectional predictive signal");
            } else {
                println!("  ❌ No meaningful cross-sectional predictive relationship");
            }
        }

        // Summary statistics using individual masking and evenly spaced samples
        println!("\n📈 SUMMARY STATISTICS (INDIVIDUAL MASKING, EVENLY SPACED SAMPLES):");
        println!("======================================================================");

        if !all_correlations.is_empty() {
            let mean_corr = all_correlations.iter().sum::<f64>() / all_correlations.len() as f64;
            let max_corr = all_correlations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_corr = all_correlations.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            // Count strong correlations
            let strong_correlations = all_correlations.iter().filter(|&&c| c.abs() > 0.1).count();
            let weak_correlations = all_correlations.iter().filter(|&&c| c.abs() > 0.05 && c.abs() <= 0.1).count();
            let no_signal = all_correlations.len() - strong_correlations - weak_correlations;

            println!("🌍 ALL CRYPTOS ({} total) - INDIVIDUAL MASKING RESULTS:", all_correlations.len());
            println!("  - Mean correlation: {:.4}", mean_corr);
            println!("  - Max correlation: {:.4}", max_corr);
            println!("  - Min correlation: {:.4}", min_corr);
            println!("  - Strong signals (|corr| > 0.1): {} ({:.1}%)",
                     strong_correlations,
                     strong_correlations as f64 / all_correlations.len() as f64 * 100.0);
            println!("  - Weak signals (0.05 < |corr| ≤ 0.1): {} ({:.1}%)",
                     weak_correlations,
                     weak_correlations as f64 / all_correlations.len() as f64 * 100.0);
            println!("  - No signal (|corr| ≤ 0.05): {} ({:.1}%)",
                     no_signal,
                     no_signal as f64 / all_correlations.len() as f64 * 100.0);
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



fn main() -> Result<()> {
    println!("📊 QUANTITATIVE ANALYSIS - Cross-Sectional Crypto Inference");
    println!("======================================================================");
    println!("This analysis focuses on inferring currency movements from others,");
    println!("not next-step prediction. Pure analysis without trading simulation.");
    println!("======================================================================");

    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let model_path = "current_model_large_r2_ep1.safetensors";

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
    let analysis_start = 0;
    let analysis_end = test_timesteps - 1;

    analyzer.analyze_inference_quality(&test_data, analysis_start, analysis_end)?;

    // Display detailed predicted vs real values for inspection
    analyzer.display_prediction_comparison(&test_data, analysis_start, analysis_end)?;

    println!("\n✅ Quantitative analysis complete!");

    Ok(())
}
