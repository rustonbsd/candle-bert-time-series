use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_bert_time_series::backtest::extract_test_split;
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{VarBuilder, VarMap};
// Removed unused rand imports
use plotters::prelude::*;
use plotters::element::Circle;

// Include the financial_bert module
#[path = "../src/financial_bert.rs"]
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

const SEQUENCE_LENGTH: usize = 240; // 240
const MODEL_DIMS: usize = 128; // 384
const NUM_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;

struct PredictionVisualizer {
    model: FinancialTransformerForMaskedRegression,
    device: Device,
    selected_crypto_idx: usize,
}

impl PredictionVisualizer {
    fn new(
        model: FinancialTransformerForMaskedRegression,
        device: Device,
        selected_crypto_idx: usize,
    ) -> Self {
        println!("🎯 Visualizer setup for CRYPTO_{}", selected_crypto_idx);
        Self {
            model,
            device,
            selected_crypto_idx,
        }
    }

    /// Create masked input where the selected crypto has its last 75% of the sequence masked
    /// All other cryptos remain fully available as context
    fn create_sequence_masked_input(&self, data: &Tensor, timestamp: usize) -> Result<Tensor> {
        if timestamp < SEQUENCE_LENGTH {
            return Err(candle_core::Error::Msg("Not enough history for sequence".to_string()));
        }

        let start_idx = timestamp - SEQUENCE_LENGTH;
        let input_sequence = data.narrow(0, start_idx, SEQUENCE_LENGTH)?;
        let mut masked_sequence = input_sequence.clone();

        // Calculate masking boundaries: first 25% is context, last 75% is masked
        let context_length = SEQUENCE_LENGTH / 4; // 25% context
        let mask_start = context_length; // Start masking after 25%
        let mask_length = SEQUENCE_LENGTH - context_length; // 75% to mask

        // Only print setup info once (when called for the first time)
        static SETUP_PRINTED: std::sync::Once = std::sync::Once::new();
        SETUP_PRINTED.call_once(|| {
            println!("🎯 Masking setup for CRYPTO_{}:", self.selected_crypto_idx);
            println!("  - Sequence length: {}", SEQUENCE_LENGTH);
            println!("  - Context length (25%): {}", context_length);
            println!("  - Masked length (75%): {}", mask_length);
            println!("  - Mask starts at position: {}", mask_start);
        });

        // Create zeros for the masked portion of the selected crypto
        let zeros = Tensor::zeros((mask_length, 1), DType::F32, &self.device)?;

        // Extract the selected crypto column
        let selected_crypto_col = masked_sequence.narrow(1, self.selected_crypto_idx, 1)?;

        // Keep the first 25% (context), zero out the last 75%
        let context_part = selected_crypto_col.narrow(0, 0, context_length)?;
        let masked_crypto_col = Tensor::cat(&[&context_part, &zeros], 0)?;

        // Reconstruct the full sequence with the partially masked crypto
        let before_cols = if self.selected_crypto_idx > 0 {
            Some(masked_sequence.narrow(1, 0, self.selected_crypto_idx)?)
        } else {
            None
        };
        let after_cols = if self.selected_crypto_idx + 1 < masked_sequence.dims()[1] {
            Some(masked_sequence.narrow(1, self.selected_crypto_idx + 1, masked_sequence.dims()[1] - self.selected_crypto_idx - 1)?)
        } else {
            None
        };

        // Reconstruct the tensor with the partially masked crypto column
        masked_sequence = match (before_cols, after_cols) {
            (Some(before), Some(after)) => Tensor::cat(&[&before, &masked_crypto_col, &after], 1)?,
            (Some(before), None) => Tensor::cat(&[&before, &masked_crypto_col], 1)?,
            (None, Some(after)) => Tensor::cat(&[&masked_crypto_col, &after], 1)?,
            (None, None) => masked_crypto_col,
        };

        Ok(masked_sequence)
    }

    /// Get predictions for the masked portion (last 75%) of the selected crypto
    fn get_sequence_predictions(&self, data: &Tensor, timestamp: usize) -> Result<Vec<f64>> {
        let masked_input = self.create_sequence_masked_input(data, timestamp)?;
        let input_batch = masked_input.unsqueeze(0)?; // Add batch dimension

        // Get model predictions for the entire sequence
        let predictions = self.model.forward(&input_batch)?;

        // Extract predictions for the selected crypto across the entire sequence
        let crypto_predictions = predictions.get(0)?.narrow(1, self.selected_crypto_idx, 1)?;
        let crypto_predictions_vec: Vec<f32> = crypto_predictions.squeeze(1)?.to_vec1()?;

        // Return only the predictions for the masked portion (last 75%)
        let context_length = SEQUENCE_LENGTH / 4; // 25% context
        let masked_predictions: Vec<f64> = crypto_predictions_vec
            .iter()
            .skip(context_length) // Skip the context portion
            .map(|&x| x as f64)
            .collect();

        Ok(masked_predictions)
    }

    /// Generate predictions across the entire test set using sliding window
    fn generate_full_test_predictions(&self, data: &Tensor) -> Result<(Vec<f64>, Vec<f64>, Vec<usize>)> {
        let total_timesteps = data.dims()[0];
        let context_length = SEQUENCE_LENGTH / 4; // 25% context
        let masked_length = SEQUENCE_LENGTH - context_length; // 75% masked

        println!("📊 Generating predictions for entire test set:");
        println!("  - Total timesteps: {}", total_timesteps);
        println!("  - Context length (25%): {}", context_length);
        println!("  - Masked length (75%): {}", masked_length);
        println!("  - Selected crypto: CRYPTO_{}", self.selected_crypto_idx);

        let mut all_predictions = Vec::new();
        let mut all_real_values = Vec::new();
        let mut all_timestamps = Vec::new();

        // Slide window across test set, using ALL masked predictions (75% of each sequence)
        // We'll step by masked_length to avoid overlapping the same predictions
        let start_time = SEQUENCE_LENGTH;
        let step_size = masked_length; // Step by masked length to avoid overlap
        let mut current_timestamp = start_time;
        let mut sequences_processed = 0;

        println!("  - Step size: {} (to avoid overlapping masked regions)", step_size);

        while current_timestamp < total_timesteps {
            if sequences_processed % 10 == 0 {
                println!("  - Processing sequence {} at timestep {}", sequences_processed + 1, current_timestamp);
            }

            // Get ALL predictions for the masked portion (75%) of this sequence
            if let Ok(sequence_predictions) = self.get_sequence_predictions(data, current_timestamp) {
                let sequence_start = current_timestamp - SEQUENCE_LENGTH;
                let masked_start = sequence_start + context_length;

                // Add ALL masked predictions and their corresponding real values
                for (i, &prediction) in sequence_predictions.iter().enumerate() {
                    let target_timestamp = masked_start + i;

                    if target_timestamp < total_timesteps {
                        // Get the real value for this timestamp
                        if let Ok(real_row) = data.get(target_timestamp) {
                            if let Ok(real_vec) = real_row.to_vec1::<f32>() {
                                all_predictions.push(prediction);
                                all_real_values.push(real_vec[self.selected_crypto_idx] as f64);
                                all_timestamps.push(target_timestamp);
                            }
                        }
                    }
                }

                sequences_processed += 1;
            }

            // Move to next non-overlapping sequence
            current_timestamp += step_size;
        }

        println!("✅ Generated {} total predictions from {} sequences (using full 75% masked portions)",
                 all_predictions.len(), sequences_processed);

        // Print some sample values for debugging
        if !all_predictions.is_empty() {
            println!("\n🔍 Sample values (first 10 points):");
            for i in 0..10.min(all_predictions.len()) {
                println!("  Point {}: Predicted={:.6}, Real={:.6}, Diff={:.6}",
                         i, all_predictions[i], all_real_values[i], all_predictions[i] - all_real_values[i]);
            }

            println!("\n🔍 Sample values (last 10 points):");
            let start_idx = all_predictions.len().saturating_sub(10);
            for i in start_idx..all_predictions.len() {
                println!("  Point {}: Predicted={:.6}, Real={:.6}, Diff={:.6}",
                         i, all_predictions[i], all_real_values[i], all_predictions[i] - all_real_values[i]);
            }
        }

        Ok((all_predictions, all_real_values, all_timestamps))
    }

    /// Create visualization plot with two subplots: predictions vs real, and difference
    fn create_plot(&self, predictions: &[f64], real_values: &[f64], timestamps: &[usize]) -> Result<()> {
        let output_path = format!("prediction_vs_real_crypto_{}.png", self.selected_crypto_idx);

        println!("🎨 Creating dual-panel visualization plot...");

        // Create the drawing area with larger height for two subplots
        let root = BitMapBackend::new(&output_path, (2600, 2600)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| candle_core::Error::Msg(format!("Plot error: {}", e)))?;

        // Split into three subplots (vertically)
        let areas = root.split_evenly((3, 1));
        let upper = &areas[0];
        let middle = &areas[1];
        let lower = &areas[2];

        // Calculate difference values
        let differences: Vec<f64> = predictions.iter().zip(real_values.iter())
            .map(|(p, r)| p - r)
            .collect();

        // Calculate rolling correlation (window size of 500 points)
        let window_size = 500;
        let mut rolling_correlations = Vec::new();
        let mut rolling_timestamps = Vec::new();

        for i in window_size..predictions.len() {
            let window_pred = &predictions[i-window_size..i];
            let window_real = &real_values[i-window_size..i];
            let corr = self.calculate_correlation(window_pred, window_real);
            rolling_correlations.push(corr);
            rolling_timestamps.push(timestamps[i]);
        }

        let min_time = *timestamps.first().unwrap_or(&0) as f64;
        let max_time = *timestamps.last().unwrap_or(&1) as f64;

        // === UPPER PLOT: Scatter Plot (Predicted vs Real) ===
        let min_val = predictions.iter().chain(real_values.iter()).fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = predictions.iter().chain(real_values.iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let val_range = max_val - min_val;
        let val_margin = val_range * 0.1;

        let mut upper_chart = ChartBuilder::on(upper)
            .caption(&format!("Predicted vs Real Values (Scatter) - CRYPTO_{}", self.selected_crypto_idx), ("sans-serif", 35))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(
                (min_val - val_margin)..(max_val + val_margin),
                (min_val - val_margin)..(max_val + val_margin)
            ).map_err(|e| candle_core::Error::Msg(format!("Upper chart creation error: {}", e)))?;

        upper_chart
            .configure_mesh()
            .x_desc("Real Values")
            .y_desc("Predicted Values")
            .draw().map_err(|e| candle_core::Error::Msg(format!("Upper mesh draw error: {}", e)))?;

        // Draw perfect prediction line (diagonal)
        upper_chart
            .draw_series(LineSeries::new(
                vec![(min_val - val_margin, min_val - val_margin), (max_val + val_margin, max_val + val_margin)],
                BLACK.stroke_width(2),
            )).map_err(|e| candle_core::Error::Msg(format!("Perfect prediction line error: {}", e)))?
            .label("Perfect Prediction")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], BLACK.stroke_width(2)));

        // Sample points for scatter plot (use every 10th point to avoid overcrowding)
        let sample_step = (predictions.len() / 2000).max(1); // Show max 2000 points
        let sampled_data: Vec<(f64, f64)> = predictions.iter()
            .zip(real_values.iter())
            .enumerate()
            .filter(|(i, _)| i % sample_step == 0)
            .map(|(_, (p, r))| (*r, *p))
            .collect();

        // Plot scatter points
        upper_chart
            .draw_series(
                sampled_data.iter().map(|&(real, pred)| Circle::new((real, pred), 2, BLUE.mix(0.6).filled()))
            ).map_err(|e| candle_core::Error::Msg(format!("Scatter plot error: {}", e)))?
            .label(&format!("Data Points (n={})", sampled_data.len()))
            .legend(|(x, y)| Circle::new((x + 5, y), 3, BLUE.mix(0.6).filled()));

        upper_chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw().map_err(|e| candle_core::Error::Msg(format!("Upper legend draw error: {}", e)))?;

        // === MIDDLE PLOT: Rolling Correlation ===
        let mut middle_chart = ChartBuilder::on(middle)
            .caption("Rolling Correlation (500-point window)", ("sans-serif", 35))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(80)
            .build_cartesian_2d(
                min_time..max_time,
                -1.0..1.0
            ).map_err(|e| candle_core::Error::Msg(format!("Middle chart creation error: {}", e)))?;

        middle_chart
            .configure_mesh()
            .x_desc("Timestamp")
            .y_desc("Correlation")
            .draw().map_err(|e| candle_core::Error::Msg(format!("Middle mesh draw error: {}", e)))?;

        // Add zero reference line for correlation
        middle_chart
            .draw_series(LineSeries::new(
                vec![(min_time, 0.0), (max_time, 0.0)],
                BLACK.stroke_width(1),
            )).map_err(|e| candle_core::Error::Msg(format!("Correlation zero line error: {}", e)))?
            .label("Zero Correlation")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], BLACK));

        // Plot rolling correlation
        if !rolling_correlations.is_empty() {
            middle_chart
                .draw_series(LineSeries::new(
                    rolling_timestamps.iter().zip(rolling_correlations.iter()).map(|(&t, &c)| (t as f64, c)),
                    MAGENTA.stroke_width(3),
                )).map_err(|e| candle_core::Error::Msg(format!("Rolling correlation plot error: {}", e)))?
                .label("Rolling Correlation")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], MAGENTA.stroke_width(3)));
        }

        middle_chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw().map_err(|e| candle_core::Error::Msg(format!("Middle legend draw error: {}", e)))?;

        // === LOWER PLOT: Difference (Predicted - Real) ===
        let min_diff = differences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_diff = differences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let diff_range = max_diff - min_diff;
        let diff_margin = diff_range * 0.1;

        let mut lower_chart = ChartBuilder::on(lower)
            .caption("Prediction Error (Predicted - Real)", ("sans-serif", 35))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(
                min_time..max_time,
                (min_diff - diff_margin)..(max_diff + diff_margin)
            ).map_err(|e| candle_core::Error::Msg(format!("Lower chart creation error: {}", e)))?;

        lower_chart
            .configure_mesh()
            .x_desc("Timestamp")
            .y_desc("Difference (Predicted - Real)")
            .draw().map_err(|e| candle_core::Error::Msg(format!("Lower mesh draw error: {}", e)))?;

        // Add zero reference line
        lower_chart
            .draw_series(LineSeries::new(
                vec![(min_time, 0.0), (max_time, 0.0)],
                BLACK.stroke_width(1),
            )).map_err(|e| candle_core::Error::Msg(format!("Zero line plot error: {}", e)))?
            .label("Zero Reference")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], BLACK));

        // Plot differences (green/red based on positive/negative)
        lower_chart
            .draw_series(LineSeries::new(
                timestamps.iter().zip(differences.iter()).map(|(&t, &d)| (t as f64, d)),
                GREEN.mix(0.8).stroke_width(2),
            )).map_err(|e| candle_core::Error::Msg(format!("Difference plot error: {}", e)))?
            .label("Prediction Error")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], GREEN.mix(0.8).stroke_width(2)));

        lower_chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw().map_err(|e| candle_core::Error::Msg(format!("Lower legend draw error: {}", e)))?;

        root.present().map_err(|e| candle_core::Error::Msg(format!("Present error: {}", e)))?;
        println!("✅ Plot saved to: {}", output_path);
        
        // Print some statistics
        let correlation = self.calculate_correlation(predictions, real_values);
        let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let mean_real = real_values.iter().sum::<f64>() / real_values.len() as f64;
        let mean_divergence = predictions.iter().zip(real_values.iter())
            .map(|(p, r)| p - r)
            .sum::<f64>() / predictions.len() as f64;
        
        println!("\n📈 VISUALIZATION STATISTICS:");
        println!("  - Data points plotted: {}", predictions.len());
        println!("  - Correlation: {:.4}", correlation);
        println!("  - Mean predicted: {:.6}", mean_pred);
        println!("  - Mean real: {:.6}", mean_real);
        println!("  - Mean divergence: {:.6}", mean_divergence);
        
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
    println!("🎨 PREDICTION VISUALIZATION TOOL");
    println!("======================================================================");
    println!("This tool loads a trained model and visualizes predicted vs real values");
    println!("for a selected cryptocurrency using sequence masking:");
    println!("- First 25% of sequence: provided as context");
    println!("- Last 75% of sequence: masked and predicted");
    println!("- All other cryptos: fully available as context");
    println!("======================================================================");

    // Setup device
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    // Configuration
    let data_path = "/mnt/storage-box/15m/transformed_dataset.parquet";
    let model_path = "training_saves_15m/current_model_middle_r3_ep396.safetensors";

    // Load data
    println!("\n📊 Loading cryptocurrency data...");
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

    // Select a crypto for visualization (you can change this index)
    let selected_crypto_idx = 23; // Change this to visualize different cryptos
    println!("🎯 Selected CRYPTO_{} for visualization", selected_crypto_idx);

    if selected_crypto_idx >= num_time_series {
        return Err(candle_core::Error::Msg(format!("Selected crypto index {} is out of range (max: {})", selected_crypto_idx, num_time_series - 1)));
    }
    
    // Initialize visualizer
    let visualizer = PredictionVisualizer::new(model, device.clone(), selected_crypto_idx);

    // Generate predictions for the entire test set
    let (predictions, real_values, timestamps) = visualizer.generate_full_test_predictions(&test_data)?;

    // Create visualization
    visualizer.create_plot(&predictions, &real_values, &timestamps)?;

    println!("\n🎉 Visualization complete!");
    println!("The plot shows predicted (red) vs real (blue) values for CRYPTO_{}.", selected_crypto_idx);
    println!("Each prediction uses a sliding window where:");
    println!("- The first 25% of the sequence provides context for the selected crypto");
    println!("- The last 75% of the selected crypto is masked");
    println!("- All other cryptocurrencies are fully available as context");
    println!("- Predictions cover the entire test set timeline.");
    
    Ok(())
}
