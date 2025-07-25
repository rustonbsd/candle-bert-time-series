// main_orderbook_financialbert.rs
// Training FinancialBERT on orderbook data
//
// This implementation trains a BERT-style transformer on normalized orderbook data
// from Bybit. The model learns to predict masked orderbook levels using self-attention.
//
// Key features:
// - Loads multiple parquet files with orderbook data (200ms intervals, 20 levels)
// - Uses orderbook-level masking (masks entire bid/ask pairs)
// - Supports temporal masking and random masking strategies
// - Includes data validation and inspection utilities
// - Saves checkpoints after each epoch
// - Includes test mode for quick validation
//
// Usage:
//   cargo run --bin main_orderbook_financialbert
//
// Data format expected:
//   - Parquet files with columns: bid_price_0, bid_qty_0, ask_price_0, ask_qty_0, ..., bid_price_19, bid_qty_19, ask_price_19, ask_qty_19
//   - Data should be normalized (z-score normalization)
//   - 200ms tick intervals (5 ticks per second)
//   - Optional timestamp column (will be filtered out)
pub mod orderbook_dataset;

use candle_bert_time_series::batcher::Batcher;
use candle_bert_time_series::financial_bert::{Config, FinancialTransformerForMaskedRegression, HiddenAct, PositionEmbeddingType};
use orderbook_dataset::load_and_prepare_orderbook_data;

use candle_core::{DType, Device, Result, Tensor, IndexOp};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use rand::Rng;

// --- Configuration ---
const SEQUENCE_LENGTH: usize = 240; // 240 timesteps (48 seconds at 200ms intervals)
const MODEL_DIMS: usize = 128; // Hidden size
const NUM_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;
const NUM_EPOCHS: usize = 500;
const TEST_MODE: bool = false; // Set to true for quick testing with fewer epochs
const LEARNING_RATE: f64 = 5e-5;
const MASK_PROB: f32 = 0.15;
const ORDERBOOK_MASK_PROB: f32 = 0.15; // Percentage of orderbook levels to mask
const BATCH_SIZE: usize = 256;

// Data file path - update this to point to your orderbook parquet files
// Using the properly normalized files with quantile normalization
const DATA_PATHS: &[&str] = &[
    "/mnt/storage-box/bybit/orderbooks/output/2025-04-30_BTCUSDC_financialbert_quantile_normal.parquet",
    "/mnt/storage-box/bybit/orderbooks/output/2025-05-01_BTCUSDC_financialbert_quantile_normal.parquet",
    "/mnt/storage-box/bybit/orderbooks/output/2025-05-02_BTCUSDC_financialbert_quantile_normal.parquet",
    "/mnt/storage-box/bybit/orderbooks/output/2025-05-03_BTCUSDC_financialbert_quantile_normal.parquet",
    "/mnt/storage-box/bybit/orderbooks/output/2025-05-04_BTCUSDC_financialbert_quantile_normal.parquet",
    // Add more files as needed - the loader will skip missing files
];

// --- Masking Functions ---

/// Standard random masking function for orderbook features
fn mask_data_random(
    input: &Tensor, // Expects a 3D tensor: [BATCH_SIZE, SEQUENCE_LENGTH, NUM_FEATURES]
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let rand_mask = Tensor::rand(0f32, 1f32, shape, device)?;
    let mask_bool = (rand_mask.lt(MASK_PROB))?;
    let mask = mask_bool.to_dtype(DType::U8)?; // Convert to u8 for where_cond
    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // Extract true labels for masked positions
    let true_labels = mask.where_cond(input, &zeros)?;

    // Create masked input by zeroing out the masked positions
    let ones = Tensor::ones(shape, DType::F32, device)?;
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let inverted_mask = ones.sub(&mask_f32)?;
    let masked_input = input.broadcast_mul(&inverted_mask)?;

    Ok((masked_input, true_labels, mask))
}

/// Orderbook-level masking: mask entire orderbook levels (bid/ask pairs)
fn mask_data_orderbook_levels(
    input: &Tensor, // [BATCH_SIZE, SEQUENCE_LENGTH, NUM_FEATURES] where NUM_FEATURES = 80 (20 levels * 4 features)
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let num_features = shape.dims()[2];
    let num_levels = num_features / 4; // Each level has 4 features: bid_price, bid_qty, ask_price, ask_qty
    
    // Create level mask: randomly select levels to mask
    let mut rng = rand::thread_rng();
    let mut level_mask_vec = vec![0u8; num_levels];
    for i in 0..num_levels {
        if rng.gen::<f32>() < ORDERBOOK_MASK_PROB {
            level_mask_vec[i] = 1;
        }
    }
    
    // Expand level mask to feature mask
    let mut feature_mask_vec = vec![0u8; num_features];
    for level in 0..num_levels {
        if level_mask_vec[level] == 1 {
            let base_idx = level * 4;
            feature_mask_vec[base_idx] = 1;     // bid_price
            feature_mask_vec[base_idx + 1] = 1; // bid_qty
            feature_mask_vec[base_idx + 2] = 1; // ask_price
            feature_mask_vec[base_idx + 3] = 1; // ask_qty
        }
    }
    
    // Create mask tensor and broadcast to full shape
    let feature_mask = Tensor::from_vec(feature_mask_vec, &[num_features], device)?;
    let mask_3d = feature_mask.unsqueeze(0)?.unsqueeze(0)?;
    let mask_u8 = mask_3d.broadcast_as(shape)?;

    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // Extract true labels for masked positions
    let true_labels = mask_u8.where_cond(input, &zeros)?;

    // Create masked input
    let ones = Tensor::ones(shape, DType::F32, device)?;
    let mask_f32 = mask_u8.to_dtype(DType::F32)?;
    let inverted_mask = ones.sub(&mask_f32)?;
    let masked_input = input.broadcast_mul(&inverted_mask)?;

    Ok((masked_input, true_labels, mask_u8))
}

/// Temporal masking: mask the most recent timesteps
fn mask_data_temporal(
    input: &Tensor, // [BATCH_SIZE, SEQUENCE_LENGTH, NUM_FEATURES]
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let seq_len = shape.dims()[1];
    
    // Mask the last few timesteps (most recent orderbook data)
    let mask_timesteps = (seq_len as f32 * MASK_PROB) as usize;
    let start_mask = seq_len - mask_timesteps;
    
    // Create temporal mask
    let mut temporal_mask_vec = vec![0u8; seq_len];
    for i in start_mask..seq_len {
        temporal_mask_vec[i] = 1;
    }

    let temporal_mask = Tensor::from_vec(temporal_mask_vec, &[seq_len], device)?;
    let mask_3d = temporal_mask.unsqueeze(0)?.unsqueeze(2)?;
    let mask_u8 = mask_3d.broadcast_as(shape)?;

    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // Extract true labels for masked positions
    let true_labels = mask_u8.where_cond(input, &zeros)?;

    // Create masked input
    let ones = Tensor::ones(shape, DType::F32, device)?;
    let mask_f32 = mask_u8.to_dtype(DType::F32)?;
    let inverted_mask = ones.sub(&mask_f32)?;
    let masked_input = input.broadcast_mul(&inverted_mask)?;

    Ok((masked_input, true_labels, mask_u8))
}

/// Evaluation function for model performance
fn evaluate_model<F>(
    model: &FinancialTransformerForMaskedRegression,
    data: &Tensor,
    device: &Device,
    dataset_name: &str,
    mask_fn: F,
) -> Result<f64>
where
    F: Fn(&Tensor, &Device) -> Result<(Tensor, Tensor, Tensor)>,
{
    let mut batcher = Batcher::new_r2(data.clone(), SEQUENCE_LENGTH, BATCH_SIZE, false)?;
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    while let Some(batch_result) = batcher.next() {
        let batch = batch_result?;
        let (masked_input, true_labels, mask) = mask_fn(&batch, device)?;
        
        let predictions = model.forward(&masked_input)?;
        
        // Calculate loss only on masked positions
        let mask_u8 = if mask.dtype() != DType::U8 { mask.to_dtype(DType::U8)? } else { mask.clone() };
        let masked_predictions = mask_u8.where_cond(&predictions, &Tensor::zeros_like(&predictions)?)?;
        let loss = loss::mse(&masked_predictions, &true_labels)?;
        
        total_loss += loss.to_scalar::<f32>()? as f64;
        batch_count += 1;
    }

    let avg_loss = total_loss / batch_count as f64;
    println!("  {} Loss: {:.10} (averaged over {} batches)", dataset_name, avg_loss, batch_count);

    Ok(avg_loss)
}

/// Detailed data analysis to identify potential issues
fn detailed_data_analysis(data: &Tensor) -> Result<()> {
    println!("Analyzing data for potential issues...");

    let shape = data.shape();
    let flattened = data.flatten_all()?;

    // Basic statistics
    let mean = flattened.mean_all()?.to_scalar::<f32>()?;
    let std = flattened.var(0)?.sqrt()?.to_scalar::<f32>()?;
    let min_val = flattened.min(0)?.to_scalar::<f32>()?;
    let max_val = flattened.max(0)?.to_scalar::<f32>()?;

    println!("  Overall stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
             mean, std, min_val, max_val);

    // Check for extreme outliers (values beyond 4 standard deviations)
    let outlier_threshold = 4.0;
    let lower_bound = mean - outlier_threshold * std;
    let upper_bound = mean + outlier_threshold * std;

    // Count outliers (approximate method since candle doesn't have advanced filtering)
    let total_elements = flattened.elem_count();
    println!("  Outlier bounds: [{:.6}, {:.6}] (±{} std)",
             lower_bound, upper_bound, outlier_threshold);

    if max_val > upper_bound || min_val < lower_bound {
        println!("  ⚠️  EXTREME OUTLIERS DETECTED!");
        println!("     Max value {:.6} exceeds upper bound {:.6}", max_val, upper_bound);
        println!("     This could cause training instability!");
    }

    // Analyze feature-wise statistics
    println!("  Analyzing individual features...");
    let num_features = shape.dims()[1];
    let num_timesteps = shape.dims()[0];

    for feature_idx in 0..num_features.min(10) { // Check first 10 features
        let feature_data = data.narrow(1, feature_idx, 1)?.flatten_all()?;
        let f_mean = feature_data.mean_all()?.to_scalar::<f32>()?;
        let f_std = feature_data.var(0)?.sqrt()?.to_scalar::<f32>()?;
        let f_min = feature_data.min(0)?.to_scalar::<f32>()?;
        let f_max = feature_data.max(0)?.to_scalar::<f32>()?;

        println!("    Feature {}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
                 feature_idx, f_mean, f_std, f_min, f_max);

        if f_max > f_mean + 10.0 * f_std {
            println!("      ⚠️  Feature {} has extreme outliers!", feature_idx);
        }
    }

    // Check for constant features
    println!("  Checking for constant/near-constant features...");
    for feature_idx in 0..num_features.min(20) {
        let feature_data = data.narrow(1, feature_idx, 1)?.flatten_all()?;
        let f_std = feature_data.var(0)?.sqrt()?.to_scalar::<f32>()?;

        if f_std < 1e-6 {
            println!("    ⚠️  Feature {} is nearly constant (std={:.8})", feature_idx, f_std);
        }
    }

    // Sample some actual values to inspect
    println!("  Sample values from first timestep:");
    for i in 0..num_features.min(10) {
        let val = data.i((0, i))?;
        println!("    Feature {}: {:.6}", i, val.to_scalar::<f32>()?);
    }

    Ok(())
}

// --- The Main Training Function ---

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Training on device: {:?}", device);

    // --- Data Validation First ---
    println!("Validating orderbook data files...");
    if let Err(e) = orderbook_dataset::validate_orderbook_files(DATA_PATHS) {
        println!("⚠️  Data validation failed: {}", e);
        println!("Continuing anyway, but expect potential issues...");
    }

    // Inspect first file for structure
    if !DATA_PATHS.is_empty() {
        if let Err(e) = orderbook_dataset::inspect_orderbook_file(DATA_PATHS[0]) {
            println!("⚠️  File inspection failed: {}", e);
        }
    }

    // --- Data Loading First to Determine Dimensions ---
    println!("Loading orderbook data...");
    let (full_data_sequence, num_features) = load_and_prepare_orderbook_data(DATA_PATHS, &device)?;

    // DETAILED DATA ANALYSIS FOR DEBUGGING
    println!("\n🔍 DETAILED DATA ANALYSIS:");
    detailed_data_analysis(&full_data_sequence)?;

    let total_timesteps = full_data_sequence.dims()[0];

    // Split data into train (70%), validation (15%), test (15%)
    let train_split = (total_timesteps as f32 * 0.7) as usize;
    let val_split = (total_timesteps as f32 * 0.85) as usize;

    let train_data = full_data_sequence.narrow(0, 0, train_split)?;
    let val_data = full_data_sequence.narrow(0, train_split, val_split - train_split)?;
    let test_data = full_data_sequence.narrow(0, val_split, total_timesteps - val_split)?;

    println!("Data splits - Train: {}, Validation: {}, Test: {}",
             train_data.dims()[0], val_data.dims()[0], test_data.dims()[0]);

    println!("Detected {} orderbook features in the dataset", num_features);
    println!("Expected features: 80 (20 levels × 4 features per level)");

    // --- Model and Optimizer Setup ---
    let config = Config {
        num_time_series: num_features, // Use orderbook features instead of time series
        hidden_size: MODEL_DIMS,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        intermediate_size: MODEL_DIMS * 4,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.1,
        max_position_embeddings: SEQUENCE_LENGTH,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: false,
        model_type: Some("financial_transformer_orderbook".to_string()),
    };

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FinancialTransformerForMaskedRegression::load(vb, &config)?;

    // Optional: Load checkpoint if available
    // let checkpoint_path = format!("training_saves_orderbook/current_model_ep0.safetensors");
    // if std::path::Path::new(&checkpoint_path).exists() {
    //     varmap.load(checkpoint_path.clone())?;
    //     println!("Loaded checkpoint: {}", checkpoint_path);
    // }

    let adamw_params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        weight_decay: 0.02,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    // Create checkpoint directory
    std::fs::create_dir_all("training_saves_orderbook")?;

    let actual_epochs = if TEST_MODE { 2 } else { NUM_EPOCHS };
    println!("Starting training with {} epochs...", actual_epochs);
    if TEST_MODE {
        println!("🧪 TEST MODE: Running only {} epochs for quick validation", actual_epochs);
    }

    // --- Training Loop ---
    for epoch in 0..actual_epochs {
        println!("\n=== EPOCH {} ===", epoch + 1);

        // --- TRAINING PHASE ---
        let mut train_batcher = Batcher::new_r2(train_data.clone(), SEQUENCE_LENGTH, BATCH_SIZE, true)?;
        let mut total_train_loss = 0.0;
        let mut train_batch_count = 0;

        while let Some(batch_result) = train_batcher.next() {
            let batch = batch_result?;

            // Use orderbook-level masking for better representation learning
            let (masked_input, true_labels, mask) = mask_data_orderbook_levels(&batch, &device)?;

            // --- FORWARD PASS ---
            let predictions = model.forward(&masked_input)?;

            // --- LOSS CALCULATION ---
            // Calculate loss only on masked positions
            let mask_u8 = if mask.dtype() != DType::U8 { mask.to_dtype(DType::U8)? } else { mask.clone() };
            let masked_predictions = mask_u8.where_cond(&predictions, &Tensor::zeros_like(&predictions)?)?;
            let loss = loss::mse(&masked_predictions, &true_labels)?;

            // --- BACKWARD PASS ---
            optimizer.backward_step(&loss)?;

            total_train_loss += loss.to_scalar::<f32>()? as f64;
            train_batch_count += 1;

            if train_batch_count % 100 == 0 {
                println!("  Batch {}: Loss = {:.10}", train_batch_count, loss.to_scalar::<f32>()? as f64);
            }
        }

        let avg_train_loss = total_train_loss / train_batch_count as f64;

        // Save model checkpoint after each epoch
        println!("Saving checkpoint...");
        let checkpoint_path = format!("training_saves_orderbook/current_model_ep{}.safetensors", epoch);
        varmap.save(&checkpoint_path)?;
        println!("Saved checkpoint to: {}", checkpoint_path);

        println!("Training completed: {} batches processed", train_batch_count);
        println!("  Training Loss: {:.10} (averaged over {} batches)", avg_train_loss, train_batch_count);

        // --- VALIDATION PHASE ---
        println!("Running validation...");
        let val_loss = evaluate_model(&model, &val_data, &device, "Validation", mask_data_orderbook_levels)?;

        // --- EPOCH SUMMARY ---
        println!("Epoch {} Summary:", epoch + 1);
        println!("  Train Loss: {:.10}", avg_train_loss);
        println!("  Val Loss:   {:.10}", val_loss);

        // Early stopping or learning rate scheduling could be added here
        if val_loss < 0.001 {
            println!("Validation loss below threshold, stopping early");
            break;
        }
    }

    // --- FINAL TEST EVALUATION ---
    println!("\n=== FINAL TEST EVALUATION ===");
    let test_loss = evaluate_model(&model, &test_data, &device, "Test", mask_data_orderbook_levels)?;
    println!("Final Test Loss: {:.10}", test_loss);

    println!("Training completed successfully!");
    Ok(())
}

/// Test function to validate the setup without full training
#[allow(dead_code)]
fn test_setup() -> Result<()> {
    println!("🧪 Testing orderbook FinancialBERT setup...");

    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    // Test data loading
    println!("Testing data loading...");
    let (data, num_features) = load_and_prepare_orderbook_data(&DATA_PATHS[0..1], &device)?;
    println!("✅ Data loaded: {:?}, features: {}", data.shape(), num_features);

    // Test model creation
    println!("Testing model creation...");
    let config = Config {
        num_time_series: num_features,
        hidden_size: MODEL_DIMS,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        intermediate_size: MODEL_DIMS * 4,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.1,
        max_position_embeddings: SEQUENCE_LENGTH,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: false,
        model_type: Some("financial_transformer_orderbook_test".to_string()),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FinancialTransformerForMaskedRegression::load(vb, &config)?;
    println!("✅ Model created successfully");

    // Test masking functions
    println!("Testing masking functions...");
    let test_batch = data.narrow(0, 0, BATCH_SIZE.min(data.dims()[0]))?
        .unsqueeze(1)?
        .expand(&[BATCH_SIZE.min(data.dims()[0]), SEQUENCE_LENGTH.min(data.dims()[0]), num_features])?;

    let (masked_input, true_labels, mask) = mask_data_orderbook_levels(&test_batch, &device)?;
    println!("✅ Masking functions work: input {:?}, labels {:?}, mask {:?}",
             masked_input.shape(), true_labels.shape(), mask.shape());

    // Test forward pass
    println!("Testing forward pass...");
    let predictions = model.forward(&masked_input)?;
    println!("✅ Forward pass successful: predictions {:?}", predictions.shape());

    println!("🎉 All tests passed! Setup is ready for training.");
    Ok(())
}

// Uncomment the line below to run tests instead of training
// fn main() -> Result<()> { test_setup() }
