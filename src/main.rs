// main.rs
pub mod dataset;
mod financial_bert;
use std::process::exit;

use candle_bert_time_series::batcher::Batcher;
use dataset::load_and_prepare_data;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

use candle_core::{scalar::TensorOrScalar, DType, Device, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};

// --- Configuration ---
// NUM_TIME_SERIES will be determined dynamically from the data
const SEQUENCE_LENGTH: usize = 240; // 240
const MODEL_DIMS: usize = 384; // 384
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const NUM_EPOCHS: usize = 32;
const LEARNING_RATE: f64 = 1e-4;
const MASK_PROB: f32 = 0.15;
const CRYPTO_MASK_PROB: f32 = 0.15; // Percentage of cryptos to mask in crypto-column masking
const BATCH_SIZE: usize = 128;   // Entire dataset rn: 1594848

// Data file path - update this to point to your parquet file
const DATA_PATH: &str = "/home/i3/Downloads/transformed_dataset.parquet";

// --- Data Generation and Preparation ---

/// Corrected and simplified masking function for 3D batched input.
fn mask_data(
    input: &Tensor, // Expects a 3D tensor: [BATCH_SIZE, SEQUENCE_LENGTH, NUM_TIME_SERIES] -> [32, 120, 190]
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let rand_mask = Tensor::rand(0f32, 1f32, shape, device)?;
    // `mask` is a boolean tensor of shape [32, 120, 190]
    let mask = (rand_mask.lt(MASK_PROB))?;
    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // `masked_select` correctly gathers only the masked values into a 1D tensor.
    // Shape: [num_masked_elements] (flattened from all batch elements)
    let true_labels = mask.where_cond(input, &zeros)?;

    // Zero out the values in the input where the mask is true.
    // This preserves the original shape of the input.
    // Shape: [32, 120, 190]
    // 1. Create a tensor of ones with the same shape as the mask.
    //    The `lt` operation produces a U8 tensor, so we use DType::U8.
    let ones = Tensor::ones(shape, DType::U8, device)?;
    // 2. Subtract the mask from ones. If mask is [1, 0, 1], result is [0, 1, 0].
    let inverted_mask = ones.sub(&mask)?;
    // 3. Use this inverted mask to create the model input.
    let masked_input = input.broadcast_mul(&inverted_mask.to_dtype(DType::F32)?)?;

    Ok((masked_input, true_labels, mask))
}



/// Masks entire cryptocurrencies (15% of them) rather than random data points.
/// This is much closer to the inference task where we predict one crypto's movements
/// based on other cryptos' movements.
fn mask_data_crypto_columns(
    input: &Tensor, // Expects a 3D tensor: [BATCH_SIZE, SEQUENCE_LENGTH, NUM_TIME_SERIES] -> [32, 120, 190]
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let batch_size = shape.dims()[0];
    let sequence_length = shape.dims()[1];
    let num_cryptos = shape.dims()[2];

    // Calculate how many cryptos to mask (15% of total)
    let num_cryptos_to_mask = ((num_cryptos as f32) * CRYPTO_MASK_PROB).ceil() as usize;

    // Create a random permutation to select which cryptos to mask
    let mut crypto_indices: Vec<usize> = (0..num_cryptos).collect();

    // Simple random shuffle using tensor operations
    let rand_values = Tensor::rand(0f32, 1f32, &[num_cryptos], device)?;
    let rand_vec: Vec<f32> = rand_values.to_vec1()?;

    // Sort indices by random values to create a shuffle
    crypto_indices.sort_by(|&a, &b| rand_vec[a].partial_cmp(&rand_vec[b]).unwrap());

    // Take the first num_cryptos_to_mask indices as the ones to mask
    let cryptos_to_mask = &crypto_indices[..num_cryptos_to_mask];

    // Create a boolean mask for which cryptos to mask
    // Shape: [NUM_TIME_SERIES]
    let mut crypto_mask_vec = vec![0u8; num_cryptos];
    for &crypto_idx in cryptos_to_mask {
        crypto_mask_vec[crypto_idx] = 1;
    }
    let crypto_mask = Tensor::from_vec(crypto_mask_vec, &[num_cryptos], device)?;

    // Broadcast the crypto mask to full tensor shape
    // Shape: [1, 1, NUM_TIME_SERIES] -> [BATCH_SIZE, SEQUENCE_LENGTH, NUM_TIME_SERIES]
    let crypto_mask_3d = crypto_mask.unsqueeze(0)?.unsqueeze(0)?;
    let mask = crypto_mask_3d.broadcast_as(shape)?;

    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // Extract true labels for masked positions
    let true_labels = mask.where_cond(input, &zeros)?;

    // Create masked input by zeroing out the masked crypto columns
    let ones = Tensor::ones(shape, DType::U8, device)?;
    let inverted_mask = ones.sub(&mask)?;
    let masked_input = input.broadcast_mul(&inverted_mask.to_dtype(DType::F32)?)?;

    Ok((masked_input, true_labels, mask.to_dtype(DType::U8)?))
}

fn mask_data_last_col(
    input: &Tensor, // Expects a 3D tensor: [BATCH_SIZE, SEQUENCE_LENGTH, NUM_TIME_SERIES] -> [32, 120, 190]
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();

    // Create mask only for the newest minute (last time step in sequence)
    // Shape: [batch_size, 1, num_time_series] -> [32, 1, 190]
    let last_timestep_shape = &[shape.dims()[0], 1, shape.dims()[2]];
    let last_timestep_mask = Tensor::ones(last_timestep_shape, DType::F32, device)?;

    // Create full mask tensor with zeros everywhere except the last timestep
    let mut full_mask = Tensor::zeros(shape, DType::F32, device)?;
    // Set the last timestep (index 119 for sequence length 120) to our random mask
    let last_idx = shape.dims()[1] - 1; // 119 for sequence length 120

    // Use narrow and cat to insert the mask at the last timestep
    let before_last = full_mask.narrow(1, 0, last_idx)?;
    let after_last = if last_idx + 1 < shape.dims()[1] {
        Some(full_mask.narrow(1, last_idx + 1, shape.dims()[1] - last_idx - 1)?)
    } else {
        None
    };

    full_mask = if let Some(after) = after_last {
        Tensor::cat(&[&before_last, &last_timestep_mask, &after], 1)?
    } else {
        Tensor::cat(&[&before_last, &last_timestep_mask], 1)?
    };

    // Convert to U8 for boolean operations
    let full_mask = full_mask.to_dtype(DType::U8)?;

    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // `masked_select` correctly gathers only the masked values from the last timestep.
    // Shape: [num_masked_elements] (only from the newest minute across all batches)
    let true_labels = full_mask.where_cond(input, &zeros)?;

    // Zero out the values in the input where the mask is true (only in the newest minute).
    // This preserves the original shape of the input.
    // Shape: [32, 120, 190]
    // 1. Create a tensor of ones with the same shape as the full mask.
    let ones = Tensor::ones(shape, DType::U8, device)?;
    // 2. Subtract the mask from ones. If mask is [1, 0, 1], result is [0, 1, 0].
    let inverted_mask = ones.sub(&full_mask)?;
    // 3. Use this inverted mask to create the model input.
    let masked_input = input.broadcast_mul(&inverted_mask.to_dtype(DType::F32)?)?;

    Ok((masked_input, true_labels, full_mask))
}

/// Evaluate the model on a dataset (validation or test) without updating weights
fn evaluate_model(
    model: &FinancialTransformerForMaskedRegression,
    data: &Tensor,
    device: &Device,
    dataset_name: &str,
    masking_fn: fn(&Tensor, &Device) -> Result<(Tensor, Tensor, Tensor)>,
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    let mut batcher = Batcher::new(data, SEQUENCE_LENGTH, BATCH_SIZE);

    while let Some(batch_result) = batcher.next() {
        let batch = batch_result?;

        // Apply masking
        let (masked_input, true_labels, mask) = mask_data(&batch, device)?;

        // Forward pass (no gradient computation needed for evaluation)
        let predictions = model.forward(&masked_input)?;

        // Calculate loss and sign accuracy
        let zeros = Tensor::zeros(predictions.shape(), predictions.dtype(), device)?;
        let predicted_values = mask.where_cond(&predictions, &zeros)?;
        let loss = loss::mse(&predicted_values, &true_labels)?;

        total_loss += loss.to_scalar::<f32>()?;
        batch_count += 1;
    }

    let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };

    println!("  {} Loss: {:.10} (averaged over {} batches)", dataset_name, avg_loss, batch_count);

    Ok(avg_loss)
}

// --- The Main Training Function ---

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Training on device: {:?}", device);

    // --- Data Loading First to Determine Dimensions ---
    println!("Loading cryptocurrency data...");
    let (full_data_sequence, num_time_series) = load_and_prepare_data(DATA_PATH, &device)?;
    let total_timesteps = full_data_sequence.dims()[0];

    // Split data into train (70%), validation (15%), test (15%)
    // NOTE: This split logic is replicated in backtest::extract_test_split() to prevent data leakage
    let train_split = (total_timesteps as f32 * 0.7) as usize;
    let val_split = (total_timesteps as f32 * 0.85) as usize;

    let train_data = full_data_sequence.narrow(0, 0, train_split)?;
    let val_data = full_data_sequence.narrow(0, train_split, val_split - train_split)?;
    let test_data = full_data_sequence.narrow(0, val_split, total_timesteps - val_split)?;

    println!("Data splits - Train: {}, Validation: {}, Test: {}",
             train_data.dims()[0], val_data.dims()[0], test_data.dims()[0]);

    println!("Detected {} cryptocurrencies in the dataset", num_time_series);

    // --- Model and Optimizer Setup ---
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

    
    let checkpoint_path = format!("current_model_large_r2_ep1.safetensors");
    varmap.load(checkpoint_path.clone())?;
    println!("Loaded checkpoint: {}", checkpoint_path);
    

    let adamw_params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        weight_decay: 0.02,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;



    /*
    println!("Running test...");    
    let val_loss = evaluate_model(&model, &val_data, &device, "Val", mask_data_last_col)?;
    println!("Last Col Val Loss:   {:.10}", val_loss);
    let test_loss = evaluate_model(&model, &test_data, &device, "Test", mask_data_last_col)?;
    println!("Last Col Test Loss: {:.10}", test_loss);

    let val_loss = evaluate_model(&model, &val_data, &device, "Val", mask_data)?;
    println!(" Val Loss:   {:.10}", val_loss);
    let test_loss = evaluate_model(&model, &test_data, &device, "Test", mask_data)?;
    println!(" Test Loss: {:.10}", test_loss);
    
    return Ok(());
    */

    println!("Starting training...");
    for epoch in 0..NUM_EPOCHS {
        println!("\n--- Epoch {} ---", epoch + 1);

        // --- TRAINING PHASE: Process all batches in the training set ---
        let mut epoch_train_loss = 0.0;
        let mut epoch_sign_loss = 0.0;
        let mut train_batch_count = 0;

        let mut train_batcher = Batcher::new(&train_data, SEQUENCE_LENGTH, BATCH_SIZE);
        let mut batch_index = 0;

        while let Some(batch_result) = train_batcher.next() {
            let batch = batch_result?;

            // `masked_input` Shape: [32, 120, 190]
            // `true_labels` Shape: [num_masked] (1D, flattened from all batch elements)
            // `mask` Shape: [32, 120, 190] (boolean)
            let (masked_input, true_labels, mask) = mask_data_crypto_columns(&batch, &device)?;

            // --- FORWARD PASS ---
            // The model expects a 3D tensor: (batch_size, seq_len, features).
            // The batch is already properly shaped, no need to unsqueeze.
            // `model_input` Shape: [32, 120, 190]
            let model_input = masked_input;

            // `predictions` will have the same shape as the input.
            // `predictions` Shape: [32, 120, 190]
            let predictions = model.forward(&model_input)?;

            // --- LOSS CALCULATION ---
            // To compare with our labels, we must isolate the predictions at the masked positions.
            // Use the same boolean mask to select the predicted values.
            // This flattens the tensor, matching the shape of `true_labels`.
            // `predicted_values` Shape: [num_masked] (1D, flattened from all batch elements)
            let zeros = Tensor::zeros(predictions.shape(), predictions.dtype(), &device)?;
            let predicted_values = mask.where_cond(&predictions, &zeros)?;

            // Now we can compare the two 1D tensors.
            let loss_mse = loss::mse(&predicted_values, &true_labels)?;
            let loss = loss_mse;

            // Calculate sign accuracy and sign loss to improve directional correlation
            //let correct_signs = (predicted_values.sign()? * true_labels.sign()?)?.eq(1.0)?.to_dtype(DType::F32)?.sum_all()?;
            //let loss_sign = predicted_values.ones_like()?.sum_all()?.div(&correct_signs)?;

            // Sign loss: penalize incorrect directional predictions
            // Convert sign agreement to loss (1.0 - accuracy gives us the error rate)

            // Combine MSE loss with sign loss for better directional correlation
            // Weight the sign loss to emphasize directional accuracy
            //let sign_weight = Tensor::new(0.05f32, &device)?;
            // let mse_weight = Tensor::new(0.95f32, &device)?;
            // let loss = loss_mse.mul(&mse_weight)?.add(&loss_sign.mul(&sign_weight)?)?;

            /*
            // Transformer learnign rate warmup
            let num_batches_per_epoch = train_batcher.clone().count() as usize;
            let total_steps = NUM_EPOCHS * num_batches_per_epoch;
            let warmup_steps = 1000;
            let current_step = epoch * num_batches_per_epoch + batch_index;
            batch_index += 1;

            let new_lr = if current_step < warmup_steps {
                // Linear warmup
                LEARNING_RATE * (current_step as f64 / warmup_steps as f64)
            } else {
                // Cosine decay (example)
                LEARNING_RATE * 0.5 * (1.0 + (std::f64::consts::PI * (current_step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64).cos())
            };
            optimizer.set_learning_rate(new_lr);
            */

            // --- BACKWARD PASS ---
            optimizer.backward_step(&loss)?;

            // Accumulate training loss and sign accuracy
            epoch_train_loss += loss.to_scalar::<f32>()?;
            train_batch_count += 1;

            if train_batch_count % 100 == 0 {
                println!("  Batch {}% processed", (train_batch_count as f32 / (1594848.0 / BATCH_SIZE as f32))*100.0);
            }
        }

        // Calculate average training loss and sign accuracy for this epoch
        let avg_train_loss = if train_batch_count > 0 {
            epoch_train_loss / train_batch_count as f32
        } else {
            0.0
        };
        
        println!("Training completed: {} batches processed", train_batch_count);
        println!("  Training Loss: {:.10} (averaged over {} batches)", avg_train_loss, train_batch_count);
        
        // --- VALIDATION PHASE ---
        println!("Running validation...");
        let val_loss = evaluate_model(&model, &val_data, &device, "Validation", mask_data)?;

        // --- EPOCH SUMMARY ---
        println!("Epoch {} Summary:", epoch + 1);
        println!("  Train Loss: {:.10}", avg_train_loss);
        println!("  Val Loss:   {:.10}", val_loss);
        
        // Save model checkpoint after each epoch
        let checkpoint_path = format!("current_model_large.safetensors");
        varmap.save(&checkpoint_path)?;
        println!("Saved checkpoint to: {}", checkpoint_path);
    }

    // --- TEST PHASE ---
    println!("Running test...");
    let val_loss = evaluate_model(&model, &val_data, &device, "Val", mask_data_last_col)?;
    println!("Last Col Val Loss:   {:.10}", val_loss);
    let test_loss = evaluate_model(&model, &test_data, &device, "Test", mask_data_last_col)?;
    println!("Last Col Test Loss: {:.10}", test_loss);

    let val_loss = evaluate_model(&model, &val_data, &device, "Val", mask_data)?;
    println!(" Val Loss:   {:.10}", val_loss);
    let test_loss = evaluate_model(&model, &test_data, &device, "Test", mask_data)?;
    println!(" Test Loss: {:.10}", test_loss);

    Ok(())
}