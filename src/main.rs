// main.rs

// Import the new model structs from our model.rs file
mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};

// --- Hyperparameters and Configuration ---
// I've used your new name for the primary dimension.
const NUM_TIME_SERIES: usize = 100;
const SEQUENCE_LENGTH: usize = 64;
const MODEL_DIMS: usize = 256; // The hidden dimension of the transformer
const NUM_LAYERS: usize = 4;   // Number of transformer blocks
const NUM_HEADS: usize = 8;    // Number of attention heads
const NUM_EPOCHS: usize = 50;
const LEARNING_RATE: f64 = 1e-4;
const MASK_PROB: f32 = 0.15; // Probability of masking a token

// --- Data Generation and Preparation ---

/// Generates a dummy time series of returns.
/// Updated to use NUM_TIME_SERIES.
fn generate_dummy_data(timesteps: usize, device: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 0.02f32, (timesteps, NUM_TIME_SERIES), device)
}

/// Takes a sequence of returns and applies masking.
/// Updated to use NUM_TIME_SERIES.
fn mask_data(
    input: &Tensor, // Shape: (SEQUENCE_LENGTH, NUM_TIME_SERIES)
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let rand_mask = Tensor::rand(0f32, 1f32, shape, device)?;
    let mask = (rand_mask.lt(MASK_PROB))?;

    // Use where_cond to select masked values: where mask is true, use input, otherwise use zeros
    let zeros = Tensor::zeros(shape, input.dtype(), device)?;
    let true_labels = mask.where_cond(input, &zeros)?;

    // Create inverted mask (1 - mask) for masking input
    let ones = Tensor::ones(shape, DType::U8, device)?;
    let inverted_mask = ones.sub(&mask)?;
    let masked_input = input.broadcast_mul(&inverted_mask.to_dtype(DType::F32)?)?;


    Ok((masked_input, true_labels, mask))
}

// --- The Main Training Function ---

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Training on device: {:?}", device);

    // 1. Create the Model Configuration
    // This is the new, crucial step. We create a config struct that defines
    // the architecture of the model we're about to build.
    let config = Config {
        num_time_series: NUM_TIME_SERIES,
        hidden_size: MODEL_DIMS,
        num_hidden_layers: NUM_LAYERS,
        num_attention_heads: NUM_HEADS,
        intermediate_size: MODEL_DIMS * 4, // A common practice for transformers
        hidden_act: financial_bert::HiddenAct::Gelu,
        hidden_dropout_prob: 0.1,
        max_position_embeddings: SEQUENCE_LENGTH,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        position_embedding_type: financial_bert::PositionEmbeddingType::Absolute,
        use_cache: false, // Not needed for training
        model_type: Some("financial_transformer".to_string()),
    };

    // 2. Initialize the Model and Optimizer
    // We now use the `FinancialTransformerForMaskedRegression::load` method,
    // passing it the VarBuilder and our new config.
    let mut varmap = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = FinancialTransformerForMaskedRegression::load(var_builder.clone(), &config)?;

    // The optimizer gets the trainable variables from the VarBuilder that was used to create the model.
    // In Candle, the VarBuilder tracks all variables created through it.
    let adamw_params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    // 3. Load the data
    let full_data_sequence = generate_dummy_data(1024, &device)?;

    // 4. The Training Loop
    // This part of the code remains almost identical because the tensor shapes
    // for the inputs and outputs are the same as before.
    println!("Starting training...");
    for epoch in 0..NUM_EPOCHS {
        let batch_start = 0;
        let batch = full_data_sequence.narrow(0, batch_start, SEQUENCE_LENGTH)?;

        let (masked_input, true_labels, mask) = mask_data(&batch, &device)?;
        println!("masked_input shape: {:?}", masked_input.shape());
        let model_input = masked_input; //.unsqueeze(0)?;

        println!("model_input shape: {:?}", model_input.shape());

        // Forward Pass
        let predictions = model.forward(&model_input)?;

        println!("predictions shape: {:?}", predictions.shape());

        // Isolate predictions for the masked positions
        let predictions_squeezed = predictions.squeeze(0)?;
        let zeros = Tensor::zeros(predictions_squeezed.shape(), predictions_squeezed.dtype(), &device)?;
        let predicted_values_at_masked_positions = mask.where_cond(&predictions_squeezed, &zeros)?;

        // Calculate Loss (MSE for regression)
        let loss = loss::mse(&predicted_values_at_masked_positions, &true_labels)?;

        // Backward Pass and Optimizer Step
        optimizer.backward_step(&loss)?;

        println!(
            "Epoch: {:4} | Loss: {:8.5}",
            epoch,
            loss.to_scalar::<f32>()?
        );
    }

    Ok(())
}