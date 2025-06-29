// main.rs

mod financial_bert;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};

// --- Configuration (no changes) ---
const NUM_TIME_SERIES: usize = 100;
const SEQUENCE_LENGTH: usize = 64;
const MODEL_DIMS: usize = 256;
const NUM_LAYERS: usize = 4;
const NUM_HEADS: usize = 8;
const NUM_EPOCHS: usize = 50;
const LEARNING_RATE: f64 = 1e-4;
const MASK_PROB: f32 = 0.15;

// --- Data Generation and Preparation ---

/// This function is correct. It produces a 2D tensor.
/// Shape: [timesteps, NUM_TIME_SERIES] -> e.g., [1024, 100]
fn generate_dummy_data(timesteps: usize, device: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 0.02f32, (timesteps, NUM_TIME_SERIES), device)
}

/// Corrected and simplified masking function.
fn mask_data(
    input: &Tensor, // Expects a 2D tensor: [SEQUENCE_LENGTH, NUM_TIME_SERIES] -> [64, 100]
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = input.shape();
    let rand_mask = Tensor::rand(0f32, 1f32, shape, device)?;
    // `mask` is a boolean tensor of shape [64, 100]
    let mask = (rand_mask.lt(MASK_PROB))?;
    let zeros = Tensor::zeros(shape, input.dtype(), device)?;

    // `masked_select` correctly gathers only the masked values into a 1D tensor.
    // Shape: [num_masked_elements]
    let true_labels = mask.where_cond(input, &zeros)?;

    // Zero out the values in the input where the mask is true.
    // This preserves the original shape of the input.
    // Shape: [64, 100]
    //let masked_input = input.broadcast_mul(&mask.logical_not()?.to_dtype(DType::F32)?)?;
    // 1. Create a tensor of ones with the same shape as the mask.
    //    The `lt` operation produces a U8 tensor, so we use DType::U8.
    let ones = Tensor::ones(shape, DType::U8, device)?;
    // 2. Subtract the mask from ones. If mask is [1, 0, 1], result is [0, 1, 0].
    let inverted_mask = ones.sub(&mask)?;
    // 3. Use this inverted mask to create the model input.
    let masked_input = input.broadcast_mul(&inverted_mask.to_dtype(DType::F32)?)?;

    Ok((masked_input, true_labels, mask))
}

// --- The Main Training Function ---

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Training on device: {:?}", device);

    // --- Model and Optimizer Setup (no changes) ---
    let config = Config {
        num_time_series: NUM_TIME_SERIES,
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

    let adamw_params = candle_nn::ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    // --- Data Loading ---
    // `full_data_sequence` Shape: [1024, 100]
    let full_data_sequence = generate_dummy_data(1024, &device)?;

    println!("Starting training...");
    for epoch in 0..NUM_EPOCHS {
        // --- Batch Preparation ---
        // `batch` Shape: [64, 100] (A 2D slice of the full data)
        let batch = full_data_sequence.narrow(0, 0, SEQUENCE_LENGTH)?;

        // `masked_input` Shape: [64, 100]
        // `true_labels` Shape: [num_masked] (1D)
        // `mask` Shape: [64, 100] (boolean)
        let (masked_input, true_labels, mask) = mask_data(&batch, &device)?;

        // --- FORWARD PASS ---
        // The model expects a 3D tensor: (batch_size, seq_len, features).
        // We add a batch dimension of 1.
        // `model_input` Shape: [64, 100] -> [1, 64, 100]
        let model_input = masked_input.unsqueeze(0)?;
        

        // `predictions` will have the same shape as the input.
        // `predictions` Shape: [1, 64, 100]
        let predictions = model.forward(&model_input)?;

        // --- LOSS CALCULATION ---
        // To compare with our labels, we must isolate the predictions at the masked positions.
        // First, remove the batch dimension.
        // `predictions_squeezed` Shape: [1, 64, 100] -> [64, 100]
        let predictions_squeezed = predictions.squeeze(0)?;

        // Now, use the same boolean mask to select the predicted values.
        // This flattens the tensor, matching the shape of `true_labels`.
        // `predicted_values` Shape: [num_masked] (1D)
        let zeros = Tensor::zeros(predictions_squeezed.shape(), predictions_squeezed.dtype(), &device)?;
        let predicted_values = mask.where_cond(&predictions_squeezed, &zeros)?;

        // Now we can compare the two 1D tensors.
        let loss = loss::mse(&predicted_values, &true_labels)?;

        // --- BACKWARD PASS ---
        optimizer.backward_step(&loss)?;

        println!(
            "Epoch: {:4} | Loss: {:8.5}",
            epoch,
            loss.to_scalar::<f32>()?
        );
    }

    Ok(())
}