// main.rs
pub mod dataset;
mod financial_bert;
use std::process::exit;

use candle_bert_time_series::batcher::Batcher;
use dataset::load_and_prepare_data;
use financial_bert::{Config, FinancialTransformerForMaskedRegression};

use candle_core::{scalar::TensorOrScalar, DType, Device, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use hftbacktest::prelude::*;

// --- Configuration ---
// NUM_TIME_SERIES will be determined dynamically from the data
const SEQUENCE_LENGTH: usize = 240; // 240
const MODEL_DIMS: usize = 128; // 384
const NUM_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;
const NUM_EPOCHS: usize = 500;
const LEARNING_RATE: f64 = 3e-6;
const BATCH_SIZE: usize = 256;   // Entire dataset rn: 1594848

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

   
    Ok(())
}