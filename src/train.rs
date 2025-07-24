//! Transformer training module for financial BERT
//!
//! This module implements the training pipeline:
//! 1. Pre-train transformer on orderbook prediction (masked language modeling)
//! 2. Fine-tune with PPO for trading decisions
//!
//! The training follows a two-stage approach similar to NLP models.

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, AdamW, ParamsAdamW, Optimizer};
use crate::download::OrderbookSnapshot;
use crate::financial_bert::{FinancialTransformerForMaskedRegression, Config, HiddenAct, PositionEmbeddingType};
use rand::Rng;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub warmup_steps: usize,
    pub save_every: usize,
    pub eval_every: usize,
    pub mask_probability: f64,
    pub orderbook_depth: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            sequence_length: 128,
            learning_rate: 1e-4,
            num_epochs: 10,
            warmup_steps: 1000,
            save_every: 1000,
            eval_every: 500,
            mask_probability: 0.15,
            orderbook_depth: 10,
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
        }
    }
}

/// Orderbook data preprocessor
pub struct OrderbookPreprocessor {
    pub config: TrainingConfig,
    pub feature_size: usize, // orderbook_depth * 4 (bid_price, bid_qty, ask_price, ask_qty)
}

impl OrderbookPreprocessor {
    pub fn new(config: TrainingConfig) -> Self {
        let feature_size = config.orderbook_depth * 4;
        Self { config, feature_size }
    }

    /// Convert orderbook snapshot to feature vector
    pub fn snapshot_to_features(&self, snapshot: &OrderbookSnapshot) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.feature_size);
        
        // Extract raw orderbook levels: [bid_price_1, bid_qty_1, ask_price_1, ask_qty_1, ...]
        for i in 0..self.config.orderbook_depth {
            let (bid_price, bid_qty) = if i < snapshot.bid_levels.len() {
                (snapshot.bid_levels[i].price as f32, snapshot.bid_levels[i].quantity as f32)
            } else { (0.0, 0.0) };

            let (ask_price, ask_qty) = if i < snapshot.ask_levels.len() {
                (snapshot.ask_levels[i].price as f32, snapshot.ask_levels[i].quantity as f32)
            } else { (0.0, 0.0) };

            features.push(bid_price);
            features.push(bid_qty);
            features.push(ask_price);
            features.push(ask_qty);
        }

        features
    }

    /// Create training sequences from orderbook data
    pub fn create_sequences(&self, snapshots: &[OrderbookSnapshot]) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut sequences = Vec::new();
        
        if snapshots.len() < self.config.sequence_length {
            return Ok(sequences);
        }

        for i in 0..=(snapshots.len() - self.config.sequence_length) {
            let mut sequence = Vec::new();
            
            for j in 0..self.config.sequence_length {
                let features = self.snapshot_to_features(&snapshots[i + j]);
                sequence.push(features);
            }
            
            sequences.push(sequence);
        }

        println!("Created {} training sequences of length {}", sequences.len(), self.config.sequence_length);
        Ok(sequences)
    }

    /// Apply masking for pre-training (mask random orderbook levels)
    pub fn apply_masking(&self, sequence: &[Vec<f32>], device: &Device) -> Result<(Tensor, Tensor)> {
        let mut masked_sequence = sequence.to_vec();
        let mut mask = vec![vec![false; self.feature_size]; sequence.len()];
        let mut rng = rand::thread_rng();

        // Mask random positions
        for t in 0..sequence.len() {
            for i in 0..self.feature_size {
                if rng.gen::<f64>() < self.config.mask_probability {
                    masked_sequence[t][i] = 0.0; // Mask with zero
                    mask[t][i] = true;
                }
            }
        }

        // Convert to tensors
        let input_data: Vec<f32> = masked_sequence.into_iter().flatten().collect();
        let mask_data: Vec<f32> = mask.into_iter()
            .flatten()
            .map(|m| if m { 1.0 } else { 0.0 })
            .collect();

        let input_tensor = Tensor::from_vec(
            input_data,
            (sequence.len(), self.feature_size),
            device,
        )?;

        let mask_tensor = Tensor::from_vec(
            mask_data,
            (sequence.len(), self.feature_size),
            device,
        )?;

        Ok((input_tensor, mask_tensor))
    }
}

/// Transformer trainer
pub struct TransformerTrainer {
    pub config: TrainingConfig,
    pub preprocessor: OrderbookPreprocessor,
    pub model: FinancialTransformerForMaskedRegression,
    pub optimizer: AdamW,
    pub device: Device,
    pub varmap: VarMap,
}

impl TransformerTrainer {
    pub fn new(config: TrainingConfig, device: Device) -> Result<Self> {
        let preprocessor = OrderbookPreprocessor::new(config.clone());

        // Create financial transformer config
        let model_config = Config {
            num_time_series: preprocessor.feature_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: config.sequence_length,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: false,
            model_type: Some("financial_bert".to_string()),
        };

        // Initialize model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = FinancialTransformerForMaskedRegression::load(vb, &model_config)?;

        // Initialize optimizer
        let params = ParamsAdamW {
            lr: config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self {
            config,
            preprocessor,
            model,
            optimizer,
            device,
            varmap,
        })
    }

    /// Pre-train transformer on orderbook prediction
    pub fn pretrain(&mut self, train_data: &[OrderbookSnapshot], val_data: &[OrderbookSnapshot]) -> Result<()> {
        println!("🚀 Starting transformer pre-training...");
        println!("Training samples: {}, Validation samples: {}", train_data.len(), val_data.len());

        // Create training sequences
        let train_sequences = self.preprocessor.create_sequences(train_data)?;
        let val_sequences = self.preprocessor.create_sequences(val_data)?;

        let mut step = 0;
        let mut best_val_loss = f32::INFINITY;

        for epoch in 0..self.config.num_epochs {
            println!("\n📚 Epoch {}/{}", epoch + 1, self.config.num_epochs);
            
            // Training loop
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for batch_start in (0..train_sequences.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(train_sequences.len());
                let batch_sequences = &train_sequences[batch_start..batch_end];

                let mut batch_loss = 0.0;

                for sequence in batch_sequences {
                    // Apply masking
                    let (masked_input, mask) = self.preprocessor.apply_masking(sequence, &self.device)?;
                    
                    // Forward pass
                    let predictions = self.model.forward(&masked_input)?;
                    
                    // Calculate reconstruction loss (MSE on masked positions)
                    let target = Tensor::from_vec(
                        sequence.iter().flatten().cloned().collect::<Vec<f32>>(),
                        (sequence.len(), self.preprocessor.feature_size),
                        &self.device,
                    )?;
                    
                    let loss = self.calculate_masked_loss(&predictions, &target, &mask)?;
                    batch_loss += loss.to_scalar::<f32>()?;

                    // Backward pass
                    self.optimizer.backward_step(&loss)?;
                }

                batch_loss /= batch_sequences.len() as f32;
                epoch_loss += batch_loss;
                num_batches += 1;
                step += 1;

                // Logging
                if step % 100 == 0 {
                    println!("  Step {}: Loss = {:.6}", step, batch_loss);
                }

                // Evaluation
                if step % self.config.eval_every == 0 {
                    let val_loss = self.evaluate(&val_sequences)?;
                    println!("  📊 Validation Loss: {:.6}", val_loss);
                    
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        self.save_checkpoint(&format!("checkpoints/best_model_step_{}.safetensors", step))?;
                    }
                }

                // Save checkpoint
                if step % self.config.save_every == 0 {
                    self.save_checkpoint(&format!("checkpoints/model_step_{}.safetensors", step))?;
                }
            }

            epoch_loss /= num_batches as f32;
            println!("📈 Epoch {} Average Loss: {:.6}", epoch + 1, epoch_loss);
        }

        println!("✅ Pre-training completed!");
        Ok(())
    }

    /// Calculate masked reconstruction loss
    fn calculate_masked_loss(&self, predictions: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // MSE loss only on masked positions
        let diff = (predictions - targets)?;
        let squared_diff = diff.sqr()?;
        let masked_loss = (squared_diff * mask)?;
        let total_loss = masked_loss.sum_all()?;
        let num_masked = mask.sum_all()?;

        // Average over masked positions
        Ok(total_loss.div(&num_masked)?)
    }

    /// Evaluate model on validation data
    fn evaluate(&self, val_sequences: &[Vec<Vec<f32>>]) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut num_samples = 0;

        for sequence in val_sequences.iter().take(100) { // Evaluate on subset for speed
            let (masked_input, mask) = self.preprocessor.apply_masking(sequence, &self.device)?;
            let predictions = self.model.forward(&masked_input)?;
            
            let target = Tensor::from_vec(
                sequence.iter().flatten().cloned().collect::<Vec<f32>>(),
                (sequence.len(), self.preprocessor.feature_size),
                &self.device,
            )?;
            
            let loss = self.calculate_masked_loss(&predictions, &target, &mask)?;
            total_loss += loss.to_scalar::<f32>()?;
            num_samples += 1;
        }

        Ok(total_loss / num_samples as f32)
    }

    /// Save model checkpoint
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Save model weights using safetensors
        self.varmap.save(path)?;
        println!("💾 Saved checkpoint: {}", path);
        Ok(())
    }

    /// Load model checkpoint
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        println!("📂 Loaded checkpoint: {}", path);
        Ok(())
    }
}
