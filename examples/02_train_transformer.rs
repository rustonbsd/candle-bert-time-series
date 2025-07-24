//! Train the transformer on orderbook prediction
//!
//! This example shows how to pre-train the financial BERT transformer
//! on orderbook data using masked language modeling approach.
//!
//! Usage:
//!   cargo run --example 02_train_transformer

use candle_bert_time_series::download::OrderbookDownloader;
use candle_bert_time_series::train::{TransformerTrainer, TrainingConfig};
use candle_core::Device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Transformer Pre-Training");
    println!("===========================");

    // Setup device
    let device = Device::Cpu; // Use CPU for now, can be changed to CUDA
    println!("🖥️  Using device: {:?}", device);

    // Load orderbook data
    let data_file = "data/orderbook_btcusdt_1_days.json";
    let demo_file = "data/orderbook_btcusdt_demo.json";
    
    let downloader = OrderbookDownloader::new();
    let orderbook_data = if Path::new(data_file).exists() {
        println!("📂 Loading real orderbook data from: {}", data_file);
        downloader.load_orderbook_data(data_file)?
    } else if Path::new(demo_file).exists() {
        println!("📂 Loading demo orderbook data from: {}", demo_file);
        downloader.load_orderbook_data(demo_file)?
    } else {
        println!("❌ No orderbook data found!");
        println!("Please run: cargo run --example 01_download_orderbook_data");
        return Ok(());
    };

    println!("✅ Loaded {} orderbook snapshots", orderbook_data.len());

    // Split data into train/validation
    let split_idx = (orderbook_data.len() as f64 * 0.8) as usize;
    let (train_data, val_data) = orderbook_data.split_at(split_idx);
    
    println!("📊 Data split:");
    println!("  Training: {} snapshots", train_data.len());
    println!("  Validation: {} snapshots", val_data.len());

    // Training configuration
    let config = TrainingConfig {
        batch_size: 16,
        sequence_length: 64,
        learning_rate: 1e-4,
        num_epochs: 5,
        warmup_steps: 500,
        save_every: 1000,
        eval_every: 200,
        mask_probability: 0.15,
        orderbook_depth: 10,
        hidden_size: 256,
        num_attention_heads: 8,
        num_hidden_layers: 6,
        intermediate_size: 1024,
    };

    println!("\n⚙️  Training Configuration:");
    println!("  Batch Size: {}", config.batch_size);
    println!("  Sequence Length: {}", config.sequence_length);
    println!("  Learning Rate: {}", config.learning_rate);
    println!("  Epochs: {}", config.num_epochs);
    println!("  Mask Probability: {}", config.mask_probability);
    println!("  Orderbook Depth: {}", config.orderbook_depth);
    println!("  Hidden Size: {}", config.hidden_size);
    println!("  Attention Heads: {}", config.num_attention_heads);
    println!("  Hidden Layers: {}", config.num_hidden_layers);

    // Initialize trainer
    println!("\n🏗️  Initializing transformer trainer...");
    let mut trainer = match TransformerTrainer::new(config, device) {
        Ok(trainer) => {
            println!("✅ Trainer initialized successfully");
            trainer
        }
        Err(e) => {
            println!("❌ Failed to initialize trainer: {}", e);
            return Err(e.into());
        }
    };

    // Start pre-training
    println!("\n🚀 Starting transformer pre-training...");
    println!("This will train the model to predict masked orderbook levels");
    println!("The model learns to understand orderbook patterns and dynamics");
    
    match trainer.pretrain(train_data, val_data) {
        Ok(()) => {
            println!("\n🎉 Pre-training completed successfully!");
            
            // Save final model
            let final_model_path = "checkpoints/final_pretrained_model.safetensors";
            trainer.save_checkpoint(final_model_path)?;
            
            println!("💾 Final model saved to: {}", final_model_path);
            
            println!("\n📊 Training Summary:");
            println!("  ✅ Transformer successfully pre-trained on orderbook data");
            println!("  ✅ Model learned to predict masked orderbook levels");
            println!("  ✅ Ready for PPO fine-tuning for trading decisions");
            
            println!("\n✨ Next Steps:");
            println!("  1. Run: cargo run --example 03_train_ppo_agent");
            println!("  2. Fine-tune the pre-trained transformer with PPO");
            println!("  3. The pre-trained model is saved in: {}", final_model_path);
        }
        Err(e) => {
            println!("❌ Pre-training failed: {}", e);
            println!("\n🔧 Troubleshooting:");
            println!("  - Check if you have enough memory");
            println!("  - Try reducing batch_size or sequence_length");
            println!("  - Ensure orderbook data is properly formatted");
            println!("  - Check device compatibility (CPU vs CUDA)");
            
            return Err(e.into());
        }
    }

    println!("\n🎯 Key Insights:");
    println!("  • The transformer learned orderbook patterns without human-engineered features");
    println!("  • Masked language modeling helps the model understand market microstructure");
    println!("  • Pre-training creates a strong foundation for trading strategy learning");
    println!("  • The model can now be fine-tuned with PPO for specific trading objectives");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_bert_time_series::download::{OrderbookSnapshot, OrderbookLevel};

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.sequence_length, 128);
        assert!(config.learning_rate > 0.0);
        assert!(config.mask_probability > 0.0 && config.mask_probability < 1.0);
    }

    #[test]
    fn test_data_preparation() {
        // Create sample orderbook data
        let mut orderbook_data = Vec::new();
        for i in 0..100 {
            orderbook_data.push(OrderbookSnapshot {
                timestamp: i,
                symbol: "BTCUSDT".to_string(),
                bid_levels: vec![
                    OrderbookLevel { price: 50000.0, quantity: 1.0 },
                    OrderbookLevel { price: 49999.0, quantity: 2.0 },
                ],
                ask_levels: vec![
                    OrderbookLevel { price: 50001.0, quantity: 1.5 },
                    OrderbookLevel { price: 50002.0, quantity: 2.5 },
                ],
                tick_size: 0.01,
                lot_size: 0.001,
            });
        }

        // Test data split
        let split_idx = (orderbook_data.len() as f64 * 0.8) as usize;
        let (train_data, val_data) = orderbook_data.split_at(split_idx);
        
        assert_eq!(train_data.len(), 80);
        assert_eq!(val_data.len(), 20);
        assert_eq!(train_data.len() + val_data.len(), orderbook_data.len());
    }
}
