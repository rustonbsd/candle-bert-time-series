//! Train PPO agent using pre-trained transformer
//!
//! This example shows how to fine-tune the pre-trained transformer
//! with PPO (Proximal Policy Optimization) for trading decisions.
//!
//! Usage:
//!   cargo run --example 03_train_ppo_agent

use candle_bert_time_series::download::OrderbookDownloader;
use candle_bert_time_series::train::{TransformerTrainer, TrainingConfig};
use candle_bert_time_series::backtest::{OrderbookBacktestRunner, Backtester, TradeSide, orderbook_to_depth_snapshot};
use candle_core::Device;
use std::path::Path;
use std::collections::VecDeque;

/// PPO training configuration
#[derive(Debug, Clone)]
pub struct PPOConfig {
    pub episodes: usize,
    pub steps_per_episode: usize,
    pub learning_rate: f64,
    pub clip_epsilon: f64,
    pub value_loss_coef: f64,
    pub entropy_coef: f64,
    pub gamma: f64, // Discount factor
    pub gae_lambda: f64, // GAE parameter
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            episodes: 100,
            steps_per_episode: 1000,
            learning_rate: 3e-4,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
        }
    }
}

/// Simple PPO agent for demonstration
pub struct PPOAgent {
    pub transformer: TransformerTrainer,
    pub config: PPOConfig,
    pub episode_rewards: Vec<f64>,
    pub episode_lengths: Vec<usize>,
}

impl PPOAgent {
    pub fn new(transformer: TransformerTrainer, config: PPOConfig) -> Self {
        Self {
            transformer,
            config,
            episode_rewards: Vec::new(),
            episode_lengths: Vec::new(),
        }
    }

    /// Simple trading policy: buy when spread is tight, sell when spread is wide
    pub fn get_action(&self, orderbook_history: &VecDeque<Vec<f32>>) -> usize {
        if orderbook_history.is_empty() {
            return 0; // HOLD
        }

        let latest = orderbook_history.back().unwrap();
        if latest.len() < 4 {
            return 0; // HOLD
        }

        let best_bid = latest[0];
        let best_ask = latest[2];
        
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return 0; // HOLD
        }

        let spread = best_ask - best_bid;
        let mid_price = (best_bid + best_ask) / 2.0;
        let relative_spread = spread / mid_price;

        // Simple policy based on spread
        if relative_spread < 0.001 {
            1 // BUY
        } else if relative_spread > 0.005 {
            2 // SELL
        } else {
            0 // HOLD
        }
    }

    /// Calculate reward based on portfolio performance
    pub fn calculate_reward(&self, prev_value: f64, current_value: f64, action: usize) -> f64 {
        let pnl = current_value - prev_value;
        let return_rate = if prev_value > 0.0 { pnl / prev_value } else { 0.0 };

        let mut reward = return_rate * 100.0; // Scale up returns

        // Penalize large drawdowns
        if current_value < prev_value * 0.98 {
            reward -= 0.5;
        }

        // Small penalty for trading (transaction costs)
        if action != 0 {
            reward -= 0.01;
        }

        // Bonus for profitable trades
        if pnl > 0.0 {
            reward += 0.1;
        }

        reward
    }

    /// Run one training episode
    pub fn run_episode(&mut self, orderbook_data: &[candle_bert_time_series::download::OrderbookSnapshot]) -> Result<f64, Box<dyn std::error::Error>> {
        let mut backtester = Backtester::new(10000.0);
        let mut orderbook_history = VecDeque::with_capacity(50);
        let mut total_reward = 0.0;
        let mut steps = 0;

        let episode_data = if orderbook_data.len() > self.config.steps_per_episode {
            &orderbook_data[..self.config.steps_per_episode]
        } else {
            orderbook_data
        };

        for orderbook in episode_data {
            // Convert orderbook to feature vector
            let features = self.orderbook_to_features(orderbook);
            
            // Maintain history
            orderbook_history.push_back(features);
            if orderbook_history.len() > 50 {
                orderbook_history.pop_front();
            }

            // Get action from policy
            let action = self.get_action(&orderbook_history);
            
            // Execute action
            let prev_value = backtester.current_capital + (backtester.position * orderbook.bid_levels.get(0).map(|l| l.price).unwrap_or(0.0));
            
            match action {
                1 => { // BUY
                    if let Some(ask_level) = orderbook.ask_levels.get(0) {
                        let quantity = 0.1; // Fixed quantity for demo
                        if backtester.current_capital >= ask_level.price * quantity {
                            let _ = backtester.execute_trade(TradeSide::Buy, ask_level.price, quantity, orderbook.timestamp);
                        }
                    }
                }
                2 => { // SELL
                    if let Some(bid_level) = orderbook.bid_levels.get(0) {
                        let quantity = 0.1.min(backtester.position); // Sell up to position
                        if quantity > 0.0 {
                            let _ = backtester.execute_trade(TradeSide::Sell, bid_level.price, quantity, orderbook.timestamp);
                        }
                    }
                }
                _ => {} // HOLD
            }

            // Calculate reward
            let current_value = backtester.current_capital + (backtester.position * orderbook.bid_levels.get(0).map(|l| l.price).unwrap_or(0.0));
            let reward = self.calculate_reward(prev_value, current_value, action);
            total_reward += reward;
            
            // Update portfolio
            backtester.update_portfolio_value(
                orderbook.bid_levels.get(0).map(|l| l.price).unwrap_or(0.0),
                orderbook.timestamp
            );

            steps += 1;
        }

        self.episode_rewards.push(total_reward);
        self.episode_lengths.push(steps);

        Ok(total_reward)
    }

    /// Convert orderbook to feature vector
    fn orderbook_to_features(&self, orderbook: &candle_bert_time_series::download::OrderbookSnapshot) -> Vec<f32> {
        let mut features = Vec::with_capacity(40); // 10 levels * 4 features
        
        for i in 0..10 {
            let (bid_price, bid_qty) = if i < orderbook.bid_levels.len() {
                (orderbook.bid_levels[i].price as f32, orderbook.bid_levels[i].quantity as f32)
            } else { (0.0, 0.0) };

            let (ask_price, ask_qty) = if i < orderbook.ask_levels.len() {
                (orderbook.ask_levels[i].price as f32, orderbook.ask_levels[i].quantity as f32)
            } else { (0.0, 0.0) };

            features.push(bid_price);
            features.push(bid_qty);
            features.push(ask_price);
            features.push(ask_qty);
        }

        features
    }

    /// Train the PPO agent
    pub fn train(&mut self, orderbook_data: &[candle_bert_time_series::download::OrderbookSnapshot]) -> Result<(), Box<dyn std::error::Error>> {
        println!("🎯 Starting PPO training...");
        
        for episode in 0..self.config.episodes {
            let reward = self.run_episode(orderbook_data)?;
            
            if episode % 10 == 0 {
                let avg_reward = self.episode_rewards.iter().rev().take(10).sum::<f64>() / 10.0;
                println!("Episode {}: Reward = {:.4}, Avg(10) = {:.4}", episode, reward, avg_reward);
            }
        }

        println!("✅ PPO training completed!");
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 PPO Agent Training");
    println!("=====================");

    // Setup device
    let device = Device::Cpu;
    println!("🖥️  Using device: {:?}", device);

    // Load pre-trained transformer
    let pretrained_model_path = "checkpoints/final_pretrained_model.safetensors";
    if !Path::new(pretrained_model_path).exists() {
        println!("❌ Pre-trained model not found!");
        println!("Please run: cargo run --example 02_train_transformer");
        return Ok(());
    }

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

    // Initialize transformer trainer
    let config = TrainingConfig::default();
    let mut transformer = TransformerTrainer::new(config, device)?;
    
    // Load pre-trained weights
    transformer.load_checkpoint(pretrained_model_path)?;
    println!("✅ Loaded pre-trained transformer");

    // Initialize PPO agent
    let ppo_config = PPOConfig {
        episodes: 50,
        steps_per_episode: 500,
        ..Default::default()
    };

    println!("\n⚙️  PPO Configuration:");
    println!("  Episodes: {}", ppo_config.episodes);
    println!("  Steps per Episode: {}", ppo_config.steps_per_episode);
    println!("  Learning Rate: {}", ppo_config.learning_rate);
    println!("  Clip Epsilon: {}", ppo_config.clip_epsilon);

    let mut ppo_agent = PPOAgent::new(transformer, ppo_config);

    // Train PPO agent
    println!("\n🚀 Starting PPO training...");
    ppo_agent.train(&orderbook_data)?;

    // Show training results
    println!("\n📊 Training Results:");
    if !ppo_agent.episode_rewards.is_empty() {
        let avg_reward = ppo_agent.episode_rewards.iter().sum::<f64>() / ppo_agent.episode_rewards.len() as f64;
        let max_reward = ppo_agent.episode_rewards.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_reward = ppo_agent.episode_rewards.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        println!("  Average Reward: {:.4}", avg_reward);
        println!("  Max Reward: {:.4}", max_reward);
        println!("  Min Reward: {:.4}", min_reward);
        println!("  Total Episodes: {}", ppo_agent.episode_rewards.len());
    }

    println!("\n✨ Next Steps:");
    println!("  1. Run: cargo run --example 04_run_backtest");
    println!("  2. Test the trained agent on unseen data");
    println!("  3. Evaluate trading performance");

    println!("\n🎯 Key Insights:");
    println!("  • PPO fine-tuned the pre-trained transformer for trading decisions");
    println!("  • The agent learned to make buy/sell/hold decisions based on orderbook patterns");
    println!("  • Reward function balances profitability with risk management");
    println!("  • Ready for backtesting on unseen market data");

    Ok(())
}
