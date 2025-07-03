use candle_bert_time_series::backtest::{Backtester, TradingFees, TradeSide, PerformanceMetrics, extract_test_split, get_data_split_info};
use candle_bert_time_series::dataset::load_and_prepare_data;
use candle_core::{Device, Result};
use std::collections::HashMap;

/// Simple momentum strategy example
/// Buys assets with positive returns and sells those with negative returns
struct SimpleMomentumStrategy {
    lookback_period: usize,
    position_size: f64, // Fraction of portfolio to allocate per position
    rebalance_frequency: usize, // Rebalance every N periods
}

impl SimpleMomentumStrategy {
    fn new(lookback_period: usize, position_size: f64, rebalance_frequency: usize) -> Self {
        Self {
            lookback_period,
            position_size,
            rebalance_frequency,
        }
    }

    /// Calculate momentum signals for all assets
    fn calculate_signals(
        &self,
        backtester: &Backtester,
        current_timestamp: usize,
    ) -> Result<HashMap<String, f64>> {
        let mut signals = HashMap::new();
        
        if current_timestamp < self.lookback_period {
            return Ok(signals);
        }

        // Calculate momentum for each asset over the lookback period
        for (asset_idx, symbol) in backtester.get_symbol_names().iter().enumerate() {
            let mut cumulative_return = 1.0;
            
            // Calculate cumulative return over lookback period
            for t in (current_timestamp - self.lookback_period)..current_timestamp {
                let returns_row = backtester.returns_data.get(t)?;
                let returns_vec: Vec<f32> = returns_row.to_vec1()?;
                let asset_return = returns_vec[asset_idx] as f64;
                cumulative_return *= 1.0 + asset_return;
            }
            
            // Convert to total return
            let total_return = cumulative_return - 1.0;
            
            // Simple momentum signal: positive if return > 0, negative otherwise
            let signal = if total_return > 0.01 { 1.0 } else if total_return < -0.01 { -1.0 } else { 0.0 };
            signals.insert(symbol.clone(), signal);
        }

        Ok(signals)
    }

    /// Execute strategy for one time step
    fn execute_step(
        &self,
        backtester: &mut Backtester,
        current_timestamp: usize,
    ) -> Result<()> {
        // Only rebalance at specified frequency
        if current_timestamp % self.rebalance_frequency != 0 {
            return Ok(());
        }

        // Calculate signals
        let signals = self.calculate_signals(backtester, current_timestamp)?;

        // Get portfolio info before making trades
        let (total_value, positions_to_close) = {
            let current_portfolio = backtester.get_current_portfolio();
            let total_value = current_portfolio.total_value;
            let positions_to_close: Vec<(String, f64)> = current_portfolio.positions
                .iter()
                .map(|(symbol, position)| (symbol.clone(), position.quantity))
                .collect();
            (total_value, positions_to_close)
        };

        // Close all existing positions first
        for (symbol, quantity) in positions_to_close {
            backtester.execute_trade(
                &symbol,
                TradeSide::Sell,
                quantity,
                current_timestamp,
            )?;
        }

        // Open new positions based on signals
        for (symbol, signal) in signals {
            if signal.abs() > 0.5 { // Only trade if signal is strong enough
                let position_value = total_value * self.position_size;
                let current_price = backtester.get_current_price(&symbol).unwrap_or(100.0);
                let quantity = position_value / current_price;
                
                if signal > 0.0 && quantity > 0.0 {
                    // Buy signal
                    backtester.execute_trade(
                        &symbol,
                        TradeSide::Buy,
                        quantity,
                        current_timestamp,
                    )?;
                }
            }
        }

        Ok(())
    }
}

/// Run a complete backtest with the simple momentum strategy
fn run_backtest() -> Result<()> {
    println!("🚀 Starting Simple Momentum Strategy Backtest");
    println!("{}", "=".repeat(60));
    println!("⚠️  IMPORTANT: Using ONLY test split data to prevent data leakage!");
    println!("🔒 This ensures fair evaluation - no training/validation data used");
    println!("{}", "=".repeat(60));

    // Configuration
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    let initial_capital = 100_0000.0; // $100k starting capital
    let device = Device::Cpu; // Use CPU for this example

    // Load data
    println!("Loading cryptocurrency data...");
    let (full_data_sequence, num_assets) = load_and_prepare_data(data_path, &device)?;
    let total_timesteps = full_data_sequence.dims()[0];

    // Extract ONLY the test split to prevent data leakage
    let test_data = extract_test_split(&full_data_sequence)?;
    let test_timesteps = test_data.dims()[0];

    // Get split information for reporting
    let (train_size, val_size, test_size) = get_data_split_info(total_timesteps);

    println!("Data loaded: {} total timesteps, {} assets", total_timesteps, num_assets);
    println!("Using ONLY test split: {} timesteps ({}% of data)", test_timesteps,
             (test_timesteps as f32 / total_timesteps as f32) * 100.0);
    println!("Data splits - Train: {}, Validation: {}, Test: {}",
             train_size, val_size, test_size);
    println!("No data leakage - test split extracted using same logic as training loop");

    // Create symbol names (assuming they follow the pattern from dataset creation)
    let symbol_names: Vec<String> = (0..num_assets)
        .map(|i| format!("ASSET_{}", i))
        .collect();

    // Initialize backtester with Binance fees using ONLY test data
    // Allow negative cash (leverage/margin trading) to see full P&L potential
    let fees = TradingFees::default(); // 0.1% maker/taker fees
    let mut backtester = Backtester::new_with_leverage(
        initial_capital,
        test_data, // Only test split - no leakage!
        symbol_names,
        Some(fees),
        true, // Allow negative cash - no "insufficient funds" errors
    )?;

    // Initialize strategy
    let strategy = SimpleMomentumStrategy::new(
        60,    // 60-minute lookback period
        0.05,  // 5% position size per asset
        30,    // Rebalance every 30 minutes
    );

    println!("Running backtest...");
    println!("Strategy parameters:");
    println!("  - Lookback period: {} minutes", strategy.lookback_period);
    println!("  - Position size: {:.1}% per asset", strategy.position_size * 100.0);
    println!("  - Rebalance frequency: {} minutes", strategy.rebalance_frequency);
    println!("  - Leverage allowed: YES (can go negative cash)");

    // Run backtest on TEST DATA ONLY
    let start_time = std::time::Instant::now();
    let mut progress_counter = 0;

    for timestamp in 1..test_timesteps {
        // Step forward in time (update prices)
        backtester.step_forward(timestamp)?;

        // Execute strategy
        strategy.execute_step(&mut backtester, timestamp)?;

        // Progress reporting
        progress_counter += 1;
        if progress_counter % 1000 == 0 {
            let progress = (timestamp as f64 / test_timesteps as f64) * 100.0;
            println!("  Progress: {:.1}% ({}/{})", progress, timestamp, test_timesteps);
        }
    }

    let elapsed = start_time.elapsed();
    println!("✅ Backtest completed in {:.2} seconds", elapsed.as_secs_f64());

    // Calculate and display performance metrics
    println!("\n📊 PERFORMANCE RESULTS");
    println!("{}", "=".repeat(60));
    
    let metrics = backtester.calculate_metrics()?;
    display_performance_metrics(&metrics, initial_capital);

    // Display trade summary
    let trades = backtester.get_trades();
    println!("\n📋 TRADE SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Total trades executed: {}", trades.len());
    println!("Total fees paid: ${:.2}", metrics.total_fees);
    
    if trades.len() > 0 {
        let avg_trade_size = trades.iter().map(|t| t.quantity * t.price).sum::<f64>() / trades.len() as f64;
        println!("Average trade size: ${:.2}", avg_trade_size);
        
        let buy_trades = trades.iter().filter(|t| t.side == TradeSide::Buy).count();
        let sell_trades = trades.iter().filter(|t| t.side == TradeSide::Sell).count();
        println!("Buy trades: {}, Sell trades: {}", buy_trades, sell_trades);
    }

    // Display final portfolio state
    let final_portfolio = backtester.get_current_portfolio();
    println!("\n💼 FINAL PORTFOLIO");
    println!("{}", "=".repeat(60));
    println!("Cash: ${:.2}", final_portfolio.cash);
    if final_portfolio.cash < 0.0 {
        println!("  ⚠️  NEGATIVE CASH (Margin used: ${:.2})", final_portfolio.margin_used);
    }
    println!("Positions: {}", final_portfolio.positions.len());
    println!("Total value: ${:.2}", final_portfolio.total_value);
    println!("Unrealized PnL: ${:.2}", final_portfolio.unrealized_pnl);
    println!("Realized PnL: ${:.2}", final_portfolio.realized_pnl);
    println!("Current leverage: {:.2}x", final_portfolio.leverage_ratio);

    Ok(())
}

/// Display performance metrics in a formatted way
fn display_performance_metrics(metrics: &PerformanceMetrics, initial_capital: f64) {
    println!("Initial Capital: ${:.2}", initial_capital);
    println!("Final Portfolio Value: ${:.2}", metrics.final_portfolio_value);
    println!("Total Return: {:.2}%", metrics.total_return * 100.0);
    println!("Annualized Return: {:.2}%", metrics.annualized_return * 100.0);
    println!("Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("Maximum Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("Max Drawdown Duration: {} periods", metrics.max_drawdown_duration);
    println!("Win Rate: {:.1}%", metrics.win_rate * 100.0);
    println!("Profit Factor: {:.2}", metrics.profit_factor);
    println!("Total Trades: {}", metrics.total_trades);
    println!("Total Fees: ${:.2}", metrics.total_fees);

    // Leverage and margin metrics
    println!("\n📊 LEVERAGE & MARGIN METRICS");
    println!("Max Leverage Used: {:.2}x", metrics.max_leverage);
    println!("Max Margin Used: ${:.2}", metrics.max_margin_used);
    println!("Min Cash Balance: ${:.2}", metrics.min_cash_balance);
    if metrics.min_cash_balance < 0.0 {
        println!("  ⚠️  Strategy went into margin (negative cash)");
    }

    // Risk-adjusted metrics
    let profit_loss = metrics.final_portfolio_value - initial_capital;
    println!("\n📈 RISK-ADJUSTED METRICS");
    println!("Net Profit/Loss: ${:.2}", profit_loss);

    if metrics.max_drawdown > 0.0 {
        let calmar_ratio = metrics.annualized_return / metrics.max_drawdown;
        println!("Calmar Ratio: {:.3}", calmar_ratio);
    }
}

fn main() -> Result<()> {
    match run_backtest() {
        Ok(()) => {
            println!("\n🎉 Backtest completed successfully!");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Backtest failed: {}", e);
            Err(e)
        }
    }
}
