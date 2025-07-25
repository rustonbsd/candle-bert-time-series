// analyze_quantity_distribution.rs
// Analyze the actual distribution of orderbook quantities to understand the normalization problem

use polars::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "/mnt/storage-box/bybit/orderbooks/output/2025-04-30_BTCUSDC_financialbert.parquet";
    
    if !Path::new(file_path).exists() {
        println!("File not found: {}", file_path);
        return Ok(());
    }

    println!("🔍 Analyzing quantity distribution in: {}", file_path);
    
    // Load the data
    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .collect()?;

    // Get all quantity columns
    let column_names = df.get_column_names();
    let qty_columns: Vec<_> = column_names.iter()
        .filter(|name| name.contains("_qty_"))
        .collect();

    println!("Found {} quantity columns", qty_columns.len());

    // Collect all quantity values
    let mut all_quantities = Vec::new();
    for col_name in &qty_columns {
        let column = df.column(col_name)?;
        let values: Vec<f64> = column
            .f32()?
            .into_iter()
            .filter_map(|opt_val| opt_val.map(|v| v as f64))
            .collect();
        all_quantities.extend(values);
    }

    println!("Total quantity values: {}", all_quantities.len());

    // Sort for percentile analysis
    all_quantities.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate detailed statistics
    let mean = all_quantities.iter().sum::<f64>() / all_quantities.len() as f64;
    let median = all_quantities[all_quantities.len() / 2];
    
    let variance = all_quantities.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / all_quantities.len() as f64;
    let std_dev = variance.sqrt();

    println!("\n📊 QUANTITY DISTRIBUTION ANALYSIS:");
    println!("  Mean: {:.6}", mean);
    println!("  Median: {:.6}", median);
    println!("  Std Dev: {:.6}", std_dev);
    println!("  Min: {:.6}", all_quantities[0]);
    println!("  Max: {:.6}", all_quantities[all_quantities.len() - 1]);

    // Percentiles
    let percentiles = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 99.99];
    println!("\n📈 PERCENTILES:");
    for p in percentiles {
        let idx = ((p / 100.0) * all_quantities.len() as f64) as usize;
        let idx = idx.min(all_quantities.len() - 1);
        println!("  P{:5.2}: {:12.6}", p, all_quantities[idx]);
    }

    // Check how many values are beyond different std dev thresholds
    println!("\n⚠️  OUTLIER ANALYSIS:");
    let thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
    for threshold in thresholds {
        let lower_bound = mean - threshold * std_dev;
        let upper_bound = mean + threshold * std_dev;
        
        let outliers = all_quantities.iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .count();
        
        let outlier_pct = (outliers as f64 / all_quantities.len() as f64) * 100.0;
        println!("  Beyond ±{} std: {} values ({:.3}%)", threshold, outliers, outlier_pct);
    }

    // Check distribution shape
    println!("\n📐 DISTRIBUTION SHAPE:");
    let skewness = calculate_skewness(&all_quantities, mean, std_dev);
    let kurtosis = calculate_kurtosis(&all_quantities, mean, std_dev);
    println!("  Skewness: {:.3} (0=normal, >0=right-skewed, <0=left-skewed)", skewness);
    println!("  Kurtosis: {:.3} (3=normal, >3=heavy-tailed, <3=light-tailed)", kurtosis);

    // Analyze zero values
    let zero_count = all_quantities.iter().filter(|&&x| x.abs() < 1e-6).count();
    let zero_pct = (zero_count as f64 / all_quantities.len() as f64) * 100.0;
    println!("  Zero values: {} ({:.2}%)", zero_count, zero_pct);

    // Check if log-normal distribution might be better
    let positive_values: Vec<f64> = all_quantities.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x.ln())
        .collect();
    
    if !positive_values.is_empty() {
        let log_mean = positive_values.iter().sum::<f64>() / positive_values.len() as f64;
        let log_variance = positive_values.iter()
            .map(|&x| (x - log_mean).powi(2))
            .sum::<f64>() / positive_values.len() as f64;
        let log_std = log_variance.sqrt();
        
        println!("\n🔬 LOG-NORMAL ANALYSIS (for positive values only):");
        println!("  Log mean: {:.6}", log_mean);
        println!("  Log std: {:.6}", log_std);
        println!("  Positive values: {} ({:.2}%)", 
                 positive_values.len(), 
                 (positive_values.len() as f64 / all_quantities.len() as f64) * 100.0);
    }

    println!("\n💡 RECOMMENDATIONS:");
    if skewness > 2.0 {
        println!("  - Data is heavily right-skewed, consider log transformation");
    }
    if kurtosis > 10.0 {
        println!("  - Data has heavy tails, robust normalization needed");
    }
    if zero_pct > 10.0 {
        println!("  - High percentage of zeros, consider separate handling");
    }

    Ok(())
}

fn calculate_skewness(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    let n = values.len() as f64;
    let sum_cubed = values.iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>();
    sum_cubed / n
}

fn calculate_kurtosis(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    let n = values.len() as f64;
    let sum_fourth = values.iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>();
    sum_fourth / n
}
