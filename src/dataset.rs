//! Dataset loading and preparation utilities for crypto time series data
//!
//! This module provides functionality to load parquet files containing cryptocurrency
//! return data and convert them into tensors suitable for the financial BERT model.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use polars::prelude::*;
use std::path::Path;

/// Load and inspect a parquet file structure
pub fn inspect_parquet_file(file_path: &str) -> Result<()> {
    println!("🔍 Inspecting parquet file: {}", file_path);
    
    if !Path::new(file_path).exists() {
        return Err(anyhow!("File does not exist: {}", file_path));
    }

    // Load just the schema first
    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .limit(5) // Just get a few rows for inspection
        .collect()?;

    println!("📊 File structure:");
    println!("  • Shape: {} rows × {} columns", df.height(), df.width());
    println!("  • Columns: {}", df.get_column_names().len());
    
    // Show first few column names
    let column_names = df.get_column_names();
    println!("  • First 10 columns: {:?}", &column_names[..column_names.len().min(10)]);
    
    // Check for timestamp column
    if column_names.iter().any(|name| name.as_str() == "timestamp") {
        println!("  • ✅ Timestamp column found");
    } else {
        println!("  • ⚠️  No timestamp column found");
    }
    
    // Count return columns (crypto data)
    let return_columns: Vec<_> = column_names.iter()
        .filter(|name| name.ends_with("_return"))
        .collect();
    println!("  • Return columns: {} (crypto assets)", return_columns.len());
    
    // Show data types
    println!("  • Data types:");
    for (name, dtype) in df.get_column_names().iter().zip(df.dtypes().iter()) {
        if name.ends_with("_return") || *name == "timestamp" {
            println!("    - {}: {:?}", name, dtype);
        }
    }
    
    // Show sample data
    println!("  • Sample data (first 3 rows):");
    println!("{}", df.head(Some(3)));
    
    Ok(())
}

/// Load parquet data and prepare it as a tensor for the model
pub fn load_and_prepare_data(file_path: &str, device: &Device) -> Result<(Tensor, usize)> {
    println!("📊 Loading dataset from: {}", file_path);
    
    if !Path::new(file_path).exists() {
        return Err(anyhow!("File does not exist: {}", file_path));
    }

    // Load the full dataset
    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .collect()?;

    println!("✅ Dataset loaded: {} rows × {} columns", df.height(), df.width());

    // Get return columns (cryptocurrency data)
    let return_columns: Vec<String> = df.get_column_names()
        .iter()
        .filter(|name| name.ends_with("_return"))
        .map(|s| s.to_string())
        .collect();

    if return_columns.is_empty() {
        return Err(anyhow!("No return columns found in dataset"));
    }

    println!("📈 Found {} cryptocurrency return columns", return_columns.len());

    // Extract the return data as a matrix
    let mut data_matrix = Vec::new();
    
    for col_name in &return_columns {
        let series = df.column(col_name)?;
        let values: Vec<f32> = series
            .f64()?
            .into_iter()
            .map(|opt_val| opt_val.unwrap_or(0.0) as f32)
            .collect();
        data_matrix.push(values);
    }

    // Transpose the matrix: from [num_cryptos, timesteps] to [timesteps, num_cryptos]
    let num_timesteps = data_matrix[0].len();
    let num_cryptos = data_matrix.len();
    
    let mut transposed_data = Vec::with_capacity(num_timesteps * num_cryptos);
    for t in 0..num_timesteps {
        for c in 0..num_cryptos {
            transposed_data.push(data_matrix[c][t]);
        }
    }

    // Create tensor with shape [timesteps, num_cryptos]
    let tensor = Tensor::from_vec(
        transposed_data,
        (num_timesteps, num_cryptos),
        device,
    )?;

    println!("🎯 Tensor created: shape {:?}, dtype {:?}", tensor.shape(), tensor.dtype());
    
    // Basic statistics
    if let Ok(min_val) = tensor.min(0)?.min(0)?.to_scalar::<f32>() {
        if let Ok(max_val) = tensor.max(0)?.max(0)?.to_scalar::<f32>() {
            println!("📊 Data range: {:.6} to {:.6}", min_val, max_val);
        }
    }

    Ok((tensor, num_cryptos))
}

/// Extract validation split from the full dataset (middle 20%)
pub fn extract_validation_split(data: &Tensor) -> Result<Tensor> {
    let total_timesteps = data.dims()[0];
    let start_idx = (total_timesteps as f64 * 0.6) as usize;
    let end_idx = (total_timesteps as f64 * 0.8) as usize;

    Ok(data.narrow(0, start_idx, end_idx - start_idx)?)
}

/// Extract test split from the full dataset (last 20%)
pub fn extract_test_split(data: &Tensor) -> Result<Tensor> {
    let total_timesteps = data.dims()[0];
    let start_idx = (total_timesteps as f64 * 0.8) as usize;

    Ok(data.narrow(0, start_idx, total_timesteps - start_idx)?)
}

/// Extract training split from the full dataset (first 60%)
pub fn extract_training_split(data: &Tensor) -> Result<Tensor> {
    let total_timesteps = data.dims()[0];
    let end_idx = (total_timesteps as f64 * 0.6) as usize;

    Ok(data.narrow(0, 0, end_idx)?)
}

/// Load cryptocurrency data and convert to orderbook-like format for hftbacktest
/// This is a simplified conversion - in practice you'd need real orderbook data
pub fn load_crypto_as_orderbook_data(file_path: &str) -> Result<Vec<CryptoTimestep>> {
    println!("📊 Loading crypto data for orderbook simulation: {}", file_path);
    
    if !Path::new(file_path).exists() {
        return Err(anyhow!("File does not exist: {}", file_path));
    }

    let df = LazyFrame::scan_parquet(file_path, Default::default())?
        .collect()?;

    // Get timestamp column
    let timestamps = if df.get_column_names().iter().any(|name| name.as_str() == "timestamp") {
        df.column("timestamp")?
            .i64()?
            .into_iter()
            .map(|opt_val| opt_val.unwrap_or(0))
            .collect::<Vec<_>>()
    } else {
        // Generate synthetic timestamps if none exist
        (0..df.height() as i64).collect()
    };

    // Get return columns
    let return_columns: Vec<String> = df.get_column_names()
        .iter()
        .filter(|name| name.ends_with("_return"))
        .map(|s| s.to_string())
        .collect();

    if return_columns.is_empty() {
        return Err(anyhow!("No return columns found in dataset"));
    }

    // Convert to timestep format
    let mut timesteps = Vec::new();
    let num_timesteps = df.height();

    for i in 0..num_timesteps {
        let mut crypto_returns = Vec::new();
        
        for col_name in &return_columns {
            let series = df.column(col_name)?;
            let value = series.f64()?.get(i).unwrap_or(0.0) as f32;
            crypto_returns.push(value);
        }

        timesteps.push(CryptoTimestep {
            timestamp: timestamps[i],
            crypto_returns,
        });
    }

    println!("✅ Loaded {} timesteps with {} cryptocurrencies", timesteps.len(), return_columns.len());
    Ok(timesteps)
}

/// Represents a single timestep of cryptocurrency return data
#[derive(Debug, Clone)]
pub struct CryptoTimestep {
    pub timestamp: i64,
    pub crypto_returns: Vec<f32>,
}

impl CryptoTimestep {
    /// Convert crypto returns to a simulated mid-price
    /// This is a simplified approach - real implementations would use actual price data
    pub fn simulate_mid_price(&self, crypto_idx: usize, base_price: f64) -> f64 {
        if crypto_idx < self.crypto_returns.len() {
            // Convert return to price: new_price = base_price * (1 + return)
            base_price * (1.0 + self.crypto_returns[crypto_idx] as f64)
        } else {
            base_price
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_data_splits() {
        let device = Device::Cpu;
        
        // Create test tensor with 1000 timesteps and 10 cryptos
        let data = vec![1.0f32; 1000 * 10];
        let tensor = Tensor::from_vec(data, (1000, 10), &device).unwrap();
        
        // Test splits
        let train = extract_training_split(&tensor).unwrap();
        let val = extract_validation_split(&tensor).unwrap();
        let test = extract_test_split(&tensor).unwrap();
        
        assert_eq!(train.dims()[0], 600); // 60% of 1000
        assert_eq!(val.dims()[0], 200);   // 20% of 1000
        assert_eq!(test.dims()[0], 200);  // 20% of 1000
        
        // All should have same number of cryptos
        assert_eq!(train.dims()[1], 10);
        assert_eq!(val.dims()[1], 10);
        assert_eq!(test.dims()[1], 10);
    }
}
