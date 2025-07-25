// orderbook_dataset.rs
// Dataset loading and preparation for orderbook data

use candle_core::{Device, Result, Tensor};
use polars::prelude::*;
use std::path::Path;

/// Load and prepare orderbook data from multiple parquet files
/// Returns (data_tensor, num_features)
pub fn load_and_prepare_orderbook_data(
    file_paths: &[&str],
    device: &Device,
) -> Result<(Tensor, usize)> {
    println!("Loading orderbook data from {} files...", file_paths.len());
    
    let mut all_dataframes = Vec::new();
    
    // Load all parquet files
    for (i, path) in file_paths.iter().enumerate() {
        if !Path::new(path).exists() {
            println!("Warning: File does not exist: {}", path);
            continue;
        }
        
        println!("Loading file {}/{}: {}", i + 1, file_paths.len(), path);
        
        let df = LazyFrame::scan_parquet(path, Default::default())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to scan parquet file {}: {}", path, e)))?
            .collect()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load parquet file {}: {}", path, e)))?;
        
        println!("  Loaded {} rows × {} columns", df.height(), df.width());
        all_dataframes.push(df);
    }
    
    if all_dataframes.is_empty() {
        return Err(candle_core::Error::Msg("No valid data files found".to_string()));
    }
    
    // Concatenate all dataframes
    println!("Concatenating {} dataframes...", all_dataframes.len());
    let mut combined_df = all_dataframes.into_iter().reduce(|mut acc, df| {
        acc.vstack_mut(&df).expect("Failed to concatenate dataframes");
        acc
    }).unwrap();
    
    println!("Combined dataset: {} rows × {} columns", combined_df.height(), combined_df.width());
    
    // Remove timestamp column if present
    let feature_columns: Vec<String> = combined_df.get_column_names()
        .iter()
        .filter(|name| !name.to_lowercase().contains("timestamp"))
        .map(|s| s.to_string())
        .collect();
    
    println!("Feature columns: {} (excluding timestamp)", feature_columns.len());
    
    // Select only feature columns
    combined_df = combined_df.select(&feature_columns)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to select feature columns: {}", e)))?;
    
    // Convert to tensor
    let num_rows = combined_df.height();
    let num_features = combined_df.width();
    
    println!("Converting to tensor: {} timesteps × {} features", num_rows, num_features);
    
    // Extract data as f32 vectors
    let mut data_matrix = Vec::with_capacity(num_rows * num_features);
    
    for col_name in &feature_columns {
        let column = combined_df.column(col_name)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get column {}: {}", col_name, e)))?;
        
        let values: Vec<f32> = column
            .f32()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to convert column {} to f32: {}", col_name, e)))?
            .into_iter()
            .map(|opt_val| opt_val.unwrap_or(0.0))
            .collect();
        
        data_matrix.extend(values);
    }
    
    // Reshape data: from column-major to row-major
    let mut row_major_data = vec![0.0f32; num_rows * num_features];
    for row in 0..num_rows {
        for col in 0..num_features {
            row_major_data[row * num_features + col] = data_matrix[col * num_rows + row];
        }
    }
    
    // Create tensor with shape [num_timesteps, num_features]
    let data_tensor = Tensor::from_vec(row_major_data, &[num_rows, num_features], device)?;
    
    println!("✅ Data tensor created: {:?}", data_tensor.shape());
    
    // Print some statistics
    print_data_statistics(&data_tensor)?;
    
    Ok((data_tensor, num_features))
}

/// Print basic statistics about the loaded data
fn print_data_statistics(data: &Tensor) -> Result<()> {
    let shape = data.shape();
    println!("\n📊 Data Statistics:");
    println!("  Shape: {:?}", shape);
    
    // Calculate basic statistics
    let flattened = data.flatten_all()?;
    let mean = flattened.mean_all()?.to_scalar::<f32>()?;
    let std = flattened.var(0)?.sqrt()?.to_scalar::<f32>()?;
    let min_val = flattened.min(0)?.to_scalar::<f32>()?;
    let max_val = flattened.max(0)?.to_scalar::<f32>()?;
    
    println!("  Mean: {:.6}", mean);
    println!("  Std:  {:.6}", std);
    println!("  Min:  {:.6}", min_val);
    println!("  Max:  {:.6}", max_val);
    
    // Check for NaN or infinite values (simplified check)
    // Note: candle doesn't have isnan/isinf methods, so we'll skip this check for now
    // In a production system, you might want to implement custom NaN/Inf detection
    println!("  ℹ️  NaN/Inf checking skipped (not available in candle)");
    
    // Check data distribution
    let zero_mask = flattened.eq(&Tensor::zeros_like(&flattened)?)?;
    let zero_count = zero_mask.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    let total_elements = flattened.elem_count() as f32;
    let zero_percentage = (zero_count / total_elements) * 100.0;
    
    println!("  Zero values: {:.2}% ({} out of {})", 
             zero_percentage, zero_count as usize, total_elements as usize);
    
    Ok(())
}

/// Load a single orderbook parquet file for inspection
pub fn inspect_orderbook_file(file_path: &str) -> Result<()> {
    println!("🔍 Inspecting orderbook file: {}", file_path);
    
    if !Path::new(file_path).exists() {
        return Err(candle_core::Error::Msg(format!("File does not exist: {}", file_path)));
    }
    
    let df = LazyFrame::scan_parquet(file_path, Default::default())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to scan parquet file: {}", e)))?
        .limit(5) // Just look at first 5 rows
        .collect()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load parquet file: {}", e)))?;
    
    println!("📊 File structure:");
    println!("  Columns: {:?}", df.get_column_names());
    println!("  Shape: {:?}", df.shape());
    println!("  Data types: {:?}", df.dtypes());
    
    println!("\n📋 Sample data (first 5 rows):");
    println!("{}", df);
    
    // Check for expected orderbook structure
    let column_names = df.get_column_names();
    let has_timestamp = column_names.iter().any(|name| name.to_lowercase().contains("timestamp"));
    let bid_price_cols: Vec<_> = column_names.iter().filter(|name| name.contains("bid_price")).collect();
    let ask_price_cols: Vec<_> = column_names.iter().filter(|name| name.contains("ask_price")).collect();
    
    println!("\n🔍 Orderbook structure analysis:");
    println!("  Has timestamp: {}", has_timestamp);
    println!("  Bid price columns: {}", bid_price_cols.len());
    println!("  Ask price columns: {}", ask_price_cols.len());
    
    if bid_price_cols.len() == ask_price_cols.len() {
        println!("  ✅ Balanced bid/ask structure with {} levels", bid_price_cols.len());
    } else {
        println!("  ⚠️  Unbalanced bid/ask structure");
    }
    
    let expected_features = bid_price_cols.len() * 4; // bid_price, bid_qty, ask_price, ask_qty per level
    let actual_features = column_names.len() - if has_timestamp { 1 } else { 0 };
    
    println!("  Expected features: {} ({}×4)", expected_features, bid_price_cols.len());
    println!("  Actual features: {}", actual_features);
    
    if expected_features == actual_features {
        println!("  ✅ Feature count matches expected orderbook structure");
    } else {
        println!("  ⚠️  Feature count mismatch - check data format");
    }
    
    Ok(())
}

/// Validate that all files have consistent structure
pub fn validate_orderbook_files(file_paths: &[&str]) -> Result<()> {
    println!("🔍 Validating {} orderbook files...", file_paths.len());
    
    let mut reference_columns: Option<Vec<String>> = None;
    
    for (i, path) in file_paths.iter().enumerate() {
        if !Path::new(path).exists() {
            println!("⚠️  File {}: Does not exist - {}", i + 1, path);
            continue;
        }
        
        let df = LazyFrame::scan_parquet(path, Default::default())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to scan file {}: {}", path, e)))?
            .limit(1) // Just check structure
            .collect()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load file {}: {}", path, e)))?;
        
        let columns: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        let shape = df.shape();
        
        match &reference_columns {
            None => {
                reference_columns = Some(columns.clone());
                println!("✅ File {}: Reference structure set - {} columns", i + 1, shape.1);
            }
            Some(ref_cols) => {
                if columns == *ref_cols {
                    println!("✅ File {}: Structure matches reference", i + 1);
                } else {
                    println!("❌ File {}: Structure mismatch", i + 1);
                    println!("   Expected columns: {}", ref_cols.len());
                    println!("   Actual columns: {}", columns.len());
                    return Err(candle_core::Error::Msg(format!("File structure mismatch: {}", path)));
                }
            }
        }
    }
    
    println!("✅ All files have consistent structure");
    Ok(())
}
