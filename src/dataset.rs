use candle_core::{Device, Result, Tensor};
use polars::prelude::*;

/// Inspects a Parquet file to show its structure without loading all data
pub fn inspect_parquet_file(path: &str) -> Result<()> {
    println!("Inspecting Parquet file: {}", path);

    let df = LazyFrame::scan_parquet(path, Default::default())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to scan Parquet file: {}", e)))?
        .limit(5) // Only load first 5 rows for inspection
        .collect()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to collect LazyFrame: {}", e)))?;

    println!("File structure:");
    println!("   Rows: {} (showing first 5)", df.height());
    println!("   Columns: {}", df.width());

    println!("\nColumn names and types:");
    for (i, (name, dtype)) in df.get_column_names().iter().zip(df.dtypes().iter()).enumerate() {
        println!("   {}: {} ({})", i + 1, name, dtype);
    }

    println!("\nSample data (first 5 rows):");
    println!("{}", df);

    Ok(())
}

/// Loads data from a Parquet file and converts it to a Candle tensor.
/// Returns a 2D tensor with shape [timesteps, num_cryptocurrencies]
/// Also returns the number of cryptocurrencies found
pub fn load_and_prepare_data(
    path: &str,
    device: &Device,
) -> Result<(Tensor, usize)> {
    // Load the parquet file
    let df = LazyFrame::scan_parquet(path, Default::default())
        .expect("Failed to scan Parquet file")
        .collect()
        .expect("Failed to collect LazyFrame");

    let num_rows = df.height();
    let num_cols = df.width();

    println!("Data loaded. Shape: {} rows × {} columns", num_rows, num_cols);
    println!("Detected {} cryptocurrencies", num_cols);

    // Convert DataFrame to Vec<Vec<f32>>
    let mut data_vec: Vec<f32> = Vec::with_capacity(num_rows * num_cols);

    // Iterate through columns and collect data
    for col in df.get_columns() {
        match col.dtype() {
            DataType::Float64 => {
                let float_series = col.f64().expect("Failed to convert to f64");
                for value in float_series.iter() {
                    data_vec.push(value.unwrap_or(0.0) as f32);
                }
            },
            DataType::Float32 => {
                let float_series = col.f32().expect("Failed to convert to f32");
                for value in float_series.iter() {
                    data_vec.push(value.unwrap_or(0.0));
                }
            },
            DataType::Int64 => {
                let int_series = col.i64().expect("Failed to convert to i64");
                for value in int_series.iter() {
                    data_vec.push(value.unwrap_or(0) as f32);
                }
            },
            _ => {
                // Try to cast to f64 first
                let casted = col.cast(&DataType::Float64).expect("Failed to cast to f64");
                let float_series = casted.f64().expect("Failed to convert to f64");
                for value in float_series.iter() {
                    data_vec.push(value.unwrap_or(0.0) as f32);
                }
            }
        }
    }

    // Create tensor from the flattened data
    // Note: Polars stores data column-wise, so we need to transpose
    let tensor = Tensor::from_vec(data_vec, (num_cols, num_rows), device)?;
    let transposed_tensor = tensor.transpose(0, 1)?; // Transpose to [rows, cols]

    println!("Data tensor shape: {:?}", transposed_tensor.shape());
    Ok((transposed_tensor, num_cols))
}