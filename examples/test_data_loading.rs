use candle_bert_time_series::dataset::{inspect_parquet_file, load_and_prepare_data};
use candle_core::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Update this path to point to your actual parquet file
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    
    println!("🚀 Testing Data Loading");
    println!("{}", "=".repeat(50));
    
    // First, inspect the file structure
    println!("Step 1: Inspecting file structure...");
    match inspect_parquet_file(data_path) {
        Ok(_) => println!("✅ File inspection complete!"),
        Err(e) => {
            println!("❌ Error inspecting file: {}", e);
            println!("\n💡 Make sure the file path is correct:");
            println!("   Current path: {}", data_path);
            println!("   Update the path in examples/test_data_loading.rs");
            return Ok(());
        }
    }
    
    println!("\n{}", "=".repeat(50));
    
    // Then, try to load the data
    println!("Step 2: Loading data as tensor...");
    let device = Device::Cpu; // Use CPU for testing
    
    match load_and_prepare_data(data_path, &device) {
        Ok((tensor, num_cryptos)) => {
            println!("✅ Data loaded successfully!");
            println!("   Tensor shape: {:?}", tensor.shape());
            println!("   Number of cryptocurrencies: {}", num_cryptos);
            println!("   Data type: {:?}", tensor.dtype());
            
            // Show some basic statistics
            if let Ok(min_val) = tensor.min(0)?.min(0)?.to_scalar::<f32>() {
                if let Ok(max_val) = tensor.max(0)?.max(0)?.to_scalar::<f32>() {
                    println!("   Data range: {:.6} to {:.6}", min_val, max_val);
                }
            }
        },
        Err(e) => {
            println!("❌ Error loading data: {}", e);
        }
    }
    
    println!("\n🎉 Test complete!");
    Ok(())
}
