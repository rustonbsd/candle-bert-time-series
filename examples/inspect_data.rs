use candle_bert_time_series::dataset::inspect_parquet_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Update this path to point to your actual parquet file
    let data_path = "/home/i3/Downloads/transformed_dataset.parquet";
    
    println!("🚀 Data File Inspector");
    println!("{}", "=".repeat(50));
    
    match inspect_parquet_file(data_path) {
        Ok(_) => println!("\n✅ Inspection complete!"),
        Err(e) => {
            println!("❌ Error inspecting file: {}", e);
            println!("\n💡 Make sure the file path is correct:");
            println!("   Current path: {}", data_path);
            println!("   Update the path in examples/inspect_data.rs");
        }
    }
    
    Ok(())
}
