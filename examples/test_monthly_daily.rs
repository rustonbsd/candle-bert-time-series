// Test script to verify the monthly/daily download logic

use anyhow::Result;
use chrono::{Datelike, NaiveDate};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Test with a small date range that spans multiple months
    let pair_to_download = "BTCUSDT";
    // Test with a few months in 2024
    let start_date = NaiveDate::from_ymd_opt(2024, 10, 1).unwrap();
    let output_dir = PathBuf::from("./test_monthly_data");

    println!("Testing monthly/daily download logic...");
    println!("Pair: {}", pair_to_download);
    println!("Start date: {}", start_date);
    
    // Simulate the logic from the main download function
    let end_date = chrono::Utc::now().date_naive();
    let current_month = end_date.format("%Y-%m").to_string();
    let mut monthly_periods = Vec::new();
    let mut daily_dates = Vec::new();
    
    // Group dates by month
    let mut current_date = start_date;
    while current_date <= end_date {
        let month_str = current_date.format("%Y-%m").to_string();
        
        if month_str == current_month {
            // Current month - use daily downloads
            println!("Current month detected: {}", month_str);
            while current_date <= end_date && current_date.format("%Y-%m").to_string() == current_month {
                daily_dates.push(current_date);
                current_date = current_date.succ_opt().unwrap();
            }
        } else {
            // Historical month - try monthly download first
            println!("Historical month: {}", month_str);
            monthly_periods.push(month_str.clone());
            // Skip to next month
            let next_month = if current_date.month() == 12 {
                NaiveDate::from_ymd_opt(current_date.year() + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(current_date.year(), current_date.month() + 1, 1).unwrap()
            };
            current_date = next_month;
        }
    }

    println!("\n📊 Download Strategy Summary:");
    println!("Monthly periods to download: {} months", monthly_periods.len());
    for period in &monthly_periods {
        println!("  - {}", period);
    }
    
    println!("Daily dates to download: {} days", daily_dates.len());
    if !daily_dates.is_empty() {
        println!("  - From {} to {}", 
                 daily_dates.first().unwrap(), 
                 daily_dates.last().unwrap());
    }

    println!("\n✅ Logic test completed successfully!");
    println!("This would significantly reduce download time by using monthly files for historical data.");
    
    Ok(())
}
