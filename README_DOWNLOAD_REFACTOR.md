# Binance Data Download Refactor

This document describes the refactoring of the Binance aggregate trades data downloader to use a single Parquet file with Polars and support for resumable downloads.

## Key Changes

### 1. Single Parquet File Output
- **Before**: Downloaded data was saved as individual CSV files per day (e.g., `2024_01_01.csv`, `2024_01_02.csv`)
- **After**: All data is consolidated into a single Parquet file per trading pair (e.g., `BTCUSDT_agg_trades.parquet`)

### 2. Resumable Downloads
- **Before**: No resume capability - would re-download existing files or skip them
- **After**: Automatically detects the latest date in existing Parquet file and resumes from the next day

### 3. Polars Integration with Parquet
- **Before**: Raw CSV file handling
- **After**: Uses Polars DataFrames with Parquet format for efficient data processing, concatenation, and sorting

### 4. Smart Monthly/Daily Download Strategy
- **Before**: Always used daily endpoint for all historical data
- **After**: Uses monthly endpoint for historical months (much faster) and daily endpoint for current month
- Automatically falls back to daily downloads if monthly data isn't available yet

### 5. Incremental Batch Saving
- **Before**: All data was downloaded first, then saved at the end (risk of losing progress)
- **After**: Data is saved in batches to prevent data loss during long downloads

### 6. Improved Data Structure
The CSV file contains the following columns (based on Binance aggTrades format):
- `agg_trade_id`: Aggregate trade ID
- `price`: Trade price
- `quantity`: Trade quantity  
- `first_trade_id`: First trade ID in the aggregate
- `last_trade_id`: Last trade ID in the aggregate
- `timestamp`: Trade timestamp (microseconds for 2025+ data, milliseconds for older data)
- `is_buyer_maker`: Whether the buyer was the maker
- `is_best_match`: Whether the trade was the best price match

## Usage

### Running the Download
```bash
cargo run --example download_data
```

### Configuration
Edit the `main()` function in `examples/download_data.rs` to configure:
- Trading pair (default: "BTCUSDT")
- Start date (default: 2017-08-17 for BTC)
- Output directory (default: "./crypto_data")

### Testing CSV Reading
```bash
cargo run --example test_download
```

## Technical Implementation

### Resume Logic
1. Check if Parquet file exists at `{output_dir}/{pair}_agg_trades.parquet`
2. If exists, read the maximum timestamp and convert to date
3. Resume downloading from the day after the latest date
4. If no existing file, start from the configured start date

### Smart Download Strategy
1. **Monthly Downloads**: For historical months (not current month)
   - Uses `https://data.binance.vision/data/spot/monthly/aggTrades/{pair}/{pair}-aggTrades-{YYYY-MM}.zip`
   - Much faster as one file contains entire month's data
   - Falls back to daily downloads if monthly data not available

2. **Daily Downloads**: For current month and fallback scenarios
   - Uses `https://data.binance.vision/data/spot/daily/aggTrades/{pair}/{pair}-aggTrades-{YYYY-MM-DD}.zip`
   - Processed in batches with concurrent downloads (max 20 concurrent requests)

### Data Processing Pipeline
1. Download monthly ZIP files for historical months
2. Download daily ZIP files for current month and fallbacks
3. Extract and parse CSV content from each ZIP file
4. Convert to Polars DataFrames with proper column names
5. Concatenate all new data
6. Merge with existing data (if any)
7. Sort by timestamp to maintain chronological order
8. Write back to single CSV file

### Error Handling
- Gracefully handles missing data (404 responses from Binance)
- Continues processing other dates if individual downloads fail
- Maintains data integrity by sorting and deduplicating

## Benefits

1. **Efficiency**: Single file is easier to work with for analysis
2. **Speed**: Monthly downloads are much faster than daily downloads for historical data
3. **Resumability**: Can restart downloads without losing progress
4. **Data Integrity**: Automatic sorting ensures chronological order
5. **Performance**: Polars provides fast data processing
6. **Scalability**: Concurrent downloads with configurable limits
7. **No Runtime Conflicts**: Removed Tokio dependency to avoid async runtime issues

## File Structure
```
crypto_data/
└── BTCUSDT_agg_trades.csv  # Single consolidated file
```

## Dependencies
The refactor uses these dependencies:
- `polars` (v0.49.1) with features: "lazy", "csv", "temporal"
- `reqwest` with "blocking" feature for synchronous HTTP requests
- `chrono` for date handling
- `zip` for extracting downloaded files

**Removed dependencies:**
- `tokio` - Removed to avoid async runtime conflicts
- `futures` - No longer needed without async

## Future Enhancements
- Support for multiple trading pairs in parallel
- Data compression options
- Incremental backup strategies
- Data validation and quality checks
