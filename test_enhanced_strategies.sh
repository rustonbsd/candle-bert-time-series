#!/bin/bash

echo "🚀 TESTING ALL ENHANCED DIVERGENCE STRATEGIES"
echo "=============================================="

# Set CUDA environment variables
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda"

# Build the enhanced strategy
echo "Building enhanced divergence strategy with CUDA support..."
cargo build --release --example enhanced_divergence_strategy

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""

# Test Path Average Method
echo "🎯 Testing Path Average Method (Original)"
echo "=========================================="
./target/release/examples/enhanced_divergence_strategy path_average > results_path_average.txt 2>&1
echo "✅ Path Average results saved to results_path_average.txt"
echo ""

# Test Path MAE Method
echo "🎯 Testing Path MAE Method"
echo "=========================="
./target/release/examples/enhanced_divergence_strategy path_mae > results_path_mae.txt 2>&1
echo "✅ Path MAE results saved to results_path_mae.txt"
echo ""

# Test Next Timestep Method
echo "🎯 Testing Next Timestep Method"
echo "==============================="
./target/release/examples/enhanced_divergence_strategy next_timestep > results_next_timestep.txt 2>&1
echo "✅ Next Timestep results saved to results_next_timestep.txt"
echo ""

echo "🏁 All tests completed!"
echo "======================="
echo "Results saved in:"
echo "  - results_path_average.txt"
echo "  - results_path_mae.txt"
echo "  - results_next_timestep.txt"
echo ""

# Extract key metrics from results
echo "📊 SUMMARY OF RESULTS"
echo "===================="

for method in path_average path_mae next_timestep; do
    file="results_${method}.txt"
    if [ -f "$file" ]; then
        echo ""
        echo "Method: ${method^^}"
        echo "-------------------"
        grep "Final Value:" "$file" || echo "Final Value: Not found"
        grep "Total Return:" "$file" || echo "Total Return: Not found"
        grep "Sharpe Ratio:" "$file" || echo "Sharpe Ratio: Not found"
        grep "Max Drawdown:" "$file" || echo "Max Drawdown: Not found"
        grep "Total Trades:" "$file" || echo "Total Trades: Not found"
        grep "Win Rate:" "$file" || echo "Win Rate: Not found"
    else
        echo ""
        echo "Method: ${method^^}"
        echo "-------------------"
        echo "❌ Results file not found"
    fi
done

echo ""
echo "🔍 For detailed results, check the individual result files."
