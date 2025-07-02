# candle-bert-time-series
goal: train bert transformer with masking on financial time series


build and run:
```sh
export PATH="/usr/local/cuda/bin:$PATH" && export CUDA_ROOT="/usr/local/cuda" && cargo build --release
export PATH="/usr/local/cuda/bin:$PATH" && export CUDA_ROOT="/usr/local/cuda" && ./target/release/candle-bert-time-series
``