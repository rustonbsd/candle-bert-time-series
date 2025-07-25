apt update
apt install -y build-essential pkg-config libssl-dev davfs2 libfontconfig1-dev llvm clang libclang-dev

mkdir /mnt/storage-box

# Linux

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

#cargo build --release --example download_bybit_orderbook

mount -t davfs https://u470372-sub1.your-storagebox.de /mnt/storage-box
