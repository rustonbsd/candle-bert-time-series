// batcher.rs
// Simple batching utility for tensor data

use candle_core::{Result, Tensor};

/// Simple batcher for creating sequences from tensor data
pub struct Batcher {
    data: Tensor,
    sequence_length: usize,
    batch_size: usize,
    shuffle: bool,
    current_position: usize,
    indices: Vec<usize>,
}

impl Batcher {
    /// Create a new batcher (legacy interface for compatibility)
    pub fn new(data: &Tensor, sequence_length: usize, batch_size: usize) -> Self {
        Self::new_r2(data.clone(), sequence_length, batch_size, false).unwrap()
    }

    /// Create a new batcher with shuffle option
    pub fn new_r2(data: Tensor, sequence_length: usize, batch_size: usize, shuffle: bool) -> Result<Self> {
        let total_timesteps = data.dims()[0];
        
        // Calculate how many valid sequences we can create
        let max_sequences = if total_timesteps >= sequence_length {
            total_timesteps - sequence_length + 1
        } else {
            0
        };
        
        // Create indices for all possible starting positions
        let mut indices: Vec<usize> = (0..max_sequences).collect();
        
        // Shuffle if requested
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        Ok(Self {
            data,
            sequence_length,
            batch_size,
            shuffle,
            current_position: 0,
            indices,
        })
    }

    /// Get the next batch of sequences
    pub fn next(&mut self) -> Option<Result<Tensor>> {
        if self.current_position >= self.indices.len() {
            return None;
        }
        
        // Calculate how many sequences to include in this batch
        let remaining = self.indices.len() - self.current_position;
        let actual_batch_size = remaining.min(self.batch_size);
        
        if actual_batch_size == 0 {
            return None;
        }
        
        // Collect the sequences for this batch
        let mut batch_sequences = Vec::new();
        
        for i in 0..actual_batch_size {
            let start_idx = self.indices[self.current_position + i];
            
            match self.data.narrow(0, start_idx, self.sequence_length) {
                Ok(sequence) => batch_sequences.push(sequence),
                Err(e) => return Some(Err(e)),
            }
        }
        
        self.current_position += actual_batch_size;
        
        // Stack sequences into a batch tensor
        // Shape: [batch_size, sequence_length, features]
        match Tensor::stack(&batch_sequences, 0) {
            Ok(batch) => Some(Ok(batch)),
            Err(e) => Some(Err(e)),
        }
    }

    /// Reset the batcher to start from the beginning
    pub fn reset(&mut self) {
        self.current_position = 0;
        
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the total number of batches that will be produced
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get the total number of sequences
    pub fn num_sequences(&self) -> usize {
        self.indices.len()
    }
}

/// Iterator implementation for Batcher
impl Iterator for Batcher {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_batcher_basic() -> Result<()> {
        let device = Device::Cpu;
        
        // Create test data: 10 timesteps, 3 features
        let data = Tensor::arange(0f32, 30f32, &device)?.reshape(&[10, 3])?;
        
        let mut batcher = Batcher::new_r2(data, 3, 2, false)?;
        
        // Should be able to create 8 sequences (10 - 3 + 1 = 8)
        assert_eq!(batcher.num_sequences(), 8);
        assert_eq!(batcher.num_batches(), 4); // 8 sequences / 2 batch_size = 4 batches
        
        // Test first batch
        let batch1 = batcher.next().unwrap()?;
        assert_eq!(batch1.dims(), &[2, 3, 3]); // [batch_size, seq_len, features]
        
        Ok(())
    }

    #[test]
    fn test_batcher_shuffle() -> Result<()> {
        let device = Device::Cpu;
        let data = Tensor::arange(0f32, 30f32, &device)?.reshape(&[10, 3])?;
        
        let mut batcher1 = Batcher::new_r2(data.clone(), 3, 2, false)?;
        let mut batcher2 = Batcher::new_r2(data, 3, 2, true)?;
        
        // Both should have same number of sequences
        assert_eq!(batcher1.num_sequences(), batcher2.num_sequences());
        
        Ok(())
    }

    #[test]
    fn test_batcher_edge_cases() -> Result<()> {
        let device = Device::Cpu;
        
        // Test with data smaller than sequence length
        let small_data = Tensor::arange(0f32, 6f32, &device)?.reshape(&[2, 3])?;
        let mut small_batcher = Batcher::new_r2(small_data, 5, 2, false)?;
        assert_eq!(small_batcher.num_sequences(), 0);
        assert!(small_batcher.next().is_none());
        
        // Test with exact sequence length
        let exact_data = Tensor::arange(0f32, 9f32, &device)?.reshape(&[3, 3])?;
        let mut exact_batcher = Batcher::new_r2(exact_data, 3, 2, false)?;
        assert_eq!(exact_batcher.num_sequences(), 1);
        
        Ok(())
    }
}
