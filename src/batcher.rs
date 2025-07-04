use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct Batcher<'a> {
    data: &'a Tensor,
    sequence_length: usize,
    batch_size: usize,
    position: usize,
}

impl<'a> Batcher<'a> {
    pub fn new(
        data: &'a Tensor,
        sequence_length: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            data,
            sequence_length,
            batch_size,
            position: 0,
        }
    }
}

impl<'a> Iterator for Batcher<'a> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        let total_len = self.data.dims()[0];
        if self.position + self.sequence_length >= total_len {
            return None; // Not enough data left to form a full sequence
        }

        let mut sequences = Vec::with_capacity(self.batch_size);
        for i in 0..self.batch_size {
            let start = self.position + i;
            if start + self.sequence_length >= total_len {
                break; // Stop if we run out of data for this batch
            }
            // `narrow` creates a view, not a copy. This is very memory efficient.
            match self.data.narrow(0, start, self.sequence_length) {
                Ok(seq) => sequences.push(seq),
                Err(e) => return Some(Err(e)),
            }
        }

        if sequences.is_empty() {
            return None;
        }

        // Move the position forward for the next batch
        self.position += sequences.len();

        // Stack the individual sequences into a single batch tensor
        Some(Tensor::stack(&sequences, 0))
    }
}