//! Numeric feature normalizer (z-score). Fit on the training split; applied at
//! both train and serve time, so it lives in the shared crate.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Result, Tensor};

pub struct NumericNormalizer {
    pub mean: Tensor,
    pub std: Tensor,
}

impl NumericNormalizer {
    /// Fit per-feature mean/std over the selected rows (mean over batch + time).
    pub fn fit(numeric: &Tensor, train_indices: &[u32]) -> Result<Self> {
        let ids = Tensor::from_slice(train_indices, train_indices.len(), numeric.device())?;
        let train = numeric.index_select(&ids, 0)?;
        let mean = train.mean_keepdim((0, 1))?;
        let centered = train.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim((0, 1))?;
        let std = (variance + 1e-6)?.sqrt()?;
        Ok(Self { mean, std })
    }

    pub fn apply(&self, numeric: &Tensor) -> Result<Tensor> {
        numeric.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensors = HashMap::from([
            ("mean".to_owned(), self.mean.clone()),
            ("std".to_owned(), self.std.clone()),
        ]);
        candle_core::safetensors::save(&tensors, path)
    }

    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let mut tensors = candle_core::safetensors::load(path, device)?;
        let mean = tensors
            .remove("mean")
            .ok_or_else(|| candle_core::Error::Msg("normalizer missing mean".to_owned()))?;
        let std = tensors
            .remove("std")
            .ok_or_else(|| candle_core::Error::Msg("normalizer missing std".to_owned()))?;
        Ok(Self { mean, std })
    }
}
