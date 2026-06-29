//! Tensor containers shared by training and serving so the forward pass sees the
//! same layout in both places.

use candle_core::Tensor;

/// Model input tensors.
/// - `categorical`: `[batch, seq, n_categorical]`, u32 vocabulary ids.
/// - `numeric`: `[batch, seq, n_numeric]`, f32 (normalized before the model).
pub struct ModelInputs {
    pub categorical: Tensor,
    pub numeric: Tensor,
}
