mod prediction;

use candle_core::Tensor;

pub use prediction::{
    ModelResponse, OutcomeProbabilities, PredictionStatus, RangeEstimate, UnknownReason,
};

pub struct ModelInputs {
    pub categorical: Tensor,
    pub numeric: Tensor,
}

pub struct TrainingTargets {
    pub categorical: Tensor,
    pub boolean: Tensor,
    pub numeric: Tensor,
}

pub struct TrainingTensors {
    pub inputs: ModelInputs,
    pub targets: TrainingTargets,
}
