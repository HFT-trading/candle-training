mod prediction;

use candle_core::Tensor;

pub use prediction::{
    HistoricalRangeMetadata, MarketStateAssessment, ModelResponse, MoveOutlook, PredictionStatus,
    StatePrediction, UnknownReason,
};

pub struct ModelInputs {
    pub categorical: Tensor,
    pub numeric: Tensor,
}

pub struct TrainingTargets {
    pub state_categorical: Tensor,
    pub categorical: Tensor,
    pub boolean: Tensor,
    pub numeric: Tensor,
}

pub struct TrainingTensors {
    pub inputs: ModelInputs,
    pub targets: TrainingTargets,
}
