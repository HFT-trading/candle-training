use candle_core::Tensor;

pub struct ModelInputs {
    pub categorical: Tensor,
    pub numeric: Tensor,
}

pub struct TrainingTargets {
    pub boolean: Tensor,
    pub numeric: Tensor,
}

pub struct TrainingTensors {
    pub inputs: ModelInputs,
    pub targets: TrainingTargets,
}
