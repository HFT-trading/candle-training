use crate::core::TrainingTensors;

pub struct TrainingDataSummary {
    pub sequences: usize,
    pub sequence_length: usize,
    pub categorical_features: usize,
    pub numeric_features: usize,
    pub boolean_targets: usize,
    pub numeric_targets: usize,
}

impl TrainingDataSummary {
    pub fn from_tensors(tensors: &TrainingTensors) -> Self {
        let categorical = tensors.inputs.categorical.dims();
        let numeric = tensors.inputs.numeric.dims();
        let boolean_targets = tensors.targets.boolean.dims();
        let numeric_targets = tensors.targets.numeric.dims();

        Self {
            sequences: categorical[0],
            sequence_length: categorical[1],
            categorical_features: categorical[2],
            numeric_features: numeric[2],
            boolean_targets: boolean_targets[1],
            numeric_targets: numeric_targets[1],
        }
    }
}
