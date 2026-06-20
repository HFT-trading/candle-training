use std::path::Path;

use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};

use crate::config::ModelConfig;
use crate::core::ModelInputs;
use crate::dataset::FeatureSchema;
use crate::model::{MarketStateModel, ModelOutput, NumericNormalizer};

pub struct ModelRuntime {
    model: MarketStateModel,
    normalizer: NumericNormalizer,
    _variables: VarMap,
}

impl ModelRuntime {
    pub fn load(
        config: &ModelConfig,
        schema: &FeatureSchema,
        device: &Device,
        model_path: impl AsRef<Path>,
        normalizer_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut variables = VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, device);
        let model = MarketStateModel::new(config, schema, vb)?;
        variables.load(model_path)?;
        let normalizer = NumericNormalizer::load(normalizer_path, device)?;

        Ok(Self {
            model,
            normalizer,
            _variables: variables,
        })
    }

    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        let normalized = ModelInputs {
            categorical: inputs.categorical.clone(),
            numeric: self.normalizer.apply(&inputs.numeric)?,
        };
        self.model.forward(&normalized)
    }
}
