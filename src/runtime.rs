use std::collections::{HashMap, VecDeque};
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::config::ModelConfig;
use crate::core::ModelInputs;
use crate::dataset::FeatureSchema;
use crate::model::{MarketStateModel, ModelOutput, NumericNormalizer};

#[derive(Debug, Clone, Deserialize)]
pub struct MarketStep {
    #[serde(default)]
    pub step_id: Option<String>,
    #[serde(default)]
    pub time: Option<String>,
    pub state: Map<String, Value>,
    pub range: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct SequenceRequest {
    #[serde(default)]
    pub cycle_id: Option<String>,
    pub steps: Vec<MarketStep>,
}

#[derive(Debug)]
pub struct SequencePrediction {
    pub cycle_id: Option<String>,
    pub observed_steps: usize,
    pub output: ModelOutput,
}

pub struct ModelRuntime {
    model: MarketStateModel,
    normalizer: NumericNormalizer,
    categorical_features: Vec<String>,
    numeric_features: Vec<String>,
    categorical_vocab: HashMap<String, HashMap<String, u32>>,
    categorical_unknown_id: u32,
    max_sequence_steps: usize,
    device: Device,
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
            categorical_features: schema.model_input.categorical_features.clone(),
            numeric_features: schema.model_input.numeric_features.clone(),
            categorical_vocab: schema.categorical_vocab.clone(),
            categorical_unknown_id: schema.model_input.categorical_unknown_id,
            max_sequence_steps: config.max_sequence_steps,
            device: device.clone(),
            _variables: variables,
        })
    }

    pub fn predict(&self, request: &SequenceRequest) -> Result<SequencePrediction> {
        let inputs = self.build_inputs(&request.steps)?;
        let output = self.forward(&inputs)?;
        Ok(SequencePrediction {
            cycle_id: request.cycle_id.clone(),
            observed_steps: request.steps.len(),
            output,
        })
    }

    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        let normalized = ModelInputs {
            categorical: inputs.categorical.clone(),
            numeric: self.normalizer.apply(&inputs.numeric)?,
        };
        self.model.forward(&normalized)
    }

    fn build_inputs(&self, steps: &[MarketStep]) -> Result<ModelInputs> {
        if steps.is_empty() || steps.len() > self.max_sequence_steps {
            candle_core::bail!(
                "sequence must contain between 1 and {} steps, got {}",
                self.max_sequence_steps,
                steps.len()
            )
        }

        let mut categorical = Vec::with_capacity(steps.len() * self.categorical_features.len());
        let mut numeric = Vec::with_capacity(steps.len() * self.numeric_features.len());
        for step in steps {
            for feature in &self.categorical_features {
                let value = step_value(step, feature)?;
                let category = value.as_str().ok_or_else(|| {
                    candle_core::Error::Msg(format!("{feature} must be a string"))
                })?;
                categorical.push(
                    self.categorical_vocab
                        .get(feature)
                        .and_then(|vocab| vocab.get(category))
                        .copied()
                        .unwrap_or(self.categorical_unknown_id),
                );
            }
            for feature in &self.numeric_features {
                let value = step_value(step, feature)?;
                let number = match value {
                    Value::Number(number) => number.as_f64().map(|value| value as f32),
                    Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
                    _ => None,
                }
                .ok_or_else(|| candle_core::Error::Msg(format!("{feature} must be numeric")))?;
                numeric.push(number);
            }
        }

        Ok(ModelInputs {
            categorical: Tensor::from_slice(
                &categorical,
                (1, steps.len(), self.categorical_features.len()),
                &self.device,
            )?,
            numeric: Tensor::from_slice(
                &numeric,
                (1, steps.len(), self.numeric_features.len()),
                &self.device,
            )?,
        })
    }
}

fn step_value<'a>(step: &'a MarketStep, feature: &str) -> Result<&'a Value> {
    step.state
        .get(feature)
        .or_else(|| step.range.get(feature))
        .ok_or_else(|| candle_core::Error::Msg(format!("step is missing {feature}")))
}

pub struct BufferedModelService {
    runtime: ModelRuntime,
    max_sequence_steps: usize,
    cycles: HashMap<String, VecDeque<MarketStep>>,
}

impl BufferedModelService {
    pub fn new(runtime: ModelRuntime) -> Self {
        let max_sequence_steps = runtime.max_sequence_steps;
        Self {
            runtime,
            max_sequence_steps,
            cycles: HashMap::new(),
        }
    }

    pub fn start_cycle(&mut self, cycle_id: impl Into<String>) -> Result<()> {
        let cycle_id = cycle_id.into();
        if self.cycles.contains_key(&cycle_id) {
            candle_core::bail!("cycle {cycle_id} is already active")
        }
        self.cycles.insert(cycle_id, VecDeque::new());
        Ok(())
    }

    pub fn push_step(&mut self, cycle_id: &str, step: MarketStep) -> Result<SequencePrediction> {
        let Some(buffer) = self.cycles.get_mut(cycle_id) else {
            candle_core::bail!("cycle {cycle_id} has not been started")
        };
        buffer.push_back(step);
        while buffer.len() > self.max_sequence_steps {
            buffer.pop_front();
        }
        let request = SequenceRequest {
            cycle_id: Some(cycle_id.to_owned()),
            steps: buffer.iter().cloned().collect(),
        };
        self.runtime.predict(&request)
    }

    pub fn end_cycle(&mut self, cycle_id: &str) -> Result<()> {
        if self.cycles.remove(cycle_id).is_none() {
            candle_core::bail!("cycle {cycle_id} is not active")
        }
        Ok(())
    }
}
