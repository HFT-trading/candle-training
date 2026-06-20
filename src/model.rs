use std::collections::HashMap;
use std::path::Path;

use candle_core::{Result, Tensor};
use candle_nn::{
    Embedding, GRU, GRUConfig, Linear, Module, RNN, VarBuilder, embedding, gru, linear,
};

use crate::config::ModelConfig;
use crate::core::ModelInputs;
use crate::dataset::FeatureSchema;

#[derive(Debug)]
pub struct EncodedStepFeatures {
    pub categorical: Tensor,
    pub numeric: Tensor,
}

#[derive(Debug)]
pub struct SequenceEncoding {
    pub step_representations: Tensor,
    pub hidden_states: Tensor,
    pub final_hidden: Tensor,
}

#[derive(Debug)]
pub struct StateHeadLogits {
    pub regime: Tensor,
    pub quality: Tensor,
    pub stage: Tensor,
}

#[derive(Debug)]
pub struct ModelOutput {
    pub state: StateHeadLogits,
    pub move_outlook: Tensor,
}

#[derive(Debug)]
pub struct NumericNormalizer {
    pub mean: Tensor,
    pub std: Tensor,
}

impl NumericNormalizer {
    pub fn fit(numeric: &Tensor, train_indices: &[u32]) -> Result<Self> {
        let ids = Tensor::from_slice(train_indices, train_indices.len(), numeric.device())?;
        let train_numeric = numeric.index_select(&ids, 0)?;
        let mean = train_numeric.mean_keepdim((0, 1))?;
        let centered = train_numeric.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim((0, 1))?;
        let std = (variance + 1e-6)?.sqrt()?;
        Ok(Self { mean, std })
    }

    pub fn load(path: impl AsRef<Path>, device: &candle_core::Device) -> Result<Self> {
        let mut tensors = candle_core::safetensors::load(path, device)?;
        let Some(mean) = tensors.remove("mean") else {
            candle_core::bail!("numeric normalizer is missing mean")
        };
        let Some(std) = tensors.remove("std") else {
            candle_core::bail!("numeric normalizer is missing std")
        };
        Ok(Self { mean, std })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        candle_core::safetensors::save(
            &HashMap::from([("mean", self.mean.clone()), ("std", self.std.clone())]),
            path,
        )
    }

    pub fn apply(&self, numeric: &Tensor) -> Result<Tensor> {
        numeric.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }
}

impl ModelOutput {
    pub fn validate_shapes(
        &self,
        batch_size: usize,
        sequence_length: usize,
        schema: &FeatureSchema,
    ) -> Result<()> {
        validate_shape(
            "regime logits",
            self.state.regime.dims(),
            &[
                batch_size,
                sequence_length,
                known_state_class_count(schema, "current.regime")?,
            ],
        )?;
        validate_shape(
            "quality logits",
            self.state.quality.dims(),
            &[
                batch_size,
                sequence_length,
                known_state_class_count(schema, "current.quality")?,
            ],
        )?;
        validate_shape(
            "stage logits",
            self.state.stage.dims(),
            &[
                batch_size,
                sequence_length,
                known_state_class_count(schema, "cycle.stage")?,
            ],
        )?;
        validate_shape(
            "move-outlook logits",
            self.move_outlook.dims(),
            &[batch_size, schema.target_groups.boolean.len()],
        )?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct FeatureEncoder {
    categorical_embeddings: Vec<Embedding>,
    numeric_projection: Linear,
    categorical_feature_count: usize,
    numeric_feature_count: usize,
}

impl FeatureEncoder {
    pub fn new(config: &ModelConfig, schema: &FeatureSchema, vb: VarBuilder<'_>) -> Result<Self> {
        let mut categorical_embeddings =
            Vec::with_capacity(schema.model_input.categorical_features.len());

        for feature in &schema.model_input.categorical_features {
            let Some(vocab) = schema.categorical_vocab.get(feature) else {
                candle_core::bail!("missing categorical vocabulary for {feature}")
            };
            let vocab_size = vocab.values().copied().max().unwrap_or(0) as usize + 1;
            let layer = embedding(
                vocab_size,
                config.categorical_embedding_dim,
                vb.pp(format!("categorical.{feature}")),
            )?;
            categorical_embeddings.push(layer);
        }

        let numeric_feature_count = schema.model_input.numeric_features.len();
        let numeric_projection = linear(
            numeric_feature_count,
            config.numeric_projection_dim,
            vb.pp("numeric_projection"),
        )?;

        Ok(Self {
            categorical_feature_count: categorical_embeddings.len(),
            categorical_embeddings,
            numeric_projection,
            numeric_feature_count,
        })
    }

    pub fn forward(&self, inputs: &ModelInputs) -> Result<EncodedStepFeatures> {
        let (batch_size, sequence_length, categorical_count) = inputs.categorical.dims3()?;
        if categorical_count != self.categorical_feature_count {
            candle_core::bail!(
                "expected {} categorical features, got {categorical_count}",
                self.categorical_feature_count
            )
        }

        let (_, _, numeric_count) = inputs.numeric.dims3()?;
        if numeric_count != self.numeric_feature_count {
            candle_core::bail!(
                "expected {} numeric features, got {numeric_count}",
                self.numeric_feature_count
            )
        }

        let mut categorical_parts = Vec::with_capacity(self.categorical_feature_count);
        for (feature_index, layer) in self.categorical_embeddings.iter().enumerate() {
            let ids = inputs
                .categorical
                .narrow(2, feature_index, 1)?
                .reshape((batch_size, sequence_length))?;
            categorical_parts.push(layer.forward(&ids)?);
        }

        Ok(EncodedStepFeatures {
            categorical: Tensor::cat(&categorical_parts, 2)?,
            numeric: self.numeric_projection.forward(&inputs.numeric)?,
        })
    }
}

#[derive(Debug)]
pub struct SequenceEncoder {
    feature_encoder: FeatureEncoder,
    step_projection: Linear,
    gru: GRU,
}

impl SequenceEncoder {
    pub fn new(config: &ModelConfig, schema: &FeatureSchema, vb: VarBuilder<'_>) -> Result<Self> {
        let feature_encoder = FeatureEncoder::new(config, schema, vb.pp("features"))?;
        let categorical_width =
            schema.model_input.categorical_features.len() * config.categorical_embedding_dim;
        let fusion_width = categorical_width + config.numeric_projection_dim;
        let step_projection = linear(
            fusion_width,
            config.step_representation_dim,
            vb.pp("step_projection"),
        )?;
        let gru = gru(
            config.step_representation_dim,
            config.gru_hidden_dim,
            GRUConfig::default(),
            vb.pp("gru"),
        )?;

        Ok(Self {
            feature_encoder,
            step_projection,
            gru,
        })
    }

    pub fn forward(&self, inputs: &ModelInputs) -> Result<SequenceEncoding> {
        let encoded = self.feature_encoder.forward(inputs)?;
        let fused = Tensor::cat(&[&encoded.categorical, &encoded.numeric], 2)?;
        let step_representations = self.step_projection.forward(&fused)?.gelu_erf()?;
        let states = self.gru.seq(&step_representations)?;
        let Some(final_state) = states.last() else {
            candle_core::bail!("cannot encode an empty sequence")
        };
        let hidden = states
            .iter()
            .map(|state| state.h().clone())
            .collect::<Vec<_>>();

        Ok(SequenceEncoding {
            step_representations,
            hidden_states: Tensor::stack(&hidden, 1)?,
            final_hidden: final_state.h().clone(),
        })
    }
}

#[derive(Debug)]
pub struct MarketStateModel {
    sequence_encoder: SequenceEncoder,
    regime_head: Linear,
    quality_head: Linear,
    stage_head: Linear,
    move_outlook_head: Linear,
}

impl MarketStateModel {
    pub fn new(config: &ModelConfig, schema: &FeatureSchema, vb: VarBuilder<'_>) -> Result<Self> {
        let sequence_encoder = SequenceEncoder::new(config, schema, vb.pp("sequence_encoder"))?;
        let hidden_dim = config.gru_hidden_dim;
        let regime_head = linear(
            hidden_dim,
            known_state_class_count(schema, "current.regime")?,
            vb.pp("heads.regime"),
        )?;
        let quality_head = linear(
            hidden_dim,
            known_state_class_count(schema, "current.quality")?,
            vb.pp("heads.quality"),
        )?;
        let stage_head = linear(
            hidden_dim,
            known_state_class_count(schema, "cycle.stage")?,
            vb.pp("heads.stage"),
        )?;
        let outlook_count = schema.target_groups.boolean.len();
        if outlook_count == 0 {
            candle_core::bail!("move-outlook head requires at least one boolean target")
        }
        let move_outlook_head = linear(hidden_dim, outlook_count, vb.pp("heads.move_outlook"))?;

        Ok(Self {
            sequence_encoder,
            regime_head,
            quality_head,
            stage_head,
            move_outlook_head,
        })
    }

    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        let encoded = self.sequence_encoder.forward(inputs)?;

        Ok(ModelOutput {
            state: StateHeadLogits {
                regime: self.regime_head.forward(&encoded.hidden_states)?,
                quality: self.quality_head.forward(&encoded.hidden_states)?,
                stage: self.stage_head.forward(&encoded.hidden_states)?,
            },
            move_outlook: self.move_outlook_head.forward(&encoded.final_hidden)?,
        })
    }
}

fn known_state_class_count(schema: &FeatureSchema, target: &str) -> Result<usize> {
    let Some(vocab) = schema.debug_semantics.categorical_vocab.get(target) else {
        candle_core::bail!("missing state-target vocabulary for {target}")
    };
    let count = vocab
        .keys()
        .filter(|category| category.as_str() != "__UNK__")
        .count();
    if count == 0 {
        candle_core::bail!("state-target vocabulary for {target} has no known classes")
    }
    Ok(count)
}

fn validate_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        candle_core::bail!("invalid {name} shape: expected {expected:?}, got {actual:?}")
    }
    Ok(())
}
