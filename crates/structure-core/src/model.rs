//! The market-structure model: hierarchical encoder + per-block / per-relation
//! heads (head A). Head B (next-block forecast) will hang off the same encoder.
//!
//! Flow: embed categorical + project numeric -> fuse (GELU) -> GRU over the 32
//! steps (hidden carries across block boundaries) -> mean-pool each block's 8
//! steps -> block representations -> linear heads.

use candle_core::{Result, Tensor};
use candle_nn::{
    Embedding, GRU, GRUConfig, Linear, Module, RNN, VarBuilder, embedding, gru, linear,
};

use crate::config::ModelConfig;
use crate::sequence::{
    BLOCK_LABEL_FIELDS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RELATION_LABEL_FIELDS,
};
use crate::tensors::ModelInputs;
use crate::vocab::FeatureVocab;

/// Logits per supervised field. Each list is in `BLOCK_LABEL_FIELDS` /
/// `RELATION_LABEL_FIELDS` order.
pub struct ModelOutput {
    /// One tensor per block field, shape `[batch, blocks, classes]`.
    pub block_logits: Vec<Tensor>,
    /// One tensor per relation field, shape `[batch, relations, classes]`.
    pub relation_logits: Vec<Tensor>,
}

pub struct MarketStructureModel {
    categorical_embeddings: Vec<Embedding>,
    numeric_projection: Linear,
    step_projection: Linear,
    gru: GRU,
    block_heads: Vec<Linear>,
    relation_heads: Vec<Linear>,
    block_size: usize,
}

impl MarketStructureModel {
    pub fn new(
        config: &ModelConfig,
        vocab: &FeatureVocab,
        block_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let mut categorical_embeddings = Vec::with_capacity(CATEGORICAL_FEATURES.len());
        for feature in CATEGORICAL_FEATURES {
            let classes = vocab
                .get(&format!("sequence.{feature}"))
                .map(|entry| entry.class_count())
                .unwrap_or(0);
            // +1 so the unknown id (0) has a row.
            let layer = embedding(
                classes + 1,
                config.categorical_embedding_dim,
                vb.pp(format!("embed.{feature}")),
            )?;
            categorical_embeddings.push(layer);
        }

        let numeric_projection = linear(
            NUMERIC_FEATURES.len(),
            config.numeric_projection_dim,
            vb.pp("numeric_projection"),
        )?;
        let fusion_dim = CATEGORICAL_FEATURES.len() * config.categorical_embedding_dim
            + config.numeric_projection_dim;
        let step_projection = linear(
            fusion_dim,
            config.step_representation_dim,
            vb.pp("step_projection"),
        )?;
        let gru = gru(
            config.step_representation_dim,
            config.gru_hidden_dim,
            GRUConfig::default(),
            vb.pp("gru"),
        )?;

        let mut block_heads = Vec::with_capacity(BLOCK_LABEL_FIELDS.len());
        for field in BLOCK_LABEL_FIELDS {
            let classes = head_classes(vocab, &format!("blocks.{field}"))?;
            block_heads.push(linear(
                config.gru_hidden_dim,
                classes,
                vb.pp(format!("head.block.{field}")),
            )?);
        }
        let mut relation_heads = Vec::with_capacity(RELATION_LABEL_FIELDS.len());
        for field in RELATION_LABEL_FIELDS {
            let classes = head_classes(vocab, &format!("relations.{field}"))?;
            relation_heads.push(linear(
                config.gru_hidden_dim * 2,
                classes,
                vb.pp(format!("head.relation.{field}")),
            )?);
        }

        Ok(Self {
            categorical_embeddings,
            numeric_projection,
            step_projection,
            gru,
            block_heads,
            relation_heads,
            block_size,
        })
    }

    pub fn forward(&self, inputs: &ModelInputs) -> Result<ModelOutput> {
        let (batch, seq_len, _) = inputs.categorical.dims3()?;

        let mut parts = Vec::with_capacity(self.categorical_embeddings.len());
        for (index, layer) in self.categorical_embeddings.iter().enumerate() {
            let ids = inputs
                .categorical
                .narrow(2, index, 1)?
                .reshape((batch, seq_len))?;
            parts.push(layer.forward(&ids)?);
        }
        let categorical = Tensor::cat(&parts, 2)?;
        let numeric = self.numeric_projection.forward(&inputs.numeric)?;
        let fused = Tensor::cat(&[&categorical, &numeric], 2)?;
        let step = self.step_projection.forward(&fused)?.gelu_erf()?;

        let states = self.gru.seq(&step)?;
        let hiddens = states.iter().map(|state| state.h().clone()).collect::<Vec<_>>();
        let hidden = Tensor::stack(&hiddens, 1)?.contiguous()?; // [batch, seq, hidden]
        let hidden_dim = hidden.dim(2)?;

        let blocks = seq_len / self.block_size;
        // Pool each block's steps; the GRU has already mixed earlier blocks in.
        let block_repr = hidden
            .reshape((batch, blocks, self.block_size, hidden_dim))?
            .mean(2)?; // [batch, blocks, hidden]

        let block_logits = self
            .block_heads
            .iter()
            .map(|head| head.forward(&block_repr))
            .collect::<Result<Vec<_>>>()?;

        let relations = blocks - 1;
        let left = block_repr.narrow(1, 0, relations)?;
        let right = block_repr.narrow(1, 1, relations)?;
        let pairs = Tensor::cat(&[&left, &right], 2)?.contiguous()?; // [batch, relations, 2*hidden]
        let relation_logits = self
            .relation_heads
            .iter()
            .map(|head| head.forward(&pairs))
            .collect::<Result<Vec<_>>>()?;

        Ok(ModelOutput {
            block_logits,
            relation_logits,
        })
    }
}

fn head_classes(vocab: &FeatureVocab, key: &str) -> Result<usize> {
    match vocab.get(key) {
        Some(entry) if entry.class_count() > 0 => Ok(entry.class_count()),
        _ => candle_core::bail!("missing or empty vocab for head {key}"),
    }
}
