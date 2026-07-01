//! The market-structure model: hierarchical encoder + per-block / per-relation
//! structure heads.
//!
//! Flow: embed categorical + project numeric -> fuse (GELU) -> GRU over the 32
//! steps (hidden carries across block boundaries) -> mean-pool each block's 8
//! steps -> block representations -> linear heads.

use candle_core::{Result, Tensor};
use candle_nn::{
    Embedding, GRU, GRUConfig, Linear, Module, RNN, VarBuilder, embedding, gru, linear,
};

use crate::config::ModelConfig;
use crate::sequence::{BLOCK_LABEL_FIELDS, CATEGORICAL_FEATURES, NUMERIC_FEATURES};
use crate::tensors::ModelInputs;
use crate::vocab::FeatureVocab;

/// Logits per supervised field, in `BLOCK_LABEL_FIELDS` order.
pub struct ModelOutput {
    /// One tensor per block field, shape `[batch, blocks, classes]`.
    pub block_logits: Vec<Tensor>,
}

pub struct MarketStructureModel {
    categorical_embeddings: Vec<Embedding>,
    numeric_projection: Linear,
    step_projection: Linear,
    gru: GRU,
    block_heads: Vec<Linear>,
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

        // block_repr = concat[mean-pool, last-step] over each block's 8 steps, so
        // the ending (needed to read intra-block process: reclaim, fade, ...) is
        // preserved alongside the average the other heads relied on.
        let block_repr_dim = config.gru_hidden_dim * 2;

        let mut block_heads = Vec::with_capacity(BLOCK_LABEL_FIELDS.len());
        for field in BLOCK_LABEL_FIELDS {
            let classes = head_classes(vocab, &format!("blocks.{field}"))?;
            block_heads.push(linear(
                block_repr_dim,
                classes,
                vb.pp(format!("head.block.{field}")),
            )?);
        }
        Ok(Self {
            categorical_embeddings,
            numeric_projection,
            step_projection,
            gru,
            block_heads,
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
        // Per block: concat[mean over its 8 steps, last step]. Mean gives the
        // average (what earlier heads used); the last GRU hidden carries the
        // ordered trajectory/ending that intra-block process reads need.
        let per_step = hidden.reshape((batch, blocks, self.block_size, hidden_dim))?;
        let mean_pool = per_step.mean(2)?; // [batch, blocks, hidden]
        let last_pool = per_step
            .narrow(2, self.block_size - 1, 1)?
            .squeeze(2)?; // [batch, blocks, hidden]
        let block_repr = Tensor::cat(&[&mean_pool, &last_pool], 2)?.contiguous()?; // [batch, blocks, 2*hidden]

        let block_logits = self
            .block_heads
            .iter()
            .map(|head| head.forward(&block_repr))
            .collect::<Result<Vec<_>>>()?;

        Ok(ModelOutput { block_logits })
    }
}

fn head_classes(vocab: &FeatureVocab, key: &str) -> Result<usize> {
    match vocab.get(key) {
        Some(entry) if entry.class_count() > 0 => Ok(entry.class_count()),
        _ => candle_core::bail!("missing or empty vocab for head {key}"),
    }
}
