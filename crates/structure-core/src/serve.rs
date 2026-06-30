//! Serving entry point: load a self-contained artifact bundle and turn a stream
//! of steps into StructureReports. This is what the hft-bot service depends on.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;
use crate::input::{StepInput, encode_steps};
use crate::model::MarketStructureModel;
use crate::normalizer::NumericNormalizer;
use crate::report::{StructureReport, build_report};
use crate::sequence::BLOCK_LABEL_FIELDS;
use crate::tensors::ModelInputs;
use crate::vocab::FeatureVocab;

/// Self-describing artifact metadata saved next to the weights so that
/// `StructureModel::load(dir)` needs nothing else.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServeMeta {
    pub model: ModelConfig,
    pub block_size: usize,
    pub context_blocks: usize,
}

pub struct StructureModel {
    model: MarketStructureModel,
    normalizer: NumericNormalizer,
    vocab: FeatureVocab,
    device: Device,
    block_size: usize,
    context_blocks: usize,
}

impl StructureModel {
    /// Load `meta.json`, `vocab.json`, `numeric-normalizer.safetensors` and
    /// `model.safetensors` from `dir`.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self, Box<dyn Error>> {
        let dir = dir.as_ref();
        let meta: ServeMeta = serde_json::from_reader(File::open(dir.join("meta.json"))?)?;
        let vocab: FeatureVocab = serde_json::from_reader(File::open(dir.join("vocab.json"))?)?;
        let normalizer =
            NumericNormalizer::load(dir.join("numeric-normalizer.safetensors"), device)?;

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = MarketStructureModel::new(&meta.model, &vocab, meta.block_size, vb)?;
        varmap.load(dir.join("model.safetensors"))?;

        Ok(Self {
            model,
            normalizer,
            vocab,
            device: device.clone(),
            block_size: meta.block_size,
            context_blocks: meta.context_blocks,
        })
    }

    /// Convenience loader that runs inference on the CPU, so callers that only
    /// serve (e.g. hft-bot) never have to name a `candle_core::Device`.
    pub fn load_cpu(dir: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        Self::load(dir, &Device::Cpu)
    }

    /// Full window size the model was trained on (block_size * context_blocks).
    pub fn max_steps(&self) -> usize {
        self.block_size * self.context_blocks
    }

    /// One block of steps; a report is emitted each time the window grows by this.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Encode an ordered window of `StepInput`s with the model's own vocab and
    /// device, so callers don't need access to either.
    pub fn encode(&self, steps: &[StepInput]) -> CandleResult<ModelInputs> {
        encode_steps(steps, &self.vocab, &self.device)
    }

    /// Read a pre-encoded context (steps a multiple of block_size) into a report.
    pub fn read(&self, inputs: &ModelInputs) -> CandleResult<StructureReport> {
        let numeric = self.normalizer.apply(&inputs.numeric)?;
        let normalized = ModelInputs {
            categorical: inputs.categorical.clone(),
            numeric,
        };
        let output = self.model.forward(&normalized)?;

        let blocks = output.block_logits[0].dim(1)?;
        let last_block = blocks - 1;
        let mut labels = HashMap::new();
        for (index, field) in BLOCK_LABEL_FIELDS.iter().enumerate() {
            let class = argmax_class(&output.block_logits[index].i((0, last_block))?)?;
            if let Some(value) = self.vocab.label(&format!("blocks.{field}"), class) {
                labels.insert((*field).to_owned(), value.to_owned());
            }
        }
        let relations = output.relation_logits[0].dim(1)?;
        let relation_class = argmax_class(&output.relation_logits[0].i((0, relations - 1))?)?;
        let relation = self
            .vocab
            .label("relations.relation", relation_class)
            .unwrap_or("-")
            .to_owned();

        Ok(build_report(&labels, &relation))
    }

    /// Start a streaming session: push one `StepInput` at a time.
    pub fn session(&self) -> Session<'_> {
        Session {
            model: self,
            buffer: Vec::with_capacity(self.max_steps()),
        }
    }
}

/// Streaming read: holds a sliding window of recent steps; emits a
/// StructureReport each time a block completes, `None` in between.
pub struct Session<'a> {
    model: &'a StructureModel,
    buffer: Vec<StepInput>,
}

impl Session<'_> {
    pub fn push(&mut self, step: StepInput) -> CandleResult<Option<StructureReport>> {
        self.buffer.push(step);
        if self.buffer.len() > self.model.max_steps() {
            self.buffer.drain(0..self.model.block_size);
        }
        if !self.buffer.is_empty() && self.buffer.len() % self.model.block_size == 0 {
            let inputs = encode_steps(&self.buffer, &self.model.vocab, &self.model.device)?;
            return Ok(Some(self.model.read(&inputs)?));
        }
        Ok(None)
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

fn argmax_class(logits: &Tensor) -> CandleResult<usize> {
    Ok(logits.argmax(0)?.to_scalar::<u32>()? as usize)
}
