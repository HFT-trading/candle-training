//! Serving entry point: load the exported artifacts and turn a context into a
//! StructureReport. This is what the external service depends on.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::config::ModelConfig;
use crate::model::MarketStructureModel;
use crate::normalizer::NumericNormalizer;
use crate::report::{StructureReport, build_report};
use crate::sequence::BLOCK_LABEL_FIELDS;
use crate::tensors::ModelInputs;
use crate::vocab::FeatureVocab;

pub struct StructureModel {
    model: MarketStructureModel,
    normalizer: NumericNormalizer,
    vocab: FeatureVocab,
}

impl StructureModel {
    /// Load `model.safetensors`, `numeric-normalizer.safetensors` and
    /// `vocab.json` from `dir`. `config`/`block_size` come from the same training
    /// run (a fully self-contained artifact bundle is a later refinement).
    pub fn load(
        dir: impl AsRef<Path>,
        config: &ModelConfig,
        block_size: usize,
        device: &Device,
    ) -> Result<Self, Box<dyn Error>> {
        let dir = dir.as_ref();
        let vocab: FeatureVocab = serde_json::from_reader(File::open(dir.join("vocab.json"))?)?;
        let normalizer =
            NumericNormalizer::load(dir.join("numeric-normalizer.safetensors"), device)?;

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = MarketStructureModel::new(config, &vocab, block_size, vb)?;
        varmap.load(dir.join("model.safetensors"))?;

        Ok(Self {
            model,
            normalizer,
            vocab,
        })
    }

    /// Read the latest block of a context into a StructureReport.
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
}

fn argmax_class(logits: &Tensor) -> CandleResult<usize> {
    Ok(logits.argmax(0)?.to_scalar::<u32>()? as usize)
}
