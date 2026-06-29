//! Training: source-level split, inverse-frequency class weights, weighted
//! cross-entropy over every block/relation field, and the AdamW loop.

use std::collections::{BTreeMap, HashMap};

use candle_core::{Device, Result, Tensor};
use candle_nn::ops::log_softmax;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use structure_core::model::{MarketStructureModel, ModelOutput};
use structure_core::normalizer::NumericNormalizer;
use structure_core::sequence::{BLOCK_LABEL_FIELDS, RELATION_LABEL_FIELDS};
use structure_core::tensors::ModelInputs;
use structure_core::vocab::FeatureVocab;

use crate::builder::TrainingTensors;
use crate::config::TrainingConfig;
use crate::dataset::Context;

/// Train/validation split at the source-file level, so heavily-overlapping
/// contexts (stride = 1 block) never straddle the split.
pub struct DataSplit {
    pub train: Vec<u32>,
    pub validation: Vec<u32>,
    pub validation_source: String,
}

impl DataSplit {
    pub fn by_source(contexts: &[Context], fraction: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&fraction) || fraction == 0.0 {
            candle_core::bail!("validation_fraction must be in (0, 1)");
        }
        let mut groups: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (index, context) in contexts.iter().enumerate() {
            groups
                .entry(context.metadata.source_name.clone())
                .or_default()
                .push(index as u32);
        }
        if groups.len() < 2 {
            candle_core::bail!("source-level split needs at least two source files");
        }
        let target = (contexts.len() as f64 * fraction).round() as usize;
        let (validation_source, validation) = groups
            .iter()
            .min_by_key(|(_, indices)| indices.len().abs_diff(target))
            .map(|(source, indices)| (source.clone(), indices.clone()))
            .expect("non-empty groups");
        let train = groups
            .into_iter()
            .filter(|(source, _)| source != &validation_source)
            .flat_map(|(_, indices)| indices)
            .collect();
        Ok(Self {
            train,
            validation,
            validation_source,
        })
    }

    /// Hold out an explicitly named source (for cross-source robustness checks).
    pub fn with_source(contexts: &[Context], source: &str) -> Result<Self> {
        let mut train = Vec::new();
        let mut validation = Vec::new();
        for (index, context) in contexts.iter().enumerate() {
            if context.metadata.source_name == source {
                validation.push(index as u32);
            } else {
                train.push(index as u32);
            }
        }
        if validation.is_empty() {
            candle_core::bail!("no contexts for source {source:?}");
        }
        if train.is_empty() {
            candle_core::bail!("every context is source {source:?}; nothing left to train on");
        }
        Ok(Self {
            train,
            validation,
            validation_source: source.to_owned(),
        })
    }
}

/// Inverse-frequency (balanced) class weights per field.
pub struct ClassWeights {
    pub block: Vec<Tensor>,
    pub relation: Vec<Tensor>,
}

impl ClassWeights {
    pub fn from_vocab(vocab: &FeatureVocab, device: &Device) -> Result<Self> {
        let block = BLOCK_LABEL_FIELDS
            .iter()
            .map(|field| field_weights(vocab, &format!("blocks.{field}"), device))
            .collect::<Result<Vec<_>>>()?;
        let relation = RELATION_LABEL_FIELDS
            .iter()
            .map(|field| field_weights(vocab, &format!("relations.{field}"), device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { block, relation })
    }
}

fn field_weights(vocab: &FeatureVocab, key: &str, device: &Device) -> Result<Tensor> {
    let entry = vocab
        .get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing vocab for {key}")))?;
    let classes = entry.class_count();
    let mut counts = vec![0f64; classes];
    for (value, id) in &entry.value_to_id {
        counts[(*id - 1) as usize] = *entry.counts.get(value).unwrap_or(&0) as f64;
    }
    let total: f64 = counts.iter().sum();
    let weights: Vec<f32> = counts
        .iter()
        .map(|&count| {
            if count > 0.0 {
                (total / (classes as f64 * count)) as f32
            } else {
                1.0
            }
        })
        .collect();
    Tensor::from_vec(weights, classes, device)
}

/// Result of a training run: the normalizer plus the best-checkpoint info.
/// The best model weights are saved to `artifacts/model.safetensors` during the
/// run (at the epoch with the lowest validation loss).
pub struct TrainReport {
    pub normalizer: NumericNormalizer,
    pub best_epoch: usize,
    pub best_val: f64,
    pub best_eval: EvalReport,
}

pub fn train(
    model: &MarketStructureModel,
    variables: &VarMap,
    tensors: &TrainingTensors,
    split: &DataSplit,
    config: &TrainingConfig,
    weights: Option<&ClassWeights>,
    verbose: bool,
) -> Result<TrainReport> {
    if config.epochs == 0 || config.batch_size == 0 {
        candle_core::bail!("epochs and batch_size must be > 0");
    }
    let normalizer = NumericNormalizer::fit(&tensors.inputs.numeric, &split.train)?;
    let params = ParamsAdamW {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(variables.all_vars(), params)?;
    let mut rng = StdRng::seed_from_u64(config.shuffle_seed);

    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0;
    let mut best_eval: Option<EvalReport> = None;
    let mut stale = 0;

    for epoch in 1..=config.epochs {
        let mut train_indices = split.train.clone();
        train_indices.shuffle(&mut rng);
        let train = run_epoch(
            model,
            tensors,
            &normalizer,
            &train_indices,
            weights,
            config,
            Some(&mut optimizer),
        )?;
        let validation = run_epoch(
            model,
            tensors,
            &normalizer,
            &split.validation,
            weights,
            config,
            None,
        )?;
        if verbose {
            println!(
                "epoch {epoch}: train total={:.4} block={:.4} rel={:.4} | val total={:.4} block={:.4} rel={:.4}",
                train.0, train.1, train.2, validation.0, validation.1, validation.2
            );
        }

        if validation.0 + 1e-9 < best_val {
            best_val = validation.0;
            best_epoch = epoch;
            best_eval = Some(evaluate(
                model,
                tensors,
                &normalizer,
                &split.validation,
                config.batch_size,
            )?);
            // Checkpoint the best weights; `artifacts/` is created by the caller.
            variables.save("artifacts/model.safetensors")?;
            stale = 0;
        } else {
            stale += 1;
            if config.early_stop_patience > 0 && stale >= config.early_stop_patience {
                if verbose {
                    println!(
                        "early stop: no val improvement for {stale} epochs (best epoch {best_epoch})"
                    );
                }
                break;
            }
        }
    }

    Ok(TrainReport {
        normalizer,
        best_epoch,
        best_val,
        best_eval: best_eval.expect("at least one epoch runs"),
    })
}

fn run_epoch(
    model: &MarketStructureModel,
    tensors: &TrainingTensors,
    normalizer: &NumericNormalizer,
    indices: &[u32],
    weights: Option<&ClassWeights>,
    config: &TrainingConfig,
    optimizer: Option<&mut AdamW>,
) -> Result<(f64, f64, f64)> {
    let mut optimizer = optimizer;
    let (mut sum_total, mut sum_block, mut sum_relation, mut seen) = (0.0, 0.0, 0.0, 0usize);

    for batch in indices.chunks(config.batch_size) {
        let inputs = select_inputs(tensors, normalizer, batch)?;
        let block_targets = select_rows(&tensors.block_targets, batch)?;
        let relation_targets = select_rows(&tensors.relation_targets, batch)?;
        let output = model.forward(&inputs)?;
        let (total, block, relation) =
            total_loss(&output, &block_targets, &relation_targets, weights, config)?;
        if let Some(optimizer) = optimizer.as_deref_mut() {
            optimizer.backward_step(&total)?;
        }
        let count = batch.len() as f64;
        sum_total += total.to_scalar::<f32>()? as f64 * count;
        sum_block += block * count;
        sum_relation += relation * count;
        seen += batch.len();
    }

    let seen = seen.max(1) as f64;
    Ok((sum_total / seen, sum_block / seen, sum_relation / seen))
}

fn total_loss(
    output: &ModelOutput,
    block_targets: &Tensor,
    relation_targets: &Tensor,
    weights: Option<&ClassWeights>,
    config: &TrainingConfig,
) -> Result<(Tensor, f64, f64)> {
    let mut block_terms = Vec::with_capacity(output.block_logits.len());
    for (field, logits) in output.block_logits.iter().enumerate() {
        let targets = block_targets.narrow(2, field, 1)?.squeeze(2)?.contiguous()?;
        let weight = weights.map(|class| &class.block[field]);
        block_terms.push(field_loss(logits, &targets, weight)?);
    }
    let block_loss = average(&block_terms)?;

    let mut relation_terms = Vec::with_capacity(output.relation_logits.len());
    for (field, logits) in output.relation_logits.iter().enumerate() {
        let targets = relation_targets.narrow(2, field, 1)?.squeeze(2)?.contiguous()?;
        let weight = weights.map(|class| &class.relation[field]);
        relation_terms.push(field_loss(logits, &targets, weight)?);
    }
    let relation_loss = average(&relation_terms)?;

    let total = (block_loss.affine(config.block_loss_weight, 0.0)?
        + relation_loss.affine(config.relation_loss_weight, 0.0)?)?;
    let block_value = block_loss.to_scalar::<f32>()? as f64;
    let relation_value = relation_loss.to_scalar::<f32>()? as f64;
    Ok((total, block_value, relation_value))
}

fn field_loss(logits: &Tensor, targets: &Tensor, weights: Option<&Tensor>) -> Result<Tensor> {
    let (batch, units, classes) = logits.dims3()?;
    let logits = logits.reshape((batch * units, classes))?;
    let targets = targets.reshape(batch * units)?;
    weighted_cross_entropy(&logits, &targets, weights)
}

fn weighted_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    weights: Option<&Tensor>,
) -> Result<Tensor> {
    let log_probabilities = log_softmax(logits, 1)?;
    let picked = log_probabilities
        .gather(&targets.unsqueeze(1)?, 1)?
        .squeeze(1)?;
    let nll = picked.neg()?;
    match weights {
        None => nll.mean_all(),
        Some(weights) => {
            let sample_weights = weights.index_select(targets, 0)?;
            let weighted = nll.mul(&sample_weights)?;
            weighted.sum_all()?.broadcast_div(&sample_weights.sum_all()?)
        }
    }
}

fn average(terms: &[Tensor]) -> Result<Tensor> {
    let mut accumulator = terms[0].clone();
    for term in &terms[1..] {
        accumulator = (accumulator + term)?;
    }
    accumulator.affine(1.0 / terms.len() as f64, 0.0)
}

fn select_inputs(
    tensors: &TrainingTensors,
    normalizer: &NumericNormalizer,
    indices: &[u32],
) -> Result<ModelInputs> {
    let categorical = select_rows(&tensors.inputs.categorical, indices)?;
    let numeric = normalizer.apply(&select_rows(&tensors.inputs.numeric, indices)?)?;
    Ok(ModelInputs {
        categorical,
        numeric,
    })
}

fn select_rows(tensor: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let ids = Tensor::from_slice(indices, indices.len(), tensor.device())?;
    tensor.index_select(&ids, 0)
}

/// Per-field accuracy vs the majority-class baseline (the bar to beat).
pub struct FieldAccuracy {
    pub field: String,
    pub accuracy: f64,
    /// Mean per-class recall — fair to rare classes (acc hides them).
    pub macro_recall: f64,
    pub baseline: f64,
    pub samples: usize,
}

pub struct EvalReport {
    pub block: Vec<FieldAccuracy>,
    pub relation: Vec<FieldAccuracy>,
}

pub fn evaluate(
    model: &MarketStructureModel,
    tensors: &TrainingTensors,
    normalizer: &NumericNormalizer,
    indices: &[u32],
    batch_size: usize,
) -> Result<EvalReport> {
    let mut block = vec![FieldStats::default(); BLOCK_LABEL_FIELDS.len()];
    let mut relation = vec![FieldStats::default(); RELATION_LABEL_FIELDS.len()];

    for batch in indices.chunks(batch_size.max(1)) {
        let inputs = select_inputs(tensors, normalizer, batch)?;
        let output = model.forward(&inputs)?;
        let block_targets = select_rows(&tensors.block_targets, batch)?;
        let relation_targets = select_rows(&tensors.relation_targets, batch)?;

        for (field, stats) in block.iter_mut().enumerate() {
            let predicted = output.block_logits[field]
                .argmax(2)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let actual = block_targets
                .narrow(2, field, 1)?
                .squeeze(2)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            stats.add(&predicted, &actual);
        }
        for (field, stats) in relation.iter_mut().enumerate() {
            let predicted = output.relation_logits[field]
                .argmax(2)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let actual = relation_targets
                .narrow(2, field, 1)?
                .squeeze(2)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            stats.add(&predicted, &actual);
        }
    }

    Ok(EvalReport {
        block: BLOCK_LABEL_FIELDS
            .iter()
            .zip(block)
            .map(|(field, stats)| stats.finish(field))
            .collect(),
        relation: RELATION_LABEL_FIELDS
            .iter()
            .zip(relation)
            .map(|(field, stats)| stats.finish(field))
            .collect(),
    })
}

#[derive(Default, Clone)]
struct FieldStats {
    correct: usize,
    total: usize,
    /// Per true-class occurrence count.
    histogram: HashMap<u32, usize>,
    /// Per true-class correct count (for recall).
    class_correct: HashMap<u32, usize>,
}

impl FieldStats {
    fn add(&mut self, predicted: &[u32], actual: &[u32]) {
        for (prediction, target) in predicted.iter().zip(actual) {
            self.total += 1;
            *self.histogram.entry(*target).or_insert(0) += 1;
            if prediction == target {
                self.correct += 1;
                *self.class_correct.entry(*target).or_insert(0) += 1;
            }
        }
    }

    fn finish(self, field: &str) -> FieldAccuracy {
        let total = self.total.max(1);
        let majority = self.histogram.values().copied().max().unwrap_or(0);
        let recall_sum: f64 = self
            .histogram
            .iter()
            .map(|(class, count)| {
                let hit = self.class_correct.get(class).copied().unwrap_or(0);
                hit as f64 / *count as f64
            })
            .sum();
        let classes = self.histogram.len().max(1);
        FieldAccuracy {
            field: field.to_owned(),
            accuracy: self.correct as f64 / total as f64,
            macro_recall: recall_sum / classes as f64,
            baseline: majority as f64 / total as f64,
            samples: self.total,
        }
    }
}
