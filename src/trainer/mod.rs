use std::collections::BTreeMap;

use candle_core::{Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap, loss};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use tracing::info;

use crate::config::TrainingConfig;
use crate::core::{ModelInputs, TrainingTensors};
use crate::dataset::MarketDataset;
use crate::model::{MarketStateModel, ModelOutput, NumericNormalizer};

pub struct TrainingDataSummary {
    pub sequences: usize,
    pub sequence_length: usize,
    pub categorical_features: usize,
    pub numeric_features: usize,
    pub state_targets: usize,
    pub categorical_targets: usize,
    pub boolean_targets: usize,
    pub numeric_targets: usize,
}

impl TrainingDataSummary {
    pub fn from_tensors(tensors: &TrainingTensors) -> Self {
        let categorical = tensors.inputs.categorical.dims();
        let numeric = tensors.inputs.numeric.dims();
        let state_targets = tensors.targets.state_categorical.dims();
        let categorical_targets = tensors.targets.categorical.dims();
        let boolean_targets = tensors.targets.boolean.dims();
        let numeric_targets = tensors.targets.numeric.dims();

        Self {
            sequences: categorical[0],
            sequence_length: categorical[1],
            categorical_features: categorical[2],
            numeric_features: numeric[2],
            state_targets: state_targets[2],
            categorical_targets: categorical_targets[1],
            boolean_targets: boolean_targets[1],
            numeric_targets: numeric_targets[1],
        }
    }
}

#[derive(Debug)]
pub struct DataSplit {
    pub train_indices: Vec<u32>,
    pub validation_indices: Vec<u32>,
    pub validation_source: String,
}

impl DataSplit {
    pub fn by_source(dataset: &MarketDataset, validation_fraction: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&validation_fraction) || validation_fraction == 0.0 {
            candle_core::bail!("validation_fraction must be between 0 and 1")
        }

        let mut groups = BTreeMap::<String, Vec<u32>>::new();
        for (index, sequence) in dataset.sequences.iter().enumerate() {
            groups
                .entry(sequence.source_name.clone())
                .or_default()
                .push(index as u32);
        }
        if groups.len() < 2 {
            candle_core::bail!("source-level validation requires at least two source files")
        }

        let target_size = (dataset.sequences.len() as f64 * validation_fraction).round() as usize;
        let (validation_source, validation_indices) = groups
            .iter()
            .min_by_key(|(_, indices)| indices.len().abs_diff(target_size))
            .map(|(source, indices)| (source.clone(), indices.clone()))
            .expect("source groups cannot be empty");
        let train_indices = groups
            .into_iter()
            .filter(|(source, _)| source != &validation_source)
            .flat_map(|(_, indices)| indices)
            .collect();

        Ok(Self {
            train_indices,
            validation_indices,
            validation_source,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub total: f64,
    pub state: f64,
    pub regime: f64,
    pub quality: f64,
    pub stage: f64,
    pub outlook: f64,
}

#[derive(Debug)]
pub struct TrainingReport {
    pub epochs: Vec<(EpochMetrics, EpochMetrics)>,
    pub normalizer: NumericNormalizer,
}

pub fn train(
    model: &MarketStateModel,
    variables: &VarMap,
    tensors: &TrainingTensors,
    split: &DataSplit,
    config: &TrainingConfig,
) -> Result<TrainingReport> {
    if config.epochs == 0 || config.batch_size == 0 {
        candle_core::bail!("epochs and batch_size must be greater than zero")
    }

    let normalizer = NumericNormalizer::fit(&tensors.inputs.numeric, &split.train_indices)?;
    let params = ParamsAdamW {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..ParamsAdamW::default()
    };
    let mut optimizer = AdamW::new(variables.all_vars(), params)?;
    let mut rng = StdRng::seed_from_u64(config.shuffle_seed);
    let mut epochs = Vec::with_capacity(config.epochs);

    for epoch in 1..=config.epochs {
        let mut train_indices = split.train_indices.clone();
        train_indices.shuffle(&mut rng);
        let train_metrics = run_epoch(
            model,
            tensors,
            &normalizer,
            &train_indices,
            config,
            Some(&mut optimizer),
        )?;
        let validation_metrics = run_epoch(
            model,
            tensors,
            &normalizer,
            &split.validation_indices,
            config,
            None,
        )?;

        info!(
            epoch,
            train_total = train_metrics.total,
            train_state = train_metrics.state,
            train_outlook = train_metrics.outlook,
            validation_total = validation_metrics.total,
            validation_state = validation_metrics.state,
            validation_outlook = validation_metrics.outlook,
            "training epoch completed"
        );
        epochs.push((train_metrics, validation_metrics));
    }

    Ok(TrainingReport { epochs, normalizer })
}

fn run_epoch(
    model: &MarketStateModel,
    tensors: &TrainingTensors,
    normalizer: &NumericNormalizer,
    indices: &[u32],
    config: &TrainingConfig,
    mut optimizer: Option<&mut AdamW>,
) -> Result<EpochMetrics> {
    let mut accumulator = MetricsAccumulator::default();

    for batch_indices in indices.chunks(config.batch_size) {
        let batch = select_batch(tensors, normalizer, batch_indices)?;
        let output = model.forward(&batch.inputs)?;
        let losses = compute_losses(
            &output,
            &batch.state_targets,
            &batch.outlook_targets,
            config,
        )?;
        accumulator.add(&losses, batch_indices.len())?;
        if let Some(optimizer) = optimizer.as_deref_mut() {
            optimizer.backward_step(&losses.total)?;
        }
    }

    accumulator.finish()
}

struct TrainingBatch {
    inputs: ModelInputs,
    state_targets: Tensor,
    outlook_targets: Tensor,
}

fn select_batch(
    tensors: &TrainingTensors,
    normalizer: &NumericNormalizer,
    indices: &[u32],
) -> Result<TrainingBatch> {
    let numeric = select_rows(&tensors.inputs.numeric, indices)?;
    Ok(TrainingBatch {
        inputs: ModelInputs {
            categorical: select_rows(&tensors.inputs.categorical, indices)?,
            numeric: normalizer.apply(&numeric)?,
        },
        state_targets: select_rows(&tensors.targets.state_categorical, indices)?,
        outlook_targets: select_rows(&tensors.targets.boolean, indices)?,
    })
}

fn select_rows(tensor: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let ids = Tensor::from_slice(indices, indices.len(), tensor.device())?;
    tensor.index_select(&ids, 0)
}

struct LossTensors {
    total: Tensor,
    state: Tensor,
    regime: Tensor,
    quality: Tensor,
    stage: Tensor,
    outlook: Tensor,
}

fn compute_losses(
    output: &ModelOutput,
    state_targets: &Tensor,
    outlook_targets: &Tensor,
    config: &TrainingConfig,
) -> Result<LossTensors> {
    let regime = categorical_step_loss(&output.state.regime, state_targets, 0)?;
    let quality = categorical_step_loss(&output.state.quality, state_targets, 1)?;
    let stage = categorical_step_loss(&output.state.stage, state_targets, 2)?;
    let state = ((&regime + &quality)? + &stage)?.affine(1.0 / 3.0, 0.0)?;
    let outlook = loss::binary_cross_entropy_with_logit(&output.move_outlook, outlook_targets)?;
    let total = ((&state * config.state_loss_weight)? + (&outlook * config.outlook_loss_weight)?)?;

    Ok(LossTensors {
        total,
        state,
        regime,
        quality,
        stage,
        outlook,
    })
}

fn categorical_step_loss(
    logits: &Tensor,
    state_targets: &Tensor,
    target_index: usize,
) -> Result<Tensor> {
    let (batch_size, sequence_length, class_count) = logits.dims3()?;
    let logits = logits.reshape((batch_size * sequence_length, class_count))?;
    let targets = state_targets
        .narrow(2, target_index, 1)?
        .reshape(batch_size * sequence_length)?;
    loss::cross_entropy(&logits, &targets)
}

#[derive(Default)]
struct MetricsAccumulator {
    total: f64,
    state: f64,
    regime: f64,
    quality: f64,
    stage: f64,
    outlook: f64,
    samples: usize,
}

impl MetricsAccumulator {
    fn add(&mut self, losses: &LossTensors, samples: usize) -> Result<()> {
        let weight = samples as f64;
        self.total += scalar(&losses.total)? * weight;
        self.state += scalar(&losses.state)? * weight;
        self.regime += scalar(&losses.regime)? * weight;
        self.quality += scalar(&losses.quality)? * weight;
        self.stage += scalar(&losses.stage)? * weight;
        self.outlook += scalar(&losses.outlook)? * weight;
        self.samples += samples;
        Ok(())
    }

    fn finish(self) -> Result<EpochMetrics> {
        if self.samples == 0 {
            candle_core::bail!("cannot calculate metrics for an empty split")
        }
        let samples = self.samples as f64;
        Ok(EpochMetrics {
            total: self.total / samples,
            state: self.state / samples,
            regime: self.regime / samples,
            quality: self.quality / samples,
            stage: self.stage / samples,
            outlook: self.outlook / samples,
        })
    }
}

fn scalar(tensor: &Tensor) -> Result<f64> {
    Ok(tensor.to_scalar::<f32>()? as f64)
}
