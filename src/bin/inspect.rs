use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use bot_training::builder::TrainingTensorBuilder;
use bot_training::config::AppConfig;
use bot_training::core::ModelInputs;
use bot_training::dataset::{FeatureSchema, load_dataset};
use bot_training::logging;
use bot_training::runtime::ModelRuntime;
use candle_core::{Device, IndexOp, Tensor};
use tracing::{debug, info};

fn main() -> Result<(), Box<dyn Error>> {
    logging::init();

    let sequence_index = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0".to_owned())
        .parse::<usize>()?;
    let config = AppConfig::load()?;
    let device = Device::Cpu;
    let dataset = load_dataset(
        "datasets/feature_schema.json",
        "datasets/market_sequences.jsonl",
    )?;
    let sequence = dataset.sequences.get(sequence_index).ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!(
                "sequence index {sequence_index} is out of range 0..{}",
                dataset.sequences.len()
            ),
        )
    })?;
    let tensors = TrainingTensorBuilder::new(&dataset.schema).build(&dataset, &device)?;
    let inputs = ModelInputs {
        categorical: tensors.inputs.categorical.narrow(0, sequence_index, 1)?,
        numeric: tensors.inputs.numeric.narrow(0, sequence_index, 1)?,
    };
    let runtime = ModelRuntime::load(
        &config.model,
        &dataset.schema,
        &device,
        "models/model.safetensors",
        "models/numeric-normalizer.safetensors",
    )?;
    let output = runtime.forward(&inputs)?;
    output.validate_shapes(1, sequence.seq_len, &dataset.schema)?;

    let observed = &sequence
        .debug_semantics
        .per_step
        .last()
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "sequence has no debug state"))?
        .values;
    info!(
        sequence_index,
        sample_id = sequence.sample_id,
        source = sequence.source_name,
        steps = sequence.seq_len,
        observed_regime = ?observed.get("current.regime"),
        observed_quality = ?observed.get("current.quality"),
        observed_stage = ?observed.get("cycle.stage"),
        "sequence loaded for inspection"
    );

    let last_step = sequence.seq_len - 1;
    let regime = categorical_scores(
        output.state.regime.i((0, last_step, ..))?,
        known_labels(&dataset.schema, "current.regime")?,
    )?;
    let quality = categorical_scores(
        output.state.quality.i((0, last_step, ..))?,
        known_labels(&dataset.schema, "current.quality")?,
    )?;
    let stage = categorical_scores(
        output.state.stage.i((0, last_step, ..))?,
        known_labels(&dataset.schema, "cycle.stage")?,
    )?;
    let outlook = independent_scores(
        output.move_outlook.i(0)?,
        dataset.schema.target_groups.boolean.clone(),
    )?;

    debug!(?output, "raw model output");
    info!(
        ?regime,
        ?quality,
        ?stage,
        ?outlook,
        "model inspection completed"
    );
    Ok(())
}

fn known_labels(schema: &FeatureSchema, target: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let vocab = schema
        .debug_semantics
        .categorical_vocab
        .get(target)
        .ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidData,
                format!("missing state vocabulary for {target}"),
            )
        })?;
    let mut labels = vocab
        .iter()
        .filter(|(label, _)| label.as_str() != "__UNK__")
        .map(|(label, id)| (*id, label.clone()))
        .collect::<Vec<_>>();
    labels.sort_by_key(|(id, _)| *id);
    Ok(labels.into_iter().map(|(_, label)| label).collect())
}

fn categorical_scores(
    logits: Tensor,
    labels: Vec<String>,
) -> Result<Vec<(String, f32)>, Box<dyn Error>> {
    let probabilities = candle_nn::ops::softmax(&logits, 0)?.to_vec1::<f32>()?;
    Ok(labels.into_iter().zip(probabilities).collect())
}

fn independent_scores(
    logits: Tensor,
    labels: Vec<String>,
) -> Result<Vec<(String, f32)>, Box<dyn Error>> {
    let probabilities = candle_nn::ops::sigmoid(&logits)?.to_vec1::<f32>()?;
    Ok(labels.into_iter().zip(probabilities).collect())
}
