use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

use bot_training::builder::TrainingTensorBuilder;
use bot_training::config::AppConfig;
use bot_training::core::ModelInputs;
use bot_training::dataset::{FeatureSchema, MarketSequence, load_dataset};
use bot_training::logging;
use bot_training::runtime::ModelRuntime;
use candle_core::{Device, IndexOp, Tensor};
use tracing::{debug, info};

fn main() -> Result<(), Box<dyn Error>> {
    logging::init();

    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let show_raw = arguments.iter().any(|argument| argument == "--raw");
    let sequence_index = arguments
        .iter()
        .find(|argument| !argument.starts_with("--"))
        .cloned()
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

    if show_raw {
        log_sequence(sequence_index, sequence);
    }

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

    let (regime_label, regime_probability) = top_score(&regime)?;
    let (quality_label, quality_probability) = top_score(&quality)?;
    let (stage_label, stage_probability) = top_score(&stage)?;
    let outlook_line = outlook
        .iter()
        .map(|(label, probability)| format!("{label}: {:.1}%", probability * 100.0))
        .collect::<Vec<_>>()
        .join(" | ");

    debug!(target: "MODEL", ?output, "raw tensors");
    info!(
        target: "MODEL",
        "STATE: regime {regime_label} {:.1}% | quality {quality_label} {:.1}% | stage {stage_label} {:.1}%",
        regime_probability * 100.0,
        quality_probability * 100.0,
        stage_probability * 100.0,
    );
    info!(target: "MODEL", "OUTLOOK: {outlook_line}");
    if show_raw {
        let observed = &sequence
            .debug_semantics
            .per_step
            .last()
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "sequence has no debug state"))?
            .values;
        info!(
            target: "REFERENCE",
            "STATE: regime {} | quality {} | stage {}",
            text_value(observed.get("current.regime")),
            text_value(observed.get("current.quality")),
            text_value(observed.get("cycle.stage")),
        );
    }
    Ok(())
}

fn log_sequence(sequence_index: usize, sequence: &MarketSequence) {
    info!(
        target: "REPORT",
        "SEQUENCE #{sequence_index} | {} | source: {} | steps: {}",
        sequence.sample_id,
        sequence.source_name,
        sequence.seq_len
    );

    for step in 0..sequence.seq_len {
        let state = &sequence.state_features[step];
        let telemetry = &sequence.range_telemetry[step].values;
        let semantics = &sequence.debug_semantics.per_step[step].values;
        info!(
            target: "REPORT",
            "{}/{} | {} | {} | {} | conf: {:.2} | span: {} | time: {:.2} | net_bps: {:.2} | range_bps: {:.2} | favorable_bps: {:.2} | adverse_bps: {:.2} | opposite_bps: {:.2} | retention: {:.2} | side {}",
            step + 1,
            sequence.seq_len,
            text_value(state.values.get("current.direction_hint")),
            text_value(state.values.get("current.vector_hint")),
            text_value(state.values.get("current.quality_hint")),
            number_value(state.values.get("current.confidence")),
            state.snapshot_time.as_deref().unwrap_or("-"),
            number_value(telemetry.get("current.duration_sec")),
            number_value(telemetry.get("current.net_bps")),
            number_value(telemetry.get("current.range_bps")),
            number_value(telemetry.get("current.favorable_bps")),
            number_value(telemetry.get("current.adverse_bps")),
            number_value(telemetry.get("current.opposite_bps")),
            number_value(telemetry.get("current.retention")),
            text_value(semantics.get("current.side")),
        );
    }
}

fn text_value(value: Option<&serde_json::Value>) -> &str {
    value.and_then(serde_json::Value::as_str).unwrap_or("-")
}

fn number_value(value: Option<&serde_json::Value>) -> f64 {
    value
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(f64::NAN)
}

fn top_score(scores: &[(String, f32)]) -> Result<(&str, f32), Box<dyn Error>> {
    scores
        .iter()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(label, probability)| (label.as_str(), *probability))
        .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "model returned no scores").into())
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
