use std::error::Error;
use std::fs::File;
use std::io::{Error as IoError, ErrorKind};

use bot_training::config::AppConfig;
use bot_training::dataset::{FeatureSchema, MarketSequence, load_dataset, load_schema};
use bot_training::logging;
use bot_training::runtime::{BufferedModelService, MarketStep, ModelRuntime, SequenceRequest};
use candle_core::{Device, IndexOp, Tensor};
use serde_json::Value;
use tracing::{debug, info};

fn main() -> Result<(), Box<dyn Error>> {
    logging::init();

    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let show_raw = arguments.iter().any(|argument| argument == "--raw");
    let stream = arguments.iter().any(|argument| argument == "--stream");
    let file_path = option_value(&arguments, "--file")?;
    let sequence_index = arguments
        .iter()
        .find(|argument| !argument.starts_with("--") && file_path.as_ref() != Some(*argument))
        .cloned()
        .unwrap_or_else(|| "0".to_owned())
        .parse::<usize>()?;
    let config = AppConfig::load()?;
    let device = Device::Cpu;

    let (schema, request, reference) = match file_path {
        Some(path) => {
            let request = serde_json::from_reader::<_, SequenceRequest>(File::open(path)?)?;
            (load_schema("datasets/feature_schema.json")?, request, None)
        }
        None => {
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
            let request = request_from_training_sequence(sequence);
            let reference = sequence
                .debug_semantics
                .per_step
                .last()
                .map(|step| step.values.clone());
            (dataset.schema, request, reference)
        }
    };

    let runtime = ModelRuntime::load(
        &config.model,
        &schema,
        &device,
        "models/model.safetensors",
        "models/numeric-normalizer.safetensors",
    )?;
    if show_raw {
        log_steps(&request);
    }
    if stream {
        let cycle_id = request
            .cycle_id
            .clone()
            .unwrap_or_else(|| "testing-cycle".to_owned());
        let mut service = BufferedModelService::new(runtime);
        service.start_cycle(cycle_id.clone())?;
        for step in request.steps.iter().cloned() {
            let prediction = service.push_step(&cycle_id, step)?;
            prediction
                .output
                .validate_shapes(1, prediction.observed_steps, &schema)?;
            log_prediction(
                &prediction.output,
                prediction.observed_steps,
                config.model.max_sequence_steps,
                &schema,
            )?;
        }
        service.end_cycle(&cycle_id)?;
    } else {
        let prediction = runtime.predict(&request)?;
        prediction
            .output
            .validate_shapes(1, prediction.observed_steps, &schema)?;
        log_prediction(
            &prediction.output,
            prediction.observed_steps,
            config.model.max_sequence_steps,
            &schema,
        )?;
    }
    if show_raw {
        if let Some(reference) = reference {
            info!(
                target: "REFERENCE",
                "STATE: regime {} | quality {} | stage {}",
                text_value(reference.get("current.regime")),
                text_value(reference.get("current.quality")),
                text_value(reference.get("cycle.stage")),
            );
        }
    }
    Ok(())
}

fn request_from_training_sequence(sequence: &MarketSequence) -> SequenceRequest {
    let steps = (0..sequence.seq_len)
        .map(|index| MarketStep {
            step_id: Some(sequence.state_features[index].snapshot_id.clone()),
            time: sequence.state_features[index].snapshot_time.clone(),
            state: sequence.state_features[index].values.clone(),
            range: sequence.range_telemetry[index].values.clone(),
        })
        .collect();
    SequenceRequest {
        cycle_id: Some(sequence.sample_id.clone()),
        steps,
    }
}

fn log_prediction(
    output: &bot_training::model::ModelOutput,
    observed_steps: usize,
    max_sequence_steps: usize,
    schema: &FeatureSchema,
) -> Result<(), Box<dyn Error>> {
    let last_step = observed_steps - 1;
    let regime = categorical_scores(
        output.state.regime.i((0, last_step, ..))?,
        known_labels(schema, "current.regime")?,
    )?;
    let quality = categorical_scores(
        output.state.quality.i((0, last_step, ..))?,
        known_labels(schema, "current.quality")?,
    )?;
    let stage = categorical_scores(
        output.state.stage.i((0, last_step, ..))?,
        known_labels(schema, "cycle.stage")?,
    )?;
    let outlook = independent_scores(
        output.move_outlook.i(0)?,
        schema.target_groups.boolean.clone(),
    )?;

    debug!(target: "MODEL", ?output, "raw tensors");
    info!(target: "MODEL", "CONTEXT: {observed_steps}/{max_sequence_steps} steps");
    info!(target: "MODEL", "REGIME: {}", format_scores(&regime));
    info!(target: "MODEL", "QUALITY: {}", format_scores(&quality));
    info!(target: "MODEL", "STAGE: {}", format_scores(&stage));
    info!(target: "MODEL", "OUTLOOK: {}", format_scores(&outlook));
    Ok(())
}

fn log_steps(request: &SequenceRequest) {
    info!(
        target: "REPORT",
        "CYCLE: {} | steps: {}",
        request.cycle_id.as_deref().unwrap_or("testing-input"),
        request.steps.len()
    );
    for (index, step) in request.steps.iter().enumerate() {
        info!(
            target: "REPORT",
            "{}/{} | {} | {} | {} | conf: {:.2} | time: {:.2} | net_bps: {:.2} | range_bps: {:.2} | favorable_bps: {:.2} | adverse_bps: {:.2} | opposite_bps: {:.2} | retention: {:.2}",
            index + 1,
            request.steps.len(),
            text_value(step.state.get("current.direction_hint")),
            text_value(step.state.get("current.vector_hint")),
            text_value(step.state.get("current.quality_hint")),
            number_value(step.state.get("current.confidence")),
            number_value(step.range.get("current.duration_sec")),
            number_value(step.range.get("current.net_bps")),
            number_value(step.range.get("current.range_bps")),
            number_value(step.range.get("current.favorable_bps")),
            number_value(step.range.get("current.adverse_bps")),
            number_value(step.range.get("current.opposite_bps")),
            number_value(step.range.get("current.retention")),
        );
    }
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

fn format_scores(scores: &[(String, f32)]) -> String {
    scores
        .iter()
        .map(|(label, probability)| format!("{label} {:.1}%", probability * 100.0))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn option_value(arguments: &[String], name: &str) -> Result<Option<String>, Box<dyn Error>> {
    let Some(position) = arguments.iter().position(|argument| argument == name) else {
        return Ok(None);
    };
    arguments
        .get(position + 1)
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidInput,
                format!("{name} requires a file path"),
            )
            .into()
        })
}

fn text_value(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("-")
}

fn number_value(value: Option<&Value>) -> f64 {
    value.and_then(Value::as_f64).unwrap_or(f64::NAN)
}
