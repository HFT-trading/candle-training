use std::collections::HashMap;
use std::error::Error;
use std::fs;

use bot_training::builder::TrainingTensorBuilder;
use bot_training::config::AppConfig;
use bot_training::dataset::load_dataset;
use bot_training::logging;
use bot_training::model::MarketStateModel;
use bot_training::trainer::{DataSplit, TrainingDataSummary, train};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use tracing::info;

fn main() -> Result<(), Box<dyn Error>> {
    logging::init();

    let config = AppConfig::load()?;
    info!(?config, "training config loaded");

    let device = training_device()?;
    info!(?device, "training device selected");

    let dataset = load_dataset(
        "datasets/feature_schema.json",
        "datasets/market_sequences.jsonl",
    )?;
    info!(
        schema_version = dataset.schema.version,
        sequences = dataset.sequences.len(),
        "dataset loaded"
    );

    let tensors = TrainingTensorBuilder::new(&dataset.schema).build(&dataset, &device)?;
    let summary = TrainingDataSummary::from_tensors(&tensors);
    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let model = MarketStateModel::new(&config.model, &dataset.schema, vb)?;
    let split = DataSplit::by_source(&dataset, config.training.validation_fraction)?;

    info!(
        sequences = summary.sequences,
        sequence_length = summary.sequence_length,
        features = summary.categorical_features,
        "categorical input tensor ready"
    );
    info!(
        sequences = summary.sequences,
        sequence_length = summary.sequence_length,
        features = summary.numeric_features,
        "numeric input tensor ready"
    );
    info!(
        sequences = summary.sequences,
        sequence_length = summary.sequence_length,
        targets = summary.state_targets,
        "state target tensor ready"
    );
    info!(
        sequences = summary.sequences,
        targets = summary.categorical_targets,
        "categorical outcome target tensor ready"
    );
    info!(
        sequences = summary.sequences,
        targets = summary.boolean_targets,
        "move-outlook target tensor ready"
    );
    info!(
        sequences = summary.sequences,
        metadata_fields = summary.numeric_targets,
        "numeric analysis metadata tensor ready"
    );
    info!(
        train_sequences = split.train_indices.len(),
        validation_sequences = split.validation_indices.len(),
        validation_source = split.validation_source,
        "source-level dataset split ready"
    );

    let report = train(&model, &variables, &tensors, &split, &config.training)?;
    fs::create_dir_all("models")?;
    variables.save("models/model.safetensors")?;
    candle_core::safetensors::save(
        &HashMap::from([
            ("mean", report.normalizer.mean.clone()),
            ("std", report.normalizer.std.clone()),
        ]),
        "models/numeric-normalizer.safetensors",
    )?;
    info!(
        epochs = report.epochs.len(),
        numeric_features = report.normalizer.mean.elem_count(),
        model_path = "models/model.safetensors",
        normalizer_path = "models/numeric-normalizer.safetensors",
        "training completed and artifacts saved"
    );

    Ok(())
}

fn training_device() -> candle_core::Result<Device> {
    if std::env::var("TRAIN_DEVICE").as_deref() == Ok("cpu") {
        return Ok(Device::Cpu);
    }

    #[cfg(target_os = "macos")]
    {
        Device::metal_if_available(0)
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(Device::Cpu)
    }
}
