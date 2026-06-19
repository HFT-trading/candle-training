use std::error::Error;

use bot_training::builder::TrainingTensorBuilder;
use bot_training::dataset::load_dataset;
use bot_training::logging;
use bot_training::trainer::TrainingDataSummary;
use candle_core::Device;
use tracing::info;

fn main() -> Result<(), Box<dyn Error>> {
    logging::init();

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
    info!("training data is ready; model architecture comes next");

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
