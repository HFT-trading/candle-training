use std::error::Error;

use candle_core::Device;
use candle_training::builder::TrainingTensorBuilder;
use candle_training::dataset::load_dataset;
use candle_training::trainer::TrainingDataSummary;

fn main() -> Result<(), Box<dyn Error>> {
    let device = training_device()?;
    println!("training device: {device:?}");

    let dataset = load_dataset(
        "datasets/feature_schema.json",
        "datasets/market_sequences.jsonl",
    )?;
    println!(
        "loaded schema v{} with {} sequences",
        dataset.schema.version,
        dataset.sequences.len()
    );

    let tensors = TrainingTensorBuilder::new(&dataset.schema).build(&dataset, &device)?;
    let summary = TrainingDataSummary::from_tensors(&tensors);

    println!(
        "categorical inputs: [{}, {}, {}]",
        summary.sequences, summary.sequence_length, summary.categorical_features
    );
    println!(
        "numeric inputs:     [{}, {}, {}]",
        summary.sequences, summary.sequence_length, summary.numeric_features
    );
    println!(
        "boolean targets:    [{}, {}]",
        summary.sequences, summary.boolean_targets
    );
    println!(
        "numeric targets:    [{}, {}]",
        summary.sequences, summary.numeric_targets
    );
    println!("training data is ready; model architecture comes next");

    Ok(())
}

fn training_device() -> candle_core::Result<Device> {
    #[cfg(target_os = "macos")]
    {
        Device::metal_if_available(0)
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(Device::Cpu)
    }
}
