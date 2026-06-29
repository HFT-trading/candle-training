//! Serving test harness: load the exported model via `structure-core` and print
//! the StructureReport for one dataset context. Usage: `inspect [context_index]`.

use candle_core::Device;
use structure_core::serve::StructureModel;
use training::builder::build_tensors;
use training::config::AppConfig;
use training::dataset::{build_vocab, load_contexts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(0);

    let device = Device::Cpu;
    let config = AppConfig::load()?;
    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let block_size = contexts[0].metadata.shape.block_size;

    let context = contexts
        .get(index)
        .ok_or_else(|| format!("context index {index} out of range (have {})", contexts.len()))?;

    // Encode just this one context (vocab is deterministic from the dataset).
    let vocab = build_vocab(&contexts);
    let tensors = build_tensors(std::slice::from_ref(context), &vocab, &device)?;

    let model = StructureModel::load("artifacts", &config.model, block_size, &device)?;
    let report = model.read(&tensors.inputs)?;

    println!(
        "context #{index}  source={}  time {}..{}",
        context.metadata.source_name,
        context_time(context).0,
        context_time(context).1,
    );
    println!("{report:#?}");

    Ok(())
}

fn context_time(context: &training::dataset::Context) -> (String, String) {
    let sequences = &context.training_data.sequences;
    let first = sequences
        .first()
        .and_then(|s| s.metadata.get("timestamp"))
        .and_then(|v| v.as_str())
        .unwrap_or("-")
        .to_owned();
    let last = sequences
        .last()
        .and_then(|s| s.metadata.get("timestamp"))
        .and_then(|v| v.as_str())
        .unwrap_or("-")
        .to_owned();
    (first, last)
}
