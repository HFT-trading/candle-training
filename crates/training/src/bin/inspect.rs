//! Serving test harness: load the model via `structure-core` (self-contained),
//! stream one context's steps through a Session, print the final StructureReport.
//! Usage: `inspect [context_index]`. Exercises the real serving path (StepInput).

use candle_core::Device;
use structure_core::input::{StepFeatures, StepInput};
use structure_core::serve::StructureModel;
use training::dataset::{Sequence, load_contexts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let index: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(0);

    let device = Device::Cpu;
    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let context = contexts
        .get(index)
        .ok_or_else(|| format!("context index {index} out of range (have {})", contexts.len()))?;

    let model = StructureModel::load("artifacts", &device)?;
    let mut session = model.session();

    let mut report = None;
    for sequence in &context.training_data.sequences {
        if let Some(read) = session.push(to_step(sequence))? {
            report = Some(read);
        }
    }
    let report = report.ok_or("no report produced (need at least one full block)")?;

    println!("context #{index}  source={}", context.metadata.source_name);
    println!("{report:#?}");

    Ok(())
}

/// Convert a parsed dataset sequence into the plain serving input.
fn to_step(sequence: &Sequence) -> StepInput {
    let pattern_name = match sequence.categorical("pattern.name") {
        "None" => None,
        other => Some(other.to_owned()),
    };
    StepInput {
        micro_trend: sequence.categorical("micro_trend").to_owned(),
        direction_hint: sequence.categorical("direction_hint").to_owned(),
        vector_hint: sequence.categorical("vector_hint").to_owned(),
        bias_hint: sequence.categorical("bias_hint").to_owned(),
        quality_hint: sequence.categorical("quality_hint").to_owned(),
        behavior_hint: sequence.categorical("behavior_hint").to_owned(),
        side: sequence.categorical("side").to_owned(),
        pattern_name,
        duration_sec: sequence.numeric("duration_sec") as f64,
        net_bps: sequence.numeric("net_bps") as f64,
        abs_net_bps: sequence.numeric("abs_net_bps") as f64,
        favorable_bps: sequence.numeric("favorable_bps") as f64,
        adverse_bps: sequence.numeric("adverse_bps") as f64,
        opposite_bps: sequence.numeric("opposite_bps") as f64,
        retention: sequence.numeric("retention") as f64,
        confidence: sequence.numeric("confidence") as f64,
        pattern_confidence: sequence.numeric("pattern.confidence") as f64,
        pattern_length: sequence.numeric("pattern.length") as f64,
    }
}
