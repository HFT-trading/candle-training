//! Pattern ablation: does the model actually READ the `pattern` feature?
//! Stream each context twice — once with the real pattern, once with pattern
//! wiped (name=None, confidence=0, length=0) — and count how often the read
//! changes. Only contexts that HAD a pattern are counted (wiping None is a no-op).

use candle_core::Device;
use structure_core::input::{StepFeatures, StepInput};
use structure_core::sequence::BLOCK_LABEL_FIELDS;
use structure_core::serve::StructureModel;
use training::dataset::{Sequence, load_contexts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let model = StructureModel::load("artifacts", &device)?;

    let mut with_pattern = 0usize; // contexts that actually carry a pattern
    let mut phase_changed = 0usize;
    let mut any_changed = 0usize;

    for context in &contexts {
        let real: Vec<StepInput> = context
            .training_data
            .sequences
            .iter()
            .map(to_step)
            .collect();
        let has_pattern = real.iter().any(|s| s.pattern_name.is_some());
        if !has_pattern {
            continue;
        }
        with_pattern += 1;

        let wiped: Vec<StepInput> = real
            .iter()
            .map(|s| StepInput {
                pattern_name: None,
                pattern_confidence: 0.0,
                pattern_length: 0.0,
                ..s.clone()
            })
            .collect();

        let a = model.read(&model.encode(&real)?)?;
        let b = model.read(&model.encode(&wiped)?)?;

        if a.phase.label != b.phase.label {
            phase_changed += 1;
        }
        if BLOCK_LABEL_FIELDS.iter().any(|field| {
            a.prediction(field).map(|prediction| &prediction.label)
                != b.prediction(field).map(|prediction| &prediction.label)
        }) {
            any_changed += 1;
        }
    }

    println!("contexts carrying a pattern: {with_pattern}");
    println!(
        "phase changed when pattern wiped : {phase_changed}/{with_pattern} = {:.1}%",
        pct(phase_changed, with_pattern)
    );
    println!(
        "ANY trained head changed: {any_changed}/{with_pattern} = {:.1}%",
        pct(any_changed, with_pattern)
    );
    println!("\nreading: ~0% => model ignores pattern (weight ~0); higher => it reads it.");
    Ok(())
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64 * 100.0
    }
}

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
