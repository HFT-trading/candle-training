//! Accuracy review harness: stream a held-out source through the real serving
//! path and print, per block, the MODEL's read next to the TRUE label and the
//! raw price shape — so a human can eyeball whether the read matches reality.
//! Usage: `review [source_name] [max_contexts]` (default data-test7.log, 8).

use candle_core::Device;
use serde_json::Value;
use structure_core::input::{StepFeatures, StepInput};
use structure_core::serve::StructureModel;
use training::dataset::{Context, Sequence, load_contexts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = std::env::args().nth(1).unwrap_or_else(|| "data-test7.log".to_owned());
    let max_contexts: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(8);

    let device = Device::Cpu;
    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let model = StructureModel::load("artifacts", &device)?;
    let block_size = contexts[0].metadata.shape.block_size;

    let picked: Vec<&Context> = contexts
        .iter()
        .filter(|c| c.metadata.source_name == source)
        .collect();
    if picked.is_empty() {
        return Err(format!("no contexts for source {source:?}").into());
    }
    let stride = (picked.len() / max_contexts).max(1);

    println!("source={source}  contexts={}  showing every {stride}th\n", picked.len());
    let mut hits = 0usize;
    let mut seen = 0usize;
    for (sample, context) in picked.iter().step_by(stride).take(max_contexts).enumerate() {
        let mut session = model.session();
        let mut reports = Vec::new();
        for sequence in &context.training_data.sequences {
            if let Some(report) = session.push(to_step(sequence))? {
                reports.push(report);
            }
        }
        println!("── sample {} ──", sample + 1);
        for (block_index, report) in reports.iter().enumerate() {
            let start = block_index * block_size;
            let seqs = &context.training_data.sequences[start..start + block_size];
            // Use the price path (what the label's close-to-close direction saw),
            // not summed per-sequence net_bps (a different quantity).
            let prices: Vec<f64> = seqs.iter().map(|s| s.numeric("price") as f64).collect();
            let base = prices[0].max(1e-9);
            let path: Vec<f64> = prices.iter().map(|p| (p - base) / base * 10_000.0).collect();
            let net = *path.last().unwrap();
            let truth = &context.labels.blocks[block_index];
            let true_dir = label(truth, "direction");
            let true_proc = label(truth, "phase");
            let dir_ok = report.trend_bias == true_dir;
            let proc_ok = report.phase == true_proc;
            seen += 2;
            hits += dir_ok as usize + proc_ok as usize;
            println!(
                "  blk{}  net={:+6.1}  {}  | MODEL {:>4}/{:<14} | TRUE {:>4}/{:<14} {}{}",
                block_index + 1,
                net,
                sparkline(&path),
                report.trend_bias,
                report.phase,
                true_dir,
                true_proc,
                if dir_ok { "" } else { "  dir✗" },
                if proc_ok { "" } else { "  proc✗" },
            );
        }
        println!();
    }
    println!(
        "eyeball tally: {hits}/{seen} (direction+process) matched the deterministic label \
         ({:.0}%). NOTE: this is model-vs-label; the real check is whether the SHAPE column \
         matches the read.",
        hits as f64 / seen.max(1) as f64 * 100.0
    );
    Ok(())
}

fn label(map: &serde_json::Map<String, Value>, key: &str) -> String {
    map.get(key).and_then(|v| v.as_str()).unwrap_or("-").to_owned()
}

/// Price path (bps vs block open) as a unicode sparkline (▁..█).
fn sparkline(path: &[f64]) -> String {
    let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let lo = path.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
    let hi = path.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0);
    let span = (hi - lo).max(1e-9);
    path.iter()
        .map(|v| bars[(((v - lo) / span) * (bars.len() - 1) as f64).round() as usize])
        .collect()
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
