//! Train head A end to end and export artifacts for serving.

use std::error::Error;
use std::fs;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use structure_core::model::MarketStructureModel;
use structure_core::serve::ServeMeta;
use training::builder::build_tensors;
use training::config::AppConfig;
use training::dataset::{build_vocab, load_contexts};
use structure_core::vocab::FeatureVocab;
use training::trainer::{ClassWeights, DataSplit, EvalReport, train};

fn main() -> Result<(), Box<dyn Error>> {
    let config = AppConfig::load()?;
    let device = Device::Cpu;

    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let block_size = contexts[0].metadata.shape.block_size;
    let context_blocks = contexts[0].metadata.shape.context_blocks;
    let vocab = build_vocab(&contexts);
    let tensors = build_tensors(&contexts, &vocab, &device)?;
    // Optional CLI arg forces a specific validation source (robustness checks);
    // otherwise auto-pick the source closest to validation_fraction.
    let split = match std::env::args().nth(1) {
        Some(source) => DataSplit::with_source(&contexts, &source)?,
        None => DataSplit::by_source(&contexts, config.training.validation_fraction)?,
    };
    println!(
        "contexts={} train={} val={} (val source={})",
        contexts.len(),
        split.train.len(),
        split.validation.len(),
        split.validation_source,
    );

    let weights = ClassWeights::from_config(&vocab, &config.training, &device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MarketStructureModel::new(&config.model, &vocab, block_size, vb)?;

    // Created before training so the best-checkpoint save inside `train` works.
    fs::create_dir_all("artifacts")?;
    let report = train(
        &model,
        &varmap,
        &tensors,
        &split,
        &config.training,
        Some(&weights),
        true,
    )?;

    println!(
        "\nbest epoch = {} (val total {:.4})",
        report.best_epoch, report.best_val
    );
    print_eval(&report.best_eval);
    print_block_process_recall(&report.best_eval, &vocab);

    report
        .normalizer
        .save("artifacts/numeric-normalizer.safetensors")?;
    fs::write("artifacts/vocab.json", serde_json::to_string_pretty(&vocab)?)?;
    let meta = ServeMeta {
        model: config.model.clone(),
        block_size,
        context_blocks,
    };
    fs::write("artifacts/meta.json", serde_json::to_string_pretty(&meta)?)?;
    println!(
        "saved artifacts/{{model.safetensors (best epoch {}), numeric-normalizer.safetensors, vocab.json, meta.json}}",
        report.best_epoch
    );

    Ok(())
}

fn print_eval(report: &EvalReport) {
    println!("\n=== validation: acc / macro-recall vs majority baseline ===");
    for field in &report.block {
        println!(
            "  block.{:<16} acc={:.3} macro={:.3} base={:.3} ({:+.3})",
            field.field,
            field.accuracy,
            field.macro_recall,
            field.baseline,
            field.accuracy - field.baseline,
        );
    }
}

/// The gate: per-class recall for block_process — do the rare shapes get caught,
/// or does the model collapse to the two baselines?
fn print_block_process_recall(report: &EvalReport, vocab: &FeatureVocab) {
    let Some(field) = report.block.iter().find(|f| f.field == "block_process") else {
        return;
    };
    let name = |id: u32| {
        vocab
            .label("blocks.block_process", id as usize)
            .unwrap_or("?")
            .to_owned()
    };
    println!("\n=== GATE: block_process per-class recall (macro={:.3}) ===", field.macro_recall);
    for (class, recall, support) in &field.class_recall {
        println!("  {:<16} recall={recall:.3}  (n={support})", name(*class));
    }
    println!("\n--- confusion: true -> top predictions ---");
    for (truth, _, support) in &field.class_recall {
        let mut preds: Vec<(u32, usize)> = field
            .confusion
            .iter()
            .filter(|(t, _, _)| t == truth)
            .map(|(_, p, c)| (*p, *c))
            .collect();
        preds.sort_by(|a, b| b.1.cmp(&a.1));
        let shown: Vec<String> = preds
            .iter()
            .take(3)
            .map(|(p, c)| format!("{}={} ({:.0}%)", name(*p), c, *c as f64 / *support as f64 * 100.0))
            .collect();
        println!("  {:<16} -> {}", name(*truth), shown.join("  "));
    }
}
