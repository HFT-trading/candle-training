//! Train head A end to end and export artifacts for serving.

use std::error::Error;
use std::fs;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use structure_core::model::MarketStructureModel;
use training::builder::build_tensors;
use training::config::AppConfig;
use training::dataset::{build_vocab, load_contexts};
use training::trainer::{ClassWeights, DataSplit, EvalReport, train};

fn main() -> Result<(), Box<dyn Error>> {
    let config = AppConfig::load()?;
    let device = Device::Cpu;

    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let block_size = contexts[0].metadata.shape.block_size;
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

    let weights = if config.training.use_class_weights {
        Some(ClassWeights::from_vocab(&vocab, &device)?)
    } else {
        None
    };

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
        weights.as_ref(),
        true,
    )?;

    println!(
        "\nbest epoch = {} (val total {:.4})",
        report.best_epoch, report.best_val
    );
    print_eval(&report.best_eval);

    report
        .normalizer
        .save("artifacts/numeric-normalizer.safetensors")?;
    fs::write("artifacts/vocab.json", serde_json::to_string_pretty(&vocab)?)?;
    println!(
        "saved artifacts/{{model.safetensors (best epoch {}), numeric-normalizer.safetensors, vocab.json}}",
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
    for field in &report.relation {
        println!(
            "  rel.{:<18} acc={:.3} macro={:.3} base={:.3} ({:+.3})",
            field.field,
            field.accuracy,
            field.macro_recall,
            field.baseline,
            field.accuracy - field.baseline,
        );
    }
}
