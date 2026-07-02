//! Leave-one-source-out cross-validation: hold out each source in turn, train on
//! the rest, then average accuracy / macro-recall across folds. Robust as the
//! dataset grows (more sources = more folds).

use std::collections::BTreeSet;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use structure_core::model::MarketStructureModel;
use structure_core::sequence::BLOCK_LABEL_FIELDS;
use training::builder::build_tensors;
use training::config::AppConfig;
use training::dataset::{build_vocab, load_contexts};
use training::trainer::{ClassWeights, DataSplit, EvalReport, FieldAccuracy, train};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::load()?;
    let device = Device::Cpu;

    let contexts = load_contexts("datasets/market_contexts.jsonl")?;
    let block_size = contexts[0].metadata.shape.block_size;
    let vocab = build_vocab(&contexts);
    let tensors = build_tensors(&contexts, &vocab, &device)?;
    let weights = ClassWeights::from_config(&vocab, &config.training, &device)?;

    let sources: BTreeSet<String> = contexts
        .iter()
        .map(|context| context.metadata.source_name.clone())
        .collect();
    println!(
        "leave-one-source-out CV over {} sources (class_weights={})\n",
        sources.len(),
        config.training.use_class_weights,
    );

    let mut folds: Vec<(String, f64, EvalReport)> = Vec::new();
    for source in &sources {
        let split = DataSplit::with_source(&contexts, source)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MarketStructureModel::new(&config.model, &vocab, block_size, vb)?;
        let report = train(
            &model,
            &varmap,
            &tensors,
            &split,
            &config.training,
            Some(&weights),
            false,
            false, // measurement only — never clobber artifacts/model.safetensors
        )?;
        println!(
            "fold val={:<16} train={:>4} val={:>4} | best epoch {:>2} val {:.4}",
            source,
            split.train.len(),
            split.validation.len(),
            report.best_epoch,
            report.best_val,
        );
        folds.push((source.clone(), report.best_val, report.best_eval));
    }

    println!("\n=== mean acc / macro across {} folds (±std) ===", folds.len());
    aggregate("block", &BLOCK_LABEL_FIELDS, &folds, |report| &report.block);
    let mean_val = mean(&folds.iter().map(|(_, value, _)| *value).collect::<Vec<_>>());
    println!("\nmean best_val = {mean_val:.4}");

    Ok(())
}

fn aggregate(
    label: &str,
    fields: &[&str],
    folds: &[(String, f64, EvalReport)],
    pick: impl Fn(&EvalReport) -> &Vec<FieldAccuracy>,
) {
    for (index, name) in fields.iter().enumerate() {
        let accuracies: Vec<f64> = folds.iter().map(|(_, _, r)| pick(r)[index].accuracy).collect();
        let macros: Vec<f64> = folds
            .iter()
            .map(|(_, _, r)| pick(r)[index].macro_recall)
            .collect();
        let baselines: Vec<f64> = folds.iter().map(|(_, _, r)| pick(r)[index].baseline).collect();
        println!(
            "  {label}.{name:<16} acc={:.3}±{:.3} macro={:.3}±{:.3} base={:.3}",
            mean(&accuracies),
            std(&accuracies),
            mean(&macros),
            std(&macros),
            mean(&baselines),
        );
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn std(values: &[f64]) -> f64 {
    let mean = mean(values);
    (values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len().max(1) as f64)
        .sqrt()
}
