//! Sanity check the dataset and the built vocabulary: row count, shape
//! consistency (32 / 4 / 3), and per-label class distributions. This is the
//! Rust mirror of the earlier Python scan — used to prove the data layer.

use training::dataset::{build_vocab, load_contexts};
use structure_core::sequence::{BLOCK_LABEL_FIELDS, RELATION_LABEL_FIELDS};
use structure_core::vocab::FeatureVocab;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "datasets/market_contexts.jsonl".to_owned());

    let contexts = load_contexts(&path)?;
    println!("rows: {}", contexts.len());

    let mut shape_mismatches = 0usize;
    for context in &contexts {
        let shape = &context.metadata.shape;
        let expected_relations = shape.context_blocks.saturating_sub(1);
        if context.training_data.sequences.len() != shape.sequence_len
            || context.labels.blocks.len() != shape.context_blocks
            || context.labels.relations.len() != expected_relations
        {
            shape_mismatches += 1;
        }
    }
    println!("shape mismatches: {shape_mismatches}");

    let vocab = build_vocab(&contexts);

    println!("\n=== block labels (x4 / row) ===");
    for field in BLOCK_LABEL_FIELDS {
        print_field(&vocab, &format!("blocks.{field}"));
    }
    println!("\n=== relation labels (x3 / row) ===");
    for field in RELATION_LABEL_FIELDS {
        print_field(&vocab, &format!("relations.{field}"));
    }

    Ok(())
}

fn print_field(vocab: &FeatureVocab, key: &str) {
    let Some(entry) = vocab.get(key) else {
        println!("{key}: <missing>");
        return;
    };
    let total: u64 = entry.counts.values().sum();
    let mut items: Vec<(&String, &u64)> = entry.counts.iter().collect();
    items.sort_by(|left, right| right.1.cmp(left.1));
    let rendered = items
        .iter()
        .map(|(value, count)| {
            let pct = **count as f64 / total as f64 * 100.0;
            format!("{value}:{count}({pct:.1}%)")
        })
        .collect::<Vec<_>>()
        .join(" | ");
    println!("{key} [{} classes]: {rendered}", entry.class_count());
}
