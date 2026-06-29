//! StructureReport: the product-facing read, mapped deterministically from the
//! latest block's predicted labels (head A) + the last block-to-block relation.
//! No model here — this is the rules layer the rule doc describes.

use std::collections::HashMap;

use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct StructureReport {
    pub trend_bias: String,
    pub risk_level: String,
    pub dirty_warning: bool,
    pub reversal_warning: bool,
    pub range_frame_tag: String,
    /// entry_support-style location read: Good / Watch / Bad.
    pub location_quality: String,
    pub structure_tags: Vec<String>,
    pub reason_tags: Vec<String>,
}

/// Map the latest block's labels (+ last relation) into a report.
pub fn build_report(labels: &HashMap<String, String>, relation: &str) -> StructureReport {
    let get = |key: &str| labels.get(key).map(String::as_str).unwrap_or("-");
    let direction = get("direction");
    let reversal = get("reversal_risk");
    let path = get("path_quality");
    let range = get("range_frame_tag");

    let dirty = matches!(path, "Dirty" | "Invalidating");
    let reversal_warning = reversal == "High" || relation.starts_with("Reversal");

    let mut structure_tags = Vec::new();
    if range == "Adapt" {
        structure_tags.push("AdaptRange".to_owned());
    }
    if dirty {
        structure_tags.push("DirtyPath".to_owned());
    }
    if relation.starts_with("Continuation") {
        structure_tags.push("Continuation".to_owned());
    }
    if relation.starts_with("Reversal") {
        structure_tags.push("Reversal".to_owned());
    }
    if relation.starts_with("Compression") || relation.starts_with("BreakoutAttempt") {
        structure_tags.push("Compression".to_owned());
    }

    let mut reason_tags = Vec::new();
    if reversal == "High" {
        reason_tags.push("HighReversalRisk".to_owned());
    }
    if dirty {
        reason_tags.push("DirtyPath".to_owned());
    }
    if range == "Follow" {
        reason_tags.push("ThinRange".to_owned());
    }

    let location_quality = if dirty || reversal == "High" {
        "Bad"
    } else if reversal == "Low" && range != "Follow" {
        "Good"
    } else {
        "Watch"
    };

    StructureReport {
        trend_bias: direction.to_owned(),
        risk_level: reversal.to_owned(),
        dirty_warning: dirty,
        reversal_warning,
        range_frame_tag: range.to_owned(),
        location_quality: location_quality.to_owned(),
        structure_tags,
        reason_tags,
    }
}
