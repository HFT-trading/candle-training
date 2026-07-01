//! StructureReport: the product-facing read, mapped deterministically from the
//! latest block's predicted labels (head A) + regressed trend_pressure. The
//! block-to-block transition (continuation / reversal / range expansion) is
//! derived HERE from the last two block reads — there is no relation head; it
//! was redundant with the per-block direction + pressure. No model here: this is
//! the rules layer the rule doc describes.

use std::collections::HashMap;

use serde::Serialize;

/// The minimal read of a block needed to derive its transition to the next one.
pub struct BlockRead {
    pub direction: String,
    pub range_rank: String,
}

/// Block processes whose control is resolved (path_state = "clear").
fn is_clear(process: &str) -> bool {
    matches!(process, "CleanDrive" | "PullbackHeld" | "Reclaim")
}

#[derive(Debug, Serialize)]
pub struct StructureReport {
    pub trend_bias: String,
    pub risk_level: String,
    pub dirty_warning: bool,
    pub reversal_warning: bool,
    pub range_frame_tag: String,
    /// Intra-block shape (CleanDrive / PullbackHeld / Reclaim / FailedPush /
    /// Absorption / DirtyRotation / BalancedAuction).
    pub block_process: String,
    /// Coarse resolution read derived from block_process: "clear" / "unclear".
    pub path_state: String,
    /// Strong-but-stuck (block_process == Absorption).
    pub absorption: bool,
    /// entry_support-style location read: Good / Watch / Bad.
    pub location_quality: String,
    pub structure_tags: Vec<String>,
    pub reason_tags: Vec<String>,
}

/// Narrow -> VeryWide as an ordinal so adjacent blocks' range can be compared.
fn range_ordinal(rank: &str) -> i32 {
    match rank {
        "Narrow" => 0,
        "Medium" => 1,
        "Wide" => 2,
        "VeryWide" => 3,
        _ => -1,
    }
}

/// Map the latest block's labels + regressed pressure (+ the previous block, when
/// the window has one) into a report.
pub fn build_report(
    labels: &HashMap<String, String>,
    previous: Option<&BlockRead>,
) -> StructureReport {
    let get = |key: &str| labels.get(key).map(String::as_str).unwrap_or("-");
    let direction = get("direction");
    let reversal = get("reversal_risk");
    let process = get("block_process");
    let range = get("range_frame_tag");
    let range_rank = get("range_rank");
    let absorption = process == "Absorption";
    let path_state = if is_clear(process) { "clear" } else { "unclear" };

    // "Dirty" now comes from the process shape rather than a separate label.
    let dirty = matches!(process, "DirtyRotation" | "FailedPush");

    // Derive the transition from the previous block's read.
    let mut reversal_transition = false;
    let mut continuation = false;
    let mut expansion = false;
    let mut compression = false;
    if let Some(prev) = previous {
        reversal_transition = matches!(
            (prev.direction.as_str(), direction),
            ("Up", "Down") | ("Down", "Up")
        );
        continuation = direction != "Flat" && prev.direction == direction;
        let (cur_ord, prev_ord) = (range_ordinal(range_rank), range_ordinal(&prev.range_rank));
        if cur_ord >= 0 && prev_ord >= 0 {
            expansion = cur_ord > prev_ord;
            compression = cur_ord < prev_ord;
        }
    }

    let reversal_warning = reversal == "High" || reversal_transition;

    let mut structure_tags = Vec::new();
    if process != "-" {
        structure_tags.push(process.to_owned());
    }
    if range == "Adapt" {
        structure_tags.push("AdaptRange".to_owned());
    }
    if dirty {
        structure_tags.push("DirtyPath".to_owned());
    }
    if continuation {
        structure_tags.push("Continuation".to_owned());
    }
    if reversal_transition {
        structure_tags.push("Reversal".to_owned());
    }
    if expansion {
        structure_tags.push("Expansion".to_owned());
    }
    if compression {
        structure_tags.push("Compression".to_owned());
    }

    let mut reason_tags = Vec::new();
    if reversal == "High" {
        reason_tags.push("HighReversalRisk".to_owned());
    }
    if dirty {
        reason_tags.push("DirtyPath".to_owned());
    }
    if absorption {
        reason_tags.push("EffortAbsorbed".to_owned());
    }
    if range == "Follow" {
        reason_tags.push("ThinRange".to_owned());
    }

    // Absorption = energy without resolution -> not a clean location even if the
    // categorical risk reads low.
    let location_quality = if dirty || reversal == "High" || absorption {
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
        block_process: process.to_owned(),
        path_state: path_state.to_owned(),
        absorption,
        location_quality: location_quality.to_owned(),
        structure_tags,
        reason_tags,
    }
}
