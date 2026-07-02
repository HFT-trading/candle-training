//! StructureReport: the product-facing read, mapped deterministically from the
//! latest block's predicted labels (head A). No model here — this is the rules
//! layer the rule doc describes.
//!
//! The lifecycle head `phase` (Running / Rejected / Stalling / Fading) is now the
//! engine of the read (it REPLACED the under-learned `block_process`). The three
//! product questions map off it plus the other reliable heads:
//!   1. Ai control?      -> `control` + `conviction`  (phase + direction + extension)
//!   2. Đang làm gì?     -> `phase` + `range_state`
//!   3. Tin được không?  -> `state_quality` + `usable` (phase + reversal_risk)
//!
//! The spec-doc contract fields (trend_bias / risk_level / warnings / tags /
//! location_quality) ride alongside, derived to stay coherent with `phase`.

use std::collections::HashMap;

use serde::Serialize;

/// The minimal read of a block needed to derive its transition to the next one.
pub struct BlockRead {
    pub direction: String,
    pub range_rank: String,
}

/// Buyers / Sellers from the block direction; `contested` when direction is Flat.
fn side(direction: &str, contested: &'static str) -> &'static str {
    match direction {
        "Up" => "Buyers",
        "Down" => "Sellers",
        _ => contested,
    }
}

/// Fallback lifecycle read from reliable heads, used only when the model does not
/// predict `phase` (e.g. an older artifact). Mirrors the exporter's phase rules.
fn phase_fallback(extension: &str, direction: &str, reversal: &str, range_rank: &str) -> &'static str {
    if matches!(extension, "MediumMove" | "LargeMove") && direction != "Flat" {
        return "Running";
    }
    if reversal == "High" {
        return "Rejected";
    }
    if matches!(range_rank, "Wide" | "VeryWide") {
        return "Stalling";
    }
    "Fading"
}

/// Question 1: who is in control, and how strongly, from the lifecycle phase.
fn control_of(phase: &str, direction: &str, extension: &str) -> (&'static str, &'static str) {
    match phase {
        "Running" => (
            side(direction, "Contested"),
            if extension == "LargeMove" { "Strong" } else { "Moderate" },
        ),
        "Fading" => ("Balanced", "Weak"),
        // Rejected / Stalling: effort spent, no clean winner.
        _ => ("Contested", "Weak"),
    }
}

/// Question 3: the shape mode, from the lifecycle phase.
fn state_quality_of(phase: &str) -> &'static str {
    match phase {
        "Running" => "Clean",
        "Rejected" => "Failed",
        "Stalling" => "Stuck",
        _ => "Indecisive", // Fading
    }
}

fn risk_rank(level: &str) -> i32 {
    match level {
        "High" => 2,
        "Medium" => 1,
        _ => 0,
    }
}

/// Derive risk from the phase floored against the model's reversal_risk head, so
/// risk stays coherent with the lifecycle read (a Rejected block can't read Low).
fn derive_risk(head: &str, phase: &str) -> &'static str {
    let floor = match phase {
        "Rejected" => 2, // pushed back = high risk
        "Stalling" => 1, // trapped effort = caution
        _ => 0,          // Running / Fading: let the head speak
    };
    match floor.max(risk_rank(head)) {
        2 => "High",
        1 => "Medium",
        _ => "Low",
    }
}

#[derive(Debug, Serialize)]
pub struct StructureReport {
    // --- spec-doc contract (derived to stay coherent with phase) ---
    pub trend_bias: String,
    pub risk_level: String,
    pub dirty_warning: bool,
    pub reversal_warning: bool,
    pub range_frame_tag: String,
    /// entry_support-style location read: Good / Watch / Bad.
    pub location_quality: String,
    pub structure_tags: Vec<String>,
    pub reason_tags: Vec<String>,

    // --- lifecycle: nhịp run/exhaust (learned head, engine của read) ---
    /// Running | Rejected | Stalling | Fading.
    pub phase: String,

    // --- 1. Ai control? ---
    /// Buyers | Sellers | Balanced | Contested.
    pub control: String,
    /// Strong | Moderate | Weak.
    pub conviction: String,

    /// Expanding | Compressing | Steady (vs the previous block's range).
    pub range_state: String,

    // --- 3. Tin được không? ---
    /// Clean | Failed | Stuck | Indecisive — shape mode from phase.
    pub state_quality: String,
    /// Act gate: Running phase and risk not High.
    pub usable: bool,
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

/// Map the latest block's labels (+ the previous block, when the window has one)
/// into a report.
pub fn build_report(
    labels: &HashMap<String, String>,
    previous: Option<&BlockRead>,
) -> StructureReport {
    let get = |key: &str| labels.get(key).map(String::as_str).unwrap_or("-");
    let direction = get("direction");
    let reversal_head = get("reversal_risk");
    let range = get("range_frame_tag");
    let range_rank = get("range_rank");
    let extension = get("extension_rank");

    // phase is a learned head now; fall back to reliable-head rules only if a
    // model without the phase head is loaded.
    let phase = match get("phase") {
        "-" => phase_fallback(extension, direction, reversal_head, range_rank),
        learned => learned,
    };

    // "Dirty" now means a messy / pushed-back lifecycle state.
    let dirty = matches!(phase, "Rejected" | "Stalling");

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

    // A reversal only counts when the opposing block is actually RUNNING (a
    // committed opposing move), not any adjacent flip (noise).
    let reversal_confirmed = reversal_transition && phase == "Running";
    let reversal_warning = reversal_confirmed || reversal_head == "High";

    // --- the answers, off phase ---
    let (control, conviction) = control_of(phase, direction, extension);
    let range_state = if expansion {
        "Expanding"
    } else if compression {
        "Compressing"
    } else {
        "Steady"
    };
    let state_quality = state_quality_of(phase);
    let risk_level = derive_risk(reversal_head, phase);
    let usable = phase == "Running" && risk_level != "High";

    let mut structure_tags = Vec::new();
    if phase != "-" {
        structure_tags.push(phase.to_owned());
    }
    if range == "Adapt" {
        structure_tags.push("AdaptRange".to_owned());
    }
    if continuation {
        structure_tags.push("Continuation".to_owned());
    }
    if reversal_confirmed {
        structure_tags.push("Reversal".to_owned());
    }
    if expansion {
        structure_tags.push("Expansion".to_owned());
    }
    if compression {
        structure_tags.push("Compression".to_owned());
    }

    let mut reason_tags = Vec::new();
    if risk_level == "High" {
        reason_tags.push("HighRisk".to_owned());
    }
    if phase == "Rejected" {
        reason_tags.push("PushedBack".to_owned());
    }
    if phase == "Stalling" {
        reason_tags.push("EffortStuck".to_owned());
    }
    if range == "Follow" {
        reason_tags.push("ThinRange".to_owned());
    }

    // Coarse location rollup — state_quality / usable carry the real read.
    let location_quality = match state_quality {
        "Clean" if risk_level == "Low" => "Good",
        "Clean" | "Indecisive" => "Watch",
        _ => "Bad", // Failed / Stuck
    };

    StructureReport {
        trend_bias: direction.to_owned(),
        risk_level: risk_level.to_owned(),
        dirty_warning: dirty,
        reversal_warning,
        range_frame_tag: range.to_owned(),
        location_quality: location_quality.to_owned(),
        structure_tags,
        reason_tags,
        phase: phase.to_owned(),
        control: control.to_owned(),
        conviction: conviction.to_owned(),
        range_state: range_state.to_owned(),
        state_quality: state_quality.to_owned(),
        usable,
    }
}
