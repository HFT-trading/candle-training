//! StructureReport: the product-facing read, mapped deterministically from the
//! latest block's predicted labels (head A). No model here — this is the rules
//! layer the rule doc describes.
//!
//! The core read is three answers, each a deterministic map over the predicted
//! labels (mostly `block_process`):
//!   1. Ai control?      -> `control` + `conviction`
//!   2. Đang làm gì?     -> `action` (= the block shape) + `range_state`
//!   3. Tin được không?  -> `state_quality` (shape mode) + `usable` (act gate)
//!
//! The spec-doc contract fields (trend_bias / risk_level / range_frame_tag /
//! warnings / location_quality / tags) ride ALONGSIDE the three answers, but they
//! are now DERIVED to stay coherent with them: `risk_level` is floored by
//! `state_quality` (a Failed/Dirty shape can't read Low risk), `reversal_warning`
//! only fires on a confirmed opposing drive (not any adjacent flip), and
//! `location_quality` is a coarse rollup of state_quality + risk.

use std::collections::HashMap;

use serde::Serialize;

/// The minimal read of a block needed to derive its transition to the next one.
pub struct BlockRead {
    pub direction: String,
    pub range_rank: String,
}

/// Buyers / Sellers from the block direction; `contested` when direction is Flat
/// or the process itself says nobody is cleanly in control.
fn side(direction: &str, contested: &'static str) -> &'static str {
    match direction {
        "Up" => "Buyers",
        "Down" => "Sellers",
        _ => contested,
    }
}

/// Question 1: who is in control, and how strongly.
fn control_of(process: &str, direction: &str) -> (&'static str, &'static str) {
    match process {
        "CleanDrive" | "Reclaim" => (side(direction, "Contested"), "Strong"),
        "PullbackHeld" => (side(direction, "Contested"), "Moderate"),
        "BalancedAuction" => ("Balanced", "Weak"),
        // FailedPush / Absorption / DirtyRotation: effort spent, no clean winner.
        _ => ("Contested", "Weak"),
    }
}

/// Question 2: what the market is doing. This is the block shape itself — a mere
/// direction flip between two blocks is a TRANSITION (reported via
/// reversal_warning / tags), not an action, so it does not override this.
fn action_of(process: &str) -> &'static str {
    match process {
        "CleanDrive" => "Driving",
        "PullbackHeld" => "Pullback",
        "Reclaim" => "Reclaiming",
        "FailedPush" => "FailedPush",
        "Absorption" => "Absorbing",
        "DirtyRotation" => "Rotating",
        "BalancedAuction" => "Ranging",
        _ => "Ranging",
    }
}

/// Question 3: the shape mode. Clean = readable/one-way; the rest are the bad
/// modes trust catches. This is a fact about the shape, independent of whether it
/// is act-able right now (that is `usable`).
fn state_quality_of(process: &str) -> &'static str {
    match process {
        "CleanDrive" | "PullbackHeld" | "Reclaim" => "Clean",
        "DirtyRotation" => "Dirty",
        "FailedPush" => "Failed",
        "Absorption" => "Stuck",
        "BalancedAuction" => "Indecisive",
        _ => "Indecisive",
    }
}

/// Lifecycle phase — where the block sits in the run/exhaust cycle. Discrete (the
/// feed is event-based, not a continuous stream) and built ONLY on reliable heads
/// (extension_rank, direction, reversal_risk, range_rank), never block_process.
/// "Not running" is split into three failure modes so risk is readable:
///   Rejected = pushed back (reversal pressure)      -> high risk
///   Stalling = swinging wide but no net progress    -> medium risk
///   Fading   = quiet drift                          -> low risk
fn phase_of(extension: &str, direction: &str, reversal: &str, range_rank: &str) -> &'static str {
    // First: is it actually moving with purpose?
    if matches!(extension, "MediumMove" | "LargeMove") && direction != "Flat" {
        return "Running";
    }
    // Not moving — classify how it is failing to run.
    if reversal == "High" {
        return "Rejected";
    }
    if matches!(range_rank, "Wide" | "VeryWide") {
        return "Stalling";
    }
    "Fading"
}

fn risk_rank(level: &str) -> i32 {
    match level {
        "High" => 2,
        "Medium" => 1,
        _ => 0,
    }
}

/// Derive risk from the shape mode floored against the model's reversal_risk head.
/// A bad shape sets a floor the head cannot undercut, so risk stays coherent with
/// state_quality (no more `risk=Low` under a Failed/Dirty read).
fn derive_risk(head: &str, state_quality: &str) -> &'static str {
    let floor = match state_quality {
        "Failed" | "Dirty" => 2, // High: broken shape is high risk regardless
        "Stuck" => 1,            // Medium: absorption = caution
        _ => 0,                  // Clean / Indecisive: let the head speak
    };
    match floor.max(risk_rank(head)) {
        2 => "High",
        1 => "Medium",
        _ => "Low",
    }
}

#[derive(Debug, Serialize)]
pub struct StructureReport {
    // --- spec-doc contract (derived to stay coherent with the 3 answers) ---
    pub trend_bias: String,
    pub risk_level: String,
    pub dirty_warning: bool,
    pub reversal_warning: bool,
    pub range_frame_tag: String,
    /// entry_support-style location read: Good / Watch / Bad. Coarse rollup of
    /// state_quality + risk — read state_quality/usable for the real signal.
    pub location_quality: String,
    pub structure_tags: Vec<String>,
    pub reason_tags: Vec<String>,

    /// Raw intra-block shape (CleanDrive / PullbackHeld / Reclaim / FailedPush /
    /// Absorption / DirtyRotation / BalancedAuction). Engine behind the answers.
    pub block_process: String,

    // --- 1. Ai control? ---
    /// Buyers | Sellers | Balanced | Contested.
    pub control: String,
    /// Strong | Moderate | Weak.
    pub conviction: String,

    // --- 2. Đang làm gì? ---
    /// Driving | Pullback | Reclaiming | FailedPush | Absorbing | Rotating |
    /// Ranging (1:1 with the block shape).
    pub action: String,
    /// Expanding | Compressing | Steady (vs the previous block's range).
    pub range_state: String,

    // --- Lifecycle: nhịp run/exhaust hiện tại (head khỏe, không block_process) ---
    /// Running | Rejected | Stalling | Fading.
    pub phase: String,

    // --- 3. Tin được không? ---
    /// Clean | Dirty | Failed | Stuck | Indecisive — the shape mode.
    pub state_quality: String,
    /// Act gate: Clean shape and risk not High. Orthogonal to state_quality, so
    /// `state_quality=Clean` + `usable=false` reads cleanly (readable but not
    /// act-able, e.g. reversal risk high).
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
    let process = get("block_process");
    let range = get("range_frame_tag");
    let range_rank = get("range_rank");
    let extension = get("extension_rank");
    let absorption = process == "Absorption";

    // Lifecycle phase from reliable heads only.
    let phase = phase_of(extension, direction, reversal_head, range_rank);

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

    // A reversal only counts when the opposing block is a COMMITTED drive/reclaim,
    // not any adjacent flip (which is common noise). This keeps the signal sharp.
    let reversal_confirmed = reversal_transition && matches!(process, "CleanDrive" | "Reclaim");
    let reversal_warning = reversal_confirmed || reversal_head == "High";

    // --- the three answers ---
    let (control, conviction) = control_of(process, direction);
    let action = action_of(process);
    let range_state = if expansion {
        "Expanding"
    } else if compression {
        "Compressing"
    } else {
        "Steady"
    };
    let state_quality = state_quality_of(process);

    // Risk / usability derived to agree with the shape mode.
    let risk_level = derive_risk(reversal_head, state_quality);
    let usable = state_quality == "Clean" && risk_level != "High";

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
    if dirty {
        reason_tags.push("DirtyPath".to_owned());
    }
    if absorption {
        reason_tags.push("EffortAbsorbed".to_owned());
    }
    if range == "Follow" {
        reason_tags.push("ThinRange".to_owned());
    }

    // Coarse location rollup — kept for the spec contract, but state_quality /
    // usable carry the real read.
    let location_quality = match state_quality {
        "Clean" if risk_level == "Low" => "Good",
        "Clean" | "Indecisive" => "Watch",
        _ => "Bad", // Failed / Dirty / Stuck
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
        block_process: process.to_owned(),
        control: control.to_owned(),
        conviction: conviction.to_owned(),
        action: action.to_owned(),
        range_state: range_state.to_owned(),
        phase: phase.to_owned(),
        state_quality: state_quality.to_owned(),
        usable,
    }
}
