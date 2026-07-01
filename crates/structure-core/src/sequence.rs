//! Canonical feature contract: which fields feed the model and which fields are
//! supervised. Kept here so training and serving never disagree.

/// Numeric telemetry fed per sequence step (after dropping identity fields like
/// price / timestamp / *_index). `pattern.*` come from the nested pattern object.
pub const NUMERIC_FEATURES: [&str; 10] = [
    "duration_sec",
    "net_bps",
    "abs_net_bps",
    "favorable_bps",
    "adverse_bps",
    "opposite_bps",
    "retention",
    "confidence",
    "pattern.confidence",
    "pattern.length",
];

/// Categorical (embedded) features fed per sequence step. `pattern.name` is
/// nullable; a missing pattern maps to the `"None"` class.
pub const CATEGORICAL_FEATURES: [&str; 8] = [
    "micro_trend",
    "direction_hint",
    "vector_hint",
    "bias_hint",
    "quality_hint",
    "behavior_hint",
    "side",
    "pattern.name",
];

/// Supervised per-block classification labels (head A). `block_process` is the
/// 7-state intra-block shape (CleanDrive/PullbackHeld/Reclaim/FailedPush/
/// Absorption/DirtyRotation/BalancedAuction) — it replaced both `path_quality`
/// and the `absorption` flag (subsumed). `path_state` (clear/unclear) and the
/// block-to-block `phase_change` are DERIVED from it in the report layer.
pub const BLOCK_LABEL_FIELDS: [&str; 7] = [
    "direction",
    "extension_rank",
    "range_rank",
    "range_frame_tag",
    "ended_bias",
    "reversal_risk",
    "block_process",
];

/// Category used when a categorical field is absent from a sequence.
pub const MISSING_CATEGORY: &str = "__MISSING__";

/// Category used for `pattern.name` when no pattern fired (`null`).
pub const PATTERN_NONE: &str = "None";
