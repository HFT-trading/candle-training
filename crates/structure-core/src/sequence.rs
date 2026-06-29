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

/// Supervised per-block structure labels (head A).
pub const BLOCK_LABEL_FIELDS: [&str; 8] = [
    "direction",
    "extension_rank",
    "range_rank",
    "range_frame_tag",
    "path_quality",
    "ended_bias",
    "reversal_risk",
    "trend_strength",
];

/// Supervised block-to-block relation labels (head A). Regression fields
/// (`range_ratio`, `net_delta_bps`) are intentionally excluded for now.
pub const RELATION_LABEL_FIELDS: [&str; 3] = ["relation", "from_direction", "to_direction"];

/// Category used when a categorical field is absent from a sequence.
pub const MISSING_CATEGORY: &str = "__MISSING__";

/// Category used for `pattern.name` when no pattern fired (`null`).
pub const PATTERN_NONE: &str = "None";
