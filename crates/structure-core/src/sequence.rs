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

/// Supervised per-block classification labels (head A). `phase` is the run/exhaust
/// lifecycle (Running/Rejected/Stalling/Fading) — it REPLACED `block_process`,
/// which under-learned (~0.40) because it was defined on intra-block micro-features
/// the model never sees. `phase` is defined on block aggregates the model DOES see
/// (net / opposite / effort), so it is learnable, and it is the product-facing
/// "what is the market doing" read the report is built on.
pub const BLOCK_LABEL_FIELDS: [&str; 7] = [
    "direction",
    "extension_rank",
    "range_rank",
    "range_frame_tag",
    "ended_bias",
    "reversal_risk",
    "phase",
];

/// Category used when a categorical field is absent from a sequence.
pub const MISSING_CATEGORY: &str = "__MISSING__";

/// Category used for `pattern.name` when no pattern fired (`null`).
pub const PATTERN_NONE: &str = "None";
