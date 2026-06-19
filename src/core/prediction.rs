/// The inference result for the latest item in an active market lifecycle.
///
/// `S` is deliberately generic so the state taxonomy can evolve without
/// changing the response envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResponse<S> {
    pub status: PredictionStatus,
    pub market_state: Option<MarketStateAssessment<S>>,
    pub move_outlook: Option<MoveOutlook>,
    pub range_metadata: Option<HistoricalRangeMetadata>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarketStateAssessment<S> {
    /// State labels produced directly by the upstream parser.
    pub observed: S,
    /// Sequence-aware interpretation produced from the recurrent memory.
    pub inferred: Option<StatePrediction<S>>,
    /// Calibrated lifecycle consistency, once a definition is validated.
    pub stability: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatePrediction<S> {
    pub state: S,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionStatus {
    Ready,
    Unknown {
        reason: UnknownReason,
        confidence: Option<f32>,
    },
}

/// Probabilities for the five overlapping future outcomes in the V5 dataset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveOutlook {
    pub reaches_25bps: f32,
    pub reaches_40bps: f32,
    pub returns_to_origin: f32,
    pub counter_confirm: f32,
    pub aligned_confirm: f32,
}

/// Offline statistics for comparable historical states, not model regression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoricalRangeMetadata {
    pub median_extension_bps: f32,
    pub p80_extension_bps: f32,
    pub median_drawback_bps: f32,
    pub p80_drawback_bps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownReason {
    InsufficientHistory,
    AmbiguousState,
    OutOfDistribution,
    InvalidInput,
}
