/// The inference result for the latest item in a market-state sequence.
///
/// `S` is deliberately generic so the market-state taxonomy can evolve without
/// changing the response contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResponse<S> {
    pub status: PredictionStatus<S>,
    pub outcome: Option<OutcomeProbabilities>,
    pub expected_range_atr: Option<RangeEstimate>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionStatus<S> {
    Ready {
        state: S,
        confidence: f32,
    },
    Unknown {
        reason: UnknownReason,
        confidence: Option<f32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutcomeProbabilities {
    pub continuation: f32,
    pub failed: f32,
    pub no_move: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeEstimate {
    pub median_atr: f32,
    pub p80_atr: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownReason {
    InsufficientHistory,
    AmbiguousState,
    OutOfDistribution,
    InvalidInput,
}
