# Market-State Sequence Model

## Purpose

This model follows the lifecycle of a market cycle or episode and continuously
estimates what that process currently represents, how stable it is, and how it
is likely to terminate.

It is not a price predictor, trading policy, or signal generator. It must never
return `BUY`, `SELL`, `HOLD`, position sizing, or execution advice.

The central question is:

> Given the ordered market-state observations seen so far in the active cycle,
> what state is the process in, how stable is that interpretation, and which
> terminal outcome is becoming more likely?

The system has two explicit objectives:

1. **Market-state understanding**: identify the current regime, quality, cycle
   stage, and eventually the stability of that interpretation.
2. **Move outlook**: estimate how likely the active process is to reach the
   existing movement and confirmation outcomes.

The first objective explains what the market process currently is. The second
describes how far that process may still develop. An external strategy may use
both as evidence for entry or position management, but those actions remain
outside the model.

## Business unit: cycle/episode lifecycle

An arbitrary fixed window is not the primary business unit. The primary unit is
an ordered process with a meaningful start and end:

```text
cycle starts
  -> market-state observations arrive in time order
  -> the state develops, stabilizes, weakens, or changes
  -> a touch, hit, confirm, return, or other terminal event occurs
  -> the cycle is finalized
```

The terms `cycle` and `episode` must be given an exact hierarchy by the data
producer. Until then, this document uses “cycle/episode” for the complete
tracked lifecycle.

Touch and hit fields are not isolated trading signals. They describe possible
terminal events or milestones in the lifecycle.

The upstream parser is the authority for lifecycle boundaries. The model does
not need to learn the business rules that decide whether a touch, hit, confirm,
or return ends a cycle. It only needs an ordered feature vector and an explicit
reset when the parser starts a new cycle.

## Input semantics

The model has one input contract: a chronological array containing 1 through 8
market steps. Each step contains only the data available in realtime:

```text
step = state + range
sequence = [step_1, ..., step_N], where 1 <= N <= 8
```

The test CLI supplies that array directly. The embedded service receives one
step at a time, retains the newest eight steps for each `cycle_id`, and passes
the resulting array to the same model API. Therefore testing and realtime do
not create two different model behaviors.

The model does not require `cycle_context` or `episode_context` at inference.
Lifecycle identity and boundaries are service concerns, not learned features.

Normalized price-derived measurements such as basis-point movement, range,
retention, and distance from origin may be inputs because they describe the
state. Absolute price is not the prediction objective.

Ordering is essential. Reordering observations changes their meaning and must
change the model representation.

## Runtime lifecycle and memory

The embedded service and its upstream parser own the active cycle lifecycle:

```text
raw event      -> parser -> { state, range }
parsed step    -> service buffer[cycle_id] -> model.predict(steps)
terminal event -> service.end_cycle(cycle_id) -> discard buffer
restart        -> all in-memory cycle buffers start empty
```

The service buffer is explicit RAM state. The GRU builds its learned sequence
memory again from the buffered array on each prediction. This keeps the model
API pure and makes a 1–8 step testing request identical to realtime inference.
A future optimized path may cache hidden state, but it must preserve the same
result and lifecycle reset semantics.

Memory must never cross cycle boundaries. If multiple cycles can be active at
once, memory must be keyed by `cycle_id`.

The service API exposes three lifecycle operations:

```text
start_cycle(cycle_id)
push_step(cycle_id, step) -> prediction
end_cycle(cycle_id)
```

The parser decides when to call them. The model does not decide when a cycle is
over.

## Training formulation

Training should teach the model how its interpretation evolves throughout a
cycle, not only how to classify the final observation.

For a training sequence containing ordered observations `x1..xT`, the trainer
uses different causal suffix lengths ending at the same labeled endpoint:

```text
[xT]
[x(T-1), xT]
[x(T-2), x(T-1), xT]
...
[x(T-7), ..., xT]
```

This teaches one checkpoint to accept every configured length from 1 through
8. Across eight epochs, each sequence is scheduled through all eight lengths.
The endpoint and its future-outlook label remain aligned while the amount of
history varies.

All prefixes from the same lifecycle must remain in the same dataset split.
Splitting overlapping prefixes of one cycle across training and validation is
target leakage.

## Learned outcomes

### Primary state heads

The recurrent representation should receive per-step state supervision from
the parser's existing debug semantics:

- `current.regime`: `Sideway`, `UpAttempt`, or `DownAttempt`;
- `current.quality`: `Accepted`, `Broken`, `Choppy`, `Pressured`, `Rejected`,
  or `Weak`;
- `cycle.stage`: `Compression`, `PressureBuild`, `EarlyExpansion`, `Expansion`,
  or `ExtendedExpansion`.

These heads make market-state understanding an explicit training objective
instead of hoping it emerges from price-oriented future labels. They may later
be hidden from the public response while still serving as useful auxiliary
supervision.

### Secondary move-outlook heads

The current dataset exposes five boolean future outcomes:

- `future_reaches_25bps`;
- `future_reaches_40bps`;
- `future_returns_to_origin`;
- `future_counter_confirm`;
- `future_aligned_confirm`.

These are multi-label outcomes: more than one may be true during a lifecycle.
The model may learn calibrated probabilities for them. They are secondary
outlook heads, not a complete definition of market state.

Higher-level interpretations such as `CONTINUATION`, `FAILED`, or `NO_MOVE`
should initially be derived from the learned outcome probabilities and terminal
semantics. They are not yet authoritative categorical labels in the V5 dataset.

The code supports categorical training targets as future plumbing, but the
current dataset does not yet provide a `market_outcome` categorical target.

## Numeric future outcomes are analysis metadata

The dataset also contains:

- `future_max_extension_bps`;
- `future_max_drawback_bps`.

These values must not be regression targets in the first model version. They
are analysis metadata used offline to calculate conditional range statistics,
evaluate outcomes, and describe historical behavior of similar states.

For example, the response layer may attach a median and percentile range for a
recognized state. This is a historical conditional distribution, not a precise
price forecast.

Numeric state features remain valid model inputs. This restriction applies only
to numeric future outcomes.

## Response semantics

While a cycle is active, the model response should communicate:

- the current interpreted market state, when identifiable;
- confidence or stability of that interpretation;
- calibrated probabilities for learned lifecycle outcomes;
- optional historical range metadata for comparable states.

The response is a Rust value divided into two main sections. V1 does not need a
JSON or external reporting contract:

```rust
#[derive(Debug)]
struct ModelResponse<S> {
    market_state: Option<MarketStateAssessment<S>>,
    move_outlook: Option<MoveOutlook>,
    // status and optional historical metadata
}

tracing::debug!(?response, "model inference updated");
```

`observed` is supplied by the upstream parser. `inferred` is the model's
sequence-aware interpretation. Strong disagreement between them is evidence
for low confidence or abstention, not a reason to silently overwrite either
value. Serialization can be added later if a real consumer requires it.

The current Rust response contract lives in
`src/core/prediction.rs`.

The model must be able to abstain:

```text
Ready(state, confidence)
Unknown(reason, optional confidence)
```

`NO_MOVE` and `UNKNOWN` are different:

- `NO_MOVE` means the model has evidence for a lifecycle outcome with no
  meaningful move;
- `UNKNOWN` means there is not enough reliable evidence to interpret the
  current state.

Typical abstention reasons include insufficient history, ambiguous state,
out-of-distribution input, and invalid input.

## Stability

“Stability” describes how consistently the lifecycle supports the current state
interpretation over time. It is not trading risk.

The final stability definition is still open. Candidate evidence includes:

- consistency of the predicted state across successive prefixes;
- decreasing outcome entropy as observations accumulate;
- persistence of compatible state features;
- absence of repeated state contradictions or direction flips;
- duration spent in a coherent lifecycle phase.

Stability should not be presented as a trustworthy scalar until it is defined,
calibrated, and evaluated against held-out cycles.

## Current repository status

Implemented:

- V5 schema and JSONL sequence loading;
- categorical and numeric input tensor construction;
- per-step regime, quality, and cycle-stage target tensors;
- boolean and numeric target tensor construction;
- categorical target tensor plumbing;
- categorical embeddings and numeric projection;
- train-split numeric normalization;
- causal GRU sequence memory;
- three per-step state heads and five move-outlook logits;
- source-level train/validation isolation;
- weighted state/outlook loss and AdamW mini-batch training;
- model and numeric-normalizer checkpoint artifacts;
- a generic inference response contract with explicit abstention;
- a public `predict()` API accepting 1–8 `{state, range}` steps;
- an in-memory service buffer keyed by `cycle_id`;
- suffix-length training across all supported input lengths;
- standalone and step-by-step streaming CLI inspection;
- 5,419 generated sequences of length 8 in the ignored `datasets/` directory.

Not implemented:

- a cycle identity or boundary marker exposed to the trainer;
- cycle-level train/validation/test splitting;
- causal prefix examples aligned to complete lifecycle boundaries;
- calibrated probabilities and abstention thresholds;
- stability estimation;
- conditional range-statistics artifact;
- production-calibrated streaming inference.

The current length-8 sequences are useful pipeline fixtures, but they do not yet
fully encode the lifecycle formulation described here.

## V1 architecture

The next implementation step is a small recurrent model:

```text
categorical state fields -> embeddings ---------+
                                                +-> step representation
numeric state fields     -> normalization/MLP --+
                                                        |
                                                        v
                                                   GRU memory
                                                        +-> regime head
                                                        +-> quality head
                                                        +-> cycle-stage head
                                                        +-> five move-outlook logits
```

At each time step:

```text
h_t = GRU(step_vector_t, h_(t-1))
```

The three state heads use per-step categorical supervision. The five outcome
logits use a multi-label objective because the existing boolean outcomes can
overlap. Numeric future outcomes do not receive a regression head or contribute
to training loss.

The V1 architecture, variable-length training, pure prediction API, and
in-memory streaming wrapper are implemented. Remaining model work:

1. produce lifecycle-aligned examples instead of overlapping fixed windows;
2. split by true cycle identity when the exporter exposes it;
3. measure per-head class balance and held-out metrics;
4. calibrate probabilities and abstention thresholds on held-out cycles.

The trainer still needs a cycle identifier or boundary marker so recurrent
memory can be reset correctly and every cycle stays in exactly one dataset
split. It does not need the model to reproduce the parser's terminal-event
decision table.

## Public runtime and CLI smoke test

`ModelRuntime` is the pure checkpoint consumer. It loads the trained model and
numeric normalizer, applies training-time preprocessing, and returns a
prediction from a `SequenceRequest` containing 1–8 steps.

`BufferedModelService` owns the transient cycle buffers used by an embedded
runtime. It does not add a second prediction algorithm; every `push_step()`
calls the same `ModelRuntime::predict()` used by standalone testing.

The `inspect` binary exercises this path end to end on one dataset sequence:

```bash
RUST_LOG=info cargo run --bin inspect -- --file examples/sample-sequence.json

RUST_LOG=info cargo run --bin inspect -- --file examples/sample-sequence.json --stream
```

The second command simulates steps arriving one by one and logs a new full
distribution after each step. An existing dataset item can also be inspected by
passing its zero-based index. This is a runtime smoke test, not a quality claim.
See [inspect-output.md](inspect-output.md) for the field-by-field output
reference.
