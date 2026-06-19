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

## Input semantics

The service receives one market-state snapshot at a time, in chronological
order. Each observation belongs to one active cycle/episode.

Model inputs describe market state rather than absolute price. The current V5
schema groups them as:

- compact categorical state hints;
- range telemetry;
- cycle context;
- episode context.

Normalized price-derived measurements such as basis-point movement, range,
retention, and distance from origin may be inputs because they describe the
state. Absolute price is not the prediction objective.

Ordering is essential. Reordering observations changes their meaning and must
change the model representation.

## Runtime lifecycle and memory

The embedded service owns the active cycle lifecycle:

```text
cycle start       -> initialize sequence memory
new observation   -> update the active sequence representation
terminal event    -> produce the final response and reset memory
service restart   -> active memory is lost and starts fresh
```

Persistent sequence memory is allowed and likely useful. It may eventually be
implemented with a recurrent hidden state or another causal sequence encoder.
The exact encoder architecture is not yet selected.

Memory must never cross cycle boundaries. If multiple cycles can be active at
once, memory must be keyed by `cycle_id`.

## Training formulation

Training should teach the model how its interpretation evolves throughout a
cycle, not only how to classify the final observation.

For a cycle containing ordered observations `x1..xT`, training examples are
causal prefixes:

```text
[x1]             -> final lifecycle outcome
[x1, x2]         -> final lifecycle outcome
[x1, x2, x3]     -> final lifecycle outcome
...
[x1, ..., xT]    -> observed terminal outcome
```

This formulation allows early prefixes to remain uncertain and later prefixes
to become more stable as evidence accumulates.

All prefixes from the same lifecycle must remain in the same dataset split.
Splitting overlapping prefixes of one cycle across training and validation is
target leakage.

## Learned outcomes

The current dataset exposes five boolean future outcomes:

- `future_reaches_25bps`;
- `future_reaches_40bps`;
- `future_returns_to_origin`;
- `future_counter_confirm`;
- `future_aligned_confirm`.

These are multi-label outcomes: more than one may be true during a lifecycle.
The model may learn calibrated probabilities for them.

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
- boolean and numeric target tensor construction;
- categorical target tensor plumbing;
- a generic inference response contract with explicit abstention;
- 5,419 generated sequences of length 8 in the ignored `datasets/` directory.

Not implemented:

- lifecycle-aware cycle/episode records;
- variable-length causal prefix generation;
- authoritative start and terminal-event rules;
- cycle-level train/validation/test splitting;
- sequence encoder architecture;
- calibrated outcome heads;
- stability estimation;
- conditional range-statistics artifact;
- embedded streaming inference state.

The current length-8 sequences are useful pipeline fixtures, but they do not yet
fully encode the lifecycle formulation described here.

## Next implementation step

Update the data contract and exporter so that every record has explicit
lifecycle identity and boundaries:

```text
cycle_id
ordered observations
start condition
terminal event
final outcome labels
duration
numeric analysis metadata
```

Before changing the model architecture, define:

1. what starts a cycle/episode;
2. which touch/hit/confirm events end it;
3. whether milestones can occur before the final terminal event;
4. how cycles that never reach a terminal event are closed;
5. the precise relationship between a cycle and an episode.

Once those semantics are stable, the existing tensor pipeline can be adapted to
variable-length causal sequences and the encoder can be selected from the data
requirements rather than guessed in advance.
