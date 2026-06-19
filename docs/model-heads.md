# Model Heads

## Purpose

The GRU produces one hidden vector for every observed step:

```text
hidden_states: [batch, sequence_length, hidden_dim]
```

With the current config this is `[B, T, 96]`. A hidden vector is internal
sequence memory; it is not yet a human-readable market state. A model head is a
small learned projection that reads that memory and assigns scores to one
specific question.

The model has three primary state heads and one secondary move-outlook head:

```text
GRU hidden state
├── regime head
├── quality head
├── cycle-stage head
└── move-outlook head
```

The heads share the same GRU memory but have independent weights. This lets the
GRU learn one lifecycle representation while each head specializes in a
different interpretation.

## What a logit is

Every head returns logits, not probabilities. A logit is an unrestricted score
learned by the model. It may be negative or positive and does not need to sum to
one.

For a categorical state head:

```text
logits -> softmax -> class probabilities
```

For the multi-label move-outlook head:

```text
each logit -> sigmoid -> one independent probability
```

Keeping raw logits in the model is important because the later loss functions
and calibration operate more reliably on logits. Probability conversion belongs
at the evaluation or inference boundary.

## Regime head

Question:

> What broad directional regime does the active process currently express?

Classes:

```text
Sideway
UpAttempt
DownAttempt
```

Transformation:

```text
hidden [B,T,96]
  -> Linear(96,3)
  -> regime logits [B,T,3]
```

This head is evaluated at every timestep. It does not mean buy, sell, long, or
short. `UpAttempt` and `DownAttempt` describe the direction currently expressed
by the market-state process; they do not assert that the attempt will succeed.

Examples:

- `Sideway` may coexist with building pressure.
- `UpAttempt` may coexist with `Weak` quality.
- `DownAttempt` may coexist with an early or extended cycle stage.

The other heads provide those missing dimensions.

## Quality head

Question:

> How coherent or structurally credible is the current state expression?

Classes:

```text
Accepted
Broken
Choppy
Pressured
Rejected
Weak
```

Transformation:

```text
hidden [B,T,96]
  -> Linear(96,6)
  -> quality logits [B,T,6]
```

Quality is deliberately separate from regime. A directional attempt is not
automatically strong or accepted. Keeping a separate head allows combinations
such as:

```text
UpAttempt + Pressured
UpAttempt + Weak
DownAttempt + Rejected
Sideway + Choppy
```

These combinations are more informative than forcing all state semantics into
one large flat class list.

## Cycle-stage head

Question:

> Where is the current process in its lifecycle?

Classes:

```text
Compression
PressureBuild
EarlyExpansion
Expansion
ExtendedExpansion
```

Transformation:

```text
hidden [B,T,96]
  -> Linear(96,5)
  -> stage logits [B,T,5]
```

Stage describes lifecycle position, not direction or quality. For example,
`Expansion` alone does not tell us whether the process expands upward or
downward, nor whether that expansion remains credible. Regime and quality answer
those separate questions.

Because the head runs at every timestep, its sequence can describe transitions:

```text
Compression
-> PressureBuild
-> EarlyExpansion
-> Expansion
```

or deterioration:

```text
EarlyExpansion
-> PressureBuild
-> Compression
```

## Why three state heads instead of one

A single combined class would need to represent every possible tuple:

```text
(regime, quality, stage)
```

With 3 regimes, 6 qualities, and 5 stages, that is potentially 90 combinations.
Many combinations would be rare, making the training data sparse and forcing
the model to relearn shared concepts repeatedly.

Three heads reuse the shared GRU representation and factor the state into three
questions:

```text
directional context × structural quality × lifecycle position
```

This is easier to inspect and allows the model to generalize across combinations
that appear infrequently.

## Unknown is not a state class

The schema vocabularies contain `__UNK__` for parsing unknown categorical data.
The three state heads intentionally do not allocate a learned `UNKNOWN` output
class.

At inference time, `UNKNOWN` means the model should abstain because confidence
is insufficient, states conflict, history is insufficient, or the input is
out-of-distribution. It will be derived later from calibrated probabilities and
thresholds.

Before training, the current state labels numbered `1..N` must therefore be
remapped to `0..N-1`. A target carrying the parser's `__UNK__` ID must be masked
or rejected rather than trained as another state.

## Move-outlook head

The move-outlook head is included for contrast. It reads only the final hidden
state of the supplied sequence:

```text
final hidden [B,96]
  -> Linear(96,5)
  -> move-outlook logits [B,5]
```

Its five outputs follow the current boolean target order in the schema:

```text
future_reaches_25bps
future_reaches_40bps
future_returns_to_origin
future_counter_confirm
future_aligned_confirm
```

These outcomes can overlap, so they are independent sigmoid outputs rather than
one softmax choice. This is a secondary outlook objective: it describes how the
active process may develop, while the three state heads describe what the
process currently is.

Numeric future extension and drawback fields do not have model heads. They
remain offline analysis metadata.

## Current tensor contract

For a batch of `B` sequences, each with `T` steps:

```text
regime logits:       [B,T,3]
quality logits:      [B,T,6]
cycle-stage logits:  [B,T,5]
move-outlook logits: [B,5]
```

All values are untrained random scores until a loss function and training loop
update the model parameters. Correct tensor shapes prove that information flows
through the architecture; they do not prove prediction quality.

During development these values can be inspected without a JSON contract:

```rust
tracing::debug!(?output, "model forward completed");
```
