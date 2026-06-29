# Market Event AI Training Rule

This document is the training rule and data contract for the market-structure
model.

It has two jobs:

```txt
1. define what the model is supposed to learn
2. define the JSONL input/output shape used to train that rule
```

The goal is to stop adding ad-hoc market stories and produce one stable training
contract from the existing semantic logs.

## Document Boundary

This doc is not just a file-format description.

It defines the rule:

```txt
Read a recent market context and classify its current inside-context structure.
```

It also defines the dataset contract:

```txt
metadata      = trace/debug only
training_data = model input
labels        = supervised targets for the rule
```

The generated JSONL/schema files are the concrete data artifacts produced from
this rule:

```txt
market_contexts.jsonl
market_context_schema.json
market_contexts_summary.md
```

## Dataset Rule

This is the hard contract for generated rows.

```txt
1 sequence = 1 combined REPORT line
8 sequence = 1 block
4 block    = 1 context
```

Each JSONL row represents one same-source current context:

```txt
metadata      = trace/debug only; do not train on this as market signal
training_data = model input
labels        = supervised inside-context structure targets
```

`training_data` contains:

```txt
sequences       = ordered 32-step time series
blocks          = ordered 4-step block summaries
context         = full-context summary
block_relations = relation labels/features between adjacent blocks
```

Each sequence is split into two sides:

```txt
sequence.metadata = numeric telemetry
sequence.vector   = semantic/vector structure
```

Each block/context summary is split the same way:

```txt
metadata_summary = bps/range/duration/raw numeric telemetry
vector_summary   = trend/vector/quality/behavior structure
```

Current labels are inside-context labels only:

```txt
labels.blocks    = per-block structure labels
labels.context   = full-context structure label
labels.relations = block-to-block relation labels
```

The current dataset does not train future prediction and does not train order
execution.

## Training Discussion Contract

This is the soft boundary for the training/model side. It defines what a future
blackbox consumes and emits, without prescribing architecture.

```txt
blackbox input  = current context training_data
blackbox output = StructureReport
```

The blackbox should not output:

```txt
buy/sell
entry price
position size
exit command
```

It should output a current-structure/risk report:

```json
{
  "location_quality": "Good | Watch | Bad",
  "range_frame_tag": "Follow | Enough | Adapt",
  "risk_level": "Low | Medium | High",
  "dirty_warning": true,
  "reversal_warning": false,
  "trend_bias": "Up | Down | Flat",
  "structure_tags": ["AdaptRange", "DirtyPath"],
  "reason_tags": ["HighReversalRisk", "WeakTranslation"]
}
```

The intended use is:

```txt
Data/model selects positions worth thinking about.
Human/rules decide whether to place orders.
State manager owns hold/exit.
```

Future model discussions can decide whether the blackbox is a classifier,
sequence encoder, multi-head model, hybrid rules+model, or something else.

## Objective

The core dataset should answer this question first:

```txt
Given the recent market behavior context,
what is the current inside-context market structure?
```

This is not an order-entry dataset.

It is a market-structure dataset. It selects positions worth thinking about; it
does not select orders.

## Core Ontology

The current core naming is:

```txt
1 sequence = 1 combined REPORT line / one atomic behavior sequence
1 block    = 8 sequences
1 context  = 4 blocks = 32 sequences
target     = inside-context structure labels, not future prediction
```

This matters because a `REPORT` is not a raw tick. It already contains duration,
movement telemetry, retention, vector interpretation, behavior quality, and
confidence. So it is treated as one atomic behavior sequence.

The older `market_sequences.jsonl` name is legacy V1 terminology. In that file,
one "sequence" means a training window of derived snapshots. In the new core
dataset, use `context` for the 4-block training window.

## Core Idea

Each training sample has three parts:

```txt
metadata -> training_data -> labels
```

Example:

```txt
last 32 atomic sequences / 4 blocks of behavior
-> internal structure labels for blocks, relations, and context
```

The model should first learn current structure:

```txt
Is this context usable, dirty, risky, compressed, expanding, or structurally interesting?
```

Future prediction is a later layer, not the current core target.

## Product Tasks Built On Top

The context dataset is the shared base representation. Separate model/function
heads should answer separate questions instead of one vague "what should I do?"
interface.

### `entry_support(context)`

This does not decide entry. It only scores whether the current location is worth
thinking about.

```txt
input  = current context
output = location quality / current risk / range tag / dirty warning / reason tags
```

Example output:

```json
{
  "location_quality": "Good",
  "range_frame_tag": "Enough",
  "risk_level": "Medium",
  "dirty_warning": false,
  "reason_tags": ["EnoughRange", "StrongEndBias"]
}
```

### `position_risk(context, active_side)`

This is for an already-open side. The state manager owns hold/exit logic; this
function only estimates whether the current context is becoming dangerous
against the active side.

```txt
input  = current context + active_side Long/Short
output = hold risk / reversal warning / against-position pressure / reason tags
```

Example output:

```json
{
  "active_side": "Long",
  "hold_risk": "High",
  "reversal_warning": true,
  "against_position_pressure": "High",
  "reason_tags": ["DirtyRange", "OppositePressure"]
}
```

### Later: `move_lifecycle(...)`

Move lifecycle is a later stage because one move can span multiple blocks or
contexts.

```txt
Build -> Move/Expansion -> Exhaustion -> Reverse/Rebuild
```

That requires segment-level data, not only one fixed 4-block context.

## Unit Of Data

The smallest unit is a sequence.

One sequence is one completed phase report plus its semantic interpretation.

```txt
Sequence =
  PhaseReport
  + VectorEvent
  + MoveEvent
```

A sequence should include:

```txt
timestamp
phase_state
side
net_bps
favorable_bps
adverse_bps
opposite_bps
retention
duration_seconds
vector_direction
vector_quality
move_behavior
confidence
```

## Training Sample

A training sample is a fixed-size recent context.

Initial shape:

```txt
block_size = 8 sequences
context_blocks = 4
sequence_len = 32 sequences
```

Each row means:

```txt
Given 4 blocks / 32 sequences,
label the internal structure inside that same context.
```

This avoids relying on one hand-picked story. Every sample is generated by the same rule.

## Input Features

Each sample should keep ordered sequence features and block summaries.

Top-level row shape:

```txt
metadata      = trace/debug only
training_data = actual model input
labels        = internal block/context/relation labels
```

For each sequence:

```txt
metadata:
  duration_sec
  net_bps
  abs_net_bps
  favorable_bps
  adverse_bps
  opposite_bps
  retention
  confidence

vector:
  micro_trend
  direction_hint
  vector_hint
  bias_hint
  quality_hint
  behavior_hint
  side
```

Each block and context also get summary features:

```txt
max_abs_net_bps
max_favorable_bps
max_adverse_bps
max_opposite_bps
avg_retention
direction_consistency
accepted_count
rejected_count
failed_count
opposite_expansion_count
conflict_count
noise_count
duration_total_seconds
```

These summaries are not replacements for ordered sequences. They are helper
features so the model can learn both micro rhythm and macro shape.

## Current Builder

The new core builder is:

```bash
python3 scripts/build_market_context_dataset.py logs/log-test8.log \
  --block-size 8 \
  --context-blocks 4
```

Default output:

```txt
datasets/market_contexts.jsonl
datasets/market_context_schema.json
datasets/market_contexts_summary.md
```

When run through `run_market_event_analytics.py`, the context dataset is written
next to the other analytics outputs as:

```txt
market_contexts.jsonl
market_context_schema.json
market_contexts_summary.md
```

## Later / Legacy Future Labels

The current core context dataset does not use future labels. This section is
kept as a later-stage / legacy reference for when we intentionally build a
future-prediction or move-lifecycle dataset.

Future labels should be simple and stable.

Start with these:

```txt
NoMove
PrepareUp
PrepareDown
ExpandUp
ExpandDown
OppositeInvalidation
ChopDecay
```

Suggested rules:

```txt
ExpandUp:
  future max positive net_bps >= 25

ExpandDown:
  future max negative net_bps <= -25

PrepareUp:
  future max positive net_bps >= 10 and < 25

PrepareDown:
  future max negative net_bps <= -10 and > -25

OppositeInvalidation:
  future max opposite_bps >= 15
  or future adverse_bps >= 20 with low retention

ChopDecay:
  future has repeated Conflict/Noise and no 10 bps move

NoMove:
  none of the above
```

These labels are intentionally broad. The first AI dataset should avoid too many tiny classes.

## Later / Legacy Risk Labels

Magnitude alone is not enough for holding context.

Each sample also gets a risk label:

```txt
Clean
Normal
Stressful
Dirty
Invalidating
```

Initial rules:

```txt
Clean:
  future adverse_bps <= 5
  future opposite_bps <= 5
  future retention >= 0.75

Normal:
  future adverse_bps <= 10
  future retention >= 0.55

Stressful:
  future adverse_bps > 10
  but future still expands in the labeled direction

Dirty:
  future adverse_bps > 20
  or future retention < 0.35

Invalidating:
  future opposite_bps >= 15
  against the current holding direction
```

This lets the model distinguish:

```txt
move happened cleanly
```

from:

```txt
move happened, but the path was ugly
```

That matters for holding decisions.

## Later / Legacy Entry-Relevant Filter

The first model should not train equally on every calm/noisy row.

Create a filtered dataset for entry-relevant zones.

A sample is entry-relevant if the past sequence contains at least one of:

```txt
Conflict
Rejection
FailedContinuation
OppositeExpansion
PreMoveStress
AbsorptionWin
adverse_bps >= 10
opposite_bps >= 10
```

This focuses the dataset on the region where a future entry or hold decision may become interesting:

```txt
market conflict
one side weakening
pressure building
pre-exhaustion
stress before resolution
```

## Output Record

The exported dataset should use JSONL or CSV.

Preferred initial format: JSONL.

One line:

```json
{
  "sample_id": "2026-xx-xx#000001",
  "start_ts": 0,
  "end_ts": 0,
  "sequence_length": 8,
  "future_horizon": 6,
  "frames": [],
  "summary": {
    "max_abs_net_bps": 0.0,
    "max_favorable_bps": 0.0,
    "max_adverse_bps": 0.0,
    "max_opposite_bps": 0.0,
    "avg_retention": 0.0,
    "accepted_count": 0,
    "rejected_count": 0,
    "failed_count": 0,
    "opposite_expansion_count": 0,
    "conflict_count": 0,
    "noise_count": 0
  },
  "labels": {
    "future_outcome": "NoMove",
    "future_risk": "Normal",
    "entry_relevant": true
  }
}
```

## What This Enables

This dataset can support several future model styles:

```txt
classification:
  predict future_outcome

risk model:
  predict future_risk

Bayesian/statistical model:
  estimate P(outcome | sequence features)

sequence model:
  learn behavior transitions from ordered frames

ranking model:
  rank current setups by expected future quality
```

The context dataset can serve all of these because it keeps both:

```txt
ordered atomic sequences
block/context summary features
future labels
```

## Current Implementation

Core exporter:

```txt
scripts/build_market_context_dataset.py
```

Inputs:

```txt
one or more pasted log files
```

Outputs:

```txt
market_contexts.jsonl
market_context_schema.json
market_contexts_summary.md
```

The summary should include:

```txt
total contexts
context direction labels
context path quality labels
context extension labels
context reversal risk labels
range frame tags
block relation labels
context metric stats
```

Range frame tags:

```txt
Adapt  = range_bps >= 30
Enough = range_bps >= 25 and < 30
Follow = range_bps < 25
```

The raw bps remains metadata. `range_frame_tag` is only an added tag for reading
the range frame more easily; imbalance/ratio indicators are a separate layer and
are not part of this dataset shape yet.

This is not just analytics. It is validation that the generated AI dataset is balanced enough to train on.

## What Not To Do Yet

Do not add order decisions.

Do not train before exporting and inspecting this dataset.

Do not add more tiny labels until the first dataset distribution is visible.

Do not keep asking for new data before the existing logs can be converted into the format above.

## Current Decision

The next concrete artifact should be:

```txt
AI dataset exporter
```

not:

```txt
more parser rules
more stories
more hand-picked examples
more runtime behavior
```

Once the dataset exists, the next question becomes measurable:

```txt
Which sequences historically led to useful moves,
and which sequences mostly led to bad or unstable outcomes?
```
