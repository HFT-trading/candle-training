# Model Response

## Input contract

Inference accepts one chronological array containing between 1 and 8 steps:

```json
{
  "cycle_id": "example-cycle",
  "steps": [
    {
      "step_id": "step-1",
      "time": "13:43:39",
      "state": {},
      "range": {}
    }
  ]
}
```

`cycle_id`, `step_id`, and `time` are optional identifiers. The learned input is
only `state + range`. Steps must be ordered from oldest to newest.

The input does not need `cycle_context`, `episode_context`, future targets, or
reference labels. A ready-to-edit example is
`examples/sample-sequence.json`.

## Commands

Predict once from every step in a JSON file:

```bash
RUST_LOG=info cargo run --bin inspect -- --file examples/sample-sequence.json
```

Simulate the embedded service receiving the same steps one at a time:

```bash
RUST_LOG=info cargo run --bin inspect -- --file examples/sample-sequence.json --stream
```

With four input steps, streaming mode returns four reports: after step 1, step
2, step 3, and step 4. It demonstrates how the interpretation changes as the
active cycle accumulates evidence.

An existing training sequence can still be inspected by its zero-based index:

```bash
RUST_LOG=info cargo run --bin inspect -- 0
```

Add `--raw` only when the input telemetry and dataset reference label are also
needed.

## Response

```text
MODEL: CONTEXT: 4/8 steps
MODEL: REGIME: DownAttempt 14.4% | Sideway 84.3% | UpAttempt 1.3%
MODEL: QUALITY: Accepted 1.3% | Broken 1.6% | Choppy 47.6% | Pressured 9.7% | Rejected 7.5% | Weak 32.3%
MODEL: STAGE: Compression 40.1% | EarlyExpansion 23.2% | Expansion 13.6% | ExtendedExpansion 5.8% | PressureBuild 17.3%
MODEL: OUTLOOK: future_reaches_25bps 55.9% | future_reaches_40bps 15.6% | future_returns_to_origin 42.7% | future_counter_confirm 8.3% | future_aligned_confirm 15.9%
```

`CONTEXT: 4/8` means four observations were available. It does not mean the
input is 50% valid or that the cycle is halfway complete. One step is valid but
contains less temporal evidence; eight steps provide the complete configured
memory window. If more than eight steps arrive, the service retains the newest
eight.

## STATE

STATE is split into three independent classification questions:

- `REGIME`: `Sideway`, `UpAttempt`, or `DownAttempt` — what broad directional
  condition is being expressed;
- `QUALITY`: `Accepted`, `Broken`, `Choppy`, `Pressured`, `Rejected`, or `Weak`
  — how coherent or credible that expression appears;
- `STAGE`: `Compression`, `PressureBuild`, `EarlyExpansion`, `Expansion`, or
  `ExtendedExpansion` — where the process appears to be in its lifecycle.

Every class is shown, rather than only the winning class. Percentages within
each line are a softmax distribution and add to approximately 100%. The three
lines are separate questions, so percentages across different lines must not
be added together.

`UpAttempt` and `DownAttempt` describe observed market-state direction. They do
not mean buy, sell, or that the attempt will succeed.

## OUTLOOK

OUTLOOK estimates five possible events after the current endpoint:

- `future_reaches_25bps`;
- `future_reaches_40bps`;
- `future_returns_to_origin`;
- `future_counter_confirm`;
- `future_aligned_confirm`.

These are five independent sigmoid probabilities. They do not add to 100%
because several events may occur in one lifecycle. OUTLOOK supplies evidence
about how the process may develop; it never returns buy, sell, hold, enter, or
exit.

## What the GRU contributes

The GRU does not merely average the rows. It updates an ordered memory:

```text
previous memory + next state/range step -> new memory -> STATE + OUTLOOK
```

Consequently, the same final step may produce a different result when the path
leading to it differs. `--stream` makes this evolution visible.

## Current limitation

The checkpoint proves the architecture, training, buffering, and inference
path work end to end. Its values are not production-grade probabilities yet.
They still require lifecycle-aligned data, held-out per-head metrics,
calibration, and explicit `UNKNOWN` thresholds.
