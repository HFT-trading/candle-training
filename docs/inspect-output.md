# Model Response

## Command

```bash
RUST_LOG=info cargo run --bin inspect -- 0
```

The final argument is the zero-based sequence index. The default output contains
only the two model results:

```text
INFO MODEL: STATE: regime Sideway 98.2% | quality Choppy 93.8% | stage Compression 43.9%
INFO MODEL: OUTLOOK: future_reaches_25bps: 42.2% | future_reaches_40bps: 10.5% | future_returns_to_origin: 65.2% | future_counter_confirm: 9.6% | future_aligned_confirm: 11.7%
```

## What running a sequence produces

The selected sequence contains ordered market snapshots. The model reads them
from first to last and uses the GRU to build one memory of the process seen so
far.

For the final snapshot, it returns:

1. `STATE`: what the market process appears to be now;
2. `OUTLOOK`: which future lifecycle outcomes appear likely after this point.

The sequence provides temporal context. The result is not eight independent
snapshot classifications and it is not a price forecast or trading action.

```text
ordered snapshots
       -> sequence memory
       -> current STATE + future OUTLOOK
```

## STATE

Example:

```text
STATE: regime Sideway 98.2% | quality Choppy 93.8% | stage Compression 43.9%
```

`STATE` describes the model's interpretation at the end of the sequence. It has
three separate dimensions.

### Regime

```text
Sideway | UpAttempt | DownAttempt
```

Regime answers:

> What broad directional condition is the process expressing now?

`UpAttempt` and `DownAttempt` describe an attempted market-state direction. They
do not mean buy, sell, or that the attempt will succeed.

### Quality

```text
Accepted | Broken | Choppy | Pressured | Rejected | Weak
```

Quality answers:

> How coherent or credible is the current state expression?

Regime and quality are separate because an `UpAttempt` can be `Weak`,
`Pressured`, or `Accepted`.

### Stage

```text
Compression | PressureBuild | EarlyExpansion | Expansion | ExtendedExpansion
```

Stage answers:

> Where is the process in its current lifecycle?

Stage does not describe direction. For example, `Expansion` may occur in either
direction and may still have weak or rejected quality.

### Percentages

Each percentage is the highest softmax probability from its own state head.
The three displayed percentages do not need to add to 100% because regime,
quality, and stage are three different classification questions.

`Compression 43.9%` means Compression was the highest stage class, but the
stage head was relatively uncertain. Until calibration and abstention are
implemented, the CLI still prints the top class instead of `UNKNOWN`.

## OUTLOOK

Example:

```text
OUTLOOK: future_reaches_25bps: 42.2% | future_reaches_40bps: 10.5% | future_returns_to_origin: 65.2% | future_counter_confirm: 9.6% | future_aligned_confirm: 11.7%
```

`OUTLOOK` describes possible events after the sequence endpoint:

- `future_reaches_25bps`: chance of reaching the 25-bps threshold;
- `future_reaches_40bps`: chance of reaching the 40-bps threshold;
- `future_returns_to_origin`: chance of returning near the cycle origin;
- `future_counter_confirm`: chance of a future accepted counter confirmation;
- `future_aligned_confirm`: chance of a future accepted aligned confirmation.

The five percentages are independent sigmoid outputs and do not add to 100%.
Multiple events can occur in one lifecycle. A process may reach 25 bps, later
reach 40 bps, and also produce an aligned confirmation.

`OUTLOOK` supplies evidence about how the process may develop. It does not say
buy, sell, enter, exit, or hold.

## Why the sequence matters

The final snapshot alone may look identical in two different situations:

```text
Compression -> PressureBuild -> UpAttempt -> Weak
```

and:

```text
DownAttempt -> Broken -> Sideway -> Weak
```

The GRU memory makes the final response depend on the path leading to the
current snapshot. `STATE` describes the endpoint using that history; `OUTLOOK`
uses the same history to estimate possible next outcomes.

The current fixed dataset sequence has eight steps. Future lifecycle-aligned
data may use longer or variable-length sequences without changing the meaning
of the two response groups.

## Optional raw/reference output

Raw timestep data and dataset reference labels are debug information, not part
of the model response. Show them only when needed:

```bash
RUST_LOG=info cargo run --bin inspect -- 0 --raw
```

This adds `REPORT` lines for every input timestep and one `REFERENCE` line for
the dataset labels. The default command omits them.

## Current limitation

The checkpoint currently proves that training and inference work end to end.
Its percentages are not production probabilities yet. They still require
cycle-aligned data, held-out metrics, calibration, and `UNKNOWN` thresholds.
