# candle-training

A small, local market-**structure** model for a trading engine — not a price predictor.

The trading engine does the numerical work (indicators, structure detection, positions, exposure) and exports a compact snapshot per cycle. This project trains a model that reads that snapshot and returns a fast, structured read of the current market context. Execution stays rule-based inside the engine; the model only interprets.

> **Read the why first:** [Building a Trading System with AI — Without Outsourcing Judgment](docs/blog-draft-build-with-ai.md).
> **Build with AI. Keep the judgment.**

## What it does

- **Input:** a sequence of market *blocks/contexts* exported by the engine (`datasets/market_contexts.jsonl`), each mixing categorical features (trend, behavior/direction hints, range-frame tags) and numeric features.
- **Model:** categorical embeddings + numeric projection → per-step representation → GRU over the sequence → linear classification heads.
- **Output:** structured labels describing market structure (e.g. path quality, range-frame tag) — a market/risk read, not a trade instruction.

The full data and labeling contract lives in [`rules/market-event-ai-dataset.md`](rules/market-event-ai-dataset.md) and [`rules/structure-report-contract.md`](rules/structure-report-contract.md).

## Layout

```text
crates/
  structure-core/   # shared inference contract: vocab, model, tensors, serving meta
  training/         # dataset loading, vocab build, trainer, and CLI binaries
datasets/           # market_contexts.jsonl + schema/summary/reports (git-ignored)
rules/              # dataset + report contracts (the source of truth for features/labels)
docs/               # guides and the blog draft
config.yml          # model + training hyperparameters (git-ignored; see config.sample.yml)
```

`structure-core` is deliberately split out so both training and an external serving
service can depend on the same feature/vocab/model definitions.

## Requirements

- Rust (edition 2024 — recent stable/nightly toolchain)
- macOS with Metal for GPU tensor ops (via [`candle`](https://github.com/huggingface/candle)); training itself currently runs on CPU
- `datasets/market_contexts.jsonl` present locally (git-ignored; produced by the trading engine)

## Getting started

```bash
# 1. copy the sample config and adjust hyperparameters
cp config.sample.yml config.yml

# 2. make sure datasets/market_contexts.jsonl exists (exported by the engine)

# 3. train head A end-to-end and export serving artifacts
cargo run -p training --bin train
```

Pass a source name to force a specific validation split (robustness checks):

```bash
cargo run -p training --bin train -- <validation_source>
```

## Tooling binaries

All under `training/src/bin`:

| Binary | Purpose |
| --- | --- |
| `train` | Train the model end-to-end and export artifacts for serving |
| `crossval` | Cross-validation across data sources |
| `review` | Review predictions / evaluation output |
| `inspect_data` | Inspect the parsed dataset |
| `inspect_tensors` | Inspect built input tensors |
| `inspect_model` | Inspect a trained model |
| `inspect` | General inspection entry point |
| `ablate_pattern` | Ablation study over pattern features |

Run any of them with `cargo run -p training --bin <name>`.

## Configuration

Hyperparameters live in `config.yml` (git-ignored). Start from
[`config.sample.yml`](config.sample.yml):

- `model.*` — embedding / projection / GRU dimensions
- `training.*` — epochs, batch size, learning rate, weight decay, validation fraction,
  class weighting, and early stopping

## Scope

This repo is the **model / AI** lane only. Order placement, execution, and testnet
wiring live in the trading engine, outside this repository. The boundary is intentional:
the model interprets market structure; the engine owns every decision that moves money.

---

*Build with AI. Keep the judgment.* · [github.com/hananguyn/candle-training](https://github.com/hananguyn/candle-training)
