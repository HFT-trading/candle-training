# Model guide — black box: data → model → read

Tài liệu maintain cho phần **model/AI** (black box). Phạm vi: từ **dataset/label** →
**training** → **artifacts** → **read** (`StructureReport`). Phần order/execution NGOÀI tài liệu này.

Read contract (output của box) ở riêng: [`crates/structure-core/STRUCTURE_REPORT.md`](../crates/structure-core/STRUCTURE_REPORT.md).

---

## 0. Cái gì nằm đâu

| Thành phần | Đường dẫn | Vai trò |
|---|---|---|
| Exporter (định nghĩa label, Python) | `hft-bot/scripts/build_market_context_dataset.py` | log thô → dataset + label |
| Dataset | `candle-training/datasets/market_contexts.jsonl` | 4375 context × 4 block × 32 seq |
| Crate `structure-core` | `crates/structure-core/` | input contract, vocab, model, normalizer, report (runtime) |
| Crate `training` | `crates/training/` | dataset/vocab/builder/trainer + các bin |
| Artifacts (bundle) | `candle-training/artifacts/` → copy sang `hft-bot/artifacts/` | model + vocab + normalizer + meta |

Vòng đời: **log → exporter → dataset → `bin/train` → artifacts → copy sang bot**. "Model sai" sửa
bằng cách lặp vòng này (cải tiến label/data → train lại → bundle mới), KHÔNG sửa trong bot.

---

## 1. Các bin (`crates/training/src/bin/`)

Chạy từ thư mục `candle-training/`: `cargo run --release -p training --bin <name> [args]`.

| Bin | Làm gì | Khi dùng |
|---|---|---|
| `train` | Train 7 head + export 4 artifacts. `[val_source]` optional (mặc định giữ `validation_fraction` theo source). Chỉ bin này **ghi** `artifacts/model.safetensors`. | Cut model production |
| `crossval` | Leave-one-source-out CV → acc/macro-recall trung bình mỗi head (± std). **Không** ghi artifacts. | Đo độ tin cậy / generalization |
| `review` | Stream 1 source held-out qua serve path, in **MODEL read vs TRUE label** + sparkline giá mỗi block. `[source] [max_contexts]` (mặc định `data-test7.log 8`). | Eyeball read có khớp thực tế |
| `inspect` | Stream 1 context qua `Session` → in `StructureReport` cuối (đúng serve path). `[context_index]`. | Test bundle load + read |
| `inspect_data` | Sanity dataset: số row, shape (32/4/…), phân bố class mỗi label. | Kiểm data layer |
| `inspect_tensors` | Build tensor từ dataset, in shape/dtype. | Kiểm builder |
| `inspect_model` | Model random-weight, 1 forward pass, kiểm shape output. | Kiểm kiến trúc |
| `ablate_pattern` | Cho mỗi context qua model 2 lần (pattern thật vs pattern=None) → đếm % read đổi. | Đo model có đọc `pattern` không |

---

## 2. Input contract — box NHẬN gì

Mỗi **step** (1 sequence) feed **8 categorical + 10 numeric**, định nghĩa canonical ở
`structure-core/sequence.rs` (train & serve dùng chung, không được lệch). Các field định danh
(`price` / `timestamp` / `*_index`) bị **loại** khỏi input.

| Categorical (8, embedded) | Numeric (10) |
|---|---|
| `micro_trend` | `duration_sec` |
| `direction_hint` | `net_bps` |
| `vector_hint` | `abs_net_bps` |
| `bias_hint` | `favorable_bps` |
| `quality_hint` | `adverse_bps` |
| `behavior_hint` | `opposite_bps` |
| `side` | `retention` |
| `pattern.name` | `confidence` |
| | `pattern.confidence` |
| | `pattern.length` |

- `pattern.name` nullable → class `"None"` (`PATTERN_NONE`); categorical vắng → `"__MISSING__"`
  (`MISSING_CATEGORY`). `pattern.*` model **có đọc nhưng nhẹ** (~7–12% ảnh hưởng — xem `ablate_pattern`).
- 1 context = **32 step** = **4 block × 8 step**. Model chỉ nhận `seq_len % 8 == 0`.

## 3. Label được define như thế nào

Model học **7 label / block** (`BLOCK_LABEL_FIELDS` trong `sequence.rs`). **Tất cả deterministic** —
tính bằng if-else trên block aggregate/metrics ở exporter. Model phải **học lại hàm này từ chuỗi
step thô** (nó KHÔNG được thấy micro-features/summary — *leak rule*: cho summary vào input là
trivialize task).

| Label | Công thức (exporter) | Classes |
|---|---|---|
| `direction` | dấu của `close_bps` vs `flat_bps=1.0` | Up / Down / Flat |
| `extension_rank` | `|net|` bucket: ≥30 / ≥15 / ≥5 | LargeMove / MediumMove / SmallMove / NoMove |
| `range_rank` | `range_bps` (high−low): ≥40 / ≥20 / ≥8 | VeryWide / Wide / Medium / Narrow |
| `range_frame_tag` | `range_bps`: ≥30 / ≥25 | Adapt / Enough / Follow |
| `ended_bias` | `close_position_in_range`(0..1) + trend, ngưỡng 0.70/0.30 | StrongUp/StrongDown/NearHigh/NearLow/NeutralEnd |
| `reversal_risk` | `max_opposite`≥15 hoặc `rejected`≥5 → High; ≥8 hoặc `vector_flips`≥5 → Medium | Low / Medium / High |
| `phase` | Running(`|net|`≥15 & dir≠Flat) > Rejected(`max_opposite`≥15 hoặc `rejected`≥5) > Stalling(`effort`=Σ(fav+adv)≥36) > Fading | Running / Rejected / Stalling / Fading |

**`phase`** thay `block_process` cũ — được thiết kế để tính trên **đại lượng model THẤY được**
(net/opposite/effort) nên học tốt (macro 0.73 vs block_process 0.41). Thứ tự first-match-wins;
Running/Rejected trùng ngưỡng `extension_rank`/`reversal_risk` (đảm bảo học được), Stalling/Fading
tách bằng **effort** (tín hiệu không head khác có).

**Legacy còn trong dataset nhưng model KHÔNG dùng:** `path_quality`, `block_process`, `path_state`
(giữ lại cho tương thích, `BLOCK_LABEL_FIELDS` bên Rust không liệt kê chúng).

Input (không phải label) xem §2.

---

## 4. Cách training (`bin/train`)

Pipeline:

1. `load_contexts("datasets/market_contexts.jsonl")` — đọc dataset.
2. `build_vocab(&contexts)` — quét data, dựng vocab per (level, field). **Không** tin schema.
3. `build_tensors(&contexts, &vocab, &device)` — categorical → id, numeric → matrix.
4. `DataSplit` — `by_source` (giữ `validation_fraction` theo source) hoặc `with_source(name)` (hold out 1 source).
5. `ClassWeights::from_config` — inverse-freq; chỉ áp field trong `weighted_block_fields` (kể cả khi `use_class_weights=false`).
6. `train(...)` — normalizer fit trên train split, **AdamW**, loss = Σ cross-entropy 7 head, early-stop theo best val, save best → `artifacts/model.safetensors` (chỉ khi `save_best=true`; `crossval` truyền `false`).
7. Export `numeric-normalizer.safetensors` + `vocab.json` + `meta.json` (`ServeMeta` = ModelConfig + block_size + context_blocks).

**Kiến trúc model** (`structure-core/model.rs`): embed categorical (8d) + project numeric (32d) →
GELU fuse → step_repr (64) → **GRU** (hidden 96, chạy 32 step, state xuyên block) → **block pool =
concat[mean, last-step]** (=192) → 7 Linear head (softmax per field khi serve = argmax).

**Config** (`config.yml` → `training:`): epochs 20 · batch 64 · lr 1e-3 · weight_decay 0.01 ·
validation_fraction 0.15 · seed 42 · early_stop_patience 3 · use_class_weights false.

**Output:** 4 file trong `artifacts/` → copy sang `hft-bot/artifacts/` là bot nuốt được.

Đo sau train: `crossval` (macro-recall per head, generalization) + `review` (eyeball serve vs label).

---

## 5. Gotchas cần biết (stale, nên dọn khi rảnh)

- `config.yml` → `weighted_block_fields: ["block_process"]` **STALE** — `block_process` đã bỏ khỏi
  head, nên hiện **không áp class-weight cho gì**. Nếu muốn cân class hiếm của `phase` (Rejected 5%),
  đổi thành `["phase"]`; hoặc để `[]`.
- `bin/train.rs` phần "=== GATE: block_process per-class recall ===" — nhãn in còn ghi
  `block_process`, thực chất giờ in recall của `phase`. Chỉ cosmetic.
