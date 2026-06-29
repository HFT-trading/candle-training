# v2 Coding Prep — Market Structure Model

Ghi chú chuẩn bị trước khi code. Bám `rules/market-event-ai-dataset.md`.
Trạng thái: **chưa code** — sẽ dựng theo note này.

---

# Crate layout (đã chốt)

`structure-core` là **crate độc lập** — định nghĩa contract MỘT lần, cả train lẫn serve dùng chung.

```
┌─ structure-core (crate ĐỘC LẬP) ──────────┐
│  • Sequence (input)  +  StructureReport (output)
│  • model architecture + forward (A + head B)
│  • vocab/schema + numeric normalizer
│  • streaming state (h + block_repr trước)
│  • report mapping + RULE helpers (range_frame_tag,
│      entry_support, position_risk — luật trên metadata)
└────────────────────────────────────────────┘
      ▲                              ▲
      │ depend                       │ depend
┌─────┴───────────┐        ┌─────────┴───────────────┐
│ candle-training │        │ service bot-trading     │
│ dataset+trainer │        │ init core + đẩy stream  │
│ bin/train(export)│       │ + hỏi report            │
│ bin/inspect(test)│       └─────────────────────────┘
└─────────────────┘
```
- candle-training = **chỉ train/test/export** `model.safetensors` (+ normalizer, vocab). `inspect.rs` co lại thành **test harness** gọi `structure-core`.
- Service ngoài chỉ cần `structure-core` + artifact.

Contract serve:
```rust
let model = StructureModel::load("artifacts/")?;   // safetensors + vocab + schema
let mut stream = model.stream();                    // giữ h + block_repr trước
let report: Option<StructureReport> = stream.push(seq);  // đẩy 1 sequence / đợt
stream.reset();                                     // đầu phiên / gói hỏng
```

---

# BUILD NOW — Model A (đọc cấu trúc) + Head B (forecast block kế)

## 1. Mục tiêu (đã khoá)
Đọc stream sequence → **report cấu trúc hiện tại** + **đoán block kế** ⇒ *"chỗ này có đang build cho 1 move / đáng nghĩ tới entry không"*. KHÔNG order.
- **A** = tả block hiện tại. **B** = đoán nhãn block kế (trả lời "đang build?" trực tiếp).

## 2. Data
- `datasets/market_contexts.jsonl` — 3894 contexts, 32/8/4, 8 sources (bản final).
- **Target chính = BLOCK + RELATIONS** (15.576 ví dụ block, phân bố đẹp).
- **Context label = BỎ** khỏi target (bão hoà; không sửa ngưỡng) — chỉ là *khung nhìn input*.
- Nhãn deterministic ⇒ data-efficient.

## 3. Input contract (Bước 0 — làm tay trước)
Per sequence:
- **Numeric**: `duration_sec, net_bps, abs_net_bps, favorable_bps, adverse_bps, opposite_bps, retention, confidence` + `pattern.confidence, pattern.length`.
- **Categorical**: `micro_trend, direction_hint, vector_hint, bias_hint, quality_hint, behavior_hint, side` + `pattern.name` (null → lớp riêng).
- **BỎ khỏi input**: top-level metadata, `block_relations` (leak), `price`, `timestamp`, `*_index`, **block/context summary**.

Target:
- **A** — `labels.blocks` ×4 (`direction, extension_rank, range_rank, range_frame_tag, path_quality, ended_bias, reversal_risk, trend_strength`); `labels.relations` ×3 (`relation, from_direction, to_direction`).
- **B** — nhãn block kế: trong mỗi context supervise `block 1→2, 2→3, 3→4` (target = nhãn block kế, đã có sẵn). Mức **block** (không chồng) để tránh leak.

## 4. Kiến trúc (1 model, 2 head)
```
embed categorical + normalize numeric
  → fuse step (Linear + GELU)
  → GRU 32 step (hidden = bộ nhớ, chảy xuyên ranh giới block)
  → pool 8 step → block_repr[4]
  → head A: block + relation  → softmax
  → head B: block_repr[k] → đoán nhãn block_repr[k+1]  → softmax
loss = A (cross-entropy, class-weight) + λ·B (λ NHẸ vì B học nhiễu)
```
- ⚠️ B học nhiễu (quá khứ không quyết định tương lai) → A là xương sống, B trọng số nhẹ.
- GELU ở fuse/MLP; GRU tự lo; heads để trần. Giữ variable-length training; split **by source**; vocab build từ data theo `(level, field)`.
- Lớp hiếm: giữ nhưng đánh dấu low-confidence/debug.

## 5. Runtime / streaming (Chế độ 1 — scroll liên tục)
- Đẩy **từng sequence**; bộ nhớ = `h` + `block_repr` trước (2 vector). O(1)/sequence, vô hạn, tự quên.
- Tầm nhìn hiệu dụng **~16 (2 block)** (núm tune). Output **block read mỗi 8, lăn vô tận** (read tạm ở giữa). KHÔNG "tới 32 mới kết luận".
- **Reset theo SỰ KIỆN** (gap/đầu phiên/gói hỏng), `h ← h0`. KHÔNG reset mỗi 32.

## 6. Report + lớp luật (trong structure-core, post-inference)
- `StructureReport` = map nhãn A + forecast B → `trend_bias, risk_level, dirty_warning, reversal_warning, range_frame_tag, building, structure_tags, reason_tags`. Read sống, KHÔNG buy/sell.
- `entry_support` / `position_risk` = **HÀM LUẬT trên metadata** (range, **avg range**, bps) + report — KHÔNG phải head model. Minh bạch, dễ chỉnh.

---

# PHASE 2 — để sau
- **Continual / online learning** (data còn nhỏ nên chưa cần).

# Tuỳ chọn (làm khi hữu ích, không chặn)
- Rescale ngưỡng context để context label sống lại.
- Summary làm target phụ (multi-task) ⇒ encoder mạnh hơn.
- Relation regression (`range_ratio`, `net_delta_bps`).
