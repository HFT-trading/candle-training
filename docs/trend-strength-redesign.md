# Handoff — Label redesign (exporter side)

Tài liệu **tự-chứa** cho một session mới làm việc ở **project data/exporter**
(`build_market_context_dataset.py`). Bạn KHÔNG cần context nào khác. Việc chính:
**redesign nhãn ở exporter → regen dataset**; sau đó phía Rust (`candle-training`)
mới thích ứng.

> ✅ **STATUS (2026-06-30): ĐÃ XONG cả 2 phía.**
> - **Exporter (Part A):** bỏ `trend_strength` 4-class → `trend_pressure` (float
>   [-1,1], có dấu) + `absorption` (bool). Công thức: effort `e=fav+adv+opp`,
>   carryover `M=EMA(e)` half-life 8, result `SR=2·close_position-1` (trailing 2
>   block), `trend_pressure=tanh(M/scale)·SR`, `absorption= M≥effort_hi & |SR|≤result_lo`.
>   Consts (CLI args): scale=10, effort_hi=9, result_lo=0.35. Regen → 4375 contexts.
> - **Rust (Part B):** head `trend_strength` classification → **regression**
>   (`trend_pressure`, tanh+MSE) + head `absorption` 2-class. Per-field class
>   weights (`weighted_block_fields=["absorption"]`) để flag hiếm học được mà
>   không hại các head cân. Trained val: trend_pressure MAE 0.201 (std 0.441),
>   absorption macro-recall 0.79. Mục §1 dưới đây là bản ghi thiết kế gốc.

---

## 0. Hệ thống trong 1 đoạn

`candle-training` (Rust) train một model **ĐỌC cấu trúc thị trường** từ một
`context` = **32 sequence = 4 block × 8**. Heads phân loại **per-block** +
**per-relation** (block↔block). Nó **không forecast** (đã thử "đoán block kế" →
ở đáy nhiễu, quá khứ không quyết định tương lai → đã bỏ). Model **chỉ đọc
sequence THÔ** (summary cố tình KHÔNG đưa vào input, vì nhãn suy từ summary →
đưa vào là tầm thường hoá).

Dữ liệu do **exporter** này sinh ra: `market_contexts.jsonl` + schema + summary.
Mỗi row: `metadata` (trace) / `training_data` (sequences[32], blocks[4], context,
block_relations[3]) / `labels` (blocks[4], context, relations[3]).
Nhãn được tính trong `internal_label(metrics)` từ `aggregate_metrics(frames)` —
toàn ngưỡng deterministic (`extension_rank`, `range_rank`, `range_frame_tag`,
`path_quality`, `ended_bias`, `reversal_risk`, `trend_strength`).

---

## 1. VIỆC CHÍNH — redesign `trend_strength`

### Hiện trạng (sai)
`trend_strength(metrics)` = 4 lớp `None/Weak/Medium/Strong`, tính từ
`abs_net_bps` + `vector_dominance` của **chính block đó** → **local**.

### Vì sao phải đổi (đã chứng minh bằng số)
Model **không học nổi** field này: macro-recall ~0.27 (gần như chỉ đoán
majority "Medium"). Bật class-weight thì macro nhích nhưng accuracy **sụp −0.38**
= chỉ rải đoán bừa, không phải học. → **Nhãn sai bản chất, không phải model dở.**

### Khái niệm đúng (lời chủ dự án)
> "Áp lực xu hướng **ngầm** trong chuỗi, dạng **sóng** — không chỉ độ mạnh nhìn
> thấy của block hiện tại."

- Một block **weak cục bộ** không nhất thiết làm yếu cả chuỗi.
- **Nhiều strong mà không di chuyển** = một trạng thái riêng (nén/hấp thụ —
  effort bị đốt mà không ra kết quả).
→ Nó là **quan hệ (effort ↔ result) + ngữ cảnh xuyên chuỗi**, KHÔNG phải biên độ
1 block.

### Hướng redesign (làm ở exporter)
1. **Liên tục/ordinal**, bỏ 4 lớp cứng (áp lực là **thang/sóng**, mượt).
2. **Carryover xuyên block** — EMA / tích luỹ effort theo hướng qua cả chuỗi,
   để 1 block weak **không reset** áp lực nền → biến "local" thành "latent".
3. **Tách EFFORT ↔ RESULT** — phân biệt:
   - effort cao + net cao = trend thật mạnh
   - effort cao + net ~0 = **nén/hấp thụ** ("strong-but-stuck") ← case bị gộp nhầm
   - effort thấp = im
   → có thể thành **1 score áp lực có dấu** (+up/−down, |.| = cường độ) **kèm cờ
   "stuck/absorption"**.

### ❓ Câu hỏi DOMAIN cần chủ dự án chốt trước khi code
- **EFFORT đo bằng gì?** (favorable+adverse+opposite bps? Σ|net|? vector
  streak/flips? rejected_count?)
- **RESULT đo bằng gì?** (net_bps? close_position_in_range?)
- **Carryover** horizon / tốc độ phân rã (áp lực phai nhanh/chậm)?
- **Output**: regression liên tục? ordinal buckets? signed + stuck-flag?

### Ràng buộc
Nhãn mới phải **đọc được từ sequence THÔ** (model chỉ ăn sequences, không ăn
summary). Đừng định nghĩa nhãn cần thông tin chỉ-có-ở-summary.

---

## 2. (Tuỳ chọn) Context-label bão hoà

Nhãn cấp **context** bão hoà (72% Dirty, 72% reversal High...) vì ngưỡng chỉnh
cho **block 8-step** bị áp lên **context 32-step**. Sửa: **scale ngưỡng context
theo số block**. *(Chủ dự án từng ngại "sửa lòi ra thứ khác"; chỉ làm nếu muốn
read cấp-context. `candle-training` hiện đang BỎ nhãn context khỏi target.)*

---

## 3. (Tuỳ chọn, sâu) Situation/phase target

Output per-block mô tả hiện chưa actionable (live cứ lặp "Up/Low/Watch/
Continuation"). Ý tưởng: nhãn **phase/lifecycle** (Build → Expansion →
Exhaustion → Reverse) hoặc **situation** {Calm/Building/Stress/Danger/Resolved}
xuyên block. Phải **suy từ cấu trúc quan sát được**, không phải dự đoán (vì
forecast đã fail). Ưu tiên thấp hơn `trend_strength`.

---

## 4. Sự thật về data (khỏi dò lại)

- 3894 contexts, 8 source log, shape 32/8/4.
- Block labels khác (`direction`, `path_quality`, `reversal_risk`, `extension_rank`,
  `range_rank`, `ended_bias`) **cân & học tốt** → **ĐỪNG đụng**.
- `relation`: 15 lớp, đuôi hiếm dài.
- Chỉ `trend_strength` là sai bản chất → đây là field cần redesign.

---

## 5. Workflow quay lại `candle-training`

```
[session này @ exporter]  định nghĩa effort/result + carryover
   → sửa trend_strength() / aggregate_metrics() → regen market_contexts.jsonl
        ↓ copy .jsonl mới về candle-training/datasets/
[candle-training @ Rust]  đổi head trend_strength: classification → REGRESSION
   (sửa: structure-core/sequence.rs BLOCK_LABEL_FIELDS, builder targets,
    model.rs head, trainer loss/eval, report.rs). Encoder giữ nguyên.
```

Phía Rust đã có hợp đồng đầy đủ ở `candle-training/docs/v2-coding-prep.md` và
memory của project đó.

## 6. ĐỪNG phá

- Đừng đổi mấy block label đang học tốt.
- Giữ **sequence thô là input duy nhất** của model (không rò summary).
- `trend_strength`: redesign **NHÃN**, kiến trúc model vẫn ổn (chỉ đổi 1 head sang
  regression).
