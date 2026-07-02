# StructureReport — chi tiết

`StructureReport` là **output sản phẩm** của lib `structure-core`. Nó KHÔNG phải một head học
được — nó là một **ánh xạ deterministic** (lớp rules trong [`src/report.rs`](src/report.rs))
từ các nhãn mà model dự đoán cho **block gần nhất** (+ một chút từ block liền trước).

> Model đọc 7 head → `build_report()` map thành report. Đổi report **không cần train lại**;
> chỉ sửa `report.rs`. Đổi *định nghĩa nhãn* thì mới cần regen data + train lại.

---

## 1. Ba câu hỏi cốt lõi + lifecycle

| Câu hỏi | Field | Giá trị |
|---|---|---|
| **Ai control?** | `control` · `conviction` | Buyers/Sellers/Balanced/Contested · Strong/Moderate/Weak |
| **Đang làm gì?** | **`phase`** · `range_state` | Running/Rejected/Stalling/Fading · Expanding/Compressing/Steady |
| **Tin được không?** | `state_quality` · `usable` | Clean/Failed/Stuck/Indecisive · bool |

`phase` là **head học được** và là **engine** của cả report — nó thay `block_process` cũ (yếu
~0.40 vì định nghĩa trên micro-feature model không thấy). Các field spec-doc còn lại (`risk_level`,
`dirty_warning`, `reversal_warning`, `range_frame_tag`, `location_quality`, `trend_bias`,
`structure_tags`, `reason_tags`) **derive lại cho khớp `phase`**.

---

## 2. Nguồn: 7 block head model dự đoán

| Head | Giá trị | macro (CV) | Report dùng vào |
|---|---|---|---|
| `phase` | Running, Rejected, Stalling, Fading | **0.73** | **engine** — chính nó + control/state_quality/risk/usable/dirty |
| `direction` | Up, Down, Flat | 0.62 | trend_bias, phe control, transition |
| `reversal_risk` | Low, Medium, High | 0.80 | risk (floor High), reversal_warning |
| `range_rank` | Narrow, Medium, Wide, VeryWide | 0.75 | range_state |
| `extension_rank` | NoMove, SmallMove, MediumMove, LargeMove | 0.63 | conviction (Running + LargeMove = Strong) |
| `range_frame_tag` | Follow, Enough, Adapt | 0.65 | range_frame_tag, ThinRange |
| `ended_bias` | StrongUpEnd … StrongDownEnd | 0.53 | **không dùng** |

---

## 3. `phase` — nhịp run/exhaust (head học, engine của read)

Định nghĩa nhãn (exporter, trên block aggregate model THẤY được — nên học được):

| phase | định nghĩa (exporter) | nghĩa | risk |
|---|---|---|---|
| **Running** | `abs_net ≥ 15bps` và direction rõ | đi có lực | theo hướng |
| **Rejected** | không chạy + `max_opposite ≥ 15` hoặc `rejected ≥ 5` | bị đạp / phản công | Cao |
| **Stalling** | không chạy/rejected + `effort(fav+adv) ≥ 36` | ghì: tốn sức không tiến | Trung bình |
| **Fading** | còn lại | trôi im | Thấp |

`Exhausted` cố ý **mổ 3 mode** (Rejected/Stalling/Fading) để đọc rủi ro dễ. Stalling/Fading tách
bằng **effort** — tín hiệu không head nào khác có (nên phase không thừa). Phân bố: Running 22% /
Rejected 5% / Stalling 26% / Fading 47%. **Không nói "sắp"**; reversal chỉ báo *khi* xác nhận.

---

## 4. Các field derive từ `phase`

### `control` / `conviction`
| phase | control | conviction |
|---|---|---|
| Running | Buyers/Sellers (theo direction) | LargeMove → Strong, else Moderate |
| Fading | Balanced | Weak |
| Rejected / Stalling | Contested | Weak |

### `state_quality`
Running → `Clean` · Rejected → `Failed` · Stalling → `Stuck` · Fading → `Indecisive`.

### `usable`
`usable = (phase == Running) && (risk_level != High)`. Tách bạch với `state_quality`, nên
`Running`+`usable=false` đọc rõ = "đang chạy nhưng risk cao, chưa vào".

### `risk_level` (derived, không phải head thô)
Floor theo phase (Rejected→High, Stalling→Medium, Running/Fading→0) rồi `max` với head
`reversal_risk`. → hết cảnh risk=Low khi phase xấu.

### `reversal_warning`
`(block trước ngược hướng VÀ block này Running) HOẶC reversal_risk head == High`. Chỉ báo khi có
**move ngược có xác nhận**, không phải mọi flip nhiễu.

### `dirty_warning`
`phase ∈ {Rejected, Stalling}`.

### `trend_bias` / `range_frame_tag`
= head `direction` / `range_frame_tag` thô.

### `location_quality` (rollup thô)
Clean+risk Low → Good · Clean/Indecisive → Watch · Failed/Stuck → Bad.

### Tags
`structure_tags`: `phase` · AdaptRange · Continuation · Reversal · Expansion/Compression.
`reason_tags`: HighRisk · PushedBack (Rejected) · EffortStuck (Stalling) · ThinRange.

---

## 5. Ví dụ đọc

```
phase: "Stalling", control: "Contested", conviction: "Weak",
state_quality: "Stuck", usable: false, risk_level: "Medium", trend_bias: "Down"
```
Đọc: "Không ai control (Contested), đang **ghì** — tốn sức mà không đi (Stuck), risk trung bình,
**chưa vào được**."

---

## 6. Thứ tự ưu tiên đọc
1. **Core:** `control` + `phase` + `state_quality` → đủ làm mode selector (hold/watch/avoid/tighten).
2. **Bổ trợ:** `conviction`, `range_state`, `usable`, `risk_level`.
3. **Thứ cấp:** `location_quality`, `reversal_warning`, các tag.

---

## 7. Streaming & cadence
- `StructureModel::read(window)` — cửa sổ step (bội số `block_size`) → 1 report.
- `Session` (lib): emit **mỗi block** (every-8).
- hft-bot có bản riêng `StructureInference::observe` — **continuous** (emit mỗi step, đọc suffix
  bội-8 lớn nhất). Divergence có chủ đích của hft-bot, NGOÀI crate này.

Model chỉ nhận `seq_len % block_size == 0` → mọi block trong report đều đầy đủ.
