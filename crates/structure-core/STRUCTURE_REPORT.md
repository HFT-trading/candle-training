# StructureReport — chi tiết

`StructureReport` là **output sản phẩm** của lib `structure-core`. Nó KHÔNG phải một head học
được — nó là một **ánh xạ deterministic** (lớp rules trong [`src/report.rs`](src/report.rs))
từ các nhãn mà model dự đoán cho **block gần nhất** (+ một chút từ block liền trước).

> Nguyên tắc: model đọc nhãn (7 block head) → `build_report()` map thành report. Đổi report
> **không cần train lại**; chỉ sửa `report.rs`.

---

## 1. Ba câu hỏi cốt lõi

Report được thiết kế để trả lời thẳng 3 câu. Đây là phần **nên đọc trước**:

| Câu hỏi | Field | Giá trị |
|---|---|---|
| **Ai đang control?** | `control` · `conviction` | Buyers/Sellers/Balanced/Contested · Strong/Moderate/Weak |
| **Thị trường đang làm gì?** | `action` · `range_state` | Driving/Pullback/Reclaiming/FailedPush/Absorbing/Rotating/Ranging · Expanding/Compressing/Steady |
| **Trạng thái tin được không?** | `state_quality` · `usable` | Clean/Dirty/Failed/Stuck/Indecisive · bool |

Các field còn lại (`risk_level`, `location_quality`, `reversal_warning`, `dirty_warning`,
`range_frame_tag`, `trend_bias`, `structure_tags`, `reason_tags`) là **hợp đồng theo spec doc**
(`rules/market-event-ai-dataset.md`) — vẫn giữ, nhưng được **derive lại cho khớp** với 3 câu trên,
và về giá trị đọc-nhanh thì **thấp hơn** bộ 3 câu (xem mục 5).

---

## 2. Nguồn: 7 block head model dự đoán

`build_report()` chỉ nhìn các nhãn categorical sau (argmax mỗi head):

| Head | Giá trị | Report dùng vào |
|---|---|---|
| `block_process` | CleanDrive, PullbackHeld, Reclaim, FailedPush, Absorption, DirtyRotation, BalancedAuction | **engine chính** — control, conviction, action, state_quality, dirty, floor cho risk |
| `direction` | Up, Down, Flat | trend_bias, phe của control, transition (reversal/continuation) |
| `reversal_risk` | Low, Medium, High | risk (bump lên High), reversal_warning |
| `range_rank` | Narrow, Medium, Wide, VeryWide | range_state (so với block trước) |
| `range_frame_tag` | Follow, Enough, Adapt | range_frame_tag, tag AdaptRange, reason ThinRange |
| `extension_rank` | NoMove, SmallMove, MediumMove, LargeMove | **`phase`** (magnitude cho Running) |
| `ended_bias` | StrongUpEnd, NearHighEnd, NeutralEnd, NearLowEnd, StrongDownEnd | **không dùng** (trùng block_process) |

`block_process` là **shape detector** — nó đã nuốt sẵn ngưỡng net (vd CleanDrive cần |net|≥8bps,
Absorption cần |net|≤4bps), nên phần lớn "câu chuyện block" nằm ở đây.

---

## 3. Từng field — ý nghĩa & cách derive

### 3.1 Bộ 3 câu

#### `control` — ai đang nắm
Map từ `block_process` (+ `direction` cho phe):

| block_process | control |
|---|---|
| CleanDrive / Reclaim / PullbackHeld | `Buyers` nếu direction=Up, `Sellers` nếu Down, `Contested` nếu Flat |
| BalancedAuction | `Balanced` (đấu giá cân bằng — không ai nắm) |
| FailedPush / Absorption / DirtyRotation | `Contested` (tốn lực, không ai thắng rõ) |

#### `conviction` — nắm mạnh cỡ nào
Chỉ từ `block_process`: CleanDrive/Reclaim → `Strong`; PullbackHeld → `Moderate`; còn lại → `Weak`.

#### `action` — đang làm gì (= shape thật, 1:1 với block_process)
| block_process | action |
|---|---|
| CleanDrive | `Driving` |
| PullbackHeld | `Pullback` |
| Reclaim | `Reclaiming` |
| FailedPush | `FailedPush` |
| Absorption | `Absorbing` |
| DirtyRotation | `Rotating` |
| BalancedAuction | `Ranging` |

> **Quan trọng:** một cú đảo hướng giữa 2 block (reversal) là **transition**, KHÔNG override `action`.
> `action` luôn là shape của chính block đó. Reversal được báo riêng qua `reversal_warning` + tag.

#### `range_state` — biên độ so với block trước
So `range_rank` block này với block trước (ordinal Narrow<Medium<Wide<VeryWide):
rộng ra → `Expanding`; hẹp lại → `Compressing`; bằng/không có block trước → `Steady`.

#### `state_quality` — chất lượng shape (fact về shape, không phải "có vào được không")
| block_process | state_quality | nghĩa |
|---|---|---|
| CleanDrive / PullbackHeld / Reclaim | `Clean` | đọc được, một chiều rõ |
| DirtyRotation | `Dirty` | xoay bẩn, nhiễu |
| FailedPush | `Failed` | đẩy rồi trả lại |
| Absorption | `Stuck` | tốn lực mà kẹt |
| BalancedAuction | `Indecisive` | giằng co, chưa ngã ngũ |

#### `usable` — gate hành động (bool)
`usable = (state_quality == "Clean") && (risk_level != "High")`.

> **Tách bạch với `state_quality`:** `state_quality=Clean` + `usable=false` đọc hợp lý = "shape
> sạch/đọc được, nhưng **chưa vào được** (vd risk cao)". Không còn kiểu `Clean(ok=false)` dính cục.

#### `phase` — nhịp lifecycle hiện tại (run / exhaust)
Đọc **rời rạc** (feed là chuỗi sự kiện ngắt quãng, hợp với trạng thái rời rạc hơn dải liên
tục), dựng **chỉ trên head đáng tin** (`extension_rank`/`direction`/`reversal_risk`/`range_rank`),
**không đụng `block_process`**. Xét theo thứ tự:

| phase | điều kiện | risk |
|---|---|---|
| **Running** | `extension ≥ MediumMove` **và** `direction ≠ Flat` (đi có lực — mạnh thì GIỮ, kể cả reversal cao) | theo hướng |
| **Rejected** | không chạy + `reversal_risk = High` (bị đạp / phản công) | Cao |
| **Stalling** | không chạy + `range_rank ∈ {Wide, VeryWide}` (quẫy rộng mà không tiến) | Trung bình |
| **Fading** | còn lại (im, trôi) | Thấp |

`Exhausted` cố ý **mổ ra 3 mode** (Rejected/Stalling/Fading) để đọc rủi ro dễ, mỗi cái một tier.
Phân bố thực (dataset): Fading ~70% (baseline im, đúng bản chất tick data), Running ~22%,
Rejected ~5%, Stalling ~3%. Hiện là **rules** (chưa head) — nếu live thấy whipsaw mới cân nhắc
train head `phase` cho mượt. Không nói "sắp"; `Rejected`/`reversal_warning` chỉ báo *khi* xác nhận.

### 3.2 Field theo spec doc (đã derive lại cho coherent)

#### `trend_bias`
= `direction` của block gần nhất (Up/Down/Flat).

#### `risk_level` — **derived**, không phải head thô
Lấy **sàn theo `state_quality`** rồi `max` với head `reversal_risk`:

| state_quality | sàn risk |
|---|---|
| Failed / Dirty | **High** (shape hỏng thì rủi ro cao, head không kéo xuống được) |
| Stuck | **Medium** — nhưng nếu `extension_rank`… *(hiện không xét; xem mục 6)* |
| Clean / Indecisive | không sàn — để `reversal_risk` head nói |

`risk_level = max(sàn, reversal_risk_head)`. → hết cảnh `risk=Low` khi `state_quality=Failed/Dirty`.

#### `reversal_warning` — **giảm nhiễu**
```
reversal_confirmed = (block trước ngược hướng block này) AND (block này là CleanDrive hoặc Reclaim)
reversal_warning   = reversal_confirmed OR (reversal_risk head == High)
```
→ chỉ báo reversal khi có **drive ngược có xác nhận**, không phải mọi cú flip hướng (vốn là nhiễu).

#### `dirty_warning`
`true` khi `block_process ∈ {DirtyRotation, FailedPush}`.

#### `range_frame_tag`
= `range_frame_tag` head thô (Follow/Enough/Adapt).

#### `location_quality` — rollup thô (giá trị thấp, xem mục 5)
| điều kiện | loc |
|---|---|
| state_quality=Clean & risk=Low | `Good` |
| state_quality ∈ {Clean, Indecisive} | `Watch` |
| còn lại (Failed/Dirty/Stuck) | `Bad` |

#### `block_process`
Nhãn shape thô, giữ nguyên trong report để **trace + eval** (bin `review` chấm accuracy dùng field này).

### 3.3 Tags

`structure_tags` (đọc nhanh, gồm shape + transition):
`block_process` (tên shape) · `AdaptRange` (range=Adapt) · `DirtyPath` (dirty) ·
`Continuation` (cùng hướng block trước) · `Reversal` (reversal_confirmed) ·
`Expansion` / `Compression` (range_state).

`reason_tags` (lý do cảnh báo):
`HighRisk` (risk=High) · `DirtyPath` (dirty) · `EffortAbsorbed` (Absorption) · `ThinRange` (range=Follow).

---

## 4. Ví dụ đọc

```
StructureReport {
    control: "Buyers", conviction: "Strong",     // 1. buyers đang nắm, mạnh
    action: "Driving", range_state: "Compressing", // 2. đang drive, biên co lại
    state_quality: "Clean", usable: false,        // 3. shape sạch NHƯNG chưa vào được
    // vì:
    risk_level: "High", reversal_warning: true,   // là drive ngược (reversal) rủi ro cao
    structure_tags: ["CleanDrive", "Reversal", "Compression"],
    block_process: "CleanDrive", ...
}
```
Đọc: "Buyers drive sạch & mạnh, nhưng đây là cú **đảo chiều** rủi ro cao → **đọc được nhưng chưa vào**."

---

## 5. Đọc field theo thứ tự ưu tiên

1. **Core (tin cậy nhất):** `control` + `action` + `state_quality` → đủ làm mode selector
   (hold / watch / avoid / tighten).
2. **Bổ trợ:** `conviction`, `range_state`, `usable`.
3. **Thứ cấp / debug (đã coherent nhưng giá trị thấp hơn):** `risk_level`, `location_quality`,
   `reversal_warning`. Dùng để cross-check, đừng làm tín hiệu chính.

---

## 6. `ended_bias` bỏ; `extension_rank` giờ dùng cho `phase`

- **`ended_bias`** (block đóng ở đâu trong range) — vẫn **cố ý bỏ**: ~80% trùng `block_process`
  (Reclaim vốn đóng mạnh, FailedPush vốn đóng yếu…), không thêm trục đáng kể.
- **`extension_rank`** (độ lớn |net|, **không có hướng**, khác `range_rank` = biên swing) — trước
  bỏ vì marginal cho `conviction`, nhưng **giờ được dùng làm thước magnitude cho `phase`** (Running
  cần `extension ≥ MediumMove`). Nó reliable (macro 0.62) và là trục biên độ độc lập với hướng.

Lưu ý cân bằng đầu vào cho lifecycle: `block_process` (0.41, yếu) KHÔNG nuôi `phase`; `phase` tựa
hẳn vào cụm head khỏe (`extension`/`direction`/`reversal_risk`/`range_rank` = 0.62–0.80).

---

## 7. Streaming & cadence

- `StructureModel::read(window)` — đọc một cửa sổ step (bội số của `block_size`) → 1 report.
- `Session` (trong lib): đẩy từng step, emit report **mỗi khi đủ 1 block** (every-8).
- hft-bot có bản riêng `StructureInference::observe` — **continuous** (emit mỗi step, đọc suffix
  bội-8 lớn nhất). Đây là divergence có chủ đích của hft-bot, nằm NGOÀI crate này.

Model chỉ nhận `seq_len % block_size == 0`, nên cửa sổ luôn là bội của 8 → mọi block trong report
đều là block đầy đủ.
