# ConFu++ — Tài liệu giải trình chuyên sâu (bản tiếng Việt)

> **Legacy note:** đây là bản paper lịch sử của các vòng v4/v5 và không phải source of truth cho v6.1. Bản submission hiện tại được xây từ evidence freeze ở `paper/draft/`, `results/V60_FINAL_REPORT.md`, và `PAPER_EXPERIMENT_FREEZE.md`; các claim cũ về Bird-MML/S1 không được dùng cho narrative v6.1 nếu không có artifact tương ứng trong registry mới.

**Tựa paper:** *Alignment Is Not Dependence: Measuring and Reducing Unimodal Shortcuts in Higher-Order Multimodal Fusion*
**File chính thức:** `docs/paper/ConFuPlusPlus_CVPR2026.pdf` · **Toàn bộ provenance:** `EXPERIMENTS.md`

---

## 1. Bài toán theo đuổi

### 1.1. Bối cảnh
Học biểu diễn đa phương thức (multimodal representation learning) là bài toán: cho dữ liệu gồm nhiều kênh (ảnh, âm thanh, văn bản...), học một biểu diễn chung tốt cho tác vụ xuống dòng (phân loại, retrieval...). Khi có **từ 3 modality trở lên**, câu hỏi mới xuất hiện: làm sao nắm bắt **tương tác bậc cao (higher-order interaction)** — thông tin chỉ tồn tại khi kết hợp *nhiều* modality cùng lúc, ví dụ cảm xúc mỉa mai chỉ nhận ra khi nghe ngữ điệu VÀ đọc lời nói VÀ nhìn nét mặt.

### 1.2. Paper gốc (ConFu, CVPR 2026) làm gì
ConFu ("The More, the Merrier") đề xuất **contrastive fusion**: encode riêng từng modality `r_i = E_i(X_i)`, trộn từng cặp `r12 = F(r1, r2)`, rồi dùng InfoNCE kéo `r12 ↔ r3` (fusion của 2 modality về phía modality thứ 3). Trực giác: fusion bị buộc phải "tương thích" với cả 3 → chứa thông tin bậc cao.

### 1.3. Vấn đề paper này phát hiện
**Không có gì trong objective của ConFu (hay CLIP/Symile...) buộc `r12` thực sự phụ thuộc vào cả hai modality.** Một lời giải "gian lận" hoàn toàn hợp lệ về mặt tối ưu:

```
r12 ≈ f(r1)     ← chỉ sao chép thông tin từ modality mạnh
```

Biểu diễn "gian lận" này vẫn: variance cao ✓, effective rank cao ✓, predict nhãn tốt ✓, nhạy với xáo trộn ✓, thỏa contrastive alignment ✓ — **nhưng gần như không chứa gì từ modality 2**. Toàn bộ luận điểm "fusion tích hợp đa phương thức" sụp đổ một cách âm thầm.

**Định lý thực nghiệm trung tâm của paper:**

> **Joint computation ≠ Joint information** (tính chung ≠ chứa thông tin chung)
> **Higher-order alignment ≠ Higher-order synergy** (căn chỉnh bậc cao ≠ synergy bậc cao)

---

## 2. Câu chuyện (story) của paper

Paper được kể như một chuỗi nhân quả 5 mắt xích:

1. **Phát hiện**: interaction representation trông khỏe nhưng là hàm một-modality (nghịch lý AV-MNIST: R²(image→r12) = 0.989).
2. **Định nghĩa**: tách bạch 4 khái niệm thường bị dùng lẫn nhau — *fusion, dependence, complementarity, synergy* — và chế tạo bộ diagnostics đo từng cái.
3. **Khảo sát**: 3 benchmark chuẩn (AV-MNIST, CMU-MOSI, UR-FUNNY) đều fail — shortcut là **hệ thống**, không phải sự cố riêng.
4. **Cơ chế sửa**: ConFu++ (utility + S1 adversarial) — nhưng **trung thực**: không bịa dependence khi dữ liệu không có (negative controls).
5. **Chứng minh nhân quả**: synthetic benchmark có ground truth (khi synergy tồn tại → interaction khớp bậc học trọn vẹn) + benchmark tự nhiên có headroom thật (Bird-MML → ConFu++ thu +5.76pp, ConFu gốc thu 0).

**Thông điệp kết**: shortcut trên các benchmark hiện có là **thuộc tính của dữ liệu** (thông tin vắng mặt/bị áp đảo), KHÔNG phải lỗi tối ưu của kiến trúc — và cách duy nhất tiến lên là: đo được nó + tối ưu representation-level + benchmark có cấu trúc tương tác đã biết.

---

## 3. Research Questions

| # | Câu hỏi nghiên cứu | Trả lời trong paper |
|---|---|---|
| RQ1 | Fused representation có thực sự phụ thuộc vào mọi modality đầu vào không? | **Không** — trên cả 3 benchmark chuẩn (§5) |
| RQ2 | Làm sao ĐO được dependence/complementarity một cách operational? | Diagnostic suite 5 thành phần (§3) |
| RQ3 | Có thể tối ưu để giảm shortcut mà không phá task, không bịa dependence không? | **Có** — S1 (p≈0.011, accuracy parity, honest trên controls) (§4, §6) |
| RQ4 | Khi synergy bậc cao THỰC SỰ tồn tại, kiến trúc interaction có học được không? | **Có, hoàn hảo** — order-matched interaction 100%, order-thấp = chance (§6) |
| RQ5 | Trên dữ liệu tự nhiên có headroom thật, objective mới có thắng baseline không? | **Có** — +5.76pp, p<0.001 trên Bird-MML (§7) |
| RQ6 | Gap này có riêng của ConFu không? | **Không** — CLIP-lineage cũng shortcut y hệt (§5) |

---

## 4. Các phương pháp trước đó (Related Work)

### 4.1. Contrastive alignment (dòng chính bị audit)
- **CLIP** (Radford 2021): InfoNCE theo cặp ảnh–text. Không có fusion → câu hỏi dependence không đặt ra ở cấp fusion.
- **Symile** (Meyer 2024): total-correlation objective cho ≥3 modality. Gần nhất về mặt contrastive; TC lower bound vẫn không buộc nội dung của fusion.
- **TriCLIP, ImageBind**: neo nhiều modality về 1 anchor.
- **ConFu** (2026): baseline trực tiếp của ta — contrastive fusion bậc cao.

### 4.2. Kiến trúc fusion (tối ưu *capacity*, ta đo *dependence*)
- **MFN** (gated multiplicative), **LMF** (low-rank tensor fusion — tổ tiên trực tiếp của low-rank interaction ta dùng), **MUL-T** (cross-attention), **MISA** (disentanglement). Điểm chung: không ai đo fusion có *phụ thuộc* vào đầu vào không.

### 4.3. Modality dominance (gần nhất về bài toán)
- **Wang et al. 2020**: multimodal nets overfit, gradient blending.
- **OGM-GE** (Peng 2022): giảm gradient modality thắng thế — *ta chứng minh thực nghiệm nó không đủ* (80.68% vs plain 80.88%, ns trên Bird-MML).
- **Underspecification** (D'Amour 2020): khung tổng quát; shortcut criterion của ta là detector cụ thể cho fusion.

### 4.4. Lý thuyết thông tin (anchor từ vựng)
- **PID** (Williams–Beer 2010): unique/redundant/synergistic information. Ta **tuyên bố rõ không claim đại lượng PID** — mọi metric là operational approximation.
- **MultiViz** (Liang 2022): interaction importance bằng perturbation — gần nhất về methodology; ta rẻ hơn (gate experiments) và bổ sung predictability probes.

### 4.5. Adversarial/debiasing (họ kỹ thuật của S1)
- **DANN** (Ganin 2015), **Learning Not to Learn** (Kim 2019): loại bỏ thông tin không mong muốn bằng adversary. **S1 khác ở TARGET**: cấm predictability của interaction từ *từng modality riêng lẻ*, nhưng cho phép từ cặp — asymmetry này là cấu hình mới.
- **VICReg** (2022): nguồn của anti-collapse terms.

---

## 5. Phương pháp đề xuất (ConFu++)

### 5.1. Kiến trúc nền (giữ nguyên từ baseline đã validate)
```
r1 = E1(X1), r2 = E2(X2)                        # encoders
r12 = Wo · (LN(U1·r1) ⊙ LN(U2·r2)) / √R         # low-rank multiplicative interaction (R=64)
c_base = LN(LN(r1) + LN(r2))                    # lower-order
c_full = LN(c_base + σ(a)·LN(r12))              # gated composition
```

### 5.2. Ba thành phần mới

**(a) Conditional utility loss** — ép full vượt base theo từng mẫu:
```
L_utility = ReLU(CE_full − stopgrad(CE_base) + m),   λ=0.2, m=0
```

**(b) S1 — Adversarial de-shortcutting** (cốt lõi):
```
t = chuẩn hóa(r12) bằng EMA (stop-grad)
Adversary step:   q1, q2 học dự đoán t từ r1 RIÊNG / r2 RIÊNG  (gradient chỉ vào q)
Generator step:   L_S1 = Σ_i ReLU(τ − MSE(q_i(ri), t))          (gradient chỉ vào r12)
```
- Vì t đã chuẩn hóa: MSE = 1 − R² của adversary → hinge τ phạt khi R² > 1−τ, rồi **tự tắt** → không bao giờ ép r12 thành noise.
- **Không có adversary nào nhìn cặp (r1,r2)** — joint predictability là hợp lệ, được giữ nguyên.
- Factor variance floor chặn escape route "u₂ collapse thành hằng số".
- Adversary **training-only**: +263k params lúc train, **0 params lúc inference**.

**(c) Per-pair utility** (bổ sung vòng 2): mỗi z_ij có margin riêng so với cặp constituents của nó → pair term tự mang conditional utility.

### 5.3. Nguyên tắc "Honesty by construction"
Trên dữ liệu không có complementarity, output ĐÚNG của method là báo "không có dependence" — fabricate dependence trên negative control được coi là **thất bại của method**, không phải chiến thắng.

---

## 6. Metrics (thước đo)

| Metric | Đo gì | Công thức/Cách |
|---|---|---|
| `R²(qi)` | predictability của r12 từ modality i | MLP probe hội tụ, R² |
| `D_i` (shuffle drop) | dependence vào modality i | Acc(normal) − Acc(shuffle modality i trước fusion) |
| Zero drop | necessity | Acc − Acc(r12=0) |
| Gain12 (conditional gain) | conditional utility | Probe[r1,r2,r12] − Probe[r1,r2] |
| Correction/regression | sửa lỗi vs gây lỗi | đếm per-sample |
| B12 (balance) | cân bằng dependence | 2·min(D1,D2)/(D1+D2) |
| Effective rank/variance | sức khỏe (chống collapse) | SVD entropy, VICReg-style |

**Chuẩn thống kê**: 5 seeds, mean ± sample std, paired t-test, capacity-matched probes, hyperparameter chọn bằng validation only, exploratory vs confirmatory tách bạch.

---

## 7. Thực nghiệm và số liệu

### 7.1. Audit 3 benchmark chuẩn (5 seeds)

| Benchmark | Dominant | R²(single→interaction) | Pair conditional gains |
|---|---|---|---|
| AV-MNIST | image | **0.989** | +0.02pp |
| CMU-MOSI | text | 0.953 | −0.96 / −1.43 / −2.27pp |
| UR-FUNNY | text | 0.970 | −0.66 / +0.19 / −0.76pp |

→ Cả 3 đều fail dependence hoặc utility gate.

### 7.2. S1 trên AV-MNIST (5 seeds, paired vs ablation cùng code)

| Metric | Không S1 | Có S1 | Paired Δ |
|---|---|---|---|
| Accuracy | 69.96 ± 0.59% | 70.05 ± 0.48% | +0.08 (ns) |
| **R²(image→r12)** | 0.989 ± 0.002 | **0.682 ± 0.155** | **−0.307 (t=−4.43, p≈0.011)** |
| Audio shuffle drop | +0.084 | +0.236 | +0.152 (p≈0.059) |

### 7.3. Synthetic ground truth (5 seeds)

**Order-3** (`y = a⊕b⊕c`, độc lập mọi single & pair):

| Model | Orders | Accuracy | Min shuffle drop |
|---|---|---|---|
| Additive | 1 | 0.496 ± 0.006 | ~0 |
| Pairs-only | 1+2 | 0.501 ± 0.010 | ~0 |
| Concat-MLP | joint | 1.000 | +0.50 |
| **Full (1+2+3)** | 1+2+3 | **1.0000 ± 0.0000** | **+0.50 cân bằng cả 3** |

- M-ladder M=3..6: order-restricted = chance ở MỌI M; full = 1.0000 ở mọi M; concat-MLP nứt ở M=6 (0.9975).
- Honesty: Config B (no synergy) + order-2 control (distractor) → S1 không bịa gì (drops ≈ 0.000).

### 7.4. Bird-MML (benchmark tự nhiên có headroom thật)

**Audit** (frozen ResNet50/wav2vec2/MiniLM, 71 loài × 70,796 mẫu): image 64.95%, audio 1.43% (null), text 63.74%; **image+text 80.89% → headroom +15.94pp** (sau de-leak: +4.78pp vẫn dương).

**Ma trận chính** (5 seeds, paired):

| Method | probe full | Conditional gain | R²(text→z13) |
|---|---|---|---|
| ConFu gốc | 76.90 ± 0.65% | +0.19pp | 0.59 |
| +S1 | 76.58% | −0.36 | 0.69 |
| +Utility | 83.01 ± 0.13% | +1.07 | **0.44** |
| +U+S1 | 82.66 ± 0.37% | +0.77 | 0.50 |
| **+PairU** (per-pair) | 82.84 ± 0.39% | +1.17 | pair gain **+0.39 ± 0.19pp** |
| **Full stack (U+S1+PairU)** | **83.17 ± 0.33%** | +1.23 | 0.50 |

**ConFu++ vs ConFu gốc: +6.27 ± 0.71pp, t=19.77, p<0.001** — vượt trần raw-feature (81.01%).

**Fair-play controls đầy đủ**:
- CLIP-lineage control: shortcut y hệt (R² 0.99) → gap không riêng ConFu.
- OGM-GE baseline: 80.68% (ns vs plain) → ConFu++ hơn 1.8pp.
- BLIP-only de-leak: headroom +4.78pp vẫn dương.
- AST audio re-audit: audio null robust qua 2 domain extractor.
- τ/λ ablations: phẳng.
- Missing-modality: degrade graceful (img −0.99, txt −0.71, aud +0.04pp).
- Hard-sample: corrections không tập trung ở mẫu khó (kết quả âm tính trung thực).

**End-to-end** (3 arms × 3 seeds, train cả encoders): mọi arm hội tụ về **text dominance** (~72%, shuffle text +4.9pp, image ≈0) — *modality abandonment*. Causal control: scrub tên loài khỏi caption (0% leakage) → dominance **không đổi** → đây là thuộc tính nội tại của dữ liệu, không phải leakage artifact.

### 7.5. Hai methodology pitfalls mới (contributions phụ)

1. **Post-aggregation LayerNorm là "higher-order term ẩn"**: additive + LN cuối = 0.75 trên 3-bit parity (đáng lẽ 0.50) → mọi ablation "low-order" kết thúc bằng LN đều vô hiệu. Phát hiện khi ladder v1 cho số liệu "bất khả thi lý thuyết" (0.874).
2. **Adversary gap**: online adversary cân bằng ở R²≈0.3-0.4, offline probe đọc được 0.84 → phải đo bằng probe hội tụ; gap thu hẹp được bằng adversary mạnh hơn (0.03-0.07) không phá task.

---

## 8. Ưu điểm của ConFu++ so với các phương pháp trước

| Tiêu chí | ConFu / CLIP / Symile | OGM-GE / modality dropout | MultiViz | **ConFu++** |
|---|---|---|---|---|
| Đo dependence thật sự | ✗ | ✗ (gradient proxy) | △ (perturbation) | **✓ predictability + shuffle + zero** |
| Sửa được shortcut | ✗ | ✗ (đo được: ns trên Bird-MML) | — (chỉ đo) | **✓ p≈0.011, accuracy parity** |
| Không bịa dependence trên negative control | — | — | — | **✓ chứng minh trên 3 controls** |
| Chi phí inference | — | — | — | **+0 tham số** |
| Biết khi nào data không có gì để học | ✗ | ✗ | ✗ | **✓ (honesty gates)** |
| Ground-truth validation | ✗ | ✗ | ✗ | **✓ synthetic order benchmark** |

---

## 9. Đã giải quyết được gì / còn mở gì

### ✅ Đã giải quyết
1. **Chứng minh gap tồn tại hệ thống** (3 benchmark + CLIP-lineage control).
2. **Bộ đo operational** đầy đủ, rẻ, tái dùng được.
3. **S1**: objective đầu tiên giảm shortcut có ý nghĩa thống kê, không tốn accuracy, trung thực.
4. **Chứng minh nhân quả**: shortcut = thuộc tính dữ liệu, không phải lỗi kiến trúc (synthetic order benchmark: +49.9pp).
5. **Kết quả dương tính tự nhiên đầu tiên**: Bird-MML +5.76~6.27pp (p<0.001).
6. **Hai pitfall methodology** mới có giá trị cộng đồng.

### 🔓 Còn mở (ghi rõ trong Limitations)
1. Full 149 loài khi Zenodo sửa part-ah (pipeline sẵn, 1 lệnh chạy).
2. Per-pair gain dương chỉ đạt được khi PairU tách riêng (interference với full-head utility — câu hỏi mở).
3. E2E: accuracy flat do text dominance nội tại — cần task mà text provably không đủ (benchmark mới).
4. Verify citations thủ công + liên hệ authors (việc của người).

---

## 10. Tái lập (Reproducibility)

- Code: `src/experiments/birds/` (extract/audit/confu_frozen/e2e_train/ogm_frozen/analysis), `src/experiments/synthetic/` (synergy_experiment, order3, orderM), `src/modules/models/complementarity.py`.
- Unit tests: `tests/test_complementarity.py` (9/9 pass — gradient isolation của minimax, hinge, stop-grad).
- Logs đầy đủ mọi run (kể cả failed): `EXPERIMENTS.md`; JSON artifacts: `results/`.
- Data: MultiBench (public) + Bird-MML Zenodo 18920487 (subset documented) + synthetic (code-generated).
- Compute: frozen ~phút/seed/GPU; e2e ~1.5h/seed trên RTX 5000 Ada; synthetic ~phút CPU.
