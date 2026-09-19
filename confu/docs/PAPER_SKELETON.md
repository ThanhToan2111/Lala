# Paper Skeleton — ConFu++

> **Legacy note:** skeleton này thuộc paper lịch sử trước v6.1. Skeleton submission hiện tại là `paper/draft/PAPER_SKELETON.md` và phải tuân theo `AGENT.md — ConFu++ v6.1_ Paper Construction and Calibration Confirmation.md`.

Working title candidates:

1. **"Alignment Is Not Dependence: Measuring and Reducing Unimodal Shortcuts in
   Higher-Order Multimodal Fusion"**  ← recommended (states the finding, not a method name)
2. "The More, the Merrier — But Is It? Auditing Higher-Order Multimodal Interaction"
3. "ConFu++: Complementarity-Aware Higher-Order Multimodal Representation Learning"

Target venues: NeurIPS / ICML / CVPR (main track or workshop first). All numbers below are REAL,
from this repository's runs (see EXPERIMENTS.md for full provenance: seeds, configs, git states).

---

## Abstract (draft)

Higher-order multimodal methods fuse representations of three or more modalities and align them
contrastively, on the premise that fused representations capture information unavailable to any
single modality. We show this premise is unfounded: across three established benchmarks
(AV-MNIST, CMU-MOSI, UR-FUNNY), fused interaction representations that are active, predictive,
high-rank and permutation-sensitive turn out to be almost entirely predictable from a single
modality (R² up to 0.989). We formalize the gap — fusion ≠ dependence ≠ complementarity — and
introduce an operational diagnostic suite (zero/shuffle distinction, per-modality corruption,
conditional probes, correction–regression, predictability probes). We then propose S1, a
training-time adversarial de-shortcutting objective that penalizes predictability of the
interaction from each single modality while leaving joint predictability free. S1 is the only
objective among those studied that significantly reduces the shortcut (paired p≈0.011) at zero
accuracy cost, and — critically — it stays honest when there is nothing to find: it fabricates no
dependence on negative controls. Finally, on a new controlled synthetic benchmark with
ground-truth third-order synergy (y = a⊕b⊕c), order-restricted models sit at chance while an
order-matched interaction solves the task with perfectly balanced dependence (+49.9 pp, 5/5
seeds), completing the causal chain: shortcuts arise from information absence, not from any
inability of explicit higher-order interactions to learn.

---

## 1. Introduction

- Hook: multimodal fusion is assumed to combine information; nobody checks whether it does.
- Figure 1 (teaser): the AV-MNIST paradox — r12 healthy by every standard metric, yet
  R²(r1→r12)=0.989. Two-panel schematic: "joint computation" vs "joint information".
- Contributions list (the 4 defensible items from docs/RELATED_WORK.md §7).

## 2. Related Work

Structure as in docs/RELATED_WORK.md: contrastive alignment lineage; fusion architectures;
modality dominance; PID/synergy; adversarial debiasing; benchmarks.
Careful tone: "we operationalize", never "first" (until survey verified).

## 3. The Alignment–Dependence Gap

- §3.1 Formalism: fused vs interaction vs complementary representations; the three required
  properties (activity, dependence, conditional utility). Definitions reused verbatim from the
  spec (§4 of AGENT.md — ConFu++).
- §3.2 The diagnostic suite (the paper's methodological toolkit):
  - zero vs shuffle distinction (Fig. 2),
  - per-modality corruption drops D1, D2, balance B12,
  - conditional linear/nonlinear probe gains (capacity-matched),
  - correction–regression counts,
  - predictability probes + the shortcut criterion R²(q1) ≫ R²(q2), R²(q1) ≈ R²(q12).
- §3.3 Theorem-free but rigorous: why contrastive alignment and task CE are BOTH satisfied by
  unimodal shortcuts (one-paragraph argument + the three escape routes from the spec §2).

## 4. Auditing Existing Benchmarks (negative results, the honest core)

Table 1 — main audit (5 seeds each, all from repo runs):

| Benchmark | Finding | Key numbers |
|---|---|---|
| AV-MNIST | image shortcut | R²(r1→r12)=0.988/0.031; audio shuffle drop +0.004 ± 0.027 |
| CMU-MOSI | text dominance, redundant pairs | text-only 67.7%; pair conditional gains −0.96/−1.43/−2.27 pp; R² up to 0.953 |
| UR-FUNNY | redundant pairs, no linear accessibility | ConFu 64.46%; pair gains ≈ −0.66/+0.19/−0.76; R² ≈ 0.97 |

- §4.1 each benchmark: protocol, numbers, interpretation, why no objective can fix it there.
- Fig. 3: predictability probe bars per benchmark.

## 5. S1: Adversarial De-Shortcutting

- §5.1 Mechanism: alternating minimax; EMA-standardized target; hinged penalty ReLU(τ−MSE);
  adversaries from singles only, never the pair; factor variance floor (S2).
  Pseudocode (Algorithm 1) — mirrors src/experiments/av_mnist/complementarity.py.
- §5.2 Why the hinge (avoids r12→noise collapse; τ = tolerated redundancy bandwidth).
- §5.3 Training-only cost: +263k params at train, zero at inference.

Table 2 — AV-MNIST paired confirmatory (5 seeds, vs same-codepath ablation):

| Metric | λ=0 | S1 | Paired Δ |
|---|---|---|---|
| accuracy | 69.962 ± 0.594 | 70.046 ± 0.476 | +0.084 (ns) |
| R²(r1→r12) | 0.989 ± 0.002 | 0.682 ± 0.155 | −0.307 (t=−4.43, p≈.011) |
| audio shuffle drop | +0.084 | +0.236 | +0.152 (p≈.059) |

- §5.4 Honesty controls: synthetic Config B (no synergy: nothing fabricated, R²→0.01);
  order-2 control (distractor shuffle drop exactly 0.000).
- §5.5 Negative scope statement: S1 reduces the shortcut; it cannot create conditional utility
  absent from the data. This is a feature (the method knows when there is nothing to find).
- §5.6 Mechanistic boundary (from the 30-epoch analysis): the residual shortcut on AV-MNIST is an
  adversary gap — the online adversary equilibrates below the hinge band while the converged
  offline probe reads R²=0.84; closing it trades task information on benchmarks where the dominant
  modality IS the label. Val-acc epoch-0 selection is optimal (val decays monotonically).

## 6. The Controlled Order Benchmark (positive result)

- §6.1 Design: ground-truth latents a,b,c; observable prototypes; y = a⊕b⊕c (order-3) and
  y = a⊕b with distractor (order-2 control); proper probe train/test splits; 5 seeds.
- Table 3:

| Model | Orders | Accuracy |
|---|---|---|
| Additive | 1 | 0.496 ± 0.006 |
| Pairs-only | 1+2 | 0.501 ± 0.010 |
| Concat-MLP | joint | 1.000 |
| **Full (order 1+2+3)** | 1+2+3 | **1.000 ± 0.000** |

- §6.2 The readout finding: pair representations contain the bits (concat-pair probe 0.97) yet
  the sum-aggregated order-2 model is at chance → order-matched interaction terms are necessary
  as *computation*, not just as capacity.
- §6.3 Dependence balance: all shuffle drops +0.505 (maximal, balanced).
- §6.4 Config A/B/C suite: S1 sound (A), honest (B), and the redesigned observable-flag C.
- §6.5 M-modality ladder (M=3..6): order-restricted at chance at every M; order-matched full model
  solves every M (1.0000, balanced +0.49-0.50 shuffle drops); concat-MLP cracks at M=6 (0.9975).
- §6.6 Methodology pitfall (novel, citable): post-aggregation LayerNorm silently leaks higher-order
  signal — additive+final-LN reaches 0.75 on 3-bit parity vs exactly 0.50 without. Any
  "low-order" ablation ending in LN is invalid.
- Fig. 4: schematic of the synthetic world + probe bars + shuffle drops.
- Fig. 5: ladder table→figure (accuracy vs M per variant) + adversary-gap dynamics panel
  (in-training adversary R² plateau vs offline probe 0.84, explaining S1's residual).

## 7. Discussion

- The big claim, stated exactly: shortcuts in current benchmarks are a *data property*
  (information absence/dominance); explicit higher-order interactions learn perfectly when the
  information exists; therefore the field needs (a) dependence audits in every fusion paper,
  (b) benchmarks with known interaction structure.
- Limitations: S1 residual R² 0.68 on AV-MNIST (hinge tolerance τ); conditional utility on
  natural data still unproven; Bird-MML pending data release.
- Ethics/broader impact: fusion auditing matters for deployment where a "multimodal" model may
  silently ignore a modality (medical, accessibility).

## 8. Reproducibility statement

Seeds, paired statistics, capacity matching, train-only parameter accounting, git-state logging —
all already enforced by the repo protocol (spec §98-99). Data: MultiBench public; synthetic
benchmark released as code.

---

## Missing before submission (ranked)

1. **Systematic related-work verification** (docs/RELATED_WORK.md is a draft from knowledge).
2. **Bird-MML natural positive** — the one experiment that turns this from "audit + synthetic"
   into "method wins on real data". Blocked on dataset release; pipeline prep is the next task.
3. Order-3 benchmark: M-modality ladder + noise ladder on the order-3 term (cheap, strengthens §6).
4. S1 on order-3 with pair adversaries ablation (exists; needs the paper-style table).
5. Figures 1-4 (all numbers exist; only plotting remains).

## Appendix (planned)

A. Full protocol details + configs (from configs/*.yaml).
B. The failed objectives (P1 utility at length, P2 ranking, S4 emphasis) — negative results
   appendix; EXPERIMENTS.md is already structured for this.
C. All per-seed numbers (results/ JSONs).
D. Unit-test suite description.
