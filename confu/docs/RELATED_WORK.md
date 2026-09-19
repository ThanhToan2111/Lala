# Related Work Survey — ConFu++ positioning

Status: DRAFT v1 (2026-09-17). Written from knowledge; **systematic verification still required**
(spec §96 forbids any "first" claim without it). Each entry includes how we position relative to it.

## 1. Contrastive multimodal alignment (the lineage we extend)

| Work | Core idea | Relation to us |
|---|---|---|
| CLIP (Radford et al. 2021) | pairwise InfoNCE, image–text | the pairwise alignment paradigm ConFu critiques for >2 modalities |
| Symile (Meyer et al., ICML 2024) | total-correlation objective for 3+ modalities | closest contrastive cousin; TC lower bound still does not require *interaction dependence* — our predictability probes would apply directly to their fusion |
| TriCLIP (2024) | image–text–audio tri-modal | same pair-alignment family as ConFu |
| ImageBind (Girdhar 2023), LanguageBind | bind modalities via one anchor | alignment, not interaction utility |
| **ConFu (Koutoupis et al. 2025, CVPR 2026)** | contrastive fusion of pair representations with remaining modality | our direct baseline and object of study |

Claim we CAN make: ConFu-style higher-order *alignment* permits unimodal shortcuts —
demonstrated with predictability probes (R²(r1→r12)=0.989) and per-modality shuffles on AV-MNIST.
Claim we must NOT make: "first to identify alignment/dependence gap" without checking Symile TC
analyses and MultiBench-era critiques.

## 2. Multimodal fusion architectures with multiplicative interactions

| Work | Core idea | Relation |
|---|---|---|
| MFN (Zadeh et al. 2018) | gated multiplicative view-to-view interactions | our LowRankInteraction is in this family |
| LMF (Liu et al. 2018) | low-rank tensor fusion | direct ancestor of our rank-R product |
| MUL-T (Tsai et al. 2019) | cross-modal attention transformers | architecture axis we deliberately gate OFF until complementarity is proven |
| MISA (2020), TFR-Net | disentangled/shared+private spaces | orthogonal separation objective; no conditional-utility guarantee |

Positioning: all of these optimize *capacity to fuse*; none measures whether the fused
representation *depends* on its inputs. Our contribution is measurement + objectives, not a new
fusion block.

## 3. Modality dominance, competition and shortcuts (closest problem literature)

| Work | Core idea | Relation |
|---|---|---|
| Wang et al. 2020, "What Makes Training Multi-Modal Networks Hard" | multimodal nets overfit; gradient blending | diagnoses joint-training pathology; we diagnose the *representation content* itself |
| Peng et al. 2022, "Balanced Multimodal Learning via On-the-fly Gradient Modulation" (OGM) | slow down the dominant modality's gradients | task-level pressure — same family as our failed P2; our predictability result predicts why it can't create genuine dependence |
| Wu et al. 2022, modality-wise loss imbalance | rebalance per-modality learning | same note as OGM |
| D'Amour et al. 2020, "Underspecification" | many solutions fit the loss, differ under stress | our shortcut criterion (R²(q1)≈R²(q12)) is an underspecification detector for fusion |
| Gat et al. 2020; modality dropout variants | random modality removal for robustness | robustness ≠ dependence; we show shuffle-drop and dropout-style training do not imply complementarity |

Positioning: this literature *reacts to* modality dominance at the loss/gradient level. We provide
the first (to our knowledge — verify) **representation-level proof** that a fused embedding can be
active, high-rank, predictive and permutation-sensitive while being a unimodal function, plus a
training-time mechanism (S1) that provably reduces it.

## 4. Synergy / redundancy / partial information decomposition (the theory anchor)

| Work | Core idea | Relation |
|---|---|---|
| Williams & Beer 2010 (PID); Bertschinger et al. 2014 | decompose mutual information into unique/redundant/synergistic | our conceptual vocabulary; we are explicit that our metrics are *operational approximations*, not PID |
| Liang et al. 2022, "MultiViz" (EMAP) | quantify unimodal/additive/interaction importance via perturbation | closest methodology; their interaction importance is perturbation-based — our zero-vs-shuffle + correction/regression analysis is complementary and cheaper to gate experiments |
| Liang et al. 2023+, "multimodal interactions" quantification | dataset-level interaction measures | our benchmark audit protocol (§25-26 of spec) is in this spirit |
| VICReg (Bardes et al. 2022), Barlow Twins | variance/covariance anti-collapse | our anti-collapse terms; we show health ≠ synergy (rank/variance rise without utility) |

Positioning: we do NOT claim an information-theoretic quantity. All our terms are operational
("empirical complementarity", "shortcut criterion"), stated explicitly in the spec (§16, §22).

## 5. Adversarial / debiasing representation learning (S1's cousins)

| Work | Core idea | Relation |
|---|---|---|
| Ganin & Lempitsky 2015 (DANN) | gradient reversal for domain invariance | adversarial removal technique family |
| Kim et al. 2019, "Learning Not to Learn" | adversarially remove bias-predictive info | closest mechanism cousin |
| β-VAE / TCVAE (Chen et al. 2018) | total-correlation penalties for disentanglement | redundancy reduction at representation level |

Positioning: S1 differs in TARGET, not technique: we penalize predictability of the *interaction*
from *each constituent modality alone* (never from the pair jointly), with a hinge to avoid the
"noise collapse" degenerate solution. This asymmetry (single forbidden, joint allowed) is the
novel configuration — must verify no prior art uses exactly it for multimodal fusion.

## 6. Hard-example emphasis (S4's cousins)

Focal loss (Lin 2017), hard example mining, curriculum learning (Bengio 2009).
S4 = reweighting by *lower-order* loss (a cross-model curriculum signal), not by own-model loss.
Our finding (S4 redistributes capacity to hard fractions; nets negative unless they are
under-served AND learnable) is a small but clean empirical note.

## 7. Benchmarks

MultiBench (Liang et al. NeurIPS 2021): AV-MNIST, MOSI, MOSEI, MUStARD, UR-FUNNY — our audit
targets. Bird-MML (ConFu paper's own benchmark): pending release — our planned positive test bed.

Our synthetic order benchmark (A/B/C + order-3 with ground-truth latents, proper probe splits)
fills a gap: no MultiBench dataset lets you KNOW the true interaction structure.

## Summary of defensible contributions after verification

1. The alignment–dependence gap, proven representation-level (predictability + shuffle + zero/shuffle
   distinction + correction/regression), on 3 natural benchmarks.
2. S1 adversarial de-shortcutting: the only tested objective that significantly reduces
   single-modality predictability (paired p≈0.011) at zero accuracy cost; honest on negative
   controls.
3. A controlled synthetic order benchmark demonstrating order-matched interactions are necessary
   and sufficient (+49.9 pp over order-restricted baselines, 5/5 seeds, balanced dependence).
4. A diagnostics + discipline protocol (gates, paired stats, capacity matching) others can reuse.
