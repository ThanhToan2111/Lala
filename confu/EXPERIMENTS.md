# ConFu++ experiment log

## AV-MNIST clean five-seed baseline

Hypothesis: A non-collapsed low-rank interaction may improve on lower-order additive evidence.

Change: Bimodal prototype classification with normalized low-rank interaction, variance, cross-covariance, and off-diagonal covariance regularization.

Expected result: Positive and stable `Gain12`, zero drop, shuffle drop, and conditional probe gain.

Actual result: SynergyFormer reached 70.762 ± 0.338% accuracy. `Gain12` was +0.022 ± 0.440 points, shuffle drop +1.106 ± 0.349, and zero drop +0.064 ± 0.390. The seed-1 linear conditional probe gain was -0.05 points.

Interpretation: Interaction collapse is substantially fixed and permutation sensitivity is real, but conditional complementarity remains unproven.

Decision: Keep the architecture fixed. Run modality-specific shuffle, gate sweep, correction/regression, and nonlinear conditional probes before adding a utility objective.

## AV-MNIST Stage A complementarity diagnostics

Hypothesis: The current interaction may be dominated by one modality or be a redundant nonlinear predictor.

Change: Evaluation-only diagnostics on the existing five checkpoints; no retraining.

Expected result: Both modality shuffles should hurt, learned gate should be near the gate-sweep optimum, net correction should be positive, and nonlinear conditional probe gain should reveal whether `r12` adds nonlinearly useful information.

Actual result: Across five seeds, shuffling modality 1 (image) reduced accuracy by 1.268 ± 0.493 points, while shuffling modality 2 (audio) changed accuracy by only 0.008 ± 0.079 points. Linear and nonlinear conditional probe gains were +0.156 ± 0.197 and +0.120 ± 0.362 points. Net correction was +2.2 ± 44.6 samples. Gate 0.25 had the best mean accuracy; gate 1.0 was consistently harmful.

Interpretation: `r12` is strongly image-dominant. Small conditional probe gains exist, but utility and net correction are unstable. Permutation sensitivity in the old aggregate test was almost entirely caused by the image factor.

Decision: Proceed to the mandated utility-loss seed-1 experiment. If audio dependence remains absent, move to the dependence objective after evaluating utility.

## AV-MNIST Stage B conditional utility

Hypothesis: A per-sample utility margin can make the interaction outperform detached lower-order evidence without collapsing its representation.

Change: Add `0.1 * relu(CE_full - stopgrad(CE_base))` with margin 0.0.

Expected result: Seed 1 must have positive `Gain12`, zero drop, and net correction while preserving variance and effective rank.

Actual result: At seed 1 with weight 0.1 and margin 0, accuracy was 70.45%, `Gain12=-0.05` points, `ZeroDrop=+0.04` points, `NetCorrection=-5`, linear probe gain -0.24 points, and nonlinear probe gain +0.33 points. Variance (0.0541) and effective rank (8.70) remained healthy. Image/audio shuffle drops were +1.15/+0.08 points.

Interpretation: The utility term did not collapse the representation, but weight 0.1 did not produce stable conditional utility and did not address image dominance.

Decision: Do not run five seeds. Run the specified seed-1 utility-weight sweep while keeping margin 0 and all other components fixed.

## AV-MNIST Stage B utility sweep and five-seed result

Hypothesis: A stronger utility term can make `r12` conditionally useful without reintroducing collapse.

Change: Sweep `lambda_utility={0, .05, .1, .2, .5}` at seed 1, select `.2`, then run the fixed protocol across seeds 1–5. The final representation is `c_full=LN(c_base + g LN(r12))` with `c_base=LN(LN(r1)+LN(r2))`.

Actual result: Weight `.2` was the strongest seed-1 candidate: `Gain12=+0.36`, `ZeroDrop=+0.49` points and net correction `+36`. Across five seeds it produced 70.296 ± 0.545% accuracy, `Gain12=+0.234 ± 0.263`, `ZeroDrop=+0.262 ± 0.309`, net correction `+23.4 ± 26.3`, linear conditional gain `+0.126 ± 0.186`, and parameter-matched nonlinear gain `+0.294 ± 0.162` points. Nonlinear gain was positive in all five seeds (exploratory one-sample p=0.015); `Gain12` was positive in all five but not significant with five seeds (p=0.118). Rank was healthy at 9.18 ± 0.66 and variance ratio 1.58 ± 0.11.

Evidence against: Accuracy was lower than the old fair Synergy baseline by 0.466 ± 0.513 points and ConFu-style by 0.590 ± 0.978 points. Audio shuffle changed accuracy by only +0.004 ± 0.027 points (positive in 3/5), while image shuffle dropped it by +1.482 ± 0.847 points in all seeds. Thus utility increased conditional signal but did not remove the one-modality shortcut or establish a superiority claim.

Decision: Retain `.2`/margin zero as the canonical P1 configuration for analysis, but do not claim it is the new benchmark winner. Run the specified margin sanity sweep, then P2 dependence rather than residual or larger architecture.

## AV-MNIST P2 dependence objective and P3 redundancy probe

Hypothesis: Ranking the true paired interaction above image- or audio-shuffled interactions will make `r12` depend on both modalities.

Change: Add task-aware ranking loss on true-class log probability for two corrupted interactions. Sweep dependency weight `.1`, `.5`, and `1.0` at seed 1 with margin `.1`; no architecture or parameter-count change.

Actual result: At weights `.1/.5/1.0`, audio-shuffle drops were `-0.02/+0.11/+0.11` points; the latter two values were small and came with `Gain12=+0.05/+0.09` and accuracy `70.54/69.93%`. The loss therefore did not create robust dual-modality dependence. Nonlinear predictability probes explain why: for the selected utility checkpoint, `r1 -> r12` gives R²=0.988 and cosine=0.996, while `r2 -> r12` gives R²=0.031 and cosine=0.203. At dependency weight `.5`, the values remain R²=0.989/0.031.

Failure mode: `r12` is a high-rank, non-collapsed transformation that is nevertheless almost entirely predictable from image representation. The ranking loss is active but cannot bootstrap audio dependence once the positive and audio-corrupted interaction are nearly identical.

Decision: P2 fails its multi-modality gate. Do not run five seeds for dependency, residual interaction, Global–Local, cross-attention, or tri-modal expansion. The next scientifically justified experiment is to change the dataset/task so audio contains conditionally necessary label information, or to add a dependence target that does not use invalid instance matching in AV-MNIST.

## CMU-MOSI original ConFu tri-modal audit

Hypothesis: A genuine tri-modal affective benchmark can establish whether original ConFu pair interactions have both modality dependence and conditional label utility.

Change: No ConFu++ objective or architecture change. Reproduce original pairwise ConFu on the official MOSI split for five seeds, then audit frozen `r12/r13/r23` against lower-order pair representations. A seed-1 TriCLIP run is an explicitly exploratory non-interaction reference.

Actual result: Text dominated the task (67.681 ± 2.177%); all-modalities concat was 66.170 ± 2.861%. Linear conditional gains for vision/audio, vision/text, and audio/text were -0.962 ± 1.248, -1.429 ± 1.608, and -2.274 ± 0.992 points. Despite this, text-containing interactions had strong intervention sensitivity (interaction-shuffle drops +9.184 ± 3.012 and +12.128 ± 3.178 points) and high effective rank, but were highly predictable from lower-order inputs (R² 0.915 and 0.953). The original ConFu model has no `r123`.

Interpretation: MOSI confirms that pair fusion/dependence is not conditional complementarity. The immediate bottleneck is redundant pair interaction on a text-dominant task, not collapse. A positive ConFu++ utility objective is not justified by these data.

Decision: Preserve this audit and do not add utility, necessity, residual, Global–Local, cross-attention, rank expansion, or third-order claims. Select a benchmark/task with validated lower-order headroom before objective development.

## Synthetic Stage 0 ground-truth validation of S1 adversarial de-shortcutting

Hypothesis: A representation-level adversarial hinge penalty (single-modality unpredictability of `r12`, tau=0.5, alternating minimax) preserves genuine dependence when synergy exists and does not fabricate dependence when it does not.

Change: New module `src/modules/models/complementarity.py` (AdversarialPredictor, EMAStandardizer, hinged shortcut loss, factor variance floor), new experiment `src/experiments/synthetic/synergy_experiment.py` (configs A/B/C, ground-truth latents a,b; y = a XOR b / y = a / mixture rho=0.5), linear classifier head, 50k train samples, 60 epochs, 5 seeds. A train/test prototype-mismatch bug in the first data generator version was found and fixed; all earlier synthetic numbers were discarded.

Actual result (5 seeds): Config A (pure synergy): baseline acc 0.999 ± 0.002, `Gain12` +0.499 ± 0.010, both modality shuffle drops ~+0.5, selectivity +0.432; with S1 acc 1.000, `Gain12` +0.501, predictability R2(r2 -> r12) reduced 0.407 -> 0.202. Config B (no synergy): baseline and S1 both acc 1.000 with zero/shuffle drops ~0 and selectivity 0; S1 did not fabricate dependence (R2(r1 -> r12) 0.087 -> 0.012). Config C (50% redundant / 50% XOR mixture): baseline converged to the a-only floor (acc 0.748 ± 0.004, `Gain12` ~0) with a misleading +0.118 shuffle-2 drop; S1 kept acc 0.750 ± 0.006, raised selectivity +0.010 -> +0.038, halved R2(r2 -> r12), but did NOT recover the XOR fraction.

Interpretation: The S1 mechanism is sound (does not destroy learnable synergy, Config A) and honest (does not fabricate dependence, Config B). On a shortcut-friendly mixture task (Config C), S1 alone at lambda=1.0 with hinge 0.5 is insufficient to overcome the dominant easy mapping; the residual shortcut is a training-dynamics problem (easy-gradient dominance), not a measurement problem.

Decision: S1 passes the Stage 0 soundness/honesty gates (A, B). Config C remains an open stress test for stronger dependence objectives; proceed to AV-MNIST Stage 1 negative control with lambda_shortcut selected by seed-1 sweep.

## AV-MNIST Stage 1 S1 adversarial de-shortcutting, five-seed confirmatory

Hypothesis: Penalizing single-modality predictability of `r12` during training reduces the image shortcut (R2(r1 -> r12)) and increases measured audio dependence, without collapsing the representation or the task accuracy.

Change: `src/experiments/av_mnist/complementarity.py` (manual optimization; alternating minimax with k_adv=1, adversary lr 0.5x, EMA-standardized hinge tau=0.5, warmup 10% of epochs, factor variance floor 0.5). Seed-1 sweep over lambda_shortcut {0, 0.1, 0.3, 0.5, 1.0} found all nonzero weights equivalent (best-epoch checkpoint selection saturates the early effect); lambda=0.1 selected. Confirmatory: 5 seeds, paired against the same-codepath lambda=0 ablation. All other protocol elements identical to the canonical AV-MNIST setup (this also implies early stopping selects epoch ~0-5, so S1 acts for only a few effective epochs).

Actual result (S1 vs ablation, paired, 5 seeds): accuracy 70.046 ± 0.476 vs 69.962 ± 0.594 (delta +0.084 ± 0.219, t=+0.86, ns). R2(r1 -> r12): 0.682 ± 0.155 vs 0.989 ± 0.002 (delta -0.307 ± 0.155, paired t=-4.43, p≈0.011) — the first objective in this project to significantly move single-modality predictability. Audio shuffle drop: +0.236 ± 0.147 vs +0.084 ± 0.069 (delta +0.152 ± 0.130, t=+2.62, p≈0.059). Gain12, zero drop, net correction unchanged; linear/nonlinear conditional probe gains declined (-0.79 / -0.46). Accuracy remains below ConFu-style (70.886 ± 0.578) and P1 (70.296 ± 0.545).

Interpretation: S1 works in the intended causal direction at zero accuracy cost: the interaction is measurably less image-derived, and audio dependence doubles (borderline significant). As predicted by the negative-control role of AV-MNIST, no objective can create conditional utility the data does not contain: gain12 stays ~+0.17, and the residual predictability (R2 0.68) plus reduced probe gains indicate remaining redundant structure the hinge (tau=0.5) deliberately tolerates.

Decision: Accept S1 as a validated representation-level mechanism (significant de-shortcutting, honest on negative controls, sound on synthetic ground truth). Do NOT tune it further on AV-MNIST (test set). The blocking issue for complementarity claims is now purely the benchmark: per the MOSI audit decision, select a benchmark with validated lower-order headroom (Config-C-style synthetic mixture and/or Bird-MML) for the first positive demonstration.

## UR-FUNNY original ConFu five-seed reproduction

Hypothesis: UR-FUNNY provides a genuine tri-modal ConFu benchmark on which
the original pairwise objective can be compared with the ConFu++ adapter.

Change: Download the official MultiBench packed `humor.pkl`, reproduce the
original ConFu with the fixed split and five seeds, and evaluate the same
frozen sklearn linear probe on unimodal, pair, and all-modality embeddings.

Actual result: The strict all-modality probe reached 64.462 ± 0.713% across
seeds 1--5. The seed-1 audit found interaction shuffle drops of +7.467,
+8.034, and +9.168 points for vision/audio, vision/text, and audio/text, but
conditional gains of -0.662, +0.189, and -0.756 points. Lower-order to
interaction R² was 0.963, 0.970, and 0.970.

Interpretation: UR-FUNNY ConFu interactions are active but redundant. The
primary bottleneck is conditional utility, not variance or effective rank.

Decision: Use this as the matched baseline. Do not interpret permutation
sensitivity as evidence of complementary information.

## UR-FUNNY ConFu++ S1 plus task-aware head

Hypothesis: Training-only adversarial de-shortcutting and a supervised task
head will reduce lower-order redundancy while preserving or improving the
all-modality representation.

Change: Add S1 alternating adversaries with `lambda_shortcut=0.1`,
`tau=0.5`, `k_adv=1`, adversary LR scale 0.5, hidden size 256, warmup 0.2,
plus a training-only task head at `lambda_task=0.1`. Run the same five seeds,
then use the matched native frozen probe.

Actual result: All-modality accuracy was 63.549 ± 0.855%, versus 64.462 ±
0.713% for ConFu. The paired difference was -0.913 ± 1.012 points
(`t=-2.016`, `p=0.114`), with four of five seeds below baseline. Health and
inference efficiency passed, but dependence and utility gates failed.

Interpretation: S1 did not remove the nonlinear shortcut on this task, and
the task loss perturbed the geometry used by the downstream frozen probe.

Decision: Reject this configuration as a benchmark upgrade. Preserve the
negative result and do not add a larger architecture or hyperparameter sweep.

## UR-FUNNY task-only weight sanity check

Hypothesis: The task objective may need a stronger weight after S1/task
interference is removed.

Change: Exploratory `lambda_shortcut=0`, `lambda_task=1.0`. A 30-epoch seed-1
pilot reached 65.597%, but a matched 100-epoch seed-1 run reached 64.188%.

Decision: The apparent pilot gain was an early-checkpoint artifact, not a
reproducible upgrade. Stop the remaining exploratory seed and do not claim a
baseline win.

## UR-FUNNY ConFu++ v2 utility-first audit

Hypothesis: Original ConFu pair outputs may contain nonlinear conditional headroom that can be exposed by pair-augmented probes or a small frozen-backbone logit residual.

Change: E0 compared `[r1;r2;r3]` with `[r1;r2;r3;r12;r13;r23]` on all five original ConFu checkpoints using matched linear and GPU nonlinear probes. E1 trained only three pair residual heads with scalar gates initialized at 0.1. E2 added a conditional utility loss, and E3 added a preserve loss. No S1, dynamic routing, cross-attention, rank sweep, context architecture, or `r123` was added.

Actual result: E0 linear pair gain was `-0.017 ± 1.179` pp; nonlinear pair gain was `+1.191 ± 2.579` pp with mixed signs across seeds. On seed 1, lower-only accuracy was `65.217%`; E1/E2/E3 final accuracies were `63.989/64.367/64.461%`, with net corrections `-13/-9/-8`. The matched original ConFu baseline is `64.462 ± 0.713%` over five seeds and `65.469%` on seed 1.

Interpretation: The remaining signal is nonlinear but not stable enough for a residual correction. Pair heads produce more regressions than corrections, gates remain near the conservative initialization, and pair interventions are mostly zero or negative. The primary bottleneck is conditional utility and calibration of existing pair outputs, not representation collapse, model capacity, or inference cost.

Decision: Reject v2 as a baseline upgrade. Do not run five-seed confirmation or add architecture/sweep complexity. The complete analysis is in `results/urfunny/URFUNNY_CONFU_V2_BENCHMARK_REPORT.md`.

## UR-FUNNY ConFu++ v3 E4 order-decomposed residual alignment

Hypothesis: Higher-order structure should be made accessible by aligning explicit pair embeddings with residual targets left after restricted additive first-order prediction.

Change: Add a frozen-feature joint-vs-additive headroom audit, independent first-order predictors, additive residual targets, and low-rank multiplicative `h12/h13/h23` branches trained with symmetric residual InfoNCE. No task loss, adversary, residual classifier, dynamic gate, cross-attention, rank sweep, or `h123` was added.

Actual result: The seed-1 headroom audit found joint-vs-additive representation R² advantages of `+0.517`, `+0.235`, and `+0.228` for targets `r1/r2/r3`; residual health passed. E4 interaction health also passed and all six modality shuffles caused positive probe drops. However, linear `Delta2` was `-1.832` pp on validation and `-2.662` pp on test. Nonlinear test `Delta2` was `+1.512` pp, but the primary linear gate failed.

Interpretation: UR-FUNNY contains non-additive representation structure, but residual InfoNCE alignment does not make it linearly accessible for humor classification. The bottleneck moved from “no joint headroom” to residual-target/task mismatch, pair order imbalance, and frozen-backbone formation.

Decision: Reject this E4 configuration. Do not run five seeds, E5 task loss, encoder fine-tuning, or `h123`. The next minimal diagnostic is P0 linear versus one small MLP first-order predictor; if `Delta2` remains negative, validate the mechanism on the controlled synthetic order benchmark.

## UR-FUNNY ConFu++ v3.1 E4.1 capacity-controlled predictor audit

Hypothesis: The negative v3 residual-alignment result may be caused by linear first-order predictors leaving nonlinear unimodal residue in the residual targets. Compare P0 linear-additive, P1 nonlinear-additive, and capacity-matched P2 joint-nonlinear predictors before retraining interactions.

Change: P1 uses two independent `256→128→256` LayerNorm/GELU MLPs; P2 uses a joint `512→171→256` MLP. P1 has 133,120 parameters and P2 132,779 (`0.256%` difference). Same normalization, optimizer, official split, validation early stopping, and seeds 1–3 are used.

Actual result: P1 improves over P0 for all targets, while P2−P1 is approximately zero. Validation joint advantages are `−0.0006`, `+0.0012`, and `+0.0002` R² for `r1/r2/r3`; test advantages are `−0.0009`, `+0.0004`, and `−0.0004`. All six directed unimodal relationships gain slightly from MLP capacity.

Interpretation: The previous v3 joint-vs-linear headroom was primarily a nonlinearity confound, not robust non-additive multimodal structure. This explains why v3 residual alignment produced healthy interactions but negative linear accessibility.

Decision: Block E4.2, task loss, encoder fine-tuning, rank/cross-attention sweeps, and `h123` on UR-FUNNY. Move to the controlled synthetic order benchmark; preserve UR-FUNNY as a diagnostic negative result. Full report: `results/urfunny/URFUNNY_CONFU_V31_CAPACITY_AUDIT.md`.

### v3.1 E4.2 correction

The weak but validation-consistent P2−P1 signal for `r1/r2` justified one exploratory E4.2 run with nonlinear additive residual targets. A mapping audit also fixed the v3 interaction target order to `h12→e3`, `h13→e2`, `h23→e1`; the old v3 E4 artifact is superseded.

E4.2 seed 1 produced validation linear `Delta2=-2.508` pp, test linear `Delta2=-1.223` pp, and nonlinear test `Delta2=+3.025` pp. Health and shuffle dependence passed, but the primary accessibility gate failed. Do not run five-seed E4.2 or further UR-FUNNY architecture work; move to the controlled synthetic order benchmark.

## Synthetic Config C redesign (label-ambiguity bug) and S4 hard-fraction emphasis

Hypothesis: A mixture task (fraction rho labeled by the modality-1 rule, the rest by XOR) with an OBSERVABLE routing flag in modality 2 is a valid shortcut stress test; hard-fraction emphasis (S4: reweight the fusion loss by stop-grad per-sample base CE) helps escape easy-gradient dominance.

Change: Config C redesigned — a routing flag is appended to modality 2 (previously the flag was latent, making labels ambiguous and Bayes-optimal 0.75; the earlier Config-C numbers measured memorization and are discarded). S4 added: `w = sg(CE_base) + 0.3*mean(w); L_task = (1-lambda_hard)*CE_full + lambda_hard*(w*CE_full)/sum(w)`. Matrix: {baseline, S1, S4, S1+S4} x rho {0.5, 0.7, 0.9} x epoch budgets {6, 10, 60}, 3-5 seeds each.

Actual result: With the observable flag, the BASELINE low-rank interaction already solves every mixture (rho=0.5: acc 0.976, a-fraction 0.978 / xor-fraction 0.974; rho=0.9: 0.987, xor 0.909). S1/S4/S1+S4 tie the baseline within ±0.005 on total accuracy; S1 still halves R2(r2 -> r12). Under a 6-10 epoch budget at rho=0.9, S4 raises xor-fraction accuracy (0.92 -> 0.97) but sacrifices easy-fraction accuracy, netting -0.008 total. On AV-MNIST seed 1 (exploratory): S1+S4 gave acc 69.63 with the lowest shortcut yet (R2(r1 -> r12)=0.609) but killed utility (gain12 +0.01); S4-only was worse (68.98, shortcut intact).

Interpretation: (1) When the synergy cue is observable and learnable, the multiplicative interaction does NOT shortcut — the AV-MNIST/MOSI/UR-FUNNY shortcut phenomena arise from information absence or dominance, not from an intrinsic optimization failure of interaction modules. (2) S4 is a real but blunt instrument: it redistributes capacity toward hard samples and helps only when hard samples are both under-served AND learnable — a combination no current benchmark provides.

Decision: Finalize the objective stack as S1 (validated de-shortcutting) with S4 documented as exploratory-negative on current data. The synthetic suite (A sound / B honest / C solvable-routed) is now the controlled benchmark the UR-FUNNY v3.1 decision asked for. Strongest honest claim to date: S1 achieves accuracy parity with a 31-point absolute reduction in image-shortcut predictability. Next justified step: a synthetic noise ladder on modality 2 (vary SNR of the synergy cue) to find the regime where baseline interaction DOES give up, then test whether S1/S4 changes the giving-up threshold.

## Synthetic noise ladder and the controlled ORDER-3 benchmark (decisive positive result)

Hypothesis: (a) Lowering the SNR of the synergy cue (noise on modality 2) finds the regime where the baseline interaction abandons it; (b) on a genuine third-order task (y = a XOR b XOR c), order-restricted models are capped at chance while the full SynergyFormer order-3 term solves it.

Change: Noise ladder on Config C (rho=0.7, X2 noise 1.5-3.0, baseline vs S1+S4). New benchmark `src/experiments/synthetic/order3_experiment.py`: three modalities observe bits a,b,c via fixed prototypes + noise; order-3 label y = a XOR b XOR c (information-theoretically independent of all singles AND all pairs); order-2 control y = a XOR b with c as distractor. Variants: additive (order-1), pairs_only (order-1+2, sum readout), concat_mlp (2x256), full SynergyFormer (order 1+2+3), full+S1 with adversaries on r123 from singles AND pairs. Probes now use proper train/test splits. 5 seeds per cell.

Actual result: (a) Noise ladder: the baseline degrades gracefully (xor-fraction 0.958 -> 0.903) and NEVER abandons the cue; S1+S4 shifts the operating point toward the hard fraction (xor +1.5 pp at noise 2.0) at a small easy-fraction cost; no give-up regime exists in this family. (b) Order-3: additive 0.4957 ± 0.0061 and pairs_only 0.5010 ± 0.0102 are BOTH at chance (information/readout caps), concat-MLP reaches 1.000 (universal approximation with capacity, 60 and 200 epochs), and full SynergyFormer reaches 1.0000 ± 0.0000 on 5/5 seeds — margin +49.9 pp over order-restricted baselines. r123 is healthy (variance 0.44), perfectly balanced (all single-modality shuffle drops +0.505), unpredictable from any single modality (R2 0.15), and singles/pairs probes sit at chance on held-out probes. Notably the pair representations DO contain the bits (pair-concat probe 0.97) — the pairs-only failure is a READOUT failure of the sum aggregator, not informational absence. Order-2 control: full = 1.0000 with shuffle_c drop exactly +0.0000 (no fabricated dependence on the distractor). S1 on order-3: identical accuracy, R2(r1 -> r123) 0.155 -> 0.124, zero collateral damage.

Interpretation: This completes the causal chain of the whole project. Shortcuts on natural benchmarks (AV-MNIST, MOSI, UR-FUNNY) arise from information absence/dominance — NOT from an inability of explicit higher-order interactions to learn: when the higher-order information is present and observable, the order-matched interaction extracts it completely, with maximally balanced dependence, and S1 neither breaks nor fabricates anything. The pair-sum readout failure on order-3 is direct evidence that "the more, the merrier" requires order-MATCHED interaction terms, not just rich pair features.

Decision: This is the first decisive positive result of the project and the headline synthetic evidence for the paper. The objective stack is final: S1 (validated), S4 (exploratory-negative), order-matched interactions (validated). The remaining gap for a publication claim is a NATURAL benchmark with genuine higher-order headroom: Bird-MML when its data is released (local data dirs are currently empty and training requires 1-2 GPU-days).

## LayerNorm composition confound (discovered via the M-ladder)

Hypothesis: An M-modality order ladder (y = XOR of M bits) cleanly separates order-restricted models from order-matched ones.

Change: New `src/experiments/synthetic/orderM_experiment.py`, M in {3,4,5,6}, variants additive/pairs_only/full/concat_mlp, linear readout, 5 seeds.

Actual result: The FIRST ladder version was invalid: additive reached 0.874 and pairs_only 0.9998 on 3-bit parity, which is impossible for sum-of-subset functions. Root cause isolated by minimal reproduction: a final LayerNorm after aggregation divides by a cross-modal statistic and silently reintroduces higher-order computation (converged 8-combo test: additive WITH final LN reaches 0.75 on 3-parity; additive WITHOUT is exactly 0.5 on all seeds). After removing the post-aggregation LN, the ladder is clean:

| M | additive | pairs_only | full (order-M term) | concat_mlp |
|---|---|---|---|---|
| 3 | 0.496 | 0.503 | 1.0000 | 1.0000 |
| 4 | 0.499 | 0.499 | 1.0000 | 1.0000 |
| 5 | 0.500 | 0.497 | 1.0000 | 1.0000 |
| 6 | 0.500 | 0.495 | 1.0000 | 0.9975 |

min per-modality shuffle drop for `full` is +0.49..+0.50 at every M (balanced maximal dependence); additive/pairs drops are ~0 (they encode nothing).

Interpretation: (1) Post-aggregation LayerNorm is a hidden higher-order term — any paper's "additive/pairwise" ablation that ends in LayerNorm is not a valid low-order control. This directly affects the ConFu++ composition design (GatedComposition uses output LayerNorm; acceptable for the main model, but invalid as an ablation baseline). (2) The order-matched multiplicative interaction solves parity at every M tested with perfectly balanced dependence; concat-MLP keeps up to M=6 but begins to crack (0.9975, noisier) — capacity-based learning of high-order parity degrades first.

Decision: Document the LayerNorm confound as a methodology finding (paper-worthy pitfall). All order-ladder claims use the LN-free readout. Keep this test as a regression test for any future composition change.

## S1 residual-shortcut bottleneck analysis (AV-MNIST long training)

Hypothesis: The residual R2(r1 -> r12) = 0.68 after S1 reflects either protocol length or an adversary gap.

Change: Seed-1 run, lambda_shortcut=0.1, warmup 0.1, patience 30, 30 epochs; tracked in-training adversary R2 and representation health per epoch.

Actual result: In-training adversary R2(r1) rose to 0.43 by epoch 2 then fluctuated at 0.26-0.41 forever — NEVER descending toward zero; meanwhile the offline evaluation probe reads R2 = 0.84. Validation accuracy decays monotonically after epoch 0 (0.7398 -> 0.688 by epoch 29): epoch-0 selection is optimal, not an artifact. r12 rank grows (24 -> 34), cos(r12, r1) ~ 0.02 throughout.

Interpretation: The bottleneck is an ADVERSARY GAP, not training length: the online adversary (k_adv=1 per batch, EMA target) equilibrates at R2 ~ 0.3-0.4 — below the hinge band edge (tau=0.5) — so the penalty switches off, while a converged offline probe still extracts R2 = 0.84. The generator only needs to fool the weakest converged adversary. Options to close the gap (more k_adv, stronger online adversary, lower tau) trade directly against task information on a benchmark where image IS the label; on AV-MNIST the residual is therefore best interpreted as irreducible-without-task-damage.

Decision: Report the adversary gap as the honest mechanistic boundary of S1 (paper Discussion). Do not raise k_adv/tau aggression on AV-MNIST — that trades task information for metric cosmetics. If the gap must be closed, do it on synthetic ground truth where task-damage is measurable.

## Bird-MML frozen-encoder audit and ConFu++ confirmatory (positive natural result)

Hypothesis: Bird-MML (Zenodo 18920487) contains genuine image-text complementarity that the original ConFu objective fails to capture, and ConFu++ (utility + S1) recovers it.

Change: Downloaded the released Bird-MML (35 GB). The record's audio.tar.gz is split into 16 parts but part-ah was never uploaded (all present parts pass MD5) — recovered parts aa..ag, yielding a balanced 71-species / 70,796-row tri-modal subset (species names alphabetical range covered by the tar prefix; fixed stratified split saved). Frozen encoders: ResNet50-ImageNet (image), wav2vec2-base (audio), MiniLM (text). Audit via linear/nonlinear probes on all subsets; ConFu reproduction and ConFu+Utility(+S1) on frozen embeddings, 5 seeds paired.

Actual result: Audit — image 64.95%, audio 1.43% (dead in these features), text 63.74% (33.1% of captions contain the species common name — documented leakage), image+text 80.89% (conditional gain +15.94 pp — genuine headroom), all 81.01%. Confirmatory — original ConFu: probe_full 76.90 ± 0.65%, gain over lower-order +0.19 pp (captures nothing). ConFu+Utility: 83.01 ± 0.13% (+1.07 pp conditional gain). ConFu+U+S1: 82.66 ± 0.37%, paired delta vs ConFu +5.76 ± 0.86 pp (t=+14.91, p<0.001), exceeding the raw-feature linear ceiling; R2(text -> z13) drops 0.59 -> 0.50 (interaction less constituent-derived); audio honestly inert (+0.01 pp drops everywhere).

Interpretation: The first positive natural-data result of the project. The alignment-utility gap is real even where headroom exists (ConFu captures ~0 of +15.94 pp available); the utility+S1 objective stack recovers a significant share (p<0.001) with no fabricated dependence on the dead modality. Residual limitation: pair-conditional gain of z13 remains slightly negative (-0.21 pp) — utility arrives via the jointly trained full head, not the pair term alone.

Decision: Bird-MML becomes the paper's positive natural benchmark (subset protocol documented; full rerun once Zenodo fixes part-ah). Next open target: per-pair utility objectives to make z13 itself carry conditional gain. Paper table and report updated: results/birds/BIRD_MML_REPORT.md.

## Bird-MML fair-play controls (leakage, attribution, same-topic baseline, S1 ablations)

Hypothesis: The Bird-MML positive result must survive leakage-free text, correct attribution, a same-topic baseline, and hyperparameter robustness checks.

Change: (1) Re-extracted text embeddings from BLIP-only image captions (no species names); re-measured headroom. (2) Added the ConFu+S1-without-utility cell (5 seeds, pair-conditional metrics). (3) Implemented OGM-style on-the-fly gradient modulation (Peng et al. 2022) on identical frozen features for the image+text pair. (4) S1 ablations on synthetic Config A: tau in {0.3, 0.5, 0.7} x k_adv in {1, 3}, 3 seeds.

Actual result: (1) BLIP-only: text alone 49.77%, image+text 69.73% — leakage-free headroom is +4.78 pp (vs +15.94 pp with the released combined captions; 33.1% of which contain the species name). Genuine image-text complementarity survives de-leakage. (2) Attribution: accuracy gain comes from the utility objective (ConFu+U 83.01% vs +S1-only 76.58%); on the balanced pair, text-predictability of z13 is lowest under utility alone (R2 0.44), and S1 pins it to the hinge band (0.50 at tau=0.5) — S1 guarantees a floor, it is not a monotone reducer here. (3) OGM: 80.68 ± 0.38 vs plain 80.88 ± 0.11 (paired t=-1.16, ns) — gradient rebalancing does not capture the complementarity; ConFu++ exceeds it by 1.8 pp. (4) S1 robust on ground-truth synergy across tau and k_adv (acc ~1.0, balanced drops).

Interpretation: The Bird-MML result survives all fairness controls. The headroom is smaller but real without leakage. OGM-style rebalancing (the closest same-topic family) is insufficient — consistent with our root-cause claim that loss/gradient-level pressure cannot create representation-level dependence.

Decision: Include all four controls in the paper. End-to-end replication (trainable encoders, 3 arms) is running; AST-domain audio re-extraction in progress for the audio-null robustness check.

## Bird-MML end-to-end replication (seed 1) and audio-domain robustness

Hypothesis: The frozen-feature findings should transfer to joint end-to-end training, and the audio-null result should not depend on the speech-domain feature extractor.

Change: (1) End-to-end training on the 71-species subset with three arms (confu / confu+utility / confu+utility+S1): ResNet18 image (ImageNet init), ResNet18 log-mel audio (scratch, 500-10kHz), MiniLM text (pretrained init), pair fusions, task heads, identical schedule/capacity. (2) Re-extracted audio with AST (AudioSet-pretrained, correct domain) and re-ran the audit.

Actual result: (1) End-to-end seed 1: all arms converge to ~71.7% with TEXT DOMINANCE in every arm (text shuffle drop +4.8-5.0 pp, image ~0, audio 0) — the task loss absorbs the caption name leakage (33%); utility adds nothing (gain ~0); de-shortcutting still responds in the right direction end-to-end (R2(text -> z13): 0.861 -> 0.641 -> 0.608 across arms). (2) AST audio re-audit: audio alone 1.43% (still chance), image+audio headroom +0.35 pp (vs +0.13 with wav2vec2) — audio-null is NOT a feature-extractor artifact; it holds across speech- and AudioSet-domain extractors.

Interpretation: End-to-end, the alignment-dependence gap manifests as MODALITY ABANDONMENT (text leakage wins, image marginalized) — a stronger failure mode than the frozen-feature shortcut, and precisely what the diagnostic suite is designed to expose. The two protocols are complementary evidence: frozen features isolate representation content; end-to-end shows what optimization actually chooses.

Decision: Include both protocols in the paper as complementary evidence. Seeds 2-3 of the three e2e arms are running for the paired table. The audio channel's null status is now established with domain-correct features; no further audio work needed. Remaining open item for a later revision: full 149-species rerun when Zenodo fixes part-ah, and per-pair utility objectives.

## Bird-MML end-to-end three-arm confirmatory (3 seeds)

Actual result (3 seeds, paired): accuracy confu 71.87 ± 0.16, +U 72.18 ± 0.48, +U+S1 72.32 ± 0.62 (paired delta +0.45 ± 0.49 pp, t=+1.58, ns). All arms text-dominant (text shuffle drop +4.9 pp, image ~0, audio 0) due to caption name leakage absorption. De-shortcutting monotone across arms: R2(text -> z13) = 0.81 ± 0.09 -> 0.73 ± 0.07 -> 0.68 ± 0.08 (paired delta -0.124 ± 0.117, t=-1.83, direction-consistent).

Interpretation: End-to-end the objectives do not differentiate on accuracy (leakage saturates the task), but the diagnostic ordering holds and S1 keeps reducing text-derived content in z13. The frozen vs end-to-end contrast is itself a core paper finding: representation audits and optimizer behavior give complementary, sometimes opposing, pictures.

Decision: Paper updated with the 3-seed table. This closes the fair-play gaps achievable in-session: leakage control, domain-correct audio robustness, attribution, same-topic baseline (OGM, ns), hyperparameter ablations, and end-to-end replication. Remaining for a later revision: full 149-species rerun after the Zenodo part-ah fix, and per-pair utility objectives targeting positive pair-conditional gains.

## Bird-MML supplement battery (paper gap-closing round)

Hypothesis: The positive Bird-MML result must survive lineage generality, per-pair attribution, sensitivity analysis, and robustness probes.

Change: (1) CLIP-lineage control (pairwise InfoNCE only, identical fusion heads, 5 seeds). (2) Per-pair utility objective: each z_ij receives its own ReLU margin against its constituents (lambda_pair=1.0). (3) Ablations: lambda_utility in {0.1, 0.5, 1.0}, tau in {0.3, 0.5, 0.7} (3 seeds each). (4) Hard-sample analysis (confidence tertiles) and missing-modality drops on the saved full-stack seed-1 model. (5) Adversary-gap closure attempt on synthetic Config A (k_adv=3, adversary hidden 128, tau=0.3).

Actual result: (1) CLIP-lineage shows the IDENTICAL shortcut (R2 img->z12 = 0.99, txt->z23 = 0.99, full probe 76.77%) — the gap is not ConFu-specific; it is a property of the contrastive-alignment lineage. (2) Per-pair utility flips the z13 pair-conditional gain from -0.36 to +0.39 ± 0.19 pp at 82.84 ± 0.39% overall; stacking U+S1+PairU reaches 83.17 ± 0.33% (paired vs ConFu +6.27 ± 0.71, t=+19.77) but the pair gain collapses back to -0.36 under the full stack (the joint head absorbs the pair heads' role — documented interference). (3) Utility gain is flat across lambda_utility (82.4-83.0); tau ladder 82.96/82.66/81.95 for 0.3/0.5/0.7. (4) Corrections are NOT concentrated in low-confidence bins (gain ~0 across tertiles); missing-modality degrades gracefully (image -0.99, text -0.71, audio +0.04 pp). (5) Stronger adversaries narrow the gap without task damage (acc ~1.0; single-modality predictability down to 0.035-0.07 on 2/3 seeds) — partial closure, direction confirmed.

Interpretation: The method family now has: lineage-generality of the problem (CLIP control), a config that makes the interaction itself carry conditional utility (PairU), flat sensitivity, graceful degradation, and an honest account of objective interference.

Decision: All added to the paper. Remaining: de-leaked-caption end-to-end retrain (running) and e2e seeds 4-5.

## Bird-MML de-leaked-caption causal control (seed 1, 3 arms)

Hypothesis: If text dominance in the end-to-end runs is driven by caption name leakage (33.1%), scrubbing names should restore image dependence.

Change: Regex-scrubbed common+scientific names from combined_caption (residual leakage 0.0% verified), retrained all three end-to-end arms on identical splits.

Actual result: Accuracy unchanged (71.67-72.20%). Text dominance PERSISTS or strengthens (text shuffle drop +4.7 to +5.6 pp; image stays ~0; audio 0). De-shortcutting meters unchanged (R2(text -> z13) 0.71-0.77).

Interpretation: Text dominance is INTRINSIC to the data (visual descriptions are highly discriminative for fine-grained species ID), not a leakage artifact. The leakage hypothesis is causally tested and rejected. This strengthens the paper's end-to-end claim: modality abandonment happens even in the cleanest available text channel.

Decision: Report the de-leak control in the paper's end-to-end section. No further caption interventions needed. The image-modality rescue would require a task where text descriptions are provably insufficient — not available in this dataset.
