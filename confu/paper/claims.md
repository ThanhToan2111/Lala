# Claims matrix

| Claim | Evidence | Assumptions | Limitation | Status |
|---|---|---|---|---|
| Multimodality does not imply cross-modal predictive identifiability. | MUStARD and MELD G1 screening | Frozen pooled representations and tested Product protocol | Representation- and class-relative, finite sample | Supported under tested settings |
| Cross-modal predictive interaction does not guarantee downstream task headroom. | MOSEI VA→T positive G1; sentiment G2 inconclusive/failing | Same sample split and label-free target/task audits | No claim that task headroom is absent universally | Supported as a non-implication |
| Interaction fidelity does not guarantee accessibility. | Synthetic IPIB and MOSEI D2/accessibility audits | Fidelity and task probes use frozen representations | Accessibility depends on probe and task | Supported under tested settings |
| Population nested projection advantage equals a squared projection gap. | `paper/theory/projection_theory.md` | L2 target, squared loss, closed nested subspaces, nonzero target variance | Finite neural models are operational approximations | Theorem under stated assumptions |
| Parameter matching alone does not guarantee equivalent hypothesis-class accessibility. | v5.6 MOSEI positive control and Generic MLP failure | Capacity-matched independent MLP and fixed optimization | Positive control exposed optimization/expressivity confound | Methodological observation |
| A nested residual did not reveal substantial MELD headroom beyond Product Joint. | v5.6.1 `NESTED_HEADROOM_ABSENT` | Fixed residual budget ≤25%, zero initialization, epoch-zero Product candidate | One residual class, one representation family | Supported under tested nested audit |
| Naturalized IPIB provides controlled interaction ground truth under natural source geometry. | v6.0 Naturalized IPIB artifact | Fixed MOSEI source projections and injected generator | Not a natural Type-IV task | Controlled benchmark claim |
| The operational score J is not PID synergy or causality. | Projection definition and deterministic-fusion constraint | Squared-risk interpretation | Does not estimate Shannon atoms | Explicit non-claim |

## v6.1 claim audit

| Claim | Evidence | Counter-evidence | Limitation | Allowed wording | Status |
|---|---|---|---|---|---|
| Naturalized IPIB J tracks known injected interaction strength. | Five-seed β ladder; J rises from -0.01427 at β=0 to 0.22625 at β=1. | β=0 accessibility is not a task null; JAD recovery is partial. | Fixed MOSEI source geometry and fixed generators. | “Construct-validity evidence under natural source geometry.” | Supported |
| MOSEI has an operationally positive empirical joint advantage. | Frozen VA→T validation J=0.01983 mean across three seeds; v6.1-C1 B=50. | C1 `p_hat(J≥observed)=0.09804`; observed score is not separated from the alignment-breaking null. | Historical predictor and alignment-breaking null are representation/protocol-specific. | “Operationally positive under the frozen protocol, but not separated from this alignment-breaking null.” | Supported, calibration-limited |
| Historical J>0.01 is a universal calibrated discovery threshold. | Historical usage only. | MOSEI alignment null overlaps/exceeds this magnitude. | v6.0 B=10 tail is finite-sample. | “Operational historical threshold.” | Rejected / not claimed |
| Positive predictive interaction guarantees task utility. | MOSEI D2 sentiment gain -0.034 ± 0.099 pp. | Controlled task labels and probes are task-relative. | No universal negative theorem. | “Positive predictive advantage does not guarantee demonstrated task headroom.” | Refuted |
| JAD faithfully reconstructs all injected interaction. | Naturalized direct and JAD recovery both increase with β. | JAD h→βj R² is only 0.25957 at β=1. | Fixed JAD capacity and optimization. | “Partial, increasing recovery.” | Refuted |
| Product Joint is universally sufficient. | v5.6.1 nested residual audit. | One modest residual class only; generic MLP audit inconclusive. | Does not cover all hypothesis classes. | “No substantial extra MELD headroom under the tested nested extension.” | Not claimed |
