# Reviewer-question matrix

## What exactly is interaction?

In this paper, interaction means non-additive predictive structure measured by the operational score `J = R2(q_joint,Y)-R2(q_additive,Y)` under a specified representation, hypothesis class, split, and optimization protocol. It is not an unqualified PID or causal quantity.

## Why is J not PID synergy?

`J` is a squared-risk/projection gap for a target representation. PID decomposes mutual information under a different set of axioms and redundancy functional. The paper reports the distinction explicitly.

## Why predict another modality?

The held-out modality is a representation target. A positive cross-modal `J` shows predictable non-additive structure in the representation space without using downstream task labels; it does not by itself prove task benefit.

## Why should cross-modal predictability matter?

It is a screening signal for whether joint predictive structure is measurable before distillation or task evaluation. The task-headroom gate is separate precisely because predictability need not imply utility.

## Why does JAD not improve MOSEI sentiment?

The experiments show a separation between target fidelity and accessibility/task headroom. JAD can match a predictive difference while a fixed downstream probe does not obtain a reliable gain.

## Why does MELD fail G1?

Under the tested pooled utterance representations and predictors, no substantial joint advantage passed the fixed gate. This is a negative screening result, not evidence that no interaction exists in MELD.

## Could Product Joint be too restrictive?

The independent Generic MLP audit was inconclusive because it failed the MOSEI positive control. The nested residual audit preserves Product exactly and found no substantial additional MELD headroom under its preregistered modest budget.

## Why is 0.01 the threshold?

It is a frozen operational historical gate. v6.0 reports finite-sample null fluctuations and v6.1-C1 confirms on B=50 that the MOSEI observed score is not separated from the alignment-breaking null (`p_hat=0.09804`). Threshold sensitivity and C1 do not change historical labels.

## Why not train a stronger model?

Architecture exploration is hard-frozen to avoid dataset/architecture fishing. v5.6 and v5.6.1 already provide independent and nested class audits; v6.0 focuses on theory, calibration, controlled bridging, and reproducibility.

## Why no dialogue context?

MELD screening deliberately uses the v5.5 utterance-level frozen cache. Adding context would change the representation and answer a different question.

## Why no natural Type-IV setting?

No natural benchmark with both independently supported cross-modal identifiability and downstream task headroom has been established. This limitation is stated directly; Naturalized IPIB is a controlled bridge, not a replacement for natural Type-IV evidence.

## What does Naturalized IPIB add?

It preserves real MOSEI source geometry while injecting a known interaction target, private component, and fixed interaction-strength ladder. This makes recovery and false-positive behavior testable without claiming unknown natural interaction ground truth.

## How is this different from ConFu and DMIL?

ConFu learns higher-order contrastive alignment; DMIL explicitly decomposes information-inspired interaction components. ConFu++ instead evaluates representation-conditional predictive identifiability, task headroom, fidelity, and accessibility, and avoids calling its score PID synergy.

## What did the calibration confirmation change?

v6.1-C1 is calibration-only. It reuses the v6.0 MOSEI `VA_to_T` alignment-breaking shuffle with B=50 and cannot change the architecture, representations, historical gate, or prior labels. Its role is to quantify how often the observed frozen validation score is exceeded by this operational null; it does not establish universal statistical significance.

## Why retain the 0.01 gate if the null is larger?

The gate is retained as a historical operational threshold so earlier experiment labels remain comparable. If C1 shows substantial null overlap, the paper states that explicitly and does not call 0.01 a calibrated discovery threshold.
