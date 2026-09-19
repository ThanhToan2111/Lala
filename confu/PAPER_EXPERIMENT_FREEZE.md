# ConFu++ v6.0 paper experiment freeze

Frozen on 2026-09-19. The paper uses the existing ConFu/MOSEI, MUStARD, MELD, v5.6.1 nested audit, v6.0 calibration, and v6.0 Naturalized IPIB artifacts listed in `results/registry.json`.

## Frozen protocol

- No architecture search, rank sweep, optimizer sweep, or post-result threshold tuning.
- Historical operational gate remains `τ=0.01` for comparability.
- Null calibration uses alignment-breaking source shuffles within each split, `B=10` per setting because the larger null was compute-prohibitive in this run; this is a finite-sample audit, not a replacement for a larger preregistered null.
- Naturalized IPIB uses canonical MOSEI V/A source geometry, fixed projection/generator matrices, interaction rank 32, private strength 0.25, β in `{0, 0.25, 0.5, 1}`, and seeds `{1,...,5}`.
- Test metrics are descriptive. Validation selects checkpoints or reports operational `J`; test is never used to choose a threshold.

## Main paper conclusions frozen

1. `J` is a squared-risk projection gap under explicit representation and hypothesis-class assumptions; it is not PID synergy, causality, or information creation.
2. Natural cross-modal predictive advantage is dataset/representation dependent: MOSEI `VA→T` is positive, while MUStARD and MELD screens are weak or negative under the frozen protocol.
3. The nested v5.6.1 audit found no substantial extra MELD headroom beyond Product Joint under its fixed residual budget.
4. Naturalized IPIB recovers the intended interaction-strength ordering at the predictor level, but recovery and accessibility remain separate measurements.
5. The MOSEI alignment-breaking null has `q99=0.02263`, above `τ=0.01`; therefore the historical gate is descriptive in that setting and is not a calibrated 1% false-positive threshold.

No new architecture claim is admitted after this freeze. Any future threshold or dataset extension requires a new version and a preregistered protocol.
