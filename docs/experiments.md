# Experiments: summary (as of 2026-10-22)

One line per tested idea; details in `v3_variant_log.md` and `v4_variant_log.md` (and `decisions.md` for
v1/v2). All on development data ≤ 2026-10-10; the holdout (10-11 → 11-01) is untouched until after Nov 2.
Mean net is per trade after costs; LB = one-sided block-bootstrap lower bound.

| # | Idea | Trades | Mean net | Lower bound | Folds + | Verdict |
|---|---|---|---|---|---|---|
| v1 | Barrier model, post-grad ≤ 18h | — | failed its first out-of-sample test | — | — | In holdout |
| v2 | v1 + flow + metadata host, rule R0 (p95) | 2,440 | +0.11% | −1.45% (90%) | 3/4 | Frozen, in holdout |
| v3-B R1 | v2 with a stricter cutoff (p98) | 553 | +2.93% | −0.61% (90%) | 3/4 | Best dev result → holdout as v2-R1 |
| v3-B R2–R6 | other cutoffs / venue / cost-aware rules | 49–2,275 | −0.3% to +10.1% | all < 0 | 1–3/4 | Rejected |
| v3-B R7 | top-5 per day | 124 | +8.04% | — | 4/4 | Discarded: look-ahead |
| v3-B R9 | causal top-5 (prior 24h threshold) | — | collapsed | — | — | Rejected |
| v3-10 | v2 + creator / early-buyer features | 536 | +3.16% | −1.46% (95%) | 2/4 better | Rejected |
| v3-11 | Exit ladder (half at each 2x) | 553 | +1.39% | −3.97% (95%) | 1/4 better | Rejected |
| v3-12 | Exit half at 2x, rest at 10x | 553 | +3.07% | −3.64% (95%) | 1/4 better | Rejected |
| v4-1 | Bonding curve, 1–15 min after launch | 31,746 | −1.89% | −3.50% (95%) | 0/4 | Rejected; one-bar delay → −4.5% (latency) |
| v4-2 | Post-crash rule P1 (Doug's pattern) | 3,260 | −7.76% | −9.60% (95%) | 1/4 | Rejected |
| v4-3 | Post-crash LightGBM | 219 | −2.70% | −8.60% (95%) | 2/4 | Rejected |

**What the evidence says:** buying everything loses in every window tested (−7% to −11% net); the model ranks
tokens well but the top of the ranking only reaches about break-even to +3% after 2.5–4.5% round-trip costs;
early-curve "signal" is mostly latency; exits beyond sell-all-at-2x didn't help.

**Open:** holdout verdicts for v1, v2, v2-R1 (after 2026-11-02; PREREG.md + PREREG_AMENDMENT_1.md).
**Future work (not started):** point-in-time social data (watch-list posts) and tracked wallets
(`tracked_wallets.md`).
