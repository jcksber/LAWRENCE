# Pre-registered Dune holdout: 2026-10-11 → 2026-11-01 (models v1 and v2)

Written 2026-10-18 on branch `social`. **No data after 2026-10-10 has been pulled or looked at.** This file, `src/models/v2_spec.py`, `src/backtest/holdout_eval_2026_10.py`, `notebooks/06_holdout_eval_2026_10.py` and `contracts/v2/` (model, manifest, host map from 04e) are committed together. **That commit is the freeze**, and it lands before any window-4 pull.

The runtime/shadow test (contracts/v1 §11) is on hold until a model passes here: no live office spend before an edge is shown on unseen history. A pass here still needs the live shadow test (slippage, latency, decoder parity) before real money.

## Models (both frozen before the window is pulled)
| | v1 | v2 |
|---|---|---|
| Artifact | `contracts/v1` (commit cd6a098), cutoff 0.0470 | `contracts/v2` (04e), cutoff from its manifest |
| Features | `barrier_model.FEATURE_COLUMNS` | `v2_spec.V2_FEATURES` = v1 + 8 flow + metadata host |
| Training rows | post-grad, rule 7, 2026-08-14 → 10-10 | post-grad, rule 7, 2026-09-13 → 10-10 |
| Chosen by | the earlier pre-registration | 04d's pre-set rule (V2b) |

## Window and data
- **Candidates:** `assessment_time` in [2026-10-11 00:00, 2026-11-01 00:00) UTC, 21 days, post-graduation, rule 7 (≤ 18h after graduation), pump.fun + Raydium LaunchLab.
- **Pull** (run after 2026-11-02 so the 24h post-graduation bars exist), with no other changes to the pipeline:
  - 00h `WINDOW_END` = 2026-11-02;
  - 00f adds window 4 = (2026-10-10, 2026-11-02);
  - 01b;
  - 02 with `CURVE_DATA_END` = 2026-11-02;
  - 00i (graduated scope, then the URI backfill cell) for the new pump.fun graduates' metadata hosts.
- **Host features:** the FROZEN `host_groups.json`; hosts not in it → `other_custom`; LaunchLab → none (as in training).

## Rule (identical for both models)
EV ≥ the model's frozen cutoff; one position per token; exit 2x / −50% / 6h; costs from `round_trip_cost_v1` (base 0.5%/side, stress 1%/side slippage).

## Verdict (per model, `holdout_eval_2026_10.evaluate_model`)
- **PASS** = 95% one-sided lower bound on mean net per trade > 0 at base costs AND mean net > 0 at stress costs.
  - The bound uses a block bootstrap over token-days, 10,000 draws, seed 17.
  - α = 0.05 per model because two models are tested on one window (Bonferroni of 0.10).
- **INCONCLUSIVE** if fewer than 100 trades.
- **INVALID** if more than 20% of days are excluded. A day is excluded if any 00h/00f chunk starting that day lacks a `success` row.
- **Reported, not judged:** median net, hit rate, top-token share of P&L, the exploratory top-2-per-day variant, and the count of pump.fun graduates missing a metadata host.

## What happens after
- **Both FAIL:** no live spend. Record the result and go back to the model.
- **Either PASSES:** that model gets a live contract (bars_v2 etc. for v2) and the runtime shadow test before any capital. A Dune pass is necessary, not sufficient.
- **No re-runs with changes.** A changed model or rule is a new pre-registration on a later window.
