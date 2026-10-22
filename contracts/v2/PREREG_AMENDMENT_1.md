# Pre-registration amendment 1: add "v2-R1" to the 2026-10-11 → 11-01 holdout

Written 2026-10-22 on branch `v3`. **No data after 2026-10-10 has been pulled or looked at.** This file,
`contracts/v2/amendment_r1.json` (from notebook 04i), `src/backtest/holdout_eval_2026_10_amend1.py`
(+ `tests/test_holdout_eval_amend1.py`) and the new last cell of `notebooks/06_holdout_eval_2026_10.py`
are committed together, BEFORE the holdout pull (after 2026-11-02). That commit is the amendment.
`PREREG.md`, `holdout_eval_2026_10.py` and everything in `contracts/v1/` and `contracts/v2/` stay unchanged.

## What is added
| | v2-R1 |
|---|---|
| Model | the frozen v2 model, unchanged (same file, same sha256 `3e6fdf53…288a`) |
| Cutoff | 98th percentile of that model's predicted EV on its training rows (04i), instead of v2's 95th |
| Rule otherwise | identical to PREREG.md: one position per token, exit 2x / −50% / 6h, same costs |

## Why, and the bias it carries
In v3 development (`docs/v3_variant_log.md`) the stricter cutoff, rule R1, was the best of 8 rules on the
same 4 folds: 553 trades, +2.93% mean net per trade, 90% lower bound −0.61%, 3/4 folds positive, vs +0.11%
for v2's own rule. R1 was chosen AFTER seeing those results, so its development numbers are optimistic. The
untouched holdout is exactly the test that removes that bias. The dev fold models re-estimated the
cutoff per fold; the frozen version uses one model and one cutoff, as v2 does.

## Verdict changes
- **Three tests on one window → alpha = 0.10 / 3 ≈ 0.0333 per model** (Bonferroni), for v1, v2 AND v2-R1.
  This makes v1 and v2 stricter than PREREG.md (0.05). Tightening a test before seeing data is allowed;
  loosening is not.
- PASS / INCONCLUSIVE / INVALID definitions are otherwise unchanged (lower bound > 0 at base AND mean > 0
  at stress; < 100 trades = INCONCLUSIVE; > 20% excluded days = INVALID).
- v2 and v2-R1 share a model, so their results are correlated; Bonferroni is conservative here.
- **The official verdicts are the "Amendment 1" cell of notebook 06.** The earlier cell's alpha = 0.05
  verdicts are printed only to show the original plan and are not judged.

## Safety checks built into 04i (the amendment is void if either fails)
- The training rows reproduce the manifest's `train_rows` exactly.
- The 95th percentile of the frozen model's predictions reproduces the manifest's `cutoff` within 1e-9.
