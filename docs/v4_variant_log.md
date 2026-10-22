# v4 variant log (started 2026-10-18)

v3 ended with no frozen model (see v3_variant_log.md). v4 changes the UNIVERSE, not the features.
Dev data ≤ 2026-10-10 only. Budget ~6 variants for the early track. Every run is logged here, good or bad.
Goal: if a variant passes, freeze it and amend contracts/v2/PREREG.md to add it BEFORE the 11-02 holdout pull.

## Early track — pump.fun bonding curve, 1–15 min after creation
- Rows: Gold entry candidates, pump.fun, pre-graduation, USD-priced, minutes_since_launch 1–15,
  T 09-13 → 10-10, same 4 walk-forward folds as v2/v3. Label r_up_360m (2x before −50%, 6h).
- Costs: 125 bps/side curve fee + 50 bps slippage = 3.5% round trip (stress 4.5%).
- New input: `gold.creator_launch_history_v4` (05c) — creator history at creation for ALL launches.
  `eb_*` early-buyer features are NOT used (graduates only → would leak graduation).
- Bar to beat (2026-10-02): first-5-min entries ≈ −3.8% over 15m before fees; curve model ~breakeven, tail-driven.

### Variant v4-1 (V4a) — declared before running
v1 features + curve flow features + creator history (cr4_*), v2 LightGBM params, rule R1 (p98 of the fold's
own train predictions), one position per token.
Promote to a freeze candidate only if ALL hold: pooled mean net > 0 at stress costs, 95% LB > 0 at base,
≥3/4 folds positive, still > 0 ex-top-5 tokens, AND mean net stays > 0 with entry delayed one bar.

### Results (2026-10-22) — 3,000,643 rows (67% token hash-sample of 4.47M), 641,856 tokens

Outcome check, buy every token at that age (first row per token), net after 3.5%:
| age | tokens | mean r | mean net | hit 2x | hit stop |
|---|---|---|---|---|---|
| 1-2m | 641,791 | -0.0489 | -0.0839 | 0.061 | 0.153 |
| 2-5m | 398,781 | -0.0366 | -0.0716 | 0.046 | 0.093 |
| 5-10m | 211,789 | -0.0390 | -0.0740 | 0.048 | 0.093 |
| 10-15m | 107,029 | -0.0354 | -0.0704 | 0.050 | 0.091 |

By creator: tokens whose creator already graduated a coin in the prior 7d are WORSE (mean net -10.4% to
-11.8%, stop hit 25-29% vs 8-11%) — consistent with repeat bundle/rug operators.

Variant v4-1 (V4a): 31,746 trades (~1,270/day), mean net -1.89% base / -2.89% stress, LB95 -3.50%,
median -2.31%, hit 2x 13.2%, 0/4 folds positive (-4.01, -2.96, -1.59, -0.76%). → REJECTED.
Delayed-entry check (50% of trades re-priced): -0.69% at the stale entry → -4.49% one bar later.
The stale entry price is worth ~3.8 points: the apparent early-curve signal is mostly latency, which a
1-minute-bar system cannot capture. Decision: early track stopped. Budget used 1 of ~6.

## Post-crash track — graduated pump.fun tokens 24h-14d after graduation (declared 2026-10-22, before any run)
- Data: `bronze.postcrash_bars_1h` (00n: 28 trade days 09-13 → 10-10, 602,559 hourly bars, ~460 credits) +
  first-24h high/low from `bronze.postgrad_bars_1m`. Pools graduated ≥ 09-12. Gold: `gold.postcrash_candidates_v4` (02c).
- Rows: hourly T, age 24h-14d, activity filter ≥ 20 trades and ≥ $2,000 volume in the prior 24h.
- Label: 72h barrier, 2x before -50% (hourly bars; tie = stop; gap = close). Last labelled T = 10-08.
- Costs: 120 bps/side PumpSwap + 100 bps slippage (stress 200) → 4.4% / 6.4% round trip.
- Folds: same 4 weekly folds; a fold trains only on rows whose 72h label resolved before the fold starts.
- Pass rule (same as v4-1): mean net > 0 at stress, LB95 > 0, ≥3/4 folds positive, > 0 ex-top-5 tokens,
  > 0 with entry at the next hourly bar's close.
- v4-2 = rule P1 (Doug pattern 1): dd_from_ath ≤ -70%, rebound_from_low ≥ +30%, ret_6h > 0; one position per token.
- v4-3 = V4b: LightGBM (v2 params) on post-crash features + creator/early-buyer features (graduates only,
  so `eb_*` is valid here), rule R1 p98 of the fold's own train predictions.

### Post-crash results (2026-10-22) — 128,949 rows, 2,495 tokens, T 09-13 → 10-08
Buy every token once a day (9,208 entries): mean net -10.6% (stress -12.6%), median -36.0%, hit 2x 19%, stop 40%.
Drawdown × rebound: every cell negative. Least bad: ≥90% below ATH with <30% rebound (-6.7%, stop 14%) and
-50..-20% with >100% rebound (-0.7%, 249 rows). Worst: within 20% of the ATH (-25% mean, stop 75%).
Doug's corner (-90..-70% with 30-100% rebound): -5.5%, stop 45%.
v4-2 P1: 3,260 trades, mean -7.8% (stress -9.8%), LB95 -9.6%, median -45.7%, 1/4 folds positive,
delayed entry -7.9% → REJECTED.
v4-3 V4b: 219 trades, mean -2.7% (stress -4.7%), LB95 -8.6%, 2/4 folds (+2.8, +9.9, -15.2, -26.9% on 6 trades),
delayed entry -5.6% → REJECTED. Top features: dd_from_ath, age_h, early-buyer supply shares.
Budget used 3 of ~6. Decision: no v4 model is frozen. Extending the pull back to 08-30 (~240 credits) is not
justified: no cell of the base-rate table is positive after costs.
