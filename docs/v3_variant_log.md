# v3 variant log

Every variant tried for v3 goes here: the variant, its development result and the date, **including failures**. Development data: post-grad rows 2026-09-13 → 10-10, rule 7, 4 weekly walk-forward folds (04d/04f). Budget: ~10 variants across B (rules), C (features), D (exit labels). The holdout windows (2026-10-11 → 11-01 reserved for v1/v2; later windows for v3) are never used here.

Reading guide: "mean net" is per trade after base costs (0.5%/side slippage). LB90 is the 90% block-bootstrap lower bound (token-days). The best of N variants is optimistic by construction.

## B — trade-selection rules on the v2 model (04f)
Run 2026-10-18. Variants 1–8 of the ~10 budget. Same v2 model per fold; only the selection rule differs.

| rule | trades | /day | mean net base | mean net stress | LB90 | folds>0 |
|---|---|---|---|---|---|---|
| R0 v2 rule (p95, all venues) | 2440 | 97.6 | +0.0011 | -0.0089 | -0.0145 | 3/4 |
| R1 p98 | 553 | 22.1 | +0.0293 | +0.0193 | -0.0061 | 3/4 |
| R2 p99 | 229 | 9.2 | -0.0002 | -0.0102 | -0.0574 | 1/4 |
| R3 LaunchLab only, p95 | 515 | 20.6 | +0.0059 | -0.0041 | -0.0267 | 2/4 |
| R4 pump.fun only, p95 | 1925 | 77.0 | -0.0003 | -0.0103 | -0.0178 | 2/4 |
| R5 cost-aware: EV-cost >= train p95 of EV-cost | 2275 | 91.0 | -0.0029 | -0.0129 | -0.0187 | 2/4 |
| R6 top-2/day by EV-cost (EV-cost > 0) | 49 | 2.0 | +0.1009 | +0.0909 | -0.0207 | 3/4 |
| R7 top-5/day by EV-cost (EV-cost > 0) | 124 | 5.0 | +0.0804 | +0.0704 | +0.0063 | 4/4 |

Per-fold mean net (folds 1–4):
- R6: +6.8%, +12.7%, −14.1%, +33.3%.
- R7: +11.1%, +8.7%, +0.0%, +13.7%.

Reading:
- R0 reproduces 04d (+0.11%), so the harness is consistent.
- Per-day ranking (R6/R7) beats fixed cutoffs (R1/R2). A fixed train-p95/p99 cutoff drifts as daily score levels shift; ranking within the day doesn't. That's a plausible mechanism, not proven.
- R7 is the only rule with LB90 > 0 and 4/4 positive folds. It's the best of 8, though, so its +8.0% is optimistic, and LB90 barely clears 0 on 124 trades. With multiplicity it's NOT significant.
- R6's +10% rests on 49 trades and one −14% fold. Too few to trust.
- **No further k-tuning** (k=3, 4, 6, …): that would be fitting the noise in these 124 trades.


### R9 — causal version of R7 (variant 9), 2026-10-18
R7 ranked against the whole UTC day (including later candidates), so it can't be traded live. R9 uses only the previous 24h: EV−cost ≥ the 5th-best per-token EV−cost of the prior 24h (hourly refresh) and > 0, first-come up to 5/day, one position per token. Code: `src/backtest/causal_rules.py` (+4 tests).

| rule | trades | tokens | mean net | stress | LB90 | ex top-1 token | ex top-5 tokens | top-5 share of P&L | positive days | folds>0 |
|---|---|---|---|---|---|---|---|---|---|---|
| R7 (hindsight) | 124 | 122 | +8.04% | +7.04% | +0.63% | +7.30% | +4.24% | 49% | 60% | 4/4 |
| R9 (causal) | 90 | 77 | +1.03% | +0.03% | −8.56% | −1.14% | −8.31% | 833% | 43% | 3/4 |

R9 per fold: +14.4%, +5.6%, −24.5%, +19.8%.

**Verdict: R9 collapses.** Its whole profit is 5 tokens; without them it's −8.3%. Most of R7's +8% came from ranking with hindsight across the day, not from a tradeable rule. **No B rule is frozen.**
- The best causal rule is R1 (p98): +2.9%, LB90 −0.6%, 3/4 folds. It's the best of 9, so optimistic. It's kept as the default rule for v3's feature work (C), and won't be tuned further.
- **Lesson for the log:** any per-period ranking rule must be checked for look-ahead in the ranking set itself, not only in features and labels.
- Budget used: 9 of ~10. C and D need their own small, declared budget.

### C data source fix (2026-10-18)
00k first run read only `pump_call_create` (248 launches, 10 graduates on 2026-09-20).
Coverage check: `pump_evt_createevent` = 30,783 = 248 `create` + 30,535 `create_v2`.
`creator_flow._creates` now reads the CreateEvent table (adds `quote_mint`, `is_mayhem_mode`).
One-day cost so far: cells 3-5 = 23 credits, cell 6 export = 13 (on the 248-launch subset; rerun).

### C data pulled (2026-10-18, 00l)
`bronze.pump_launches` 1,594,455 rows (08-30 → 10-10); `bronze.pump_early_buyers` 40,990 graduates
and `bronze.pump_creator_trades` 403,921 rows (created 09-06 → 10-10). ~1k Dune credits. All reads < 10-11.
Features built in `notebooks/05b_creator_features.py` → `gold.creator_token_features_v3`, `gold.creator_flow_features_v3`.
Planned comparison (declared before any fit): budget used 9 of ~10, so C gets ONE fit (variant 10):
V2b + all C features (`eb_*`, `cr_*`, `ct_*`, `launch_*`) vs the already-logged V2b, same 4 folds, rule R1 (p98) fixed.
Adopt only if pooled mean net AND its 95% lower bound beat V2b, and ≥3 of 4 folds improve. Feature
importances are reported but no feature subset is re-fit (that would be variant 11+).

### Variant 10 result (2026-10-18) — REJECTED by the pre-declared rule
| model | trades | mean net base | mean net stress | LB95 | hit 2x | LaunchLab share |
|---|---|---|---|---|---|---|
| V2b R1 | 553 | +0.0293 | +0.0193 | -0.0154 | 0.273 | 0.128 |
| V10 = V2b + C, R1 | 536 | +0.0316 | +0.0216 | -0.0146 | 0.297 | 0.058 |

Per fold (mean net, V2b → V10): f1 +0.0003 → -0.0262, f2 +0.1181 → +0.0723, f3 +0.0398 → +0.0522, f4 -0.0121 → +0.0346.
Mean and LB95 both slightly better, but only 2/4 folds improved → REJECT. Gain of +0.23 pts/trade is noise-sized.
Note (not acted on): the two later folds improved, the two earlier ones got worse — consistent with C features
needing more training history (C data starts 09-06), but testing that would be variant 11.
Budget: 10 of ~10 used. No v3 model is frozen. v1/v2 go to the 11-02 holdout as pre-registered.

### Variant 11 (declared 2026-10-22, before running) — exit ladder E1, beyond the original ~10 budget
Jack's rule: at 2x sell half (recovers the stake), then treat the rest as a new position — sell half again at
each further doubling, stop at -50% of the latest rung price, fresh 6h clock per rung. Same 553 V2b+R1 entries;
only the exit changes (`backtest.ladder_exit`, 8 tests). E0 = current sell-all-at-2x, re-simulated as a check.
Adopt only if mean net AND LB95 beat E0 at base, mean net beats E0 at stress, and ≥3/4 folds improve.
Logged honestly as an extension of the budget: one more try raises the false-positive risk slightly.

### Variant 12 (declared 2026-10-22, before running) — exit E2
Jack's rule: sell half at 2x, then the rest at 10x the entry price; the rest's stop = 1.1x the entry price
(a 45% drop from the 2x level, amended by Jack before any run so the second half also covers fees; +55% gross overall); fresh 6h clock after the first sale.
Same entries and same adoption rule as variant 11. If E1 and E2 both pass, adopt the one with the higher LB95.
Testing two exits on the same trades = two more chances for a lucky pass; noted, not corrected for.

### Variants 11-12 results (2026-10-22) — both REJECTED
Simulator check: E0 reproduces the label on 553/553 trades (mean |diff| 0.0000).
| exit | mean net base | LB95 | mean net stress | median |
|---|---|---|---|---|
| E0 sell all at 2x | +0.0284 | -0.0157 | +0.0181 | -0.1638 |
| E1 ladder | +0.0139 | -0.0397 | +0.0037 | -0.1638 |
| E2 half at 2x, rest at 10x (stop 1.1x) | +0.0307 | -0.0364 | +0.0204 | -0.1638 |
Folds improved: E1 1/4, E2 1/4. E1 rungs {0: 402, 1: 103, 2: 39, 3: 7, 5: 2}; E2 exits stop 305 / timeout 237 /
target 11; 27.3% reached 2x; 0.9% truncated at the 24h data end.
Reading: of the 151 trades that reached 2x, only 11 (7%) went on to 10x within the clock — below the ~1-in-9
needed for E2 to beat selling everything; the ladder gives back more on the ones that fade than it gains on runners.
(E0 here = +2.84% vs +2.93% from the label: the simulator charges fees on the sale's actual notional, which is
more accurate.) Exits stay E0.
