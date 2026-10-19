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
