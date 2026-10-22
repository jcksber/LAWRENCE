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
