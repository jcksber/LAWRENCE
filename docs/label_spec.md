# Label spec (v1) — regression target

## Definition

```
y = (price_at_exit - price_at_entry) / price_at_entry
```

- **entry**: first price at which a buy would realistically fill after pool
  creation (i.e. after `init`-style pool-creation event) — pulled from actual
  swap data, not the theoretical initial pool ratio, so it reflects real
  slippage on entry.
- **exit**: price at `entry_timestamp + HOLD_MINUTES`. `HOLD_MINUTES` starts as
  a placeholder (60) and should be re-derived empirically once Bronze data is
  flowing — plot the price-decay / pump curve distribution for a sample of
  launches and pick a hold window that separates "still finding out if this
  is a rug" from "outcome is basically decided."
- If liquidity is pulled (rug) before `exit`, clamp `y` at the realized loss
  at the liquidity-pull event rather than reading a stale/zero price — a
  token with no liquidity has no meaningful "price."

## Point-in-time correctness (read this before writing any feature)

Every feature used to predict `y` must be computable using **only** data
timestamped at or before `entry_timestamp`. Concretely:

- Liquidity depth, holder count, deployer wallet history: snapshot as of
  `entry_timestamp`, never a later value.
- Social signal (Phase 2): only posts/mentions timestamped before
  `entry_timestamp`.
- No feature may be derived from anything that happened between
  `entry_timestamp` and `exit_timestamp` — that's the window we're trying to
  predict, and leaking it in is the single most common way these models look
  great in backtest and fail live.

## Open questions to revisit once real data lands

- Is a single global `HOLD_MINUTES` right, or should it vary by initial
  liquidity size (thinner pools may resolve faster)?
- How do we handle tokens with a trading halt / no volume before `exit`
  (illiquid exit — can't actually sell at the "price")? Likely: model exit
  price using actual executable swap size, not last-trade price.
- Outlier handling for extreme (>100x) winners — winsorize `y` for training
  stability, but keep the raw value for backtest P&L simulation.
