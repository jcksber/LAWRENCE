# Label spec (v2) — entry and exit are separate models, any cadence

Supersedes v1's single "return from pool creation" framing. Two changes,
both from a real design conversation, not implementation detail:

1. **Not gated on pool creation.** LAWRENCE assesses whatever it's watching
   on a recurring cadence, not just once at launch. Pool creation is just
   the first possible assessment point for a token, not the only one.
2. **Entry and exit are separate models**, each with its own label and
   feature set — not one regression reused for both. They have different
   decision costs (entry scans a wide candidate set; exit only evaluates
   whatever's actually held) and different feature needs (exit needs
   position context entry doesn't), so keeping them separate lets each be
   tuned for its own latency/cost budget instead of forcing one model to
   serve both cheaply-at-scale and quickly-on-a-small-set at once.

## Core idea: assessment timestamp `T`

Both labels are defined relative to an **assessment timestamp `T`** — any
point in a token's trading life, not just its first trade. In training,
`T` is sampled across each token's history (frequency TBD, see below); in
production, `T` is "now," each time LAWRENCE runs.

**Cadence (how often `T` recurs — 15 min, 1 min, streaming/continuous) is a
deployment/cost parameter, not something baked into the label or feature
code.** The label definition is identical whether LAWRENCE runs every 15
minutes or every second; only the infra cost changes. Nothing in Silver/
Gold or the model should hardcode a specific interval.

## Entry label

```
y_entry(T) = (price_at(T + HOLD_MINUTES) - price_at(T)) / price_at(T)
```

- Candidate universe at `T`: tokens with at least one trade in a trailing
  activity window (placeholder: 24-72h — wide enough to catch tokens that
  are still "alive," narrow enough not to keep scoring dead ones forever;
  tune once real activity-decay curves are visible in Silver/Gold).
- `price_at(T)`: actual fillable price from real swap data at/near `T`, not
  a theoretical pool ratio — same slippage-aware logic as v1.
- Same rug/no-liquidity clamping as v1: if liquidity is pulled before
  `T + HOLD_MINUTES`, clamp `y_entry` at the realized loss at the pull
  event rather than reading a stale price.
- Features: trailing trade activity (liquidity depth, volume, buy/sell
  ratio, holder concentration, deployer wallet history) and, later, social
  signal — all snapshotted at-or-before `T`. No position context (there's
  no position yet).

## Exit label

```
y_exit(T) = (price_at(T + HOLD_MINUTES) - price_at(T)) / price_at(T)
```

Same forward-return shape as entry — deliberately, so "hold vs. sell now"
reduces to "is the expected forward return from here still worth the
risk," using the same kind of signal the entry model already trusts. What's
different is the **candidate universe and feature set**:

- Candidate universe at `T`: only tokens LAWRENCE currently holds a
  position in (small set, by construction) — never the full market.
- Features: everything the entry model uses, **plus position context**:
  entry price, unrealized return so far, holding duration, and (once we
  build it) whatever risk-budget state the decision agent is tracking.
  These matter for exit in a way they don't for entry — a token trading
  flat is a "pass" for entry but a live P&L decision for something already
  held.

## Point-in-time correctness (generalized from v1 — read before writing any feature)

Every feature for either label must be computable using **only** data
timestamped at or before its `T`. This is the same rule as v1, just no
longer tied to a pool-creation timestamp specifically:

- Liquidity depth, holder count, deployer wallet history, position context:
  snapshot as of `T`, never later.
- Social signal (deferred — see decisions.md): only posts/mentions
  timestamped before `T`.
- No feature may see anything in `(T, T + HOLD_MINUTES]` — that's the
  window being predicted, and leaking it in is the #1 way these models look
  great in backtest and fail live.

## Gold tables: separate for entry and exit

`lawrence.gold.entry_candidates` and `lawrence.gold.exit_candidates` are
two physically separate tables, not one table with a row-type flag --
mirrors the same reasoning already used to keep entry/exit as separate
*models* (different candidate-universe sizes, different serving-time query
shapes, different latency/cost budgets). A combined table wouldn't actually
avoid the split: every serving-time read would still filter on row type,
you'd just be filtering a column instead of picking a table, while also
carrying permanently-null position-context columns on every entry row.

What *is* shared: the underlying feature computation (trailing volume,
buy/sell ratio, price action from Silver) lives in one function, called by
both tables' build logic. No duplicated feature logic, still two tables.

## Exit label: training data via simulated entries

Exit's position-context features (entry price, unrealized return, holding
duration) need a "when did I enter" — but there's no real position history
yet (that only exists once LAWRENCE is actually paper-trading, Phase 4).
For v1 training data, entries are simulated: for a candidate token at
assessment point `T`, manufacture a synthetic entry at `T - h`:

```
entry_price       = price_at(T - h)
unrealized_return = (price_at(T) - price_at(T - h)) / price_at(T - h)
holding_duration  = h
y_exit(T)         = (price_at(T + HOLD_MINUTES) - price_at(T)) / price_at(T)   # unchanged
```

`h` is sampled from a grid (placeholder, tune later) independently of
whatever the entry model's actual trigger logic would have decided --
confirmed as a deliberate simplification, not an oversight: entry and exit
training data are generated independently, not from a coupled simulation
where "exit training only samples h values the entry model would actually
have acted on." This is standard hindsight relabeling (any point along a
token's real price trajectory can retroactively serve as "if I'd bought
here") and is explicitly a training-time stand-in -- real paper-trading
position history is the ground truth this substitutes for until it exists.

## Open questions to revisit once real data lands

- Training sample cadence: sample `T` every 15 min per token to match an
  assumed production cadence, or denser/coarser? (Matters for train/serve
  skew if production cadence ever changes — worth deciding whether training
  data should be cadence-independent, e.g. sampled densely and then
  resampled to whatever cadence is actually deployed.)
- Is a single global `HOLD_MINUTES` right for both labels, or should exit
  use a shorter horizon than entry (faster-reacting risk management vs.
  slower-reacting opportunity scanning)?
- Is a single global `HOLD_MINUTES` right, or should it vary by initial
  liquidity size (thinner pools may resolve faster)? (carried over from v1)
- How do we handle tokens with a trading halt / no volume before
  `T + HOLD_MINUTES` (illiquid exit — can't actually sell at the "price")?
  Likely: model exit price using actual executable swap size, not
  last-trade price. (carried over from v1)
- ~~Outlier handling for extreme (>100x) winners — winsorize `y` for
  training stability, keep the raw value for backtest P&L simulation.
  (carried over from v1)~~ **Resolved, 2026-09-17, see decisions.md:**
  "keep the raw value for backtest P&L simulation" didn't survive contact
  with real data. A real Gold run had 779 rows (0.061%) with `y_entry` in
  the hundreds-of-thousands-to-16.7-billion range — tokens priced at 1e-8
  to 1e-15 USD (plausible real data: huge supply, near-zero market cap),
  where a tiny absolute price move produces an astronomical percentage
  "return" no real trade could capture without knowing actual pool depth
  (the same liquidity-data gap tracked in gold_spec.md/game_plan.md's
  Phase 1). `gold_features.py::winsorize_return` now clips to
  `RETURN_WINSORIZE_CAP` (100.0, i.e. this section's own ">100x" framing)
  for BOTH training (`src/models/entry_model.py`) and backtest
  (`src/backtest/walk_forward.py`) — an unexecutable 18-billion-x number
  is exactly as misleading in a backtest P&L sum as it is during
  training. Gold's own stored values stay raw/unclipped; only consumers
  clip. Still a placeholder cap, not a measured one — revisit once real
  liquidity-depth data exists to separate genuine big winners from
  unexecutable thin-pool noise.

## 2026-10-01 changes (Jack approved)

1. **No rug-clamp where liquidity can't be pulled.** The interim rule
   "nothing trades near T+hold -> label = -100%" assumed an LP that can be
   withdrawn. That's false on a bonding curve (SOL sits in the curve
   program; you can always sell back at the curve price) and after
   pump.fun / LaunchLab graduation (LP burned or locked). For records in
   phase `curve` or `postgrad`, the exit price is the token's LAST price
   at or before T+hold, with no staleness limit. The -100% clamp now only
   applies where liquidity can actually be pulled (records with no phase,
   i.e. raw AMM trades). Slippage on exiting a thin/dead market is real
   but belongs in the backtest's execution-cost model, not the label.
2. **Censoring is not a rug.** If T+hold is past the token's
   `observed_until` (end of the data we pulled for it), that label is
   NULL / the row is skipped -- never -100%.
3. **Shorter primary horizon + several horizons.** `HOLD_MINUTES` (primary
   `y_entry` / `y_exit`) is now 15, down from 60. Every row also carries
   `y_entry_{5,15,60}m` (and `y_exit_*`), NULL where censored, so the
   horizon can be chosen from data instead of guessed.
4. **Denser sampling.** T is sampled every minute for a token's first hour
   after launch, then every 15 minutes.
5. **Activity filter (2026-10-02).** In v2 sampling, an assessment point T
   is kept only if the token had a trade/bar within the 2 minutes before T
   (`ACTIVE_WINDOW`). Gold v2's first run showed 68-71% of 15m labels were
   exactly 0 for T 15+ min after launch -- idle moments on tokens that had
   already gone quiet, which a live bot wouldn't be scoring. Labels are
   unaffected; only which moments are sampled changes.

## Barrier label (2026-10-08, Jack approved)

Why: the fixed 15-minute return (`y_entry`) never cleared ~3% round-trip
costs in any segment over 50 days, while results at a 100x cap were
positive -- the profit is in rare runs a fixed exit throws away. The new
label asks the question Jack's trade actually depends on (sell half at 2x
to recover the stake).

For each entry row at T, entry price `p` = `price_at_t`, and each horizon
N in `BARRIER_HORIZONS_MIN` = 60, 360, 1440 minutes:

- Walk the bars that finished after T (trade_time > T) up to T + N.
- **down** if a bar's low <= 0.5 p; **up** if a bar's high >= 2.0 p. A bar
  touching both counts as **down** (order unknown -- conservative).
- **timeout** if neither by T + N.
- **censored** (NULL) if neither was hit and the token is unobserved
  before T + N (`observed_until`, same rule as `y_entry`).

Columns: `y_up_{N}m` (1.0 up / 0.0 down or timeout / NULL censored),
`r_up_{N}m` (realized return of the rule, before costs: up = +1.0; down =
-0.5, or the bar's close if the whole bar traded below the stop; timeout
= price at T + N, quiet curve/postgrad tokens keeping their last price),
`t_up_{N}m` (minutes from T to resolution).

Known limits: entry at the last close at T (no fill delay or slippage in
the label); intra-bar order unknown; the full position is modelled, not
the half-sell -- the half-sell is an exit-layer rule on top of this. The
24h horizon is censored for most post-graduation rows (bars stop 24h
after graduation).
