# Gold spec (v1) — `lawrence.gold.entry_candidates` / `lawrence.gold.exit_candidates`

Two physically separate tables, both training-data outputs (not a
production-serving path yet — see "Training vs. serving" below). Design
rationale for the split, the simulated-entry approach for exit, and the
open questions this still carries are in `docs/label_spec.md`; this doc is
the concrete schema + implementation notes.

## Training vs. serving

Gold v1 is a **batch/backfill job**, not a scheduled pipeline step —
`notebooks/02_gold_entry_candidates.py` and
`notebooks/03_gold_exit_candidates.py` are *not* wired into
`resources/lawrence_job.yml`'s recurring schedule the way Silver is. Each
run reads the current full `lawrence.silver.token_trades`, samples
assessment points `T` across every token's history (a 15-min grid, matching
the deployed Bronze/Silver cadence — see `T_SAMPLE_INTERVAL` in
`src/features/gold_features.py`), and does a full `overwrite` of the Gold
table. Re-run by hand whenever Silver has accumulated enough new history to
be worth retraining on.

This is deliberately different from Bronze/Silver's incremental
append/MERGE pattern: there's no model to serve yet, so there's no
"current candidates right now" production path to build — only a training
set to backfill. A real-time serving path (`T = "now"`, one row per
actively-scanned token, incremental) is future work once a model exists to
feed it.

## Feature set (v1 — Silver-only)

Both tables share the same trailing-activity features
(`src/features/gold_features.py::trailing_activity_features`), computed
over `(T - TRAILING_WINDOW, T]`:

| column | notes |
|---|---|
| `trailing_volume_usd` | sum of `amount_usd` in the window |
| `trailing_trade_count`, `trailing_buy_count`, `trailing_sell_count` | |
| `trailing_buy_sell_ratio` | Laplace-smoothed `(buy+1)/(sell+1)` — avoids divide-by-zero on an all-buy or all-sell window |
| `trailing_price_change_pct` | first-to-last price change in the window |

**Not included, on purpose:** liquidity depth, holder concentration,
deployer wallet history — `label_spec.md` lists these as entry features,
but Silver's `token_trades` (a pure trade log) can't supply them. They need
new Bronze sources (pool-reserve-state snapshots, wallet/holder data) that
don't exist yet.

## `entry_candidates` schema

`token_id`, `chain_family`, `network`, `dex_venue`, `dex_protocol`,
`assessment_time` (T), `price_at_t`, `hold_minutes`, the trailing-feature
columns above, `y_entry`.

## `exit_candidates` schema

Same columns as `entry_candidates`, plus: `simulated_holding_minutes` (h),
`entry_price` (synthetic, `price_at(T - h)`), `unrealized_return`,
`y_exit`. One `(token_id, T)` pair can produce multiple exit rows — one
per `h` in `EXIT_SIMULATED_HOLDS` for which a synthetic entry price exists.

## KNOWN GAP: rug-clamping is not implemented — rows are dropped instead

`label_spec.md`'s point-in-time rule says a token whose liquidity is
pulled before `T + HOLD_MINUTES` should have its label **clamped at the
realized loss at the pull event**, not read from a stale price. We have no
liquidity/pool-state data to detect an actual pull event — only a trade
log — so `build_entry_row`/`build_exit_rows` instead **drop** the
`(token_id, T)` row entirely when no trade exists near `T + HOLD_MINUTES`
to serve as an exit price (see `price_at`'s `max_staleness` cutoff).

This is a real survivorship-bias risk: it quietly excludes likely-rugged
or now-dead tokens from training data instead of labeling them as the
large losses they probably are — precisely what `game_plan.md` §2 warns
backtests lie about ("smooth 50%+ average return... almost certainly
survivorship bias"). Flagged here rather than silently shipped; revisit
once Bronze ingests pool-liquidity state and an actual pull event can be
detected.

## Expected output volume right now

Both notebooks will produce few or zero rows against Bronze/Silver's
current (just-started) history — `build_entry_row` needs both a trailing
feature window *and* a real trade near `T + HOLD_MINUTES`, and
`build_exit_rows` additionally needs a synthetic entry price further back
still. That's expected given how little history has been ingested so far,
not a bug — re-run after Bronze/Silver have accumulated more.
