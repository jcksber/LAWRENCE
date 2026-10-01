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
columns above, `y_entry`, `is_rug_clamped` (see "Rug-clamping" below).

## `exit_candidates` schema

Same columns as `entry_candidates`, plus: `simulated_holding_minutes` (h),
`entry_price` (synthetic, `price_at(T - h)`), `unrealized_return`,
`y_exit`. One `(token_id, T)` pair can produce multiple exit rows — one
per `h` in `EXIT_SIMULATED_HOLDS` for which a synthetic entry price exists.
All exit rows for the same `(token_id, T)` share the same `y_exit` and
`is_rug_clamped` value (both are computed once from `T`/`T+hold_minutes`,
independent of `h`).

## Rug-clamping: interim heuristic (2026-09-17), not the real thing yet

`label_spec.md`'s point-in-time rule says a token whose liquidity is
pulled before `T + HOLD_MINUTES` should have its label **clamped at the
realized loss at the pull event**, not read from a stale price. We still
have no liquidity/pool-state data to detect an actual pull event — only a
trade log — so a *real* implementation of that rule isn't possible yet
(tracked as essential-but-deferred work in `game_plan.md`'s Phase 1).

Rather than keep silently dropping these rows (a real survivorship-bias
risk — precisely what `game_plan.md` §2 warns backtests lie about,
"smooth 50%+ average return... almost certainly survivorship bias"),
`build_entry_row`/`build_exit_rows` now use the trade log itself as a
proxy signal: if `T` has a real price but no trade exists near
`T + HOLD_MINUTES` to serve as an exit price (see `price_at`'s
`max_staleness` cutoff), the row is **kept**, `y_entry`/`y_exit` is
**clamped to `RUG_CLAMP_RETURN`** (a full realized loss) instead of
dropped, and `is_rug_clamped=True` is set so these rows stay inspectable
and filterable rather than silently mixed in.

This is explicitly a heuristic, not real liquidity-pull detection — it
can't distinguish an actual rug from a token that just stopped trading for
other reasons — but it removes the survivorship bias that dropping the
row would otherwise create. A row with a genuine data gap at `T` itself
(no trade at or before `T`) is still dropped — that's a data-availability
issue, not a rug signal. Swap this heuristic for real pull-event detection
once Bronze ingests pool-liquidity state.

## Expected output volume right now

Both notebooks will produce few or zero rows against Bronze/Silver's
current (just-started) history — `build_entry_row` needs both a trailing
feature window *and* a real trade near `T + HOLD_MINUTES`, and
`build_exit_rows` additionally needs a synthetic entry price further back
still. That's expected given how little history has been ingested so far,
not a bug — re-run after Bronze/Silver have accumulated more.

## Gold v2 input (2026-10-01)

Gold now reads `lawrence.silver.token_price_1m` (one 1-minute series per
token, single price unit) through `src/features/gold_input.py`, instead of
raw `silver.token_trades`:

- **Candidates** = tokens with a `curve` phase (launch observed). Postgrad-
  only and amm-only tokens are excluded.
- **Records** are bars: `trade_time = bar_start + 1 min` (close known only
  once the minute ends), `price = close`, `n_trades`/`n_buys` drive the
  trailing counts.
- **Censoring, not rug-clamping**: each token has an `observed_until`
  (first post-grad bar + 24h for graduates; end of the curve backfill,
  2026-09-13, otherwise). Any T with T + HOLD_MINUTES past it is skipped.
  Before v2, a graduated token's trades stopped at graduation, so its
  post-graduation labels were rug-clamped to -100% -- the opposite of what
  happened.
- **New columns** on both Gold tables: `price_unit`, `minutes_since_launch`,
  `is_postgrad_at_t`. Not yet in the model's feature list.
- `dex_venue` = launch venue (pumpdotfun / raydium_launchlab);
  `dex_protocol` is NULL (no single protocol across curve + post-grad).
