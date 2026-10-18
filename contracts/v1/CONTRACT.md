# Scoring contract v1 (answer to request R1)

Owner: model side. Code: `src/serving/contract_v1.py` (pure pandas/numpy, tested against Gold in `tests/test_contract_v1.py`). Artifacts: `contracts/v1/manifest.json` + `contracts/v1/artifacts/` (written by `notebooks/04c_export_contract_v1.py`).

> **Status: SHADOW ONLY.** v1 is the Model A spec, which FAILED its pre-registered out-of-sample test on 2026-10-16 (about break-even at base costs, negative under stress). No real money through v1, ever. The point of v1 is (a) to prove the live plumbing end to end, (b) to measure live-vs-backtest skew (slippage, feature drift, data gaps), and (c) to run a second, independent pre-registered test on truly unseen data. Expect it to fail again; that's an acceptable outcome.

## 1. Model artifact
| Field | Value |
|---|---|
| File | `artifacts/model_v1.txt`: LightGBM `Booster.save_model` text, which includes the pandas category mappings |
| Version id | `v1` |
| Hash | `manifest.json → model_sha256`; `load_model()` refuses a mismatched file |
| Spec | `LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05)`, target `r_up_360m` (realized barrier return, gross) |
| Training data | Post-graduation Gold rows with a resolved 6h label that satisfy §4 rule 7 (≤ 18h after graduation), assessment_time 2026-08-14 → ~2026-10-10 (exact range in the manifest). **The booster itself is retrained on these filtered rows**; the cutoff and baselines come from the same rows. |
| Libraries | Exact versions in `manifest.json → libraries`. Pin `lightgbm` to that version; pandas/numpy/scikit-learn as recorded. Python per manifest. |

Load: `contract_v1.load_model("contracts/v1")` → `lightgbm.Booster`; score with `booster.predict(to_model_frame(rows, manifest["categories"]))`.

## 2. Feature list (ordered: `contract_v1.FEATURE_COLUMNS`)
| # | Name | dtype into model | Null handling | Source |
|---|---|---|---|---|
| 1 | trailing_volume_usd | float | never null for a kept row | bars (48h trailing) |
| 2 | trailing_trade_count | float | never null | bars |
| 3 | trailing_buy_count | float | never null | bars |
| 4 | trailing_sell_count | float | never null | bars |
| 5 | trailing_buy_sell_ratio | float | Laplace-smoothed, never null | bars |
| 6 | trailing_price_change_pct | float | NaN if the window's first close is 0/None, left NaN | bars |
| 7 | chain_family | category | always `solana` in v1 | token metadata |
| 8 | network | category | always null | constant |
| 9 | dex_venue | category | LAUNCH venue: `pumpdotfun` / `raydium_launchlab` | token metadata |
| 10 | dex_protocol | category | always null | constant |
| 11 | minutes_since_launch | float | needs launch_time (first curve bar) | bars |
| 12 | is_postgrad_at_t | float (1.0) | always 1.0 for v1 candidates | bars (phase) |

Category sets come from `manifest.json → categories`, never re-derived live. **v1 uses only bar data.** No on-chain flow stats, no pump.fun social, no X.

## 3. Feature code
`compute_features(bars, as_of, *, launch_time, launch_venue) -> dict | None`
- `bars`: pandas, one row per CLOSED 1-minute bar of the token's primary series, columns `bar_start` (UTC), `close`, `high`, `low`, `volume_usd`, `n_trades`, `n_buys`, `phase` (`curve` | `postgrad`).
  - Primary series = the bonding curve, then the graduation pool (pump.fun → PumpSwap pool; LaunchLab → its Raydium CPMM pool). Prices in a single unit per token (USD, or quote if USD is unavailable), as in `silver.token_price_1m`.
  - USD volume per bar.
- Point-in-time is enforced inside: bars with `bar_start + 1 min > as_of` are dropped, so passing extra bars can't leak.
- Returns `None` exactly where Gold would drop the row.
- Minimum history: from launch (for `minutes_since_launch`). The trailing features use up to 48h back; tokens younger than 48h use what exists, same as training.
- Implementation = Gold's own `price_at`, `trailing_activity_features`, `_v2_features`, `_meta`. Only the pandas → records mapping is new (`bars_to_records`), and it mirrors `gold_input`.

## 4. Row selection (`select_row` → `(row, reason)`)
A token-minute `as_of` is an entry candidate only if ALL hold:
1. The token was watched **from its first curve bar** (Gold excludes tokens first seen post-graduation). `launch_time` = first curve bar's `bar_start`.
2. Launch venue ∈ {`pumpdotfun`, `raydium_launchlab`}.
3. `as_of` is on Gold's grid: first bar's `bar_start` + 1 min, then every 1 min until launch_time + 60 min, then every 15 min from the last step (`assessment_grid` / `is_assessment_time`).
4. Active: a bar closed within 2 min before `as_of`.
5. Graduated by `as_of`: the last closed bar's phase is `postgrad`.
6. `compute_features` isn't None (price within 30 min, trailing window non-empty).
7. **Within the training horizon** (R3: applied to the TRAINING rows in 04c too, so cutoff and baselines are computed on the same universe live selects from): `as_of + 6h ≤ first postgrad bar_start + 24h`, i.e. the first 18h after graduation. Training only has 24h of post-graduation bars, so later rows were never seen. *(Added for R2: missing from the first draft.)* Reason `past_training_horizon`.

Reasons (`off_grid`, `inactive`, `not_postgrad`, …) should be logged as counts, not per row.

## 5. Entry, exit and costs
- **Entry:** `decide(score, cutoff)` = predicted EV ≥ cutoff, and only one open shadow position per token (skip a token until its prior trade resolves).
- **Fill assumption:** entry at `price_at_t` = close of the last bar closed by `as_of`. Log the actual first trade price after `as_of` as well; that measures real slippage against the assumption.
- **Exit:** first of
  - +100%: exact limit fill at 2× entry;
  - −50%: 0.5× entry, or that bar's close if the whole bar traded below the stop;
  - a bar touching both barriers counts as the stop;
  - 6h timeout: price at entry + 360 min.

  Only bars closing after `as_of` count.
- **Costs:** `round_trip_cost_v1(launch_venue, is_postgrad, slippage_bps)` = 2·(venue fee + slippage)/1e4.
  - Venue fee per side: pump.fun post-grad (PumpSwap) 120 bps; LaunchLab post-grad (CPMM) 25 bps.
  - Slippage per side: 50 bps base, 100 bps stress.
  - Net = gross return − round-trip cost.

## 6. Outcome resolver
`resolve_outcome(bars, entry, *, launch_venue, observed_until) -> {barrier_hit, return, exit_time, minutes} | None`
- `entry` = `{assessment_time, price_at_t}`.
- Returns `None` while unresolved (no barrier yet and `observed_until` < entry + 6h).
- Calls Gold's `barrier_outcome`, so live labels follow training rules exactly.

**Authoritative labels for the pre-registered test come from the model side,** by re-running the standard pipeline (Dune bars → 01b → 02) over the shadow window and joining on `(token_id, assessment_time)`. Live-resolved outcomes are for monitoring and for measuring skew between live and training data. If the two disagree, the pipeline's labels win, and the gap is a finding about live data quality.

## 7. Decision cutoff
`manifest.json → cutoff`: the 95th percentile of predicted EV on the v1 training rows. Same rule as the failed test. Fixed for the life of v1.

## 8. Drift baselines (`artifacts/baselines.json`)
- **Per numeric feature:** training decile edges, the share of rows per bin, and the null rate. Use them for PSI, and alert above 0.2.
- **Per categorical feature:** frequencies.
- **Score:** distribution bins, the cutoff, and the share of rows at or above it (≈ 5% by construction).
- **Base rates:** 2x-first rate and mean gross return.
- **Expected trades per day:**
  - an in-sample replay over the last 7 training days (optimistic);
  - the failed OOS test for context: about 145/day at its own cutoff.
  - Alert if live is outside 0.33×–3× of the in-sample figure for 2 consecutive days.

## 9. Schemas
**`lawrence.gold.entry_candidates` (as built by notebook 02):**
- Key: `(token_id, assessment_time)`. `assessment_time` is an ISO-8601 UTC string.
- Columns: `token_id`, `chain_family`, `network`, `dex_venue`, `dex_protocol`, `assessment_time`, `price_at_t`, `hold_minutes`, `y_entry`, `is_rug_clamped`, `trailing_*` (6), `price_unit`, `minutes_since_launch`, `is_postgrad_at_t`, flow features (8, once rebuilt), `y_entry_{5,15,60}m`, and `y_up_` / `r_up_` / `t_up_{60,360,1440}m`.
- `dex_venue` = launch venue; `chain_family` = chain.

**Shadow log (product side writes; one row per candidate scored):**

| Column | Notes |
|---|---|
| `contract_version`, `model_sha256` | must equal the manifest |
| `token_id`, `assessment_time` | join key to Gold |
| `chain_family`, `dex_venue`, `pool_address`, `price_unit` | |
| all 12 features + `price_at_t` | as computed live |
| `score`, `cutoff`, `decision` (bool) | |
| `ingest_watermark`, `bars_lag_s` | data freshness at scoring |
| `first_trade_price_after`, `first_trade_time_after` | real fill proxy |
| `outcome_barrier_hit`, `outcome_return`, `outcome_exit_time`, `outcome_minutes`, `resolved_at` | from `resolve_outcome` |
| `net_base`, `net_stress` | |

Log every candidate (decision true or false), not only trades. Unselected rows are needed to measure drift and calibration.

**Final (R2).** Add to the table above:
- `grad_pool_id` (replaces `pool_address`), `grad_time`, `primary_quote`;
- `bars_builder` (= `"bars_v1/live"`);
- `late_trades_in_window` (trades that arrived after their bar closed, in the 6h after entry).

Plus one daily table, **`shadow_skips_daily`**: `date`, `contract_version`, `reason` (every `select_row` reason + `skipped_stale`), `count`. Also log `tokens_watched`, `tokens_graduated` and `tokens_first_seen_postgrad` per day as reasons, so coverage gaps show up.

## 10. Live data requirements
- **Freshness:** score `as_of` only when the bar watermark ≥ `as_of` (the bar ending at `as_of` has closed) and `bars_lag_s` ≤ 60. Otherwise skip and count `skipped_stale`. A late bar silently changes the features, so it's skew, not noise.
- **Staleness the features tolerate:** last price ≤ 30 min (`price_at`); active ≤ 2 min (selection rule 4).
- **Primary series continuity:** the runtime must detect graduation and switch to the graduation pool, as Silver does. Wrong-pool bars change prices.
- **pump.fun callouts poller: not an input to v1.**
  - The model side owns the client, `src/ingestion/pumpfun_social.py` (`fetch_coin_social`); the runtime wraps and schedules it.
  - Collect it from day one anyway: live callouts give an unbiased history that the backfill can't, because of the 100-cap ranking.
  - Suggested cadence: every 5 min for post-graduation tokens under 6h old; once at graduation for metadata.
  - Measured throughput: about 4 tokens/s at 4 workers (3 requests per token). Back off on 429s.
- **X: not in v1.**

## 11. Pre-registration for the v1 shadow test (APPROVED by Jack 2026-10-17)
- **Freeze:** the commit that adds `contracts/v1/manifest.json`. Record `SHADOW_START` in `docs/decisions.md`: the first full UTC day after the runtime scores continuously.
- **Fixed evaluation date:** `SHADOW_START + 21 days`. The window is about covering different market days; v1 reaches the trade floor within days. Monitoring for bugs and data gaps is allowed, but shadow P&L must not trigger any change.
- **Floor:** ≥ 100 non-overlapping trades at day 21. If not met, extend once to day 35; still short → *inconclusive*.
- **Validity:**
  - Any change to the contract, artifact, cutoff or selection rule mid-window invalidates the window and restarts it under v2.
  - Days with < 95% bar coverage for scored tokens are excluded and counted; more than 20% excluded days → *invalid*.
- **Evaluation:** on pipeline-recomputed labels (§6), using the frozen rule (one position per token, 2x / −50% / 6h) and costs.
- **Evaluator (frozen code):** `src/backtest/shadow_eval_v1.evaluate(cands, cutoff, shadow_start, extended=, excluded_days=)`, tested in `tests/test_shadow_eval_v1.py`. Committed with the freeze; the verdict is whatever it returns.
- **PASS** (judged on outcomes, with an honest error bar):
  1. The **90% one-sided lower confidence bound on mean net return per trade > 0 at base costs**, and
  2. **mean net return per trade > 0 at stress costs** (1%/side).
  - The bound comes from a block bootstrap: resample **token-days** (all trades of a token on a day together), 10,000 draws, seed 17. One lucky token or one hot day can't carry it.
  - Anything else → **FAIL**. A positive average alone does not pass.
- **Reported, not judged:** per-trade distribution, per-token and per-day P&L, top-token share of P&L, live-vs-assumed fill slippage.
- **Exploratory variant (cannot pass or fail):** *top-2-per-day*, i.e. the 2 highest-scoring candidates per UTC day among rows ≥ cutoff, same exit and costs. Reported on the same window to decide whether a selective rule deserves its own pre-registration. With ~42 trades in 21 days it's far too small to judge (SE ≈ ±8 points per trade).
- **Other models:** any model whose contract is frozen (hash committed) before `SHADOW_START` may also be evaluated on this window. All such models must be listed at freeze, and the number tested is reported with the result (multiple-testing).

## 12. Bar construction (R2): how training bars were built
Training bars for 2026-09-13 on came from **Dune aggregation** (`ingestion/curve_bars.py`, `ingestion/postgrad_bars.py`) **plus Silver** (`features/silver_price_bars.py`). Bars before 09-13 came from raw trades via `trade_bars()`, with near-identical rules. Live should reproduce the Dune path. The reference implementation is **`src/serving/bars_v1.py: build_bars()`**: the runtime should import it, not rewrite it.

**Which swaps (Dune `dex_solana.trades`):**
| Phase | Rows |
|---|---|
| curve, pump.fun | `project='pumpdotfun' AND version=1`, one side is WSOL; `pool_id` = the mint |
| curve, LaunchLab | `project='raydium_launchlab' AND version=1`, pair = (pool base mint, pool quote mint) from `raydium_launchlab_v1.stg_swaps` |
| postgrad, pump.fun | `project='pumpswap'`, `project_program_id` = graduation pool, from pool `created_at` for 24h |
| postgrad, LaunchLab | `project='raydium' AND version=5` (CPMM), `project_program_id` = graduation pool, from its first trade for 24h |

**Per trade (base = the launched mint):**
- `is_buy` = base was bought.
- `quote_amount` = the other side's amount.
- `price_usd = amount_usd / base_amount`. Dune's `amount_usd` is the trade's USD value from its minute price feed (`prices.usd`); live, use quote_amount × the quote's USD price for that minute.
- `price_q = quote_amount / base_amount`.
- Order within a minute: `block_slot·1e9 + tx_index·1e3 + outer_instruction_index`.

**Dust:** curve trades with `amount_usd < $0.01` are dropped (NULL USD is kept). Post-graduation trades keep dust; Dune aggregated it in, and the outlier band below handles it.

**Per bar (UTC minute = `date_trunc('minute', block_time)`):**
- open/close = first/last priced trade by order key; high/low = max/min trade price.
- `volume_usd` = sum of `amount_usd` over priced trades.
- `n_trades` counts all trades, including unpriced; `n_buys` counts buys.

**Silver steps:**
1. **Unit:** `price_unit='usd'` if the primary quote is SOL/USDC/USDT. That covers all of pump.fun and most of LaunchLab; the live rule is `price_unit_live()`. Otherwise 'quote', using `*_q` prices only where the bar's quote = the primary quote.
2. **Bad prices:** prices ≤ 0 are treated as missing, and a bar without a close is dropped.
3. **Outlier band:** drop a bar whose close is more than 1000× from the median close; then blank any open/high/low more than 1000× from the median of the kept bars, falling back to close-based values.
4. **Same-minute clash** (the graduation minute): keep the bar with more trades; on a tie, curve wins.
5. **Empty minutes produce no bar.** There is no carry-forward; features handle gaps (`price_at` tolerates 30 min).

**Late trades:** Dune data is final. Live, close a minute's bar only after a grace period (suggested 20s after the minute ends, with `confirmed` commitment). Count trades that arrive later instead of rewriting closed bars, because a rewrite after scoring is unlogged skew.

**Training-side look-ahead the runtime must NOT copy** (found while answering R2; logged in decisions.md):
| Silver uses | Live does | Measured by |
|---|---|---|
| outlier median over the token's full 24h | expanding median (`band_mode="live"`) | `parity_report.json → live_vs_training_bar_count_diff` |
| canonical pool = busiest pool over 24h | the migration pool created at graduation | `pumpfun_canonical_is_earliest_pool` |
| USD coverage over full history (non-major quotes only) | `price_unit_live()` | count of `price_unit='usd'` tokens with a non-major quote |

## 13. Graduation (R2)
- **pump.fun:** graduation = the migration that creates the PumpSwap pool. Bars switch to that pool from its `created_at`.
- **LaunchLab:** graduation = the first trade in the token's Raydium CPMM pool (Dune has no create event in our tables).
- `phase` flips at the first post-graduation bar. Curve bars normally stop there. If both exist in one minute, the busier one wins (§12, step 4). Bars don't overlap after the merge. A gap of a few minutes at migration is normal and is left as a gap.
- Training picked the busiest post-graduation pool; live must use the migration pool. The fixture measures how often they differ.

## 14. Token discovery (R2)
- **"Watched from its first curve bar":** the runtime saw the token's first curve trade. Subscribe to pump.fun `create` and LaunchLab `initialize`; the first curve bar is the minute of the first curve trade, which is often the creator's buy in the create transaction.
- `launch_time` = that bar's `bar_start`.
- **No other up-front exclusions in Gold.** A token without trades just never produces rows: the active rule (2 min) and the price staleness rule (30 min) drop it naturally.
- Tokens first seen after graduation (runtime restarts, gaps) are excluded (`not_observed_from_launch`). Count them daily.

## 15. Environment (R2)
`manifest.json → libraries / python` records the exact versions of the run that produced the artifact; they're filled in when you run 04c. Pin the VPS image to the same `lightgbm`, `numpy` and `pandas`, plus the same Python minor version. `load_model` checks the file hash. Startup check (R3): `contract_v1.verify_startup("contracts/v1")` scores `artifacts/startup_sample.parquet` (1,000 training rows) and raises unless every prediction matches within `manifest.startup_tolerance` = **1e-9** (R4). The runtime must refuse to start on failure. x86-64 Linux is the reference platform; arm64 (Apple Silicon) is expected to pass but is unverified until Jack runs `verify_startup` on his Mac.

## 16. Parity fixture (R2)
`notebooks/00j_parity_fixture.py` → `contracts/v1/fixtures/parity_2026-10-05/`:
- 20 tokens: 12 pump.fun graduates, 4 LaunchLab graduates and 4 non-graduates, all launched 2026-10-05 00:00–12:00 UTC;
- raw Dune swaps for 10-05 → 10-07;
- `token_context.json`;
- the expected Silver bars;
- the expected features at every grid point;
- `parity_report.json`.

R3 additions: `expected_bars_training.parquet` and `expected_bars_live.parquet` (`build_bars` output per band mode), plus `quote_prices_usd.parquet` (Dune `prices.usd` minute prices for SOL/USDC/USDT, 10-05 → 10-07) to benchmark the live USD source against.

`tests/test_parity_fixture.py` asserts that the reference builder reproduces Silver (≥ 99% of bars, OHLC/volume within 1e-6) and that features reproduce exactly. The runtime's own bar path should pass the same test, fed from its live swap decoder.
