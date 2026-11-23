# Decisions log

Running record of the calls made along the way, so the "why" doesn't get lost.

- **Separate repo from BIA.** LAWRENCE is a clean build; BIA's code is not
  reused. BIA's documented behavior is studied only as a baseline strategy
  to beat (see `docs/game_plan.md` §1).
- **v1 label = regression** (realized return at a fixed exit rule), not
  binary or multi-class. See `docs/label_spec.md`.
- **Phase 1 = DEX-only ingestion.** Social data (LunarCrush/Santiment) is
  deliberately deferred to Phase 2, to validate the on-chain pipeline and a
  baseline model before taking on a paid vendor + Twitter-timing complexity.
- **First venue = Raydium** (via Bitquery's Solana DEXTrades API), matching
  the venue BIA already targets, so early results are directly comparable
  to the "buy everything new" baseline.
- **Superseded — chain/platform-agnostic, not Solana-centric.** LAWRENCE is
  explicitly *not* a Solana bot: Solana/Jito were never meant to be the
  focus, just BIA's original venue. Ingestion is now built on a `Venue`
  abstraction (chain family + DEX filter, see `src/ingestion/bitquery_client.py`)
  with Raydium (Solana) and Uniswap v3 (Base) as the two starting venues,
  specifically to prove the abstraction generalizes rather than becoming
  "Solana with extra steps." Execution-layer, chain-specific speed tricks
  (Jito bundle tips, etc.) are deferred to a later, venue-specific execution
  adapter — never assumed by the core pipeline.
- **Success metric = risk-adjusted, not a guarantee.** Tracking EV/trade,
  Sharpe/Sortino, hit rate, max drawdown — see `docs/game_plan.md` §2 for
  why "guarantee >50% average profit" isn't the target.
- **Bitquery auth: use a "Manual"-type Application token, not client_credentials.**
  The raw dashboard API key is not a usable bearer token by itself (returns
  401). Fix: create an Application under Authorization -> Applications
  (used name `lawrence-ingestion`) -- its "Manual" type shows `secret: n/a`
  (no client_id/client_secret flow) but issues tokens via its own "Tokens"
  UI with a 1-year lifespan, long enough that no OAuth refresh logic is
  needed in `bitquery_client.py`. That generated token is what's stored as
  the `bitquery_api_key` Databricks secret.
- **Phase 1 ingestion validated end-to-end (2026-09-14).** Ran notebook 00
  manually against both starting venues: `raydium_v4` -> `dex_protocol =
  raydium_amm`, `uniswap_v3_base` -> `dex_protocol = uniswap_v3`, 1000 rows
  each, each venue showing exactly one protocol label with no
  cross-contamination -- confirms the `Venue` filter is actually scoping
  correctly, not just failing to error. Job stays `pause_status: PAUSED`
  until intentionally switched to the automated 15-min schedule.
- **Social-data Bronze source deferred indefinitely (cost), not just "until
  Phase 2."** Santiment came out as the right vendor for it (see chat), but
  it's a real $49-249/mo cost Jack isn't committing to yet. Proceeding
  straight to Silver/Gold construction (`docs/game_plan.md` Phase 2) on the
  DEX-only Bronze data already validated -- model features will be
  financial-only until/unless the social source gets added later. Note:
  README's informal "Phase 1 = DEX ingestion, Phase 2 = social ingestion"
  phrasing is a different numbering than `game_plan.md`'s master Phase
  0-6 sequence -- game_plan.md's numbering is the authoritative one going
  forward to avoid confusion between the two.
- **Entry and exit are separate models; not gated on pool creation.**
  Corrected an implicit assumption that LAWRENCE only ever decides once per
  token, at launch (same shape as BIA's reflex, just smarter). Instead:
  LAWRENCE assesses whatever it's watching on a recurring cadence, and
  entry ("is this worth buying now," wide candidate set) and exit ("should
  I sell what I hold," small candidate set, needs position context) are
  split into separate labels/models so each can be tuned for its own
  latency/cost budget rather than one model serving both cheaply-at-scale
  and quickly-on-a-small-set. See `docs/label_spec.md` v2.
- **Cadence is a deployment/cost parameter, not a label/code parameter.**
  15 min was always just an example for the Bronze job's schedule -- the
  label and feature definitions are written generically over an assessment
  timestamp `T`, identical whether LAWRENCE runs every 15 min or
  continuously. Only infra cost changes with cadence, never correctness.
- **Bronze fee/gas fields added, verified against live data first.** Solana:
  `Transaction { Fee FeeInUSD }` nests directly, no join. EVM: no direct
  `Transaction.Fee` under `DEXTrades` — requires `joinTransactions(join:
  inner, Transaction_Hash: Transaction_Hash) { Fee { SenderFee
  SenderFeeInUSD } }`, confirmed working on the existing `DEXTrades` root
  (no migration to `DEXTradeByTokens` needed). Both values are native-
  currency decimals (SOL/ETH), not raw base units — so the column is named
  `chain_native_fee`, not `chain_native_fee_raw` as first drafted in
  `docs/silver_spec.md`. Purpose: downstream competition/MEV-heat proxy and
  realistic backtest cost simulation. See `docs/silver_spec.md`.
- **Silver layer implemented (`lawrence.silver.token_trades`).** Row-shaping
  logic (quote-asset denylist applied per side, price/amount normalization,
  fee passthrough) lives in `src/features/silver_token_trades.py` — plain
  Python, no PySpark, unit-tested in `tests/test_silver_token_trades.py` so
  it actually runs in CI (unlike the Databricks notebooks, which need a
  Databricks runtime). `notebooks/01_silver_token_trades.py` is the Spark
  glue: bounded lookback read from Bronze, MERGE into Silver keyed on
  `(dex_venue, tx_hash, token_id)` — the `token_id` is required, not just
  `(dex_venue, tx_hash)`, because a trade between two non-quote tokens
  produces two Silver rows (see `docs/silver_spec.md`). Wired into
  `resources/lawrence_job.yml` as a task depending on the Bronze task, so it
  runs right after Bronze each cycle.
- **Gold: separate tables for entry/exit, shared feature code.**
  `lawrence.gold.entry_candidates` and `lawrence.gold.exit_candidates` are
  physically separate tables (not one table with a row-type flag) — same
  reasoning already used to keep entry/exit as separate models. Trailing-
  activity feature computation (volume, buy/sell ratio, price action from
  Silver) is shared code, called by both tables' build logic.
- **Exit label training data uses simulated entries, sampled
  independently of entry-model logic.** No real position history exists
  yet (that needs live paper trading, Phase 4), so exit training rows
  manufacture a synthetic entry at `T - h` and derive entry_price/
  unrealized_return/holding_duration from it. `h` is sampled independently
  — not constrained to values the entry model would actually have
  triggered on. Explicit simplification, not an oversight: this is
  standard hindsight relabeling, and a stand-in for real position history
  until Phase 4 provides it. See `docs/label_spec.md`.
- **Gold v1 implemented as a batch/backfill training-data job, not wired
  into the recurring schedule.** `notebooks/02_gold_entry_candidates.py`
  and `03_gold_exit_candidates.py` read all of `lawrence.silver.token_trades`,
  sample assessment points T on a 15-min grid per token, and full-overwrite
  `lawrence.gold.entry_candidates`/`exit_candidates` — unlike Bronze/Silver,
  not added to `resources/lawrence_job.yml`'s schedule, since there's no
  model yet to justify a production-serving (T="now") path. Shared feature
  math lives in `src/features/gold_features.py` (no PySpark — unit-tested
  in `tests/test_gold_features.py`, runs in CI).
- **KNOWN GAP, shipped intentionally rather than hidden: no rug-clamping.**
  `label_spec.md`'s rule (clamp the label at the realized loss when
  liquidity is pulled before T+HOLD_MINUTES) isn't implemented — there's no
  pool-liquidity-state data to detect a pull event. v1 instead drops a
  (token, T) row when no trade exists near T+HOLD_MINUTES. This is a real
  survivorship-bias risk (quietly excludes likely-rugged tokens rather than
  labeling their real loss) — documented in `docs/gold_spec.md`, revisit
  once Bronze ingests pool-liquidity state.
- **Paused the recurring schedule again (2026-09-15).** Continuous 15-min
  Bronze/Silver polling was projected by Bitquery to exhaust API credits
  within days of being unpaused — the account's projected exhaustion date
  (Oct 2) was based on earlier sporadic manual runs, not sustained 15-min
  polling, so real burn rate once continuous was higher than that estimate.
  Rejected going tighter (1-min) for the same reason, worse. Replacing
  perpetual live polling with a bounded, one-time historical backfill for
  this phase (accumulating training-data volume, not production
  freshness) — see the backfill design notes to follow. Bitquery's
  dashboard doesn't expose a per-query credit breakdown, so real cost is
  being measured empirically via a small bounded test query before
  committing to a full backfill.
- **Historical backfill via Bitquery abandoned (2026-09-15).** Cost-test
  notebook (`00b_bronze_backfill_test.py`) failed with `403: access
  restricted: your plan only allows "realtime", but the request uses
  "archive:solana:DEXTrades"`. Confirmed with Bitquery: historical/archive
  data requires an Enterprise account, not available on the current plan
  at any price point below that. This isn't a query-syntax bug (the
  `dataset: archive` + `till` shape was correct) -- it's a hard plan gate.
  `fetch_historical_trades()` and the `_EVM_BACKFILL_QUERY`/
  `_SOLANA_BACKFILL_QUERY` templates in `bitquery_client.py` are unusable
  on this plan and staying in the codebase unused for now, pending a
  decision on next steps (see game_plan.md / next conversation).
- **Historical backfill pivoted to Dune (2026-09-15).** Verified live
  against real bounded queries (not guessed): Dune's curated `dex.trades`
  (EVM) and `dex_solana.trades` (Solana) tables cover both venues --
  `raydium_v4` filters by an exact `project_program_id` match (identical
  to our Bitquery program address), `uniswap_v3_base` filters by
  `project = 'uniswap' AND version = '3'` (Dune's EVM rows carry per-pool
  addresses, not the shared factory address our Venue stores). Cost is
  negligible: 1.029 credits for 13 test queries against a 2,500-credit
  free trial -- not a real constraint at any scope we'd plausibly need.
  Both tables lag real time by ~1 day (irrelevant for a backfill).
  Three decisions made before writing any code: (1) accept
  `tx_fee_native`/`tx_fee_usd` = NULL on Dune-backfilled rows rather than
  chase an unverified fee join -- neither Dune table has a usable
  on-chain tx fee (EVM: no fee column at all; Solana: `fee_tier`/`fee_usd`
  exist but were blank on every sampled row, and even populated would be
  the DEX's own swap fee, not the wallet's tx fee anyway); (2) map Dune
  rows into Bronze's existing schema (`source = 'dune'`) so the
  already-tested Silver transform runs unchanged, rather than a second
  Silver path; (3) start with a 30-day window. Implemented in
  `src/ingestion/dune_client.py` (`DuneClient`, `DuneVenueFilter`,
  `to_bronze_row`, unit-tested in `tests/test_dune_client.py`) and
  `notebooks/00c_bronze_backfill_dune.py`.
  Found and fixed a real bug while wiring this up: Dune's `amount_usd` is
  null for some illiquid/exotic tokens (confirmed on real sampled Base
  rows) -- `to_silver_rows` would have thrown `TypeError` on
  `amount_usd / amount` when `amount_usd` is `None`. Fixed to skip those
  rows instead of crashing; new test case added.
- **Dune backfill cost incident + fixes (2026-09-15).** First real run of
  `00c_bronze_backfill_dune.py` (30-day window, daily chunks, `SELECT *`)
  burned through a 2,500-credit free trial partway through a SINGLE
  day/venue chunk -- paginated past row offset 415,000 before a 402. The
  earlier cost-test only measured tiny 1-hour aggregate queries
  (COUNT(*)/GROUP BY), not full-column trade-level SELECTs at real
  top-venue volume -- not a like-for-like cost estimate, and that gap
  should have been flagged before running the real backfill.
  Worse: the 402 was a raw, unhandled `requests.HTTPError` -- only
  `_submit` had retry/error-wrapping, `_wait_until_done`/`_fetch_all_rows`
  didn't, so it crashed the whole notebook. Since Bronze was written only
  once, at the very end, every credit spent produced zero rows in Bronze.
  Fixed: `dune_client.py`'s `DuneVenueFilter` now takes an explicit
  `columns` list (only what `to_bronze_row` needs, not `SELECT *`); a
  shared `_request` helper wraps every HTTP call site consistently and
  raises a new `DuneCreditsExhausted` (never retried) immediately on a
  402; the notebook now writes to Bronze after every chunk instead of
  once at the end, and aborts the whole run (not just one chunk) on
  `DuneCreditsExhausted`. `BACKFILL_DAYS` reset to 1 with hourly chunks
  as a real-cost test at the new column set, before scaling back up.
- **Correction to the entry above (2026-09-15, later same day).** The
  failed run actually got 7 full days through fine (daily chunks, both
  venues) before the 8th day's chunk spiked in volume and drained the
  remaining credits -- "a full day was already too much for one venue"
  was wrong; daily chunking itself was never the problem. That 7 days of
  fetched data never reached Bronze -- confirmed gone, not recoverable --
  because the crash happened before the (then one-shot, end-of-run) write
  ever executed. The per-chunk-write fix already made this session is
  exactly what prevents this specific loss from recurring.
  `notebooks/00c_bronze_backfill_dune.py` reset to `BACKFILL_DAYS = 5`
  with daily chunks (not hourly) as the next real-cost test, at the new
  narrower column set, once credits are available again.
- **Dune backfill: minimize loss on 402, make re-runs resumable
  (2026-09-15).** Two remaining gaps from the cost incident fixed:
  (1) `DuneCreditsExhausted` now carries `partial_rows` -- whatever pages
  of the SAME query already succeeded before a later page hit 402 (the
  actual 2026-09-15 loss was pages 1-3 of a paginated fetch being
  discarded when page 4 failed, not just "the chunk"). The notebook
  writes those salvaged rows to Bronze immediately, logged as `partial`.
  (2) Added `lawrence.bronze._dune_backfill_progress`, an append-only log
  of every chunk attempted; only a `success` row blocks re-fetching a
  chunk on a future run (`partial`/`failed` are retried wholesale --
  cheap, at most one page's re-fetch, and safer than risking permanent
  data loss by marking an incomplete chunk done). This is why the
  backfill window moved from "now minus N days" to fixed absolute dates
  (`BACKFILL_WINDOW_START`/`END`) -- chunk boundaries must be identical
  across separate runs for the skip-check to match anything.
  Operational note: plan is to run this with a friend's Dune API key
  (swapped into the `dune_api_key` secret) once the current account's
  credits are available again -- the progress table is what makes that
  handoff safe regardless of which key completes which chunk.
- **Dune backfill: CANNOT_DETERMINE_TYPE and block_time format
  (2026-09-15).** After the CANNOT_MERGE_TYPE fix, a real run hit
  `CANNOT_DETERMINE_TYPE` on every `uniswap_v3_base` chunk (200k-380k
  real rows each, fetched and paid for, not written). Root cause:
  `tx_fee_native`/`tx_fee_usd` are always `None` for every Dune-sourced
  row (the documented fee gap) -- across a whole batch that's an
  all-null column, which Spark's schema inference cannot assign any type
  to at all, unlike a type conflict (CANNOT_MERGE_TYPE) which at least
  has two real types to fail on. Fixed by giving `_write_bronze` an
  explicit schema (`_BRONZE_SCHEMA` in the notebook) instead of relying
  on inference from Row objects -- this also makes the earlier
  CANNOT_MERGE_TYPE class of bug structurally impossible here going
  forward. Separately normalized `block_time`: Dune returns
  `"YYYY-MM-DD HH:MM:SS.mmm UTC"`, a different string shape than
  Bitquery's ISO-8601 `Z`-suffixed format Bronze already stores --
  `_normalize_block_time()` converts so both sources' rows sort/compare
  identically as strings.
  OPEN QUESTION, not yet resolved: the same run's `raydium_v4` chunks for
  Aug 15-20 all reported 0 rows written. Unverified whether this is real
  (volume genuinely absent/shifted to other Raydium program versions by
  then) or a query bug -- asked to check directly against Dune's SQL
  editor before trusting it, since those chunks are already logged
  `success` in the progress table and would be silently skipped forever
  on a re-run if it turns out to be a bug.
- **Open question resolved: raydium_v4's 0-row days are real, not a bug
  (2026-09-15).** Verified independently via Dune's SQL editor directly
  (same filter, run outside the backfill notebook): `raydium_v4` genuinely
  had 0 matching trades on 2026-08-15. Not a query bug -- legacy AMM
  volume likely migrated to newer Raydium program versions (CPMM/CLMM) by
  this point. The `success`-with-0-rows chunks in the progress table are
  correct and don't need clearing.
- **Correction to the entry above: raydium_v4's 0-row days were a real bug,
  not real zeros (2026-09-15).** The independent Dune SQL editor check only
  confirmed one day (2026-08-15) against the *same broken filter* -- it
  reproduced the bug, it didn't validate the venue. A broader check (`SELECT
  version, COUNT(*), SUM(amount_usd) FROM dex_solana.trades WHERE
  project='raydium' AND block_time in August 2026, GROUP BY version`) shows
  `version=4` ("amm") with 6,064,410 real trades and $406M volume for the
  month -- nowhere near zero. `where_sql` filtered on
  `project_program_id = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'`,
  which is unreliable: likely many v4 swaps route through an aggregator, so
  Dune's `project_program_id` records the outer router program instead of
  Raydium's own address for a large share of trades -- the same class of
  problem already known for uniswap_v3_base's per-pool addresses. Fixed by
  matching `project = 'raydium' AND version = '4'` instead, mirroring
  uniswap_v3_base's existing approach. Every `raydium_v4` chunk logged
  `success` with 0 rows before this fix is now known wrong and needs
  clearing from `_dune_backfill_progress` (safe -- Bronze never actually
  got any raydium rows from those chunks, so there's nothing to
  de-duplicate) before the next run, or the corrected query will be
  silently skipped for every date already marked done.
- **raydium_v4 backfill confirmed fixed (2026-09-15).** After the
  project+version filter (365cb65) and the integer-literal fix (14a4c7d,
  `version = 4` not `'4'`) both deployed, raydium_v4 is writing real rows:
  105223 / 113365 / 162384 for Aug 16-18, consistent with the ~195k/day
  implied by the monthly aggregate. Chain of bugs, in order: wrong filter
  column (address instead of project+version) -> silently returned 0,
  misread as a real zero -> corrected filter had the right columns but
  wrong literal type (quoted '4' against an integer column) -> Trino
  query failure, invisible detail until error.message was surfaced. All
  three fixed; both venues now writing correctly for the remaining backfill.
- **Dune backfill complete (2026-09-15).** Full window
  2026-08-14T00:00:00Z - 2026-09-13T00:00:00Z, both venues, 60/60 chunks,
  no chunk left in `partial`/`failed` state. Took 5 separate Dune API keys
  (Jack's own trial, a friend's, three more) across ~4 credit-exhaustion
  cycles, all recovered cleanly via the partial-row-salvage + resumable
  progress table. Known caveats carried into Bronze, unchanged from the
  module docstring: `tx_fee_native`/`tx_fee_usd` always NULL for
  `source='dune'` rows; `buy_amount_usd`/`sell_amount_usd` both copied from
  Dune's single `amount_usd`. One outlier not yet explained:
  raydium_v4 on 2026-09-01 wrote 1,508,642 rows vs. ~200-350k on every
  other day -- worth a real look (token launch? one wallet spamming
  trades? a query artifact?) before trusting that day's data downstream.

  Next: Silver's `LOOKBACK_MINUTES` (01_silver_token_trades.py) only looks
  back ~60 min by design for the live 15-min cadence, so it won't see any
  of this on its own -- either widen it for one manual run over the
  backfill window, or query Bronze directly for `source = 'dune'` and
  feed Silver's transform that way. Not done yet.
- **CANNOT_DETERMINE_TYPE recurred in Silver + a correctness concern in
  Bronze's existing fix (2026-09-15).** Silver's catch-up run hit the same
  all-null-column type-inference failure Bronze already hit
  (tx_fee_usd/chain_native_fee always None for a 100%-Dune-sourced chunk).
  Fixed with an explicit _SILVER_SCHEMA, same idea as Bronze's
  _BRONZE_SCHEMA. While adding it, realized Bronze's existing fix
  (`spark.createDataFrame([Row(**r) for r in bronze_rows], schema=...)`)
  had an unverified assumption baked in: `Row(**kwargs)` is documented to
  sort its fields alphabetically by name internally, and it was never
  confirmed whether createDataFrame resolves that correctly against a
  separately-authored explicit schema (name-matched) or naively
  positionally (which would silently write values into the wrong
  columns -- not a crash, wrong data). Couldn't install pyspark in this
  environment to test empirically, so rather than trust memory on an
  internal Spark resolution order, switched both Bronze and Silver to
  pass plain dicts directly to `createDataFrame(data, schema=...)`,
  which is unambiguous, documented, name-matched behavior. `_log_progress`
  in the Bronze notebook is unaffected (no separate explicit schema is
  given there, so there's no cross-schema ambiguity to worry about).
  Have not gone back to re-verify the already-written ~9M Bronze rows
  column-by-column; row *counts* have been spot-checked repeatedly and
  matched Dune's own aggregates, which any full column-scramble likely
  wouldn't survive undetected (e.g. text landing in numeric columns would
  break every downstream Silver read), but this hasn't been explicitly
  proven the way row counts were. Worth a real spot-check of raw Bronze
  values (not just counts) before trusting this fully.
- **Dune backfill through Silver confirmed complete end-to-end
  (2026-09-17).** Bronze: 60/60 chunks, full Aug 14 - Sep 13 window,
  spot-checked clean. Silver: 19,024,520 Bronze rows scanned ->
  10,756,956 Silver rows written (~56.5% conversion, consistent with
  typical quote/non-quote trade patterns), verified against
  lawrence.silver.token_trades directly (count + min/max trade_time
  matched the run's own summary). CATCHUP_SINCE/CATCHUP_UNTIL reverted to
  None; live cadence path confirmed still working correctly (0/0 result
  on a paused live job, as expected). This closes out the whole Dune
  backfill effort -- see the chain of entries above for the full bug
  history (SELECT * cost incident, 402 resilience, raydium_v4 filter and
  type bugs, CANNOT_DETERMINE_TYPE x2, Row(**kwargs) ambiguity, driver
  hang, notebook-session timeout). Next: re-run Gold (02/03) against real
  historical volume for the first time.
- **Removed notebooks/00b_bronze_backfill_test.py (2026-09-17).** Its sole
  purpose was measuring real Bitquery credit cost against `dataset:
  archive` before committing to a real backfill -- moot once that path
  was confirmed Enterprise-only (403, see the 2026-09-15 entries above)
  and replaced entirely by the Dune-based backfill
  (00c_bronze_backfill_dune.py). Nothing imports or schedules it. Left
  the historical references to it in this log and in 00c's own docstring
  as-is -- they're describing what happened, not a live dependency.

- **Liquidity/holder data confirmed essential, but deferred (2026-09-17).**
  Discussed whether pool-liquidity-state and holder/dev-allocation data is
  actually needed, or whether trade-log-only features are enough. Conclusion:
  it's essential to the end goal, not a nice-to-have -- DEX trade data alone
  can train a competent chart-based trader, but Larry (LAWRENCE's human
  benchmark) uses liquidity/allocation signal at a surface level (volume is
  already covered by Silver's trade log; % supply held by devs/founders and
  real liquidity depth are not, and neither is real rug-pull detection).
  This isn't a new finding -- `gold_spec.md`'s "Not included, on purpose"
  section and `game_plan.md` §4's Gold feature list already named these same
  two missing Bronze sources (pool-reserve-state snapshots, wallet/holder
  data) -- but it had only ever appeared as background/footnote text, not as
  a tracked phase item. Added an explicit callout to `game_plan.md` Phase 1
  so it survives independent of this conversation. Decision: still don't
  block on it right now -- building that ingestion is real new engineering
  (a new on-chain data source, not a config change) -- proceed with the
  interim trade-log-only rug-clamp heuristic (see next entry) and revisit
  real liquidity/holder ingestion as a real, scheduled piece of work rather
  than an aspiration.

- **Interim rug-clamping heuristic implemented (2026-09-17).** Jack decided
  the rug-clamping gap must be fixed before the next Gold re-run ("the
  model needs to see them as losses") -- not deferred. A real fix (detecting
  an actual liquidity-pull event) needs pool-state data that doesn't exist
  yet (see the entry above), so implemented an interim heuristic in
  `src/features/gold_features.py` instead of waiting: `build_entry_row`/
  `build_exit_rows` still drop a `(token_id, T)` row when there's no real
  price at `T` itself (a genuine data gap), but no longer drop it when `T`
  has a real price and only the exit side (`T + HOLD_MINUTES`) has nothing
  to trade against -- that case is now treated as a proxy rug/dead-token
  signal: the label is clamped to `RUG_CLAMP_RETURN` (-1.0, full loss)
  instead of the row being dropped, and a new `is_rug_clamped` boolean
  column is carried on every Gold row so these are inspectable/filterable
  rather than silently mixed in with real-price-derived labels. This can't
  tell an actual rug apart from a token that simply went quiet for other
  reasons -- it's a stand-in for real liquidity-pull detection, not the
  real thing -- but it removes the survivorship-bias risk of dropping these
  rows entirely. Updated `gold_features.py`'s module docstring,
  `docs/gold_spec.md`'s schema + rug-clamping section, and
  `tests/test_gold_features.py` accordingly (couldn't run pytest in this
  sandbox -- no pytest install and no network access to install it -- so
  verified the new branches manually with a standalone script exercising
  the same cases the updated tests cover; Jack should still run the real
  test suite once he pulls this). Ready to re-run
  `02_gold_entry_candidates.py`/`03_gold_exit_candidates.py` after this.

- **pytest never had `src/` on its import path -- likely never actually
  collected in CI (2026-09-17).** Jack ran `pytest test_gold_features.py`
  locally (the first real attempt to run this suite that surfaced in this
  conversation) and hit `ModuleNotFoundError: No module named 'features'`.
  Checked: no `conftest.py`, `pytest.ini`, `pyproject.toml`, or `setup.cfg`
  has ever existed in this repo (confirmed via `git log`), and every test
  file (`test_gold_features.py`, `test_silver_token_trades.py`,
  `test_dune_client.py`) imports from `features.X`/`ingestion.X` the same
  way the notebooks do -- except the notebooks explicitly
  `sys.path.append("../src")` and the tests had no equivalent. Without an
  ini file, pytest's default "prepend" import mode puts the test file's own
  directory (`tests/`, since it has no `__init__.py`) on sys.path, not
  `src/` -- regardless of cwd. This means `.github/workflows/ci.yml`'s own
  `pytest -q` step was very likely failing at collection the entire time,
  not just Jack's local run. Fixed with a new root `pyproject.toml` setting
  `[tool.pytest.ini_options] pythonpath = ["src"]` -- resolved relative to
  the ini file's own location (rootdir), so it works the same from any cwd,
  matching how notebooks already treat `src/` as the shared import root.

- **Confirmed (2026-09-17):** all 16 tests pass locally
  (`pytest tests/test_gold_features.py`) after the pyproject.toml fix,
  including the new rug-clamping cases. Rug-clamping fix + test-path fix
  both verified end-to-end; clear to re-run 02/03 Gold notebooks.

- **Gold notebooks 02/03 hit the same driver-collect() hazard as Silver,
  caught mid-run (2026-09-17).** Jack's first real run of
  `02_gold_entry_candidates.py` against real backfill volume (~10.75M
  Silver rows) had been running 45+ minutes with the orderBy shuffle's
  Spark tasks already fully complete (25/25) but no output printed --
  the classic sign the driver is stuck materializing a huge `.collect()`
  result, the same failure mode already hit and fixed in Silver's own
  catch-up notebook. Root cause: `02`/`03` were written when
  gold_spec.md's own docs expected "few or zero rows" (no real backfill
  existed yet), so `silver_df.collect()` -- the whole table into one
  Python list -- was never actually exercised at scale before now. Had
  Jack interrupt the run rather than wait it out further.
  Fixed both notebooks the same way: replaced `.collect()` with
  `toLocalIterator()` (streams partitions to the driver one at a time
  instead of materializing everything at once) plus a single streaming
  pass that relies on the existing `orderBy("token_id", "trade_time")` --
  a token's rows are contiguous in that order, so each token's full trade
  history is built and processed as soon as the next token_id is seen,
  never holding more than one token's raw trades (plus the much smaller
  accumulating Gold output) in memory. Also added explicit schemas
  (`_ENTRY_SCHEMA`/`_EXIT_SCHEMA`) and switched the final write to plain
  dicts instead of `Row(**r)`, matching Bronze/Silver's existing
  CANNOT_DETERMINE_TYPE / Row-ordering-ambiguity fix -- this notebook's
  final createDataFrame call had neither before. Verified the streaming
  group-by logic against the real `build_entry_row` with a standalone
  script (two interleaved fake tokens, checked scan counts and per-token
  labels); couldn't test the Spark-side `toLocalIterator()` mechanics
  themselves in this sandbox (no Spark/Databricks access here) -- that
  needs a real re-run to confirm the runtime actually resolves.

- **Gold notebooks still slow after the collect()->streaming fix -- real
  root cause was O(n) rescans in gold_features.py, not the Spark read
  (2026-09-17).** After the toLocalIterator() fix, Jack's re-run showed the
  underlying Spark statement (the orderBy read) completing in ~36s for all
  10.76M rows, but the notebook cell kept running well past that -- a sign
  the driver-side Python work after the read, not Spark, was the
  bottleneck. Root cause: `price_at` and `trailing_activity_features` in
  `src/features/gold_features.py` re-scanned a token's ENTIRE trade list
  from the start (re-parsing every trade_time string along the way) on
  EVERY call, despite `trades` always being sorted ascending by
  trade_time. Called ~2-3x per assessment point, with points sampled every
  15 minutes across a token's whole active history, this made
  `build_entry_row`/`build_exit_rows` effectively O(n) per lookup x O(m)
  lookups per token -- invisible while Gold only ever ran against near-zero
  real history (per gold_spec.md's original "few or zero rows" assumption),
  but the real bottleneck once real Dune-backfill-scale token histories
  exist. Fixed by replacing the linear scans with a binary search
  (`_count_at_or_before`, O(log n)) over the same sorted invariant --
  verified against the old linear-scan logic with 3000 randomized
  (trades, t) checks (zero mismatches) and measured ~37x faster on a single
  5000-trade synthetic token (6.1s -> 0.16s for a full assessment-time
  grid); the real speedup on the busiest real tokens will be larger still,
  since the gap between O(n) and O(log n) widens with n. No public
  function signature changed, so tests/test_gold_features.py needed no
  changes and should still be the thing to run to confirm this didn't
  change behavior.

- **toLocalIterator() without caching likely recomputed the sort per
  partition -- still slow after the O(log n) fix (2026-09-17).** After the
  binary-search fix, Jack's next re-run was still going after 18 minutes,
  but this time with a materially different signal: the Databricks
  performance panel showed "Statement 0/1" (the underlying Spark query
  itself hadn't finished), versus the earlier run where the identical
  orderBy query had completed in 36s and only the driver-side Python work
  was still running. Ruled out plain Python loop overhead as the
  explanation first (benchmarked: iterating/asDict()-ing 10.76M rows takes
  ~3s on its own, nowhere near 18 minutes). The likely cause: without a
  `.cache()`, `toLocalIterator()` can force Spark to recompute the entire
  upstream query (the full orderBy shuffle) once per partition retrieved,
  instead of computing it once and streaming pieces of one materialized
  result -- for ~25 partitions that's roughly 25x the work, and 25 x ~36s
  lands right around 18+ minutes, a strong match. Fixed by caching the
  sorted DataFrame and forcing eager materialization with `.count()`
  (Spark caching is lazy otherwise) before the toLocalIterator() loop, so
  the shuffle runs exactly once regardless of partition count. Also added:
  progress logging every 500k rows (elapsed time, tokens seen, Gold rows
  built so far) so a future stalled run is visible immediately instead of
  discovered many minutes in with no signal at all, and a sanity check
  comparing the streamed row count against the cached `.count()` to catch
  a silent partial-scan. Applied identically to both 02 and 03. Could not
  verify the caching fix's actual effect from this sandbox (no
  Spark/Databricks access here) -- needs a real re-run to confirm.

- **Progress logging made a standard practice (2026-09-17).** After three
  separate "is it stuck or just slow" incidents in this one session, Jack
  asked to make progress logging standard rather than something added
  reactively each time. Wrote it up in the new `docs/conventions.md`
  (chunked-job vs. tight-loop patterns, both already in use across the
  codebase) so it's a documented expectation for new notebooks going
  forward, not just a fix applied to the two files that happened to hit
  the problem.

- **.cache() unsupported on serverless -- switched Gold 02/03 to a
  distributed groupBy/applyInPandas job instead of streaming through the
  driver at all (2026-09-17).** The `toLocalIterator()` + `.cache()` fix
  from the previous entry errored immediately: `[NOT_SUPPORTED_WITH_SERVERLESS]
  PERSIST TABLE is not supported on serverless compute`. This confirms
  Jack's cluster is Databricks serverless (Spark Connect client) --
  important context for any future fix that assumes classic-cluster Spark
  behavior (RDD-level ops and caching in particular are restricted there).
  Given this was the THIRD distinct failure mode from trying to route
  10M+ Silver rows through a single Python process on the driver
  (`.collect()` memory hang, then a suspected toLocalIterator()
  recompute-per-partition problem, now a blocked caching API), decided to
  stop patching that approach and instead push the per-token computation
  onto the cluster as a real distributed operation:
  `silver_df.groupBy("token_id").applyInPandas(fn, schema=...)`. Each
  executor now processes a subset of tokens in parallel, calling straight
  into the same `gold_features.py` functions -- no driver memory
  pressure, no caching, and Spark's own groupBy shuffle guarantees a
  token's rows land together (no hand-rolled streaming/ordering logic to
  get right). applyInPandas groups have no guaranteed row order, so the
  per-group function sorts locally first (cheap -- one token's rows, not
  the whole table); verified this is correct and order-independent with a
  standalone script (out-of-order input trades, checked the resulting
  y_entry/is_rug_clamped match the same call via gold_features.py
  directly) and that the empty-group edge case returns the right empty
  schema. pandas/pyarrow were already project dependencies
  (requirements.txt), no new dependency added.
  This also resolves the "is it stuck" visibility problem more
  fundamentally than the progress-print fix did: the heavy computation
  now happens AS the Spark job, so its Tasks X/Y progress in the
  Databricks UI is an honest signal of real completion, instead of
  misleadingly showing "done" while driver-side Python work continued
  invisibly (what made the first two stalls hard to diagnose). Rewrote
  `docs/conventions.md`'s progress-logging section to lead with "prefer a
  real distributed Spark job when the computation allows it" ahead of the
  driver-loop-plus-print-statements pattern, which still applies to
  Bronze/Silver's genuinely driver-side chunked work. Could not test the
  actual Spark-side applyInPandas execution from this sandbox (no
  Databricks access here) -- needs a real re-run to confirm.

- **applyInPandas ran fast -- next error was just a missing schema
  (2026-09-17).** Jack's re-run confirmed the applyInPandas rewrite: "went
  extremely fast!!" No further collect()/toLocalIterator()/caching issue
  came back, which is a real confirmation of that architecture (though the
  write itself hasn't succeeded yet, so per-token feature correctness at
  real scale is still unverified end-to-end). Hit `SCHEMA_NOT_FOUND: The
  schema lawrence.gold cannot be found` on `saveAsTable` -- unlike
  `lawrence.bronze`/`lawrence.silver`, the `gold` schema itself had never
  been created (no run had ever gotten far enough to write there before
  today). Fixed with an idempotent `CREATE SCHEMA IF NOT EXISTS
  lawrence.gold` right before the write in both 02/03, so this is a
  one-time no-op from here rather than something Jack has to remember to
  set up by hand.

- **Real Silver data has a mix of trade_time formats -- a resumable-backfill
  side effect, not random corruption (2026-09-17).** applyInPandas ran
  against real data and crashed inside `gold_features.py::_parse`:
  `ValueError: Invalid isoformat string: '2026-08-18 10:16:01.000 UTC'`.
  Traced the root cause: `src/features/silver_token_trades.py::to_silver_rows`
  passes Bronze's `block_time` straight through as Silver's `trade_time`
  with zero normalization (`"trade_time": bronze_row["block_time"]`).
  `src/ingestion/dune_client.py::_normalize_block_time` DOES convert Dune's
  raw `"YYYY-MM-DD HH:MM:SS.mmm UTC"` shape to a clean ISO
  `"...T...Z"` string -- but that normalization was added in commit
  `73d56aa`, which landed AFTER the backfill's resumability
  (`bf6c541`). Resumability means a chunk already written and marked
  'success' is skipped on every later re-run -- exactly the behavior we
  wanted for efficiency -- but it also means a fix added mid-backfill never
  retroactively reprocesses chunks that already succeeded under the old
  code. Net effect: the real ~10.75M-row Bronze/Silver history genuinely
  contains BOTH trade_time shapes side by side, not hypothetically -- some
  chunks pre-date the normalization fix, some postdate it. This is a real,
  general risk of the resumable-backfill pattern used throughout this
  project (also in `01_silver_token_trades.py`'s catch-up mode) worth
  keeping in mind for any future mid-backfill fix: a bug fixed today does
  not retroactively apply to chunks already marked done.
  Fixed at the READ/parse side rather than by rewriting already-verified
  Bronze/Silver data: `gold_features.py::_parse` now handles both shapes
  (ISO Z/offset, and Dune's raw space-separated + " UTC" suffix shape) and
  always returns a UTC-aware datetime so aware/naive comparisons across
  differently-sourced rows never mix and crash. Verified with a standalone
  script (both shapes parse to the identical instant; a token whose trades
  mix both shapes processes correctly) and added permanent regression
  tests to `tests/test_gold_features.py` covering both shapes and a
  mixed-shape token. Deliberately did NOT re-backfill Bronze/Silver to fix
  this at the source -- that's real, costly rework for data Jack already
  independently verified as correct (row counts, min/max), and this fix
  makes any downstream reader of trade_time correct regardless of which
  shape a given row happens to carry.

- **ZeroDivisionError in build_entry_row: a real Silver row has
  price_usd == 0.0 (2026-09-17).** applyInPandas got past the timestamp
  fix and crashed on real data again:
  `ZeroDivisionError: float division by zero` at
  `y_entry = (price_exit - price_t) / price_t`. `price_t` was `0.0` --
  `price_at` returned it as a legitimate value since `0.0 is not None`
  passed the existing null check. Root cause: `price_usd` in Silver is
  computed as `amount_usd / amount` in `silver_token_trades.py`, and a
  dust/degenerate swap (nonzero token amount, `amount_usd == 0`) produces
  a literal `0.0` price that Silver's own filter doesn't catch (it only
  checks `amount_usd is None`, not `amount_usd == 0`). A $0 price isn't a
  meaningful "this token is worth nothing" signal, it's a data artifact.
  Fixed the same way as the timestamp-format issue -- at the read side,
  not by rewriting already-verified Silver data: `price_at` now treats
  any non-positive price the same as "no usable price" (returns None).
  This reuses ALL of the existing, already-tested None-handling in
  build_entry_row/build_exit_rows for free -- a zero price_t drops the
  row (genuine gap at T), a zero price_exit goes through the interim
  rug-clamp path instead of literally dividing by it -- no new branches
  needed in either function. Verified with a standalone script (zero and
  negative price both treated as unusable; a zero price_exit clamps
  instead of crashing; build_exit_rows' existing `if not entry_price`
  check already covered a zero synthetic entry price) and added
  permanent regression tests. Could not run the real pytest suite from
  either this sandbox or the device_bash shell (neither has pytest
  installed -- device_bash is an isolated VM without Jack's own
  Anaconda/PATH setup) -- Jack should run it in his own terminal to
  confirm before trusting this fully.

- **UDF_PYSPARK_ERROR.OOM in applyInPandas -- likely one outlier token's
  trade history, not a general memory problem (2026-09-17).** After the
  price_usd==0.0 fix, the next real run crashed with "Python worker
  exited unexpectedly (crashed) due to running out of memory" inside
  `_process_token_group`. Leading theory: `gold_features.py` needs a
  token's FULL trade history in memory to sample point-in-time-correct
  assessment points, so unlike everything else in this codebase, a
  token's group genuinely can't be chunked -- one executor has to hold it
  all regardless of applyInPandas' distribution across OTHER tokens. The
  quote-asset denylist in `silver_token_trades.py` only excludes 5 known
  addresses (WSOL/USDC/USDT on Solana, USDC/WETH on Base); any other
  heavily-traded, non-memecoin asset that isn't on that list (a major
  token, a stablecoin not yet denylisted, etc.) would aggregate far more
  trades than a typical memecoin and could plausibly be the single group
  that overflowed one worker's memory.
  Added two things to both 02/03: (1) `_MAX_TRADES_PER_TOKEN = 50_000` (a
  placeholder, not a measured number) inside `_process_token_group` --
  SKIPS (never truncates -- truncating would silently break point-in-time
  correctness for early assessment points) any token whose trade count
  exceeds it, so one outlier can't take the whole job down again; losing
  one token is a negligible cost against thousands of tokens' worth of
  training data. (2) A driver-side diagnostic query (`groupBy("token_id")
  .count().orderBy(desc).limit(10)`) printed BEFORE the applyInPandas
  call, since print() inside the UDF itself goes to executor logs, not
  this notebook's visible output -- this is real visibility into the
  actual trade-count distribution, which will confirm or refute the
  outlier-token theory and tell us whether 50,000 is remotely the right
  cap once Jack re-runs. Verified the skip-vs-process branching logic
  with a standalone script (an oversized synthetic group returns the
  correctly-shaped empty result; a normal-size group is unaffected).

- **Outlier-token theory confirmed: all top-10 tokens by trade count are
  established projects/blue-chips, not memecoins evading the denylist by
  accident (2026-09-17).** The diagnostic query added in the OOM fix above
  came back non-flat -- all 10 tokens exceed `_MAX_TRADES_PER_TOKEN` by a
  wide margin (182,807 to 659,071 trades each, vs. a typical memecoin's
  much smaller count), confirming the outlier-token theory rather than a
  general memory problem. Identified all 10 via BaseScan/Solscan:

  | address | trades | identity |
  |---|---|---|
  | 0xcbb7...ed33bf | 659,071 | Coinbase Wrapped BTC (cbBTC) |
  | 0xb200...082f41d01 | 442,903 | "Basecat" (low-confidence ID -- fetched page listed multiple similarly-named token variants) |
  | 0x0b3e...4e7e1b | 385,108 | Virtual Protocol (VIRTUAL) |
  | 0xdcf5...f5bfd0 | 363,989 | ViciCoin (VCNT), ~$18/token |
  | 4k3D...QrkX6R (Solana) | 360,149 | Raydium (RAY) |
  | 0x9401...fd98631 | 326,970 | Aerodrome (AERO) -- Base's largest DEX token, itself commonly a trading pair |
  | 0x624e...cb98831 | 307,884 | SoSoValue (SOSO) -- page's own tx count (68,833) doesn't match our 307,884, worth a second look |
  | 0x1111...980afc69 | 243,626 | Zora (ZORA) |
  | 0x4137...997afbd2 | 208,982 | "EDYNE Technologies" (ETD) -- ANOMALY: contract is only ~27 days old yet already has this many trades; doesn't fit the established-blue-chip pattern of the other 9, worth manually checking for wash-trading/bot activity rather than assuming it's a legitimate denylist gap |
  | 0xacfe...cc21bf | 182,807 | Venice Token (VVV), ~$23/token |

  Net read: 8-9 of 10 are real, established DeFi/platform tokens (a
  wrapped blue-chip, DEX governance tokens, AI-platform tokens) that
  legitimately trade at volumes no early-stage memecoin would reach --
  this is a quote-asset-denylist gap, not a data bug, consistent with the
  original hypothesis. EDYNE (#9) is the one entry that doesn't fit that
  story and deserves a separate look. The existing
  `_MAX_TRADES_PER_TOKEN` skip-cap already keeps today's Gold run
  unblocked regardless of which explanation is right, since it skips by
  trade count, not by identity. Not yet decided: whether to expand
  `SOLANA_QUOTE_DENYLIST`/`EVM_QUOTE_DENYLIST` in `silver_token_trades.py`
  (prevents re-encountering these in future Silver runs, but doesn't
  retroactively clean already-written Silver data without a re-run) vs.
  leaving the Gold-layer skip-cap as the sole, lower-cost mitigation --
  flagged for Jack, not yet implemented.

- **Correction from Jack: don't denylist non-memecoin tokens -- the bot
  needs to trade both memecoins and established/blue-chip assets
  (2026-09-17).** After the outlier-token identification above, my
  framing treated "expand the quote-asset denylist" as one candidate
  fix. Jack corrected this: cbBTC/Virtual Protocol/Raydium/Aerodrome/Zora/
  etc. are legitimate assets Larry would plausibly trade, not junk to
  filter out -- excluding them from Gold would just mean the model never
  learns to handle that regime at all. Retracting the denylist-expansion
  option; it was never really the right tool here anyway, since Silver's
  quote denylist is about which side of a swap is the CURRENCY (e.g.
  WETH/USDC), not about which tokens are worth training on.
  Replaced the `_MAX_TRADES_PER_TOKEN` skip-cap's role accordingly in both
  02/03: it's now an explicit last-resort circuit breaker (raised from the
  50,000 placeholder to 2,000,000 -- comfortably above the real observed
  max of 659,071) for a genuinely pathological trade count, not a
  mechanism for excluding real assets; the skip-branch comment and the
  constant's own comment were rewritten to say so plainly, since the old
  "losing one token is a negligible loss" framing was exactly the
  reasoning Jack is pushing back on. Also made two changes that
  address the actual OOM without dropping anything: (1) both notebooks
  now `.select()` only the 9 columns `gold_features.py` reads before the
  groupBy shuffle -- Silver's `_SILVER_SCHEMA` carries 13, and the 4
  unused ones (notably `tx_hash`, a long string on every one of 10M+
  rows) were dead weight in every group's per-executor memory, not just
  the outliers'; (2) added a real distinct-token-count check (Silver vs.
  written Gold) printed after every run, so if `_MAX_TRADES_PER_TOKEN`
  (or any other future skip condition) ever actually triggers, it shows
  up as a number Jack sees, not a silent gap. EDYNE (the one anomaly that
  didn't fit the established-blue-chip pattern -- 27 days old, 208,982
  trades) is still worth a manual look, but is no longer being treated as
  a denylist candidate either; if it turns out to be wash-trading/bot
  activity, that's a Silver-layer data-quality question, not a Gold-layer
  exclusion decision, and hasn't been investigated further yet.
  Verified the notebook edits parse (`ast.parse`) and diffed cleanly
  against both files; the actual Spark run is still pending in Jack's own
  Databricks workspace, as with every notebook change so far this
  session.

- **EDYNE clarified by Jack: stale/zero-volume token, not wash-trading
  (2026-09-17).** Closing the anomaly flagged two entries up -- Jack
  confirmed EDYNE currently has zero real trading volume despite its
  208,982-trade Silver history; it's stale/dead, not a bot-driven outlier.
  Distinguishing "stale" from "actively traded" isn't needed right now and
  ties into the same deferred pool-liquidity-state gap already tracked in
  docs/game_plan.md's Phase 1 -- revisit once that data exists, not before.

- **First successful real-scale `lawrence.gold.entry_candidates` run
  (2026-09-17).** After the column-trim + cap-raise fix above, notebook
  02 completed cleanly against the real ~10.75M-row Silver history:
  1,276,167 rows written. The new distinct-token check held up as
  intended: 61,633 distinct tokens in Silver, 61,617 made it into Gold --
  a 16-token (0.026%) gap, not zero. Given `_MAX_TRADES_PER_TOKEN` is
  2,000,000 (no real token comes close), the cap did not cause this --
  the far more likely explanation is the same dust-trade/zero-price
  artifact already fixed in `price_at` (2026-09-17, above): a token whose
  entire trade history has `price_usd <= 0` produces zero valid
  assessment points and thus zero Gold rows, exactly like the gap this
  distinct-token check was added to catch and explain rather than paper
  over. Not independently confirmed against the real 16 token_ids (no
  direct Databricks SQL access from this session -- Jack would need to
  run `SELECT DISTINCT token_id FROM lawrence.silver.token_trades EXCEPT
  SELECT DISTINCT token_id FROM lawrence.gold.entry_candidates` himself
  to see them), but a 0.026% gap with a known, benign, already-fixed root
  cause is not a red flag. notebook 03 (`exit_candidates`) not yet run/
  reported.

- **The 16-token Silver->Gold gap, identified by Jack (2026-09-17).** Ran
  the EXCEPT query suggested above; the 16 token_ids are:
  `BWXrrYFhT7bMHmNBFoQFWdsSgA3yXoAnMhDK6Fn1eSEn`,
  `0x88faea256f789f8dd50de54f9c807eef24f71b16`,
  `0x162e1571b9511c1d66e58fa4880fb5e0cf46cb9b`,
  `0x06e8f93f0d0bf96629557e0264880ffc2858396c`,
  `0xd3f857208231b2d9cc54792e0aef1c69f3142fdc`,
  `7FcFaHyvuycxXBX63yoJVuvfrev2CJz62kDV3o7zpump`,
  `0xc2cc346600103733eccf18c2cdd438479e8f8d7d`,
  `GBKXKLGrxb5R6hhKjdEPyUd6jtYcdfa4y9mQ6quRQwJ1`,
  `0x412253e1f45463e1c120338563ffab44bbb1544f`,
  `GnNKwkgnM3DqCEcbLR2QKnK4waPFoeZBSLF8E3Appump`,
  `aMYwbHHWFJWCRQofBwTVyY9tjjEr29ZyGwaYCNfRY5g`,
  `CKweVATbMaCUngCs2V8oBMfuw3fRZuEPJjzRGDGoTbDt`,
  `2LQtGJgMybqWT1rQYaV1UQJffEMuhTH5etM5qdj2pump`,
  `H5RYoPzWFJfb44QJp8sBhwozyUkXSZEfEDRgdLCepump`,
  `6uQP5vMFZrkJ7rpM1XU5ugoAgA8NDWKyUccuoBnjGwD8`,
  `0x0b64174f86612d9721662297cc26f64c1692d65d`.
  9 EVM, 7 Solana -- and notably 4 of the 7 Solana addresses end in
  "pump" (`...zpump`, `...Appump`, `...2pump`, `...epump`), the vanity
  suffix pump.fun bonding-curve mints always carry. Pump.fun launches are
  exactly the profile most likely to be fully degenerate in Silver: a
  huge share never graduate off the bonding curve and die almost
  immediately, which fits the zero/negative-`price_usd` dust-trade
  hypothesis from the earlier `ZeroDivisionError` fix better than any
  other explanation so far -- a token whose ENTIRE history is dust
  produces zero valid assessment points, not just an occasional dropped
  row. Not independently confirmed against real Silver rows yet (no
  direct Databricks SQL access from this session); a query like
  `SELECT token_id, COUNT(*), MAX(price_usd) FROM
  lawrence.silver.token_trades WHERE token_id IN (...) GROUP BY
  token_id` against these 16 would confirm it outright (max_price <= 0
  for all of them = fully explained; anything with a positive max_price
  would mean something else is going on and deserves a closer look).

- **Correction: the 16-token gap is NOT 100% zero-price dust (2026-09-17).**
  Jack ran the confirmation query. 14 of 16 do have `MAX(price_usd) = 0`
  (pure dust, as hypothesized), but two don't:
  `0x06e8f93f0d0bf96629557e0264880ffc2858396c` (2 trades,
  max_price=0.0000021660948...) and
  `CKweVATbMaCUngCs2V8oBMfuw3fRZuEPJjzRGDGoTbDt` (2 trades,
  max_price=0.1525231508888889). "100% of the gap is dust trades" was
  wrong -- retracting that as the full explanation.
  The pattern that actually IS 100% consistent across all 16: every one
  has a vanishingly thin trade history -- 1 trade (12 of them), 2 trades
  (3 of them: the two above plus `6uQP5vMFZrkJ7rpM1XU5ugoAgA8NDWKyUccuoBn
  jGwD8`), or 4 trades (`0x88faea256f789f8dd50de54f9c807eef24f71b16`).
  For the 2-trade, nonzero-max-price cases, the likely mechanism (not yet
  confirmed against the actual trade_time/price_usd rows -- would need
  one more query, `SELECT token_id, trade_time, side, price_usd,
  amount_usd FROM lawrence.silver.token_trades WHERE token_id IN (...)
  ORDER BY token_id, trade_time`) is `sample_assessment_times`' own
  contract: with only 2 trades, the T-grid runs from the FIRST trade's
  timestamp to the LAST, stepping every 15 minutes -- if the earliest
  trade is the dust one (price_usd == 0) and it's also the only trade
  `price_at` can anchor to at T=start (the always-included first sample
  point), the row drops at T=start regardless of the second trade's real
  price, UNLESS a later 15-minute grid point happens to land within
  `PRICE_STALENESS` (30 min) of that second trade -- which a sparse,
  irregularly-timed 2-trade history frequently won't.
  Net conclusion, unchanged from before despite the correction: this
  isn't a bug and doesn't need a code change. A token with 1-4 total
  trades in its entire Silver history has no real trailing-activity
  signal to learn from either way -- there's nothing informative to
  recover here by relaxing `price_at`/`trailing_activity_features`,
  unlike the earlier blue-chip-token situation where real, tradeable
  assets were at risk of being silently excluded. 16/61,633 (0.026%)
  dropping out because they're single-blip tokens is exactly the kind of
  gap the distinct-token-count check exists to surface and let Jack
  judge for himself, not something to "fix" away.

- **exit_candidates Silver->Gold gap is much larger than entry_candidates'
  and this is architecturally expected, not a bug -- verified against the
  real code (2026-09-17).** notebook 03's first real run: 3,018,803 rows
  written, but only 14,907 of 61,633 distinct Silver tokens (24%) made it
  in, vs. entry_candidates' 61,617/61,633 (99.97%). Rather than guess,
  wrote a standalone script importing `build_entry_row`/`build_exit_rows`/
  `sample_assessment_times` directly from `gold_features.py` and ran both
  against several synthetic trade-spacing scenarios. Confirmed the
  mechanism: `build_entry_row` only needs ONE trade within
  `PRICE_STALENESS` (30 min) at-or-before T, and T's sampling grid always
  includes T=the token's very first trade's own timestamp -- so almost
  any token with >=1 usable-price trade gets at least one entry row
  (matches the 99.97% figure). `build_exit_rows` additionally needs a
  valid synthetic entry price at T-h for at least one h in
  `EXIT_SIMULATED_HOLDS` (15/30/60/120 min) -- which requires a SECOND
  trade roughly 15-150 minutes before that same T. A single-trade token
  (1 sampled T, 0 exit rows) or a 2-trade token with the trades <15 min
  apart (0 exit rows) both confirmed this in the script; wider or denser
  spacing produced exit rows, sometimes more than entry rows since one T
  can yield up to 4 (one per hold duration). This matches notebook 03's
  own pre-existing comment ("exit candidates need MORE trailing history
  per token than entry candidates do") -- now confirmed against the real
  functions instead of just asserted.
  What ISN'T yet confirmed: whether the *real* Silver trade-count
  distribution actually has enough single/sparse-trade tokens to justify
  a 76% gap specifically (vs. some size that would suggest a second,
  subtler issue). Suggested Jack run the real distribution query
  (bucketed trade-count histogram across all 61,633 tokens) to check
  before fully closing this out -- not yet run.

- **exit_candidates gap fully, quantitatively explained -- closing this
  out (2026-09-17).** Real Silver trade-count distribution across all
  61,633 tokens: 44,380 have exactly 1 trade (72.0%), 4,753 have 2,
  5,855 have 3-5, 3,083 have 6-20, 1,719 have 21-100, 1,843 have 100+.
  The math closes exactly: 44,380 single-trade tokens can NEVER produce
  an exit row (build_exit_rows structurally requires a second trade --
  see the verified mechanism two entries up), leaving 61,633 - 44,380 =
  17,253 tokens with >=2 trades. Of those, 14,907 made it into
  `lawrence.gold.exit_candidates`, meaning 17,253 - 14,907 = 2,346
  multi-trade tokens ALSO produced zero exit rows -- consistent with
  trades clustered too tightly together (e.g. a snipe-bot buy
  immediately followed by a sell, both within the same ~15-minute
  window, so no valid T-h anchor ever exists). 44,380 + 2,346 = 46,726,
  which is exactly 61,633 - 14,907 -- the entire gap accounted for, no
  residual unexplained by. No bug, no code change. This is a real,
  structural fact about the token universe Silver has captured: a large
  majority of tokens are one-and-done (minted, traded once, abandoned --
  consistent with the pump.fun-suffix pattern in the earlier 16-token
  gap investigation), and exit-model training data can only meaningfully
  exist for tokens with enough trade density to simulate a real holding
  period. Both `lawrence.gold.entry_candidates` (1,276,167 rows,
  61,617/61,633 tokens) and `lawrence.gold.exit_candidates` (3,018,803
  rows, 14,907/61,633 tokens) are now considered a clean first real-scale
  run -- every observed gap in both tables has a confirmed, benign
  explanation. Gold v1 is unblocked; next real step is Phase 3 (baseline
  + first trained model) per docs/game_plan.md, not further Gold
  debugging.

- **Phase 3 kickoff: entry baselines + first ML model (2026-09-17).**
  Three scope decisions made with Jack before writing any code (see
  docs/game_plan.md §6's Phase 3):
  1. **Entry only this pass**, not entry+exit together -- exit_candidates
     has 24% the token coverage entry does right now (see the gap
     investigation above) and label_spec.md already treats them as
     separate models with different needs. Exit gets its own pass once
     this one's pipeline (walk-forward, backtest metrics, MLflow logging)
     is proven out.
  2. **Baseline B is a stand-in, not the real thing.** game_plan.md §5's
     originally-envisioned "smarter" baseline ("liquidity > X and no
     prior rugs from this deployer") needs pool-liquidity-state and
     deployer-wallet data that doesn't exist yet (the same Phase 1 gap
     tracked since the rug-clamping work). Substituted the closest signal
     Gold v1 actually has: a trailing-volume + buy/sell-ratio momentum
     filter (`BASELINE_B_MIN_TRAILING_VOLUME_USD`/
     `BASELINE_B_MIN_BUY_SELL_RATIO` in `src/backtest/baselines.py`,
     placeholders, not measured). Documented plainly as a stand-in to
     swap out once real liquidity/deployer data lands, not a permanent
     design.
  3. **Regression on `y_entry`, not binary classification.** `y_entry` is
     already a continuous forward return including the interim rug-clamp
     -- predicting it directly supports ranking candidates by expected
     return (how an entry decision actually gets made) and keeps a -100%
     rug loss as real magnitude instead of collapsing it into the same
     "loss" bucket as a -5% dip.

  New code, all pure-Python/pandas (no PySpark), matching the existing
  src/features/ split -- notebooks do the Spark-table-read glue only:
  - `src/backtest/baselines.py` -- Baseline A ("buy everything," mirrors
    BIA's actual current rule) and Baseline B (above).
  - `src/backtest/walk_forward.py` -- `walk_forward_folds` (expanding-
    window time-based splits -- game_plan.md §5 is explicit that a random
    split on time-series token data is how these backtests lie) and
    `simulate_trades` (applies round-trip trading costs -- 30bps fee +
    50bps slippage placeholders -- and reports EV/trade, win rate, max
    drawdown per game_plan.md §0/§2's real success metric, not "average
    return").
  - `src/models/entry_model.py` -- `LGBMRegressor` on the 6 numeric
    trailing-activity columns Gold v1 has (chain_family/dex_venue etc.
    deliberately left out of v1 -- tracked as a follow-up, not an
    oversight). Rug-clamped rows are NOT filtered from training -- real
    signal, not noise, per gold_spec.md's whole rationale for clamping
    instead of dropping.
  - `notebooks/04_entry_baseline_and_model.py` -- reads
    `lawrence.gold.entry_candidates` (select()-trimmed before toPandas(),
    same OOM-avoidance lesson as notebooks 02/03), runs all three
    strategies through the same walk-forward folds, logs params/metrics/
    the last fold's model to MLflow (already in requirements.txt --
    Databricks-native, ties into game_plan.md §7's ML-Associate-lifecycle
    framing), prints a comparison table.

  **Bug caught before Jack ever ran it:** wrote standalone tests for
  `walk_forward_folds` first (verified against real pandas in the
  device_bash sandbox, which has pandas/numpy but not pytest/sklearn/
  lightgbm/mlflow -- no network there either, confirmed by a failed `pip
  install`). `len(folds) == n_folds` failed -- fold 0 always tested on the
  very first time bucket, which structurally has nothing before it to
  train on, so it silently got dropped every time (n_folds=4 only ever
  returned 3 folds). Fixed by reserving one extra boundary bucket purely
  as the initial training seed (`n_folds + 2` boundary points instead of
  `n_folds + 1`) so test folds are always buckets 1..n_folds, each
  guaranteed a non-empty train window before it. All 9
  baseline/walk-forward tests pass now against real pandas.
  `tests/test_entry_model.py` also written (skips cleanly via
  `pytest.importorskip` where lightgbm isn't installed, same principle as
  `test_label_point_in_time.py`'s existing scaffold) but NOT independently
  run -- lightgbm/scikit-learn aren't available in this sandbox either.
  Jack: run `pytest` in your own terminal to confirm before trusting the
  model-training path fully, same caveat as every other lightgbm-adjacent
  change this session.

- **First walk-forward run's backtest was meaningless: 779 rows with
  billions-of-percent "returns," from near-zero-priced tokens, not a bug
  (2026-09-17).** notebooks/04's first real run: LightGBM's own
  "Start training from score" line was the first tell -- fold 0's target
  mean was a sane 0.0986, but folds 1-4 jumped to 16,791.9 / 11,245.1 /
  8,612.7 / 6,986.2. The resulting backtest table confirmed it was worse
  than just the model: Baseline A alone showed `total_return` of 7.97e9
  (fold 0) and 1.83e10 (fold 4) -- not real numbers for any strategy,
  and since baselines/model all score against the same underlying
  `y_entry`, this was a label problem, not a model problem.
  Confirmed with two queries (Jack): `SELECT ... ORDER BY y_entry DESC
  LIMIT 20` showed the actual offending rows -- price_at_t values from
  6e-15 to 7e-8 (one exception at 0.0037, likely a single bad price_exit
  trade, absorbed by the same fix) -- and `SELECT COUNT(*) WHERE
  y_entry > 100` came back 779 (0.061% of entry_candidates' 1,276,167
  rows), max_y_entry = 16,693,038,859.
  Root cause: these tokens are genuinely priced at 1e-8 to 1e-15 USD --
  plausible real data (huge total supply, near-zero market cap), not a
  decimals/parsing bug (checked `silver_token_trades.py`/`dune_client.py`:
  Dune's `token_bought_amount`/`token_sold_amount` are already human-unit,
  decimal-adjusted). At that price scale, a tiny absolute price move
  produces an astronomical PERCENTAGE "return" -- mathematically real
  given the raw trade log, but not something any real trade could capture
  without knowing actual pool depth, which is the same liquidity-data gap
  already tracked in gold_spec.md's feature-set section and
  game_plan.md's Phase 1 (this is now the third distinct place that gap
  has surfaced this session, after rug-clamping and Baseline B).
  Fix: added `RETURN_WINSORIZE_CAP = 100.0` and `winsorize_return()` to
  `gold_features.py`, next to `RUG_CLAMP_RETURN` -- both
  `src/models/entry_model.py` (training target) and
  `src/backtest/walk_forward.py` (`simulate_trades`) now clip through it.
  Deliberately a CLIP, not a drop (game_plan.md §2's survivorship-bias
  warning applies to silently dropping extreme winners just as much as
  rugs), and deliberately NOT written back into Gold itself -- Gold stays
  the raw, point-in-time-correct ground truth; only consumers winsorize.
  This also corrects `label_spec.md`'s original plan ("keep the raw value
  for backtest P&L simulation") -- that assumption didn't survive contact
  with real data, updated in place there. Verified `winsorize_return`
  (scalar and pandas Series) plus the full existing gold_features/
  baselines/walk_forward suite (34/34) against real pandas in the
  device_bash sandbox. Re-running notebooks/04 with this fix in is the
  next step -- not done yet.

- **Byte-identical results across two "redeploy and re-run" cycles was
  module caching, not a deploy-sync issue or a bug in the fix itself
  (2026-09-18).** After the winsorize fix (previous entry), notebooks/04
  produced the EXACT same "Start training from score" values and the
  EXACT same backtest table as the pre-fix run -- not just similar,
  bit-for-bit identical, across two separate redeploy+re-run cycles.
  Since `train_entry_model` uses a fixed `random_state`, identical input
  -> identical output, so this was strong evidence the fix's code path
  wasn't actually running, not that it ran and failed.
  Ruled out step by step: (1) confirmed the fix was genuinely committed
  (`git status` clean at `ca5d23a`, `prepare_features` calls
  `winsorize_return` in the file on disk); (2) asked Jack to confirm a
  git push had happened before deploy (a real risk given my commits land
  locally via the device bridge, not pushed by me -- see the working-
  style constraints at the top of this doc) and to run a scratch-cell
  sanity check (`from features.gold_features import RETURN_WINSORIZE_CAP,
  winsorize_return; print(...)`) -- came back `100.0`/`100.0`, proving
  the correct code WAS present and callable in that session.
  That combination (file correct, function correct when called fresh,
  full notebook re-run still stale) pointed at Python module caching:
  `sys.path.append("../src")`-imported local modules
  (`features.*`/`backtest.*`/`models.*`) are plain Python imports, not
  pip packages -- an interactive cluster/session that stays warm across
  notebook reruns caches them in `sys.modules` on first import and does
  NOT re-read them from disk on a later run. The notebook's own `%pip
  install` cell only guarantees fresh code for pip packages; it doesn't
  address this. Fixed by adding `dbutils.library.restartPython()` right
  after the `%pip install` cell, before any `src/` imports -- this is
  exactly the call I'd originally included when first writing this
  notebook and then deliberately removed as unnecessary (see the
  original Phase 3 kickoff entry); that removal was the wrong call.
  Documented as a standing convention in docs/conventions.md rather than
  just a one-off fix, since any future notebook importing from `src/` on
  a warm session has the same exposure. Re-running notebooks/04 with this
  in is the next step -- not done yet, and this time the "Start training
  from score" values changing is itself the confirmation signal to look
  for, before even getting to the backtest table.

- **Entry model v2: categorical features + a cost-aware decision rule
  (2026-09-18).** v1's first real backtest (previous entries) showed the
  LightGBM model losing to Baseline B on ev_per_trade in every
  walk-forward fold. Jack asked to pursue optimizing the model further;
  two concrete, independently-motivated changes landed together in this
  pass (`src/models/entry_model.py`, `src/backtest/walk_forward.py`,
  `notebooks/04_entry_baseline_and_model.py`, plus their tests):
  1. **Categorical features.** v1 was deliberately numeric-only (see its
     own module docstring) -- `chain_family`/`network`/`dex_venue`/
     `dex_protocol` are real signal LightGBM can use natively via
     `category`-dtype columns (no one-hot needed), but v1 punted on the
     encoding decision to keep the first pass simple. `FEATURE_COLUMNS`
     is now `NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS`, and
     a new `set_categorical_dtypes()` casts the four categorical columns
     to pandas `category` dtype. Its docstring explains the one real
     footgun: it MUST run once on the FULL dataset before walk-forward
     splitting, never per-fold -- pandas derives a category-dtype
     column's category set from whatever values are present at cast
     time, so casting each fold's subset separately could give the same
     column inconsistent category sets across folds (a category seen
     only in a later fold's test data would look "unseen" to LightGBM
     instead of just less common). The notebook now calls it once on
     `entry_df` right after sorting, before the fold loop.
  2. **Cost-aware entry threshold.** v1's decision rule was simply
     `preds > 0`, which ignores the ~1.6% round-trip trading cost (30bps
     fee + 50bps slippage, each way) `simulate_trades` already charges --
     a model predicting a small positive return (say, +0.5%) would still
     "enter" a trade that's a guaranteed loss after real costs. Added
     `DEFAULT_FEE_BPS`/`DEFAULT_SLIPPAGE_BPS` and a `round_trip_cost()`
     helper to `walk_forward.py` (refactoring `simulate_trades` to call
     it internally instead of duplicating the formula), so the notebook's
     decision rule (`preds > round_trip_cost()`) and the backtest's own
     cost accounting are guaranteed to use the exact same numbers --
     never two cost assumptions that could quietly drift apart.
  Verification note (same device_bash sandbox limitation as every prior
  entry -- no network, no lightgbm/scikit-learn/mlflow installed): all
  non-lightgbm-dependent tests (gold_features, baselines, walk_forward,
  including new tests for `round_trip_cost` and its shared-formula
  guarantee with `simulate_trades`) ran for real against real pandas and
  passed. For `entry_model.py`'s new tests (`set_categorical_dtypes`,
  the winsorized-target check, the categorical-dtype-survives-
  prepare_features check), real `lightgbm` still isn't installable here,
  so they were additionally run against a small stand-in `LGBMRegressor`
  (plain least-squares regression on the numeric columns only) so the
  actual plumbing -- casting, column selection, winsorization, fit/
  score/reproducibility -- gets exercised end-to-end rather than just
  skipped outright. That stand-in can't validate real categorical-split
  behavior or gradient-boosting quality; Jack still needs to confirm the
  real thing in Databricks. Next step: redeploy and re-run notebooks/04,
  watching whether `lightgbm_model`'s ev_per_trade improves relative to
  Baseline B now that both changes are in.

- **v2 backtest result: still loses to Baseline B in every fold, but the
  gap closes sharply as training history grows (2026-09-18).** Jack
  re-ran notebooks/04 with both v2 changes (categorical features,
  `preds > round_trip_cost()`) in:

  | fold | lightgbm ev/trade | baseline_b ev/trade | gap | lightgbm n_trades |
  |---|---|---|---|---|
  | 0 | -0.6162 | -0.2604 | -0.356 | 5,345 |
  | 1 | -0.5266 | -0.2513 | -0.275 | 759 |
  | 2 | -0.3708 | -0.2580 | -0.113 | 676 |
  | 3 | -0.3409 | -0.2586 | -0.082 | 797 |
  | 4 | -0.2666 | -0.2533 | -0.013 | 538 |

  Two honest reads, not one triumphant one:
  1. **Not a win yet.** Pooled across all folds (`sum(total_return) /
     sum(n_trades)`), lightgbm_model's ev/trade is -0.537, worse than
     both baseline_a (-0.392) and baseline_b (-0.257). v2's changes did
     NOT flip the headline result from the first backtest -- the model
     still loses.
  2. **The gap is closing fold-over-fold, and that trend is real enough to
     act on.** `baseline_b`'s ev/trade is roughly flat across all 5 folds
     (-0.260 to -0.251) -- a fixed threshold rule, it doesn't benefit from
     more training history by construction. `lightgbm_model`'s ev/trade
     improves monotonically as the walk-forward training window expands
     (fold 0's -0.616 to fold 4's -0.267), closing a 0.356 gap down to
     0.013 -- nearly tied in the last fold. That's consistent with a model
     that's genuinely data-starved, not a model that's wrong in kind --
     more real trading history (which keeps accumulating) may be what it
     actually needs, more than another feature/threshold tweak right now.
  Caveat worth flagging, not burying: `n_trades` for `lightgbm_model`
  collapsed to 538-797 per fold (down from thousands under v1's
  `preds > 0` -- `round_trip_cost()` is a much stricter bar, as intended).
  Fold 4's win_rate (0.229) is also a clear outlier vs. every other fold's
  0.06-0.13, on only 538 trades. A trend built on a few hundred trades per
  fold could easily be noise dressed up as a pattern, not yet strong
  enough evidence to declare "getting more data will fix this" -- it's
  a hypothesis the trend supports, not a conclusion the trend proves.
  Next step (not yet decided): most likely more real trading history
  accumulating naturally over time, rather than another immediate model
  change -- revisit once meaningfully more Silver data exists.

- **Dune pricing research for expanding the historical backfill
  (2026-09-18).** Jack asked to add training data faster/cheaper than
  waiting weeks for live accumulation to fix the data-starvation read
  from the v2 backtest (previous entry). Researched Dune's current
  pricing before proposing a plan, since the existing backfill's
  "5 separate API keys" pattern (2026-09-15 entries) is exactly the kind
  of usage Dune just moved against:
  - **2026-09-10 policy change:** legacy free accounts (created before
    2026-07-21) lost query-execution ability entirely (view-only now,
    with a one-time 14-day Plus trial offered before reverting), cited as
    an anti-abuse move against "rising compute costs." Newer accounts
    aren't immediately affected, but this is explicit evidence Dune is
    actively closing the free-credit-cycling path the first backfill
    relied on -- not a foundation to keep building on for a second round.
  - **Current plan structure:** Free plan still includes 2,500
    credits/month + API access, with pay-as-you-go top-ups at a flat
    $5.00/100 credits ($0.05/credit) and NO subscription required. Plus
    is $349/month flat, ~25,000 credits included (~$0.014/credit average
    at full use); Analyst overage is $1.875/100 credits monthly. Credits
    are consumed by actual compute (rows scanned + engine time), not a
    fixed per-row rate, matching what the original COST INCIDENT already
    found empirically.
  - **Rough empirical burn rate, from the completed backfill:** 1 month
    (2026-08-14 to 09-13), both venues, trimmed columns, took ~4-5
    free-tier credit exhaustions to finish -- roughly 10,000-12,500
    credits for that window. That's already past the ~9,500-credit point
    where a single month of Plus ($349 flat) would have been cheaper than
    paying Free's $0.05/credit rate out of pocket ($500-625).
  Proposed plan (not yet decided/executed -- see next entry once Jack
  picks the parameters): (1) stop relying on further free-account
  cycling -- fragile now and the exact pattern Dune is closing; (2) run a
  cheap aggregate cost-test (COUNT(*)/GROUP BY, same playbook as the
  original cost-test, well under $1) comparing credits-per-day for
  extending raydium_v4's window further back in calendar time vs. adding
  1-2 more Solana venues (pump.fun bonding curve, Orca, Meteora -- however
  `dex_solana.trades` tags them) within the CURRENT window, since the
  entry model's actual data-starvation is in distinct tokens/candidates,
  not calendar density, and it's not yet known which lever is cheaper per
  new distinct token; (3) spend on whichever measures cheaper, via Free's
  pay-as-you-go top-up unless the measured need exceeds ~9,500 credits (in
  which case one month of Plus, then downgrade, is the cheaper option);
  (4) skip extending the EVM/uniswap_v3_base side -- blue-chip tokens
  aren't the data-starved population, the high-token-count memecoin/
  launchpad side is.

- **Wrote the cost-test tooling for the Dune backfill-expansion decision
  (2026-09-18, following the previous entry's proposed plan).** Jack
  picked: fund via Dune's Free-plan pay-as-you-go ($0.05/credit, no
  subscription), and run the cost-test before spending anything.
  Added `DuneClient.run_sql()` -- a general-purpose "run this exact SQL,
  return all rows" method, since the existing `fetch_historical_trades`
  is hard-coded to `DuneVenueFilter`'s trade-row shape (fixed columns,
  `WHERE`-fragment-only filtering) and can't express an aggregate
  `COUNT(*)`/`GROUP BY` cost-test query. `fetch_historical_trades` now
  delegates to it (pure refactor, same SQL it always built, covered by a
  regression test). New `notebooks/00d_dune_cost_test.py` runs three
  staged, cost-gated aggregate queries -- deliberately small and run one
  at a time, with an explicit instruction to check the real Dune credit
  balance after each before running the next (an unfiltered-by-project
  discovery scan is real new territory: every single-venue query this
  backfill has run so far filtered to one project, so this could scan
  more rows than Jack's used to paying for):
  1. A 1-hour unfiltered discovery query (what Solana DEXs exist at all,
     by trade count/distinct tokens/volume) -- the same "tiny window"
     size the original (pre-incident) cost-test used, run first
     specifically to confirm this query shape is cheap before widening it.
  2. A 7-day unfiltered discovery query over the last week of the
     already-backfilled window (2026-09-06 to 09-13) -- big enough to
     actually rank candidate venues (pump.fun/Orca/Meteora/whatever
     Dune's `project` labels turn out to be -- not yet verified against
     real data, deliberately discovered rather than guessed, matching
     this project's established "verify against live Dune data before
     trusting a literal" practice from the raydium_v4 filter bugs).
  3. The same 7-day-window distinct-token-count query, but for
     `raydium_v4` specifically, 7 days immediately BEFORE the current
     backfill's start (2026-08-07 to 08-14) -- the "extend calendar time"
     lever's yield, directly comparable to #2's `raydium` row for "extend
     to more venues in the same window already paid for."
  Not yet run -- Jack needs to execute this in Databricks and report back
  real credit-balance deltas and the resulting distinct-token counts for
  each lever before any real spend decision gets made.

- **Cost-test results: "more Solana venues" beats "more calendar time" by
  roughly 50x, not a close call (2026-09-18).** Jack ran all three
  `00d_dune_cost_test.py` cells. Real numbers, not estimates:

  | project (version) | 7-day n_trades | 7-day n_distinct_tokens | trades/token |
  |---|---|---|---|
  | `pumpdotfun` (1) | 20,059,502 | **206,902** | ~97 |
  | `raydium_launchlab` (1) | 3,410,279 | 42,479 | ~80 |
  | `meteora` (3) | 18,971,523 | 39,932 | ~475 |
  | `meteora` (4) | 5,862,721 | 24,438 | ~240 |
  | `pumpswap` (1) | 113,425,521 | 24,177 | ~4,692 |
  | `raydium` (4, ALREADY BACKFILLED) | 1,852,123 (current window) | 2,793 (current window) | ~663 |
  | `raydium` (4) EXTEND-TIME lever, 7 days before current window | 937,161 | 4,085 | ~229 |

  Extending `raydium_v4` 7 days further back (cell 3) buys 4,085 new
  distinct tokens. Adding `pumpdotfun` for the SAME 7 days already
  covered by the current backfill (cell 2) buys 206,902 -- ~50x more,
  for a window that's already paid for and needs no calendar extension
  at all. `raydium_launchlab` and `meteora` (v3/v4) are the next-best
  levers, each ~6-10x the extend-time yield.

  A second finding worth acting on, not just the headline ranking:
  **`pumpdotfun` and `raydium_launchlab` are also the most row-efficient
  additions**, not just the highest-token-count ones -- ~80-97
  trades/token, actually BELOW `raydium_v4`'s own ~229-663 trades/token,
  because they're literally launch venues (bonding-curve/launchpad
  mechanics), matching LAWRENCE's actual "score a new token launch"
  framing more directly than `raydium_v4` ever did. `pumpswap` and
  `meteora`, despite big token counts, carry much heavier trade volume
  per token (graduated/already-liquid tokens) -- real signal, but
  markedly more expensive to extract per unit of new training signal.
  Several other discovered venues (`bisonfi`, `humidifi`, `tessera`,
  `scorch`, `goonfi`, `alphaq` -- all from cell 1/2) have huge trade
  counts/volume but under 30 distinct tokens each -- not memecoin/launch
  venues at all (likely stablecoin/perp/MM infrastructure), explicitly
  not worth adding for this project's purpose.

  **Real cost caveat, not yet resolved:** `pumpdotfun` alone scales to
  roughly 86M trades for a full month at this rate (20.06M/week x ~4.3),
  `raydium_launchlab` to ~14.7M -- combined, roughly 10x the current
  Bronze dataset's total row volume for the SAME one-month window. Real
  Dune query cost for a trade-level extract at that row count is
  unmeasured (the cost-test cells were aggregates, not trade-level
  SELECTs -- see `dune_client.py`'s own COST INCIDENT note: aggregate
  cost doesn't predict trade-level extract cost 1:1). Credit-balance
  deltas for the three test cells themselves were not yet reported by
  Jack, so actual $ spent so far and $/row for a real extract are both
  still unknown. Next: get those numbers, then decide real scope (full
  month of `pumpdotfun`+`raydium_launchlab`, a shorter window first as a
  real-cost test at trade-level, or something in between) before touching
  `00c_bronze_backfill_dune.py`'s `VENUES`/`DUNE_VENUE_FILTERS`.

- **Correction: the cost-test itself burned through the whole free-tier
  balance, and "let it self-limit" doesn't hold once pay-as-you-go is
  active (2026-09-18).** Jack reported running out of Dune credits after
  the three cost-test cells -- the "aggregates are cheap" assumption
  `00d_dune_cost_test.py`'s own markdown leaned on was wrong for this
  table at this scale: cell 2's unfiltered 7-day discovery scanned
  roughly 200M+ rows across ALL ~26 Solana DEX projects (summing every
  project's `n_trades` in the previous entry's table) to produce one
  aggregate -- MORE rows scanned than the entire proposed
  `pumpdotfun`+`raydium_launchlab` month-long extract (~101M rows) would
  need, because an unfiltered query can't skip the 24 other projects'
  data the way a `WHERE project = ...` filter can. Lesson: never run
  another unfiltered discovery scan -- filter by project from the start
  now that the real labels are known.

  Separately, and more importantly: Jack had picked "start the full
  month now, let it self-limit" for scope, reasoning the resumable
  progress table + per-chunk writes would cap any damage the way they
  did during the original backfill. That reasoning held when the only
  currency was a fixed free-credit balance that hard-stops at zero. It
  does NOT hold now that the chosen funding path is Free-plan pay-as-you-
  go top-ups (Jack's own earlier choice, previous entries) -- pay-as-you-
  go has no ceiling by design; "self-limiting" would mean it keeps
  charging a real payment method until Jack manually stops it, not until
  some safe free-tier wall is hit. Flagged back to Jack before running
  anything at full scale, with a rough extrapolated cost estimate
  (calibrated off the completed month/2-venue backfill's own ~10,000-
  12,500-credit, ~10.75M-row economics): roughly 100,000-120,000 credits
  for a full month of `pumpdotfun`+`raydium_launchlab` at ~101M rows --
  $5,000-$6,000 on Free's $0.05/credit rate, or roughly $1,500-$1,600 on
  one month of Plus. Recommending a small real paid test slice (a single
  day, project-filtered, trade-level) to replace this rough extrapolation
  with a real number before committing to anything close to full scope --
  not yet decided.

- **Wired pumpdotfun + raydium_launchlab into the real backfill, scoped to
  a one-day test first (2026-09-18).** Per Jack's decision (top up a
  small amount, test one real day before committing to full scope):
  - Added both as new `Venue`/`DuneVenueFilter` entries in
    `00c_bronze_backfill_dune.py`, same `dex_solana.trades` table/column
    shape as `raydium_v4` (confirmed via `00d`'s discovery queries, which
    already read `token_bought_mint_address` successfully for both --
    not guessed). `version = 1` unquoted for both, matching the
    2026-09-15 lesson that `version` is an integer column.
    `dex_program_or_address` is an explicit unverified placeholder --
    these are Dune-backfill-only for now, no live Bitquery counterpart.
  - Restructured the main chunking loop from "for each time chunk: for
    each venue" to "for each venue: for each chunk in that venue's own
    window" (new `VENUE_WINDOWS` dict, defaulting to the shared
    `BACKFILL_WINDOW_START/END` for any venue not listed). This lets
    `pumpdotfun`/`raydium_launchlab` start on a single day
    (2026-09-12 to 09-13) without touching `raydium_v4`/
    `uniswap_v3_base`'s already-completed full-month window at all --
    those two keep skipping every chunk via the progress table exactly
    as before. Sanity-checked the chunk-count arithmetic standalone
    (30+30+1+1=62 total chunks, only 2 of which do real work this run).
  Not yet run. Next: Jack tops up Dune credits, runs the notebook, reports
  the real per-day cost for these two venues (check the progress table:
  `SELECT * FROM lawrence.bronze._dune_backfill_progress WHERE venue_name
  IN ('pumpdotfun', 'raydium_launchlab') AND status = 'success'` alongside
  the Dune credit-balance delta) -- then decide whether to widen
  `VENUE_WINDOWS` to the full month at that measured rate.

- **Test-day run crashed on a Databricks session timeout mid-pumpdotfun-
  fetch, not a Dune error (2026-09-18).** Jack ran the notebook from the
  previous entry; it crashed with `BAD_REQUEST: session_id is no longer
  usable... INACTIVITY_TIMEOUT_AND_FAILED_TO_RESTORE`, raised from INSIDE
  the `except Exception` write-failure handler -- meaning the original
  `_write_bronze` call failed (dead Spark session), and then the
  fallback `_log_progress("failed", ...)` call ALSO failed for the same
  reason, so the whole notebook crashed instead of cleanly logging and
  continuing. Real cost consequence: that chunk's Dune fetch had already
  succeeded (real credits spent, real rows returned) before the write
  step, but none of it reached Bronze or the progress table -- a re-run
  has to re-fetch and re-pay for that specific chunk.
  Root cause: `pumpdotfun`'s daily chunk is ~2.9M rows (00d's 7-day
  measurement), vastly more than `raydium_v4`'s ~60k-260k rows/day that
  daily chunking was originally proven against. Paginating that many rows
  from Dune is minutes of pure Python/HTTP work with zero Spark activity
  -- long enough for the compute session to go idle and get torn down
  mid-fetch.
  Fixed: new `VENUE_CHUNK_HOURS` override (same pattern as `VENUE_WINDOWS`),
  set to 1 hour for `pumpdotfun`/`raydium_launchlab` (raydium_v4/
  uniswap_v3_base keep 24h, proven fine). At the measured 7-day rate this
  puts pumpdotfun at roughly 120k rows/chunk instead of ~2.9M -- keeps
  each fetch+write cycle short enough to avoid tripping the idle timeout,
  and gives more frequent checkpoints if anything else goes wrong.
  Sanity-checked the updated chunk-count math standalone: pumpdotfun/
  raydium_launchlab now 24 hourly chunks each for the one test day
  (raydium_v4/uniswap_v3_base unaffected, still 30 daily chunks each,
  all already `success` and skipped). Not yet re-run.

- **Test-day run succeeded after the chunking fix -- real cost-per-row
  measured (2026-09-18).** Both new venues completed 24/24 hourly chunks
  for the one test day (2026-09-12): `pumpdotfun` 2,701,352 rows,
  `raydium_launchlab` 489,916 rows -- 3,191,268 rows total. Actual Dune
  spend for that day: 2,489.006 credits. That gives a real, measured
  cost-per-row for the first time (not an extrapolation from the original
  raydium_v4/uniswap_v3_base backfill): ~780 credits/million rows,
  slightly cheaper than the earlier rough estimate (~930-1,163
  credits/million rows) -- validates that estimate's ballpark accuracy.
  `pumpdotfun` is 84.65% of the combined row volume (2,701,352/3,191,268)
  and, assuming a shared per-row rate, an equivalent share of the day's
  cost (~2,107 of the 2,489 credits) -- `raydium_launchlab` alone is
  roughly 6.5x cheaper for a full month than `pumpdotfun` alone.
  Full-month (30-day) projections at this rate:
  | scope | credits | Free pay-as-you-go ($0.05/credit) | Plus overage (25k/mo included, $0.01596/credit over) |
  |---|---|---|---|
  | both venues | ~74,670 | ~\$3,734 | ~\$793 (+ \$349 base) |
  | pumpdotfun only | ~63,207 | ~\$3,160 | ~\$610 (+ \$349 base) |
  | raydium_launchlab only | ~11,463 | ~\$573 | \$0 (+ \$349 base, fits in included credits) |
  Shorter windows as a middle ground: 1 week (both venues) ~17,423
  credits / ~\$871 Free / fits inside Plus's included 25k; 2 weeks
  ~34,846 credits / ~\$1,742 Free / ~\$157 Plus overage.
  Decision on final scope/plan/window left to Jack -- not yet made.

- **Full-month widen (2026-09-19): `pumpdotfun`/`raydium_launchlab`
  `VENUE_WINDOWS` widened to the full `BACKFILL_WINDOW_START`/`END`
  range.** Test-day cost was measured and accepted (previous entry);
  funding is free-tier key-cycling, not pay-as-you-go, so no real-dollar
  exposure from widening. Run started fine (chunks already completed from
  the earlier test day correctly skipped via the progress table), but
  every NEW chunk from `2026-08-19 00:00` onward failed with a generic
  `Dune POST .../sql/execute failed after 3 attempts` -- `_request`'s
  retry loop was swallowing the actual HTTP status/response body on every
  non-402 failure, so there was no way to tell a bad/expired key (401), a
  legacy account without execute permission (403 -- see the 2026-09-10
  Dune policy change noted earlier), rate limiting (429), or a Dune-side
  5xx apart; they all rendered identically. Fixed: `_request` now catches
  `requests.HTTPError` specifically and includes `last_status`/`last_body`
  (first 500 chars of the response) in the final `RuntimeError` message.
  402 handling (`DuneCreditsExhausted`, raised immediately, never
  retried) is unchanged. Verified via mocked `requests.request` for both
  the new error-detail path and the pre-existing 402/success paths (real
  pytest not available in this sandbox, same standing limitation as
  elsewhere in this doc). Not yet re-run for real -- next step is Jack
  re-running so the real status/body tells us whether this is a bad key
  or an execute-permission issue on the newest friend's account.

- **`Java heap space` on the Spark driver during `_write_bronze`, not a
  Dune issue (2026-09-20).** Full-month run for `pumpdotfun`/
  `raydium_launchlab` was going fine (real fetches succeeding, e.g. chunk
  465 wrote 206,846 rows) until chunk 479 (only 119,638 rows) failed with
  `RESOURCE_EXHAUSTED: Java heap space` inside `spark.createDataFrame`,
  then chunk 480 immediately after. Row count didn't correlate with the
  failure (a smaller chunk failed shortly after a bigger one succeeded),
  pointing at cumulative driver JVM heap pressure across a long-running
  session (~480 chunks, one JVM the whole time) rather than any single
  chunk being too large -- likely compounded by the progress table's
  small-file/metadata bloat flagged earlier (never `OPTIMIZE`'d).
  Immediate recovery: detach/reattach the compute (fresh heap) + run
  `OPTIMIZE` on both `lawrence.bronze.dex_trades` and
  `lawrence.bronze._dune_backfill_progress` before re-running. Failed
  chunks are logged `"failed"` (not `"success"`), so re-running retries
  them safely -- costs re-paying Dune credits for those two re-fetches,
  not a data-loss risk.
  Structural fix (so a fresh session doesn't just repeat this after
  another few hundred chunks): `_write_bronze` now writes in
  `WRITE_BATCH_SIZE = 25,000`-row sub-batches instead of one
  `spark.createDataFrame` call per chunk (which for pumpdotfun chunks
  regularly means 100k-200k+ rows, `raw_json` blob included, marshalled
  through the driver's JVM heap in one shot -- `createDataFrame` on a
  Python list is a driver-local operation, not distributed). Bounds peak
  driver memory per call independent of chunk size or how long the
  session has been running. Return value/behavior unchanged for both
  call sites (`_write_bronze` itself and the `DuneCreditsExhausted`
  partial-rows salvage path).

- **Fixed the actual crash from the `SparkConnectGrpcException` traceback
  (2026-09-20): `_log_progress` inside the write-failure handler needed a
  working Spark session, same as `_write_bronze` -- when the write failed
  AND that fallback log call also failed (degraded/dead session either
  way), the second failure was uncaught and took the whole 1500-chunk run
  down instead of just skipping one chunk. Jack had Databricks' Genie
  patch this live in the notebook (guarding that `_log_progress` call in
  a try/except, plus `gc.collect()`/`del dune_rows` cleanup) before I saw
  it; ported the same guard back into the git-tracked notebook so it
  doesn't get silently reverted next pull, and applied the identical
  guard to the fetch-failure path's own `_log_progress` call too (same
  latent risk, just not hit yet). Note: `gc.collect()` manages CPython's
  own heap, not the separate JVM driver heap `RESOURCE_EXHAUSTED: Java
  heap space` actually came from -- `del dune_rows` has real value in the
  except branch (a caught exception's traceback keeps the frame, and
  therefore the fetched rows list, alive longer otherwise), but
  `WRITE_BATCH_SIZE` (previous entry) remains the actual mitigation for
  the JVM-side heap pressure, not this change.

- **`WRITE_BATCH_SIZE` dropped from 25,000 to 5,000 (2026-09-20).**
  25,000 kept hitting `RESOURCE_EXHAUSTED: Java heap space`, including
  immediately after what should have been a fresh serverless environment
  (no classic cluster exists -- `databricks clusters list` returns
  empty, confirmed this is Serverless compute). Two live possibilities,
  not fully distinguished: (1) whatever "restart" action was used only
  restarted the client-side Python kernel, not the server-side Spark
  Connect JVM Serverless keeps running behind it -- those are decoupled,
  so a Python-only restart wouldn't actually free driver heap; or (2)
  25,000 rows (with the `raw_json` blob per row) is still too large for
  Serverless's actual driver memory ceiling, restart or not. Going much
  smaller side-steps needing to resolve which one it is -- 5,000 is a
  cheap, git-tracked, reversible change with no effect on progress-table
  chunk keys or correctness, just more (smaller) Delta append calls per
  chunk.

- **Capped `raw_json`/symbol field sizes in `to_bronze_row`
  (2026-09-20).** Chunk 479 (`pumpdotfun`, 2026-08-31 10:00-11:00) failed
  with `RESOURCE_EXHAUSTED: Java heap space` identically across every
  `WRITE_BATCH_SIZE` tried (single unbatched call, 25,000, then 5,000) --
  row-count-based batching could never explain that, since chunk 465
  wrote 206,846 rows fine in one unbatched call earlier. Root-caused to
  `to_bronze_row`'s previously uncapped `json.dumps(dune_row,
  default=str)` for `raw_json`, plus the dedicated `buy_symbol`/
  `sell_symbol` columns -- pump.fun-family venues are permissionless
  token creation, so token names/symbols are fully attacker-controlled,
  arbitrary-length strings; a single spam/attack token with a
  pathologically huge symbol in this one hour's data would poison
  whatever batch it landed in regardless of batch size. Added
  `MAX_SYMBOL_LEN = 200` / `MAX_RAW_JSON_LEN = 20_000` and a `_cap_str`
  helper that truncates with a `...[truncated from N chars]` marker,
  applied to both symbol fields and the raw_json blob. Verified: 21
  existing `test_dune_client.py` tests unaffected (normal-length test
  fixtures never hit the cap) plus 3 new tests covering the cap itself,
  all passing via the sandbox's pytest-shim harness. Not yet confirmed
  against the real data (Jack running a cheap aggregate `MAX(LENGTH(...))`
  query on this exact chunk to check) but this is a real, unbounded-input
  gap independent of whether it's the exact trigger here, and the earlier
  `WRITE_BATCH_SIZE` reduction (25,000 -> 5,000) stays in place as
  additional headroom either way.

- **Dropped `mergeSchema="true"` from `_write_bronze`'s write call
  (2026-09-20).** It made sense before `_BRONZE_SCHEMA` existed (the
  2026-09-15 `CANNOT_DETERMINE_TYPE` incident), but every batch now
  writes that exact, unchanging explicit schema -- `mergeSchema` has
  nothing left to reconcile except `dex_trades`' full transaction
  history, on every single batch write. Suspect alongside a possibly
  bloated transaction log: since `WRITE_BATCH_SIZE` batching landed, a
  single ~120k-row chunk is ~24 separate append transactions instead of
  1 (at 5,000/batch), and the table hasn't been `OPTIMIZE`'d since well
  before batching existed -- asked Jack to check `dex_trades`' version
  count via `DESCRIBE HISTORY` and re-run `OPTIMIZE` regardless of the
  result. This is a real, concrete mechanism (driver has to open/reconcile
  against a growing log on every write) independent of any single chunk's
  row count or content, unlike the two theories already ruled out
  (batch size: identical failure from single-shot down to 5,000/batch;
  pathological symbol data: measured max length 13 chars, completely
  normal).

- **Root cause reassessment for the recurring `RESOURCE_EXHAUSTED: Java
  heap space` failure (2026-09-20), and a warm-up write as the actual
  fix.** Three write-side changes (single call -> 25,000 -> 5,000-row
  batches; capping raw_json/symbol fields; dropping mergeSchema) all
  made zero difference -- same chunk (pumpdotfun, 2026-08-31 10:00-11:00,
  119,638 rows) failed identically every time. The pattern those three
  attempts missed: chunks 1-478 (raydium_v4/uniswap_v3_base/pumpdotfun's
  already-`success` range) all skip before ever touching Spark, so
  whichever chunk comes right after that skip run is ALWAYS the first
  real Spark write of that session -- true on every single re-run,
  independent of any restart. Chunk 465 (206,846 rows, larger than the
  chunk that's now failing) had succeeded earlier in the SAME run as
  dozens of prior real writes -- not a cold start. This points at
  one-time per-session overhead on a table's first write (transaction-
  log/metadata load, codegen/JIT compilation, catalog connection setup)
  landing on top of a 100k+ row payload, rather than anything about that
  payload's own size or content, which is exactly why none of the
  write-side tuning changed anything -- they all optimize the write
  itself, not a fixed cost that happens once regardless of payload.
  Fix: a throwaway single-row warm-up write to `BRONZE_TABLE` (obviously
  sentinel-tagged `tx_hash`, immediately deleted after) right after
  `client = DuneClient(...)` and before the main loop, so any first-write
  tax lands on 1 row instead of 100k+. Verified `to_bronze_row` handles
  the dummy row correctly (pure Python, testable without Spark); the
  Spark-side write/delete itself needs a real run to confirm.

- **Silver rebuild for the 131M-row Bronze backfill: content-based
  `swap_id` dedup, Spark-native transform, parsed timestamps
  (2026-09-22).** Bronze now holds ~131M Dune rows (pumpdotfun 108.1M,
  uniswap_v3_base 10.7M, raydium_v4 8.6M, raydium_launchlab 3.6M) plus
  7,000 live Bitquery rows per venue. Three findings from profiling it:
  1. **The old key `(dex_venue, tx_hash, token_id)` deletes real trades.**
     Measured per venue as `n` vs `COUNT(DISTINCT xxhash64(tx_hash,
     buy_token_id, sell_token_id, buy_amount, sell_amount))` vs
     `COUNT(DISTINCT tx_hash)` (Dune rows only):
     | venue | true dupes | legit extra swaps in same tx |
     |---|---|---|
     | raydium_v4 | 36,010 (0.42%) | 413,774 (4.84%) |
     | pumpdotfun | 1,605,036 (1.48%) | 1,606,503 (1.49%) |
     | raydium_launchlab | 0 | 9,602 (0.26%) |
     | uniswap_v3_base | 130,278 (1.22%) | 1,792,904 (16.8%) |
     True dupes line up with the partial-chunk-salvage + full re-fetch
     path during the 2026-09-20 heap incidents (launchlab, which ran
     clean, has zero). Multi-swap-per-tx rows are real: router multi-hops
     on Base, and bundled launch buys on pump.fun -- likely one of the
     most predictive launch signals, so collapsing them is not OK.
     Bronze has no per-swap index (Dune `evt_index` /
     `outer_instruction_index` were never selected), so the two can only
     be separated by content.
     **Decision:** `swap_id = sha2(concat_ws('|', dex_venue, tx_hash,
     buy_token_id, sell_token_id, buy_amount, sell_amount), 256)`,
     computed on the two-sided Bronze swap before the per-side explode;
     dedupe on `swap_id`; Silver key becomes `(swap_id, token_id)`.
     sha2 not xxhash64 for the persisted key (64-bit collisions are
     non-negligible at 1e8+ rows; fine for the one-off count above).
     Accepted cost: two byte-identical swaps in one tx collapse --
     near-impossible on a bonding curve, since each buy moves price.
     Follow-up (not now): select `evt_index`/`outer_instruction_index`
     for future ingestion; not worth re-paying Dune credits to backfill.
  2. **`block_time` (Bronze) and `trade_time` (Silver) are strings in two
     formats** -- `YYYY-MM-DD HH:MM:SS.mmm UTC` (Dune rows written before
     `_normalize_block_time` existed) and `YYYY-MM-DDTHH:MM:SSZ`
     (everything since). String comparison sorts ' ' before 'T', so any
     range filter/sort/MIN/MAX on these columns is wrong on days mixing
     both. **Decision:** Silver parses both into a real `TimestampType`
     `trade_time`; Bronze is left as-is (not rewriting 131M rows).
  3. **The existing Silver notebook can't run at this scale** -- it
     `.collect()`s each Bronze chunk to the driver and transforms row by
     row in Python (same driver-heap pattern as the Bronze incidents,
     at ~3-5M rows/day for pumpdotfun). **Decision:** rewrite the
     transform as native Spark DataFrame ops (per-side select +
     `unionByName`), backfill Silver with one set-based partitioned
     overwrite, keep `MERGE` on `(swap_id, token_id)` only for live
     cadence. `to_silver_rows()` stays as the pure-Python reference
     implementation the Spark version is parity-tested against.
  Also noted: live Bitquery ingestion stops at 2026-09-15 04:15 with
  7,000 rows/venue -- expected, not a bug: the live job's schedule has
  been PAUSED since 2026-09-15 (`resources/lawrence_job.yml`).
  Implemented (2026-09-22): `src/features/silver_spark.py`
  (`build_silver`), `notebooks/01_silver_token_trades.py` rewritten with
  `MODE = "live" | "rebuild"` (live now windows on Bronze `ingested_at`,
  a real timestamp, instead of the string `block_time`; old
  `_catchup_progress` chunking removed -- `lawrence.silver._catchup_progress`
  is now unused and can be dropped). Downstream: `gold_features._parse`
  now accepts datetimes as well as strings, and notebooks 02/03 pin
  `spark.sql.session.timeZone = UTC` (applyInPandas hands naive
  Timestamps in session tz). Side effect: Gold's
  `pdf.sort_values("trade_time")` was sorting mixed-format strings
  lexically before this -- now chronological. Parity tests in
  `tests/test_silver_spark.py` (skipped where PySpark isn't installed,
  i.e. CI); not yet run -- PyPI unreachable from the sandbox. Existing
  gold/silver pure-Python tests pass.

- **Venue-mechanics research before finalizing Silver (2026-09-22)** --
  full write-up in `docs/venue_mechanics.md`. Headlines: (1) Bronze has
  launchpad *curve* trades (pumpdotfun, raydium_launchlab) but not the
  post-graduation venues (PumpSwap, Raydium CPMM), so graduated tokens --
  the winners -- vanish from our data at the moment they succeed
  (right-censoring; labels biased toward losers). (2) LaunchLab quote
  mints are per-pool and, since 2026-09-07, arbitrary (StonkFun pairs
  against SPYx/ZEC/PreStocks/other launched coins) -- the static quote
  denylist can't express that; token-of-interest should come from each
  pool's base mint. (3) raydium_v4 and uniswap_v3_base are not launch
  venues in 2026. Silver rebuild on hold until these are decided.

- **00e cost-test results, cells 1-2 (2026-09-26).** Cell 1 (1h, Sep 1
  00:00 UTC): pumpswap 548,986 trades/h, pumpdotfun 177,367, raydium v3
  (CLMM) 21,072, v4 11,657, v5 (CPMM) 7,566, raydium_launchlab 114.
  PumpSwap is ~3x the pump.fun curve (~400M trades/month) -- a
  graduates-only, bounded-horizon filter is mandatory, not an
  optimization. `raydium` v5 = CPMM confirmed from spellbook
  (`CPMMoo8L...`). LaunchLab's 114/h on Sep 1 vs ~5k/h Bronze average
  means most of our LaunchLab rows are post-2026-09-07 (any-pair /
  StonkFun). Cell 2 (pool metadata, full window) cost ~0.3 credits:
  PumpSwap pools created in window ~35k (33,124 SOL-quoted, 1,884 USDC,
  103 PUMP, then a long tail of stock/crypto/pump-token quotes);
  LaunchLab >=32k pools in the top 30 (platform, quote) groups alone --
  StonkFun dominates with dozens of quote mints (xStocks, PreStocks,
  ZEC, DOGE, PUMP, other launched coins), plus letsbonk.fun (SOL, USD1),
  Raydium's own UI, VOIDEX LaunchLab. Confirms the static quote denylist
  can't work for LaunchLab; per-pool base/quote metadata is cheap
  (tens of thousands of rows). ~35k PumpSwap pools/month vs ~4k/month
  expected pump.fun graduations (0.6%) => most PumpSwap pools are NOT
  graduates; cell 3's curve-mint join is the filter.
- **00e cell 3 (2026-09-26), ~4.1 credits:** PumpSwap pools created on
  2026-09-01: 1,235 traded, 1,209 of them on mints that traded on the
  pump.fun curve (so the curve-mint join barely filters -- nearly every
  new PumpSwap pool is for a curve token; correcting cell 2's note: most
  of the ~35k/month pools ARE curve-token pools, and graduations are far
  above the ~0.6% cited from the Sep-2025 study). Trades within 72h of
  pool creation: 11,223,337 for ONE day of pools -> ~336M rows/month,
  ~260k credits at ~780 credits/M rows. Raw trade-level pull ruled out.
  Next: cell 5 sizes per-pool OHLCV bars computed on Dune (1m/5m,
  24h/72h) instead of raw trades.
- **00e cell 5 (2026-09-26), ~1.8 credits:** same 1-day PumpSwap sample
  as bars instead of trades. 1m bars: 216,657 (24h) / 286,816 (72h);
  5m bars: 61,911 (24h) / 87,515 (72h). vs 11.2M raw trades -> 1m bars
  are ~39x fewer rows. Activity is front-loaded: 95% of 72h trades
  (10.6M) and 76% of 1m bars fall in the first 24h. Monthly estimates
  (x30, at 00c's ~780 credits/M rows exported): 1m/72h ~8.6M rows
  ~6.7k credits; 1m/24h ~6.5M ~5.1k; 5m/72h ~2.6M ~2.0k; 5m/24h ~1.9M
  ~1.5k; hybrid 1m for 0-24h + 5m for 24-72h ~7.3M ~5.7k. Plus
  ~2-4 credits scan per day queried. Choice depends on HOLD_MINUTES
  (still a placeholder in label_spec.md) and available credits.
- **00e cell 4 (2026-09-26), ~8.6 credits (inferred from balance
  delta):** LaunchLab mints with curve trades on 2026-08-31/09-01 that
  then traded on Raydium CPMM (v5) within 72h: 180 mints, 1,624 trades.
  Tiny -- but 2026-09-01 predates the 09-07 any-pair/StonkFun surge
  (LaunchLab curve volume is ~40x higher after it), so treat as a lower
  bound; even 40x is ~2M trades/month. Cost here is the scan (mint-IN
  join over all CPMM trades), not rows. Plan: same 1m-bar format as
  PumpSwap, so post-graduation data from both venues lands in one shape.

- **Post-graduation pull: 1-minute bars, 24h horizon (Jack, 2026-09-26).**
  Chosen from 00e cell 5 (~6.5M rows/month, ~5.1k credits est.) over
  72h / 5-min / hybrid. Implemented as `notebooks/00f_bronze_postgrad_bars_dune.py`
  + `src/ingestion/postgrad_bars.py` (pure SQL builders/mappers,
  `tests/test_postgrad_bars.py`). New Bronze tables:
  `lawrence.bronze.pool_metadata` (PumpSwap + LaunchLab pools: base/quote
  mint, platform, created_at) and `lawrence.bronze.postgrad_bars_1m`
  (per-pool OHLCV in USD of the BASE token, n_trades/n_buys/n_traders,
  n_unpriced). Bars are aggregated on Dune; trades with no USD price are
  counted in `n_unpriced`, not dropped silently. PumpSwap chunked 6h by
  pool creation time (120 chunks) with a progress table
  (`_postgrad_progress`) for key-swapping resumability; first run is
  capped at ONE PumpSwap chunk to check real cost vs estimate. CPMM
  (LaunchLab graduates) is one whole-window query -- small output, but
  its scan (mint-IN join over all CPMM trades) may cost tens of credits.
  Silver work still TODO: token-of-interest from pool_metadata base_mint
  (not the denylist) for LaunchLab, and folding postgrad bars into the
  per-token price series Gold reads.
- **00f first run (2026-09-29).** pools_pumpswap 109,296 rows (vs cell
  2's ~35k -- cell 2 was LIMIT 20 quote-mint groups and cell 3 only
  counted pools that TRADED; most created pools apparently never trade,
  to be confirmed by the checks cell), pools_launchlab 49,622,
  bars_pumpswap first 6h chunk 55,041 bars (00e estimate ~54k -- on
  target). bars_raydium_cpmm FAILED on DuneClient's 180s poll timeout,
  not a Dune error -- the execution kept running server-side. Fix:
  `POLL_TIMEOUT_S = 1800` and `RECOVER_EXECUTION_IDS` (fetch the already-
  paid execution's results before re-running). Credit delta: 2,245.986
  -> 2,526.873 = ~281 credits for the whole run (159k metadata rows +
  55k bars exported, plus the PumpSwap chunk scan and the timed-out CPMM
  scan, which still billed). Can't split per job from one delta; worst
  case (all attributed to rows) ~1,300 credits/M rows vs 00c's ~780.
  Remaining PumpSwap (119 chunks x ~55k bars ~6.5M rows): ~5.6k credits
  at 780/M + ~4/chunk scan, up to ~9k worst case -- i.e. several
  free-tier accounts. This account is now past the 2,500 free-tier
  mark -- confirm it isn't billing pay-as-you-go before running more.
- **00f runs 2-3 (2026-09-30, fresh Dune key).** Run with
  MAX_PUMPSWAP_CHUNKS=0: 0 credits. Run with =1: 0 -> 810.2 credits.
  Progress table now: pools_pumpswap success (109,296), pools_launchlab
  success (49,622), bars_raydium_cpmm failed x2 then success (418,593
  bars), bars_pumpswap success x2 (91,656 bars total; chunk 2 = 36,615).
  Which run did the CPMM fetch (recovered vs fresh) is not yet
  attributed -- needed before the 810 can be read as a per-chunk cost.
  Checks: pool_metadata pumpswap 109,296 pools / 35,283 distinct base
  mints (~3 pools per mint -- many duplicate/competing pools per token),
  launchlab 49,622 pools = 49,622 mints. Bars: pumpswap 503 pools,
  2.84M trades, 0 unpriced (SOL/USDC quotes, all priced). raydium_cpmm
  (LaunchLab graduates) 1,984 pools, 5.20M trades, **62% unpriced**
  (3.22M) -- Dune has no USD price for most StonkFun quote assets, so
  amount_usd is NULL. Same issue silently affects existing Bronze
  LaunchLab curve rows (Silver drops amount_usd-NULL rows).
- **Per-chunk PumpSwap cost measured (2026-09-30):** two chunks fetched
  alone (05:53 run) moved the key 810.2 -> 855.5 = ~22.7 credits/chunk
  (~43k bars each, ~530 credits/M rows). Remaining 116 chunks ~2.6k
  credits. The 810 before it was ~all the fresh whole-month CPMM query
  (~790: scan + 419k bars).
- **Keep trades Dune can't price in USD, via quote-unit prices (Jack,
  2026-09-30).** Labels are relative returns, so price in quote-token
  units (quote_amount / base_amount) is sufficient over a short hold and
  exists for every trade. Caveat: a volatile quote (another memecoin)
  mixes its own moves into the return -- to be flagged by quote type.
  Implemented:
  - Bars: `open_q/high_q/low_q/close_q/volume_quote` added to both bar
    SQLs + schema; existing bars table gets the columns via ALTER TABLE
    (the 4 PumpSwap chunks already pulled keep NULLs there -- fine, 0%
    USD-unpriced). CPMM re-pulled as job `bars_raydium_cpmm_v2`, which
    deletes the v1 `raydium_cpmm` rows first (~790 credits again).
  - Silver (`to_silver_rows` + `build_silver`): new `price_quote` and
    `quote_amount` columns; a row is dropped only if NEITHER price_usd
    nor price_quote is computable (previously any NULL amount_usd dropped
    it). Tests updated (reference + parity).
  - Gold notebooks 02/03 filter `price_usd IS NOT NULL` to keep current
    behaviour until Gold is designed to use price_quote. Open question:
    a token can trade against several quotes (e.g. SOL and USDC PumpSwap
    pools, ~3 PumpSwap pools per mint), so quote-unit prices must come
    from ONE quote per token -- likely its canonical/launch pool.
- **00f complete (2026-10-01).** pool_metadata: PumpSwap 109,296 pools /
  35,283 base mints; LaunchLab 49,622 pools = 49,622 mints.
  postgrad_bars_1m: PumpSwap 5,992,096 bars over 35,119 pools covering
  308.7M trades (3.72% USD-unpriced); CPMM v2 418,593 bars / 1,984
  pools / 5.2M trades, quote-unit price missing on only 0.21% of bars
  (vs 62% USD-unpriced) -- the quote-price fix works.
  Observations: (1) PumpSwap pools that actually traded (35,119) ~= the
  number of distinct base mints (35,283) -> ~1 traded pool per token;
  the other ~74k created pools never traded. Canonical-pool selection
  is nearly free: the pool with trades. (2) PumpSwap bars have NO
  quote-unit prices (bars_missing_quote_price = 1.0) even though the
  committed SQL produces them -- CONFIRMED from the final run's log:
  all 120 PumpSwap chunks were already 'success' (fetched by the
  pre-quote-price version of 00f) and were skipped; that run only did
  bars_raydium_cpmm_v2. Impact is limited to the 3.72% of PumpSwap
  trades without a USD price; re-pulling all PumpSwap (~2.7k credits)
  for that is not worth it now -- revisit only if those tokens matter. (3) Progress shows 1 `partial` PumpSwap chunk
  (30,000 rows) later re-fetched as success: success rows 5,962,096 +
  30,000 partial = 5,992,096 bars, i.e. ~30k duplicate bars in Bronze,
  removed by Silver's (venue, pool_id, bar_start) dedup.

- **Silver v2: pool-metadata base mints + one price series per token
  (2026-10-01, Jack approved).**
  1. `silver_spark.build_silver(bronze, pool_pairs(pool_metadata))`:
     LaunchLab token-of-interest = matching pool's base mint (pair-key
     equi-join, broadcast ~49k rows); denylist fallback when unmatched or
     ambiguous. Notebook 01 loads pool_metadata automatically and reports
     LaunchLab rows-per-swap (should be all 1).
  2. New `lawrence.silver.token_price_1m` (`src/features/silver_price_bars.py`,
     notebook `01b_silver_token_price_1m.py`): trade bars (curve/amm) +
     canonical post-grad bars (busiest pool per base mint; duplicate bars
     from the partial chunk removed by newest ingested_at). Per-token
     `price_unit` = usd if >=95% USD coverage else quote-units of the
     primary quote (launch pool quote, else most-traded quote); other-quote
     rows excluded, never mixed. This is the table Gold will move onto,
     replacing the price_usd IS NOT NULL stopgap.
  Tests: tests/test_silver_price_bars.py + a LaunchLab case in
  tests/test_silver_spark.py -- Spark-only, not run yet (no PyPI in the
  sandbox); first real run is in Databricks.
- **First Silver rebuild failed: Bronze amounts are STRING (2026-10-01).**
  `DATATYPE_MISMATCH.BINARY_OP_WRONG_TYPE` on `buy_amount_usd / buy_amount`.
  The real `bronze.dex_trades` stores buy/sell amounts, *_usd and fee
  columns as STRING (first writes were Bitquery, whose GraphQL decimals
  are JSON strings). The old Python path float()-ed everything, which hid
  it. Fix: `build_silver` try_casts those six columns to DOUBLE right after
  computing swap_id (hash still sees the stored text); added a parity test
  with an all-string Bronze schema.
- **"Fix didn't work" was a stale import (2026-10-01).** Re-running cells
  in the same Databricks session reuses the already-imported module, so
  the synced string-cast fix never executed. Notebooks 01/01b now
  `importlib.reload` our src modules on every run; otherwise restart
  Python after a `bundle sync`.
- **Silver rebuild succeeded (2026-10-01).** token_trades: pumpdotfun
  106,479,158 rows / 1,164,480 tokens; raydium_launchlab 3,625,514 /
  49,551; raydium_v4 3,465,186 / 48,278; uniswap_v3_base 9,409,517 /
  15,342. trade_time now a real timestamp (min/max chronological; v4 and
  Base run to 2026-09-15 04:15 = the paused live Bitquery rows).
  pumpdotfun ~= Bronze's 106.5M distinct swaps (the ~1.6M partial-retry
  dupes are gone). LaunchLab: every swap -> exactly 1 row (3,625,514 x 1),
  i.e. pool-metadata base mint resolved them all; previously most rows
  were split in two or dropped for missing USD. Quote check now shows
  non-denylist LaunchLab quotes (ZEC, SPYx, OpenAI PreStock, ...) as
  expected -- they are quotes, correctly. uniswap_v3_base grew 7.28M ->
  9.41M (multi-swap txs and USD-unpriced rows no longer dropped).
  (Graduation-rate note corrected by 01b below: ~1.5%, not ~3%.)
- **01b first run (2026-10-01): token_price_1m built.** By unit/phase
  (bars / tokens / trades): usd curve 7.71M / 1,175,670 / 103.6M; usd
  postgrad 4.64M / 30,546 / 251.2M; usd amm 2.59M / 61,214 / 10.2M; quote
  curve 393,738 / 25,924 / 5.80M; quote postgrad 246,601 / 651 / 3.17M;
  quote amm 1.05M / 2,405 / 1.49M. Tokens by phase: curve+postgrad
  18,626 (graduates we can now follow), curve-only 1,182,968, postgrad-
  only 12,571, amm-only 63,614. Graduation ~= 18,626 / ~1.2M curve tokens
  ~= 1.5% (understated: late-window launches had no time to graduate);
  the earlier "~3%" estimate (35k PumpSwap pools / 1.16M tokens) was
  wrong. Issues: (1) 306,057 bars with close <= 0 -- Dune amount_usd = 0
  on dust trades; fixed in silver_price_bars (price <= 0 -> missing; OHLC
  fall back to close), re-run 01b only. (2) 12,571 postgrad-only tokens --
  either launched before 2026-08-14 (window edge) or non-pump.fun tokens
  with PumpSwap pools; diagnostic query (pump-suffix / first-3-days) pending.
- **01b re-run (2026-10-01):** nonpositive_close = 0 after the fix.
  Postgrad-only tokens = 13,950 (up from 12,571: tokens whose curve
  trades were all zero-priced lost their curve bars). Breakdown: 12,721
  pump-suffix (11,579 not in the first 3 days and WITH trades in
  token_trades -> most likely older pre-2025 pump.fun graduates trading
  on raydium_v4 plus a newer PumpSwap pool), ~1.2k non-pump mints.
  **Gold rule (decided): entry candidates = tokens with a `curve` phase in
  token_price_1m** (we saw them launch). Postgrad-only and amm-only tokens
  are excluded from the entry model whatever the cause.
- **Gold v2: built on token_price_1m (2026-10-01).** `src/features/gold_input.py`
  (candidates = curve-phase tokens; bar records with trade_time =
  bar_start + 1min; per-token observed_until and launch_time) feeding
  `gold_features.py`, which now accepts bars (n_trades/n_buys, `price`)
  as well as raw trades, skips censored labels (T + hold past
  observed_until) instead of rug-clamping them, treats pandas NaN as
  missing, and adds minutes_since_launch / is_postgrad_at_t. Notebooks
  02/03 rewired (and force-reload modules); schemas gain price_unit,
  minutes_since_launch, is_postgrad_at_t. Key correction this fixes: v1
  labelled every graduated token's post-graduation window as a rug (-100%)
  because its trades stopped at graduation. 29 gold_features tests pass.
  Model feature list unchanged for now -- adding the new columns is a
  separate, measured step.
- **Gold v2 built (2026-10-01).** entry_candidates 4,868,739 rows /
  1,145,637 tokens (of 1,147,025 candidates); exit_candidates 7,148,227
  rows / 413,115 tokens (exit needs a price at T - h with h >= 15 min, so
  tokens that live < ~15 min produce none -- expected). Entry labels by
  phase/unit: pre-grad usd 4.27M rows, rug_clamped 74.7%, median y -1,
  capped mean -0.75; pre-grad quote 186k, 62.0%, -1, -0.55; post-grad usd
  373k, 37.7%, -0.19, -0.36; post-grad quote 41k, 11.8%, -0.05, -0.09.
  **Problem found: the rug-clamp proxy is wrong for these venues.** "No
  trade near T+hold" is labelled -100% ("liquidity pulled"), but a
  bonding curve has no LP to pull (SOL sits in the curve program; you can
  always sell back at the curve price), and PumpSwap / LaunchLab
  graduation pools have their LP burned or locked at migration. A quiet
  token's realizable price is its LAST price, not zero. So ~3/4 of
  pre-graduation labels are artificially -100%. Proposed fix pending
  Jack's call (see next entry).
- **Label fixes + time-scale changes (Jack approved, 2026-10-01).** See
  label_spec.md "2026-10-01 changes". gold_features: NO_RUG_PHASES =
  {curve, postgrad} -> exit price = last price (no staleness) instead of
  RUG_CLAMP_RETURN; HOLD_MINUTES 60 -> 15; LABEL_HOLD_MINUTES = (5, 15, 60)
  stored as y_entry_{h}m / y_exit_{h}m (NULL when censored); dense T
  sampling (1 min for the first 60 min after launch, then 15 min). Jack
  asked whether tuning PRICE_STALENESS would do instead: no -- any finite
  staleness limit is what converts "went quiet" into -100%; only removing
  it for non-rug phases is correct. 33 gold_features tests pass.
  Notebook 04 now samples a deterministic subset of TOKENS (xxhash64) when
  entry_candidates exceeds MAX_TRAIN_ROWS = 8M, since dense sampling can
  make Gold 5-10x larger than the driver can toPandas().
- **Gold v2 run 2 + activity filter (2026-10-02).** With the rug-clamp fix:
  entry 21,550,424 rows / 1,146,630 tokens, exit 20,296,495 / 413,307,
  rug_clamped_share 0, censored_60m 0.05% (pre-grad) / 0.57% (post-grad).
  15m label by token age at T (pre-grad): 0-5m 3.32M rows, 34.5% exactly
  zero, p10 -43.8%, p90 +0.1%, capped mean -3.63%; 5-15m 4.62M, 52.8%
  zero, mean -0.93%; 15-60m 10.21M, 68.2% zero, mean +0.09%; 60m+ 2.43M,
  70.8% zero, mean +0.19%. Reading: first-5-minute entries lose ~3.6% over
  15 min on average BEFORE fees, with winners only in the top few percent
  -- the base rate to beat; and ~half of Gold was idle samples. Added
  ACTIVE_WINDOW = 2 min (Jack approved): v2 sampling keeps T only if the
  token traded within 2 min before T. 35 gold_features tests pass.
- **Gold v2 run 3, with ACTIVE_WINDOW (2026-10-02).** entry 7,523,114 rows /
  1,146,630 tokens (all tokens kept; ~14M idle samples removed); exit
  4,232,595 / 234,701 tokens (exit needs a token still ACTIVE >= 15 min
  after a prior price, so short-lived tokens drop out -- expected). 15m
  label by age (pre-grad): 0-5m 3.18M rows, 33.4% zero, p10 -46.4%, p90
  +0.1%, mean -3.80%; 5-15m 1.71M, 29.8% zero, p90 +0.7%, mean -2.45%;
  15-60m 1.67M, 27.9% zero, p90 +5.7%, mean -0.90%; 60m+ 478k, 43.7%
  zero, p90 +2.7%, mean +0.41%. Under notebook 04's 8M cap -> trains on
  all rows. Open issue before 04: backtest cost model is 30 bps fee +
  50 bps slippage per side (AMM-style), but pump.fun's bonding curve
  charges ~1.25% per trade -- curve-phase round trips are understated ~2x.
- **Per-row trading costs by venue + phase (Jack approved, 2026-10-02).**
  Verified fees: pump.fun curve 1.25% per trade (0.30% creator + 0.95%
  protocol, pump.fun/docs/fees); PumpSwap canonical pools are tiered by
  market cap -- 1.25% under 420 SOL, 1.20% at 420-1,470 SOL, down to 0.30%
  only above ~98k SOL -- so a freshly graduated token pays ~1.2%, NOT the
  ~25-30 bps I first assumed; LaunchLab-UI curve 1% base fee (Raydium),
  with third-party platform fees on top not modelled (a floor); CPMM 25 bps
  standard tier (placeholder). `walk_forward.VENUE_PHASE_FEE_BPS` +
  `venue_fee_bps(dex_venue, is_postgrad_at_t)`; `round_trip_cost` and
  `simulate_trades` accept a per-row fee Series. Notebook 04 uses the same
  per-row fees for the model's entry rule (pred > row's round-trip cost)
  and for scoring every strategy. Slippage stays 50 bps/side (placeholder;
  thin curves are likely worse). Round trip on the pump.fun curve is now
  2 x (125 + 50) = 3.5% vs the old 1.6%.
- **First entry-model result on Gold v2 + real costs (2026-10-02).** Per
  fold (EV/trade after per-row costs; win rate; trades): model +5.0% /
  9.4% / 22,269; +9.5% / 10.5% / 35,129; +4.7% / 9.3% / 32,044; +4.5% /
  18.2% / 69,251; +4.1% / 22.1% / 90,831. Baseline A (buy everything)
  -5.4% to -6.4%; Baseline B (volume/momentum) -7.8% to -9.0% -- worse
  than A. First time the model beats both baselines in every fold, and
  the first positive EV after costs. NOT yet trusted: low win rate +
  positive EV = right-tail driven; per-minute sampling lets one token be
  entered many times; returns are capped at 100x; and the entry price is
  the last completed bar's close (up to 2 min stale), which flatters
  momentum entries. Added stress tests to notebook 04 (caps 10x/2x,
  ex-top-1%, first signal per token, EV by trailing-volume quartile).
- **Stress tests: the model's positive EV does NOT survive (2026-10-02).**
  Per fold: EV at a 10x cap -1.1% to -6.6% (all negative); at 2x -6.4% to
  -13.3%; median net trade -3.6% to -5.4%; EV without the top 1% of trades
  -8.2% to -14.6%; the top 1% of trades produce 251-408% of total profit
  (everything else loses money); first signal per token at 10x cap -7.1% to
  -15.8%. By trailing-volume quartile (pooled, 10x cap): only Q1 is
  positive (+0.9%, 28% win rate) and its median trailing volume is $0 --
  tokens with no USD volume (most likely quote-unit / StonkFun tokens,
  whose platform fees aren't modelled); Q2-Q4 (median $276-$3,631) are
  -3.6% to -5.2%. **Conclusion: no tradeable edge yet.** The headline
  +4-9.5% EV is a handful of near-100x outcomes on thin tokens, counted
  repeatedly via per-minute sampling -- not capturable at any real size.
  Recorded as-is per the project's honest-reporting rule.
- **Train and score at a 10x cap; measure ranking signal (Jack approved,
  2026-10-02).** New `gold_features.CAPTURABLE_RETURN_CAP = 10.0`
  (placeholder for "most a real position could capture" until liquidity-
  aware fills exist); `winsorize_return(y, cap)`; `prepare_features` trains
  on y capped at 10x; `simulate_trades(..., return_cap=10x)` scores every
  strategy -- model AND both baselines -- at the same cap, so the
  comparison is like-for-like. RETURN_WINSORIZE_CAP (100x) remains only as
  the outer data-sanity bound. Notebook 04 adds (a) a threshold-free
  strategy, `lightgbm_top1pct` (top 1% of scores per fold), and (b) a
  ranking table per fold: Spearman(pred, y) and mean net return (10x cap,
  after row costs) by score decile. Tests: gold_features 35, walk_forward
  12, baselines 3 pass; entry_model tests need lightgbm (CI only).
- **Planned exit rule: "recover the initial at 2x" (Jack, 2026-10-02) --
  deferred to the exit/decision layer, not the entry model.** Rule: when
  a position doubles, sell half (recovering the initial stake) and leave
  the rest for the exit model to manage. Decision: don't change entry
  labels for it now. It's a position-management policy, and its value is
  path-dependent (did the price touch 2x before it fell?), so it belongs
  in (a) a path-dependent backtest simulator over silver.token_price_1m
  and (b) as a rule-based exit baseline the exit model must beat. Notes
  for when it's built: a partial take-profit doesn't raise EV by itself
  -- it trades some upside for lower variance; it costs an extra sell fee
  (~1.2-1.25% on curve/fresh PumpSwap); selling half into a thin curve
  has its own slippage; and on current data 2x within the hold is rare
  (first-5-minute p90 over 15 min is ~+0.1%), so it would trigger on few
  trades. Revisit once the entry model shows ranking signal.
- **04 at a 10x cap (2026-10-02): real ranking signal, ~breakeven after
  costs.** Spearman(pred, y) 0.30 / 0.31 / 0.31 / 0.28 / 0.21 across folds;
  net return by score decile is monotone in every fold (d1 ~ -22% ->
  d10 -1.1% to -3.6%) -- the model orders trades far better than chance,
  but even its top decile loses after costs. Baselines at 10x: A -7.2% to
  -8.3%, B -8.9% to -10.3%. Model (threshold) EV +0.6 / -1.2 / -3.6 /
  +3.3 / +0.5%; top-1% variant +0.7 / -1.1 / -4.3 / +1.6 / +5.0%. So the
  model beats both baselines by ~7-9 points in every fold but has no
  reliable absolute edge. Still tail-dependent: median net -3.5% to -5.0%,
  negative ex-top-1%, top 1% = 147-321% of profit, negative at a 2x cap.
  Red flag: by trailing-volume quartile, Q1 AND Q2 have median trailing
  USD volume $0 -- about half the model's trades are on tokens with no
  USD volume (likely quote-unit / StonkFun tokens, whose platform fees
  aren't modelled and whose returns include the quote asset's moves). Q3
  (median $774) +2.0%, Q4 ($5.4k) -0.7%. Later folds score better
  (more training history), consistent with the earlier data-starvation read.
- **Segment breakdown added to notebook 04 (2026-10-02).** Model trades
  split by launch venue x price_unit (trades, unique tokens, EV and median
  at 10x cap, win rate, zero-USD-volume share, post-grad share, share of
  all trades) next to each segment's buy-everything EV, plus per-fold EV
  by segment. Purpose: find out whether the ~breakeven result comes from
  the trustworthy segment (pumpdotfun / usd) or from quote-unit tokens
  whose fees and prices we can't fully trust.
- **Segment breakdown result (2026-10-02).** Model trades by launch venue x
  price_unit (EV at 10x cap after costs vs the segment's buy-everything EV):
  - pumpdotfun / usd -- the trusted segment: 49,420 trades (41%), 8,271
    tokens, EV +0.44% vs base -8.91%, median -3.5%, win 12%. Per fold
    +0.6 / -1.6 / -4.2 / +9.1 / -0.1%: ~breakeven on average, inconsistent.
  - raydium_launchlab / quote: 60,212 trades (50%), 4,615 tokens, ~100%
    zero USD volume, EV +0.61% vs base -2.11%; per fold -3.0 / -8.8 / -14.0
    / +1.0 / +0.5%. Platform fees unmodelled -> true EV lower. Untrusted.
  - raydium_launchlab / usd: 8,997 trades, 1,472 tokens, EV +1.97% vs base
    -3.56%; per fold -3.3 / +7.9 / +8.6 / +1.7 / +1.6%. Most consistent, but
    small and the LaunchLab fee is a floor.
  - pumpdotfun / quote: 1,784 trades; base EV **+59.5%** over 67,679 rows --
    impossible for buy-everything; almost certainly a data bug in how
    quote-unit prices are built for pump.fun tokens that fall under the 95%
    USD-coverage line. Investigate before trusting any quote-unit series.
  Reading: the model's ranking adds ~9 points over buying everything in
  the trusted segment, landing around zero -- no robust edge yet. Half its
  trades sit in a segment we can't measure reliably.
- **Root causes of the pump.fun/quote "+59%" anomaly (2026-10-02).**
  Query A: those 5,337 tokens' 3.92M curve trades are fully USD-priced
  (price_usd NULL share 0) and all quoted in WSOL. Query B: worst series
  jump from ~3e-8 SOL/token (normal: a pump.fun curve spans ~2.8e-8 at
  launch to ~4e-7 at graduation) to EXACTLY 0.001 or 0.000001, or drop to
  ~1e-11. Two bugs:
  1. **Dust trades set absurd prices.** Round values like 0.001 = 1
     lamport / 1 raw token unit (1e-9 SOL / 1e-6 token): when a trade is a
     few raw units, rounding makes amount ratios meaningless, and one such
     trade in a minute sets that bar's high/low/close. This is not
     specific to quote mode -- USD prices are the same ratio x SOL price,
     so USD series can carry the same spikes, and they may explain part of
     the "near-100x lottery winners" the model learned to chase.
  2. **Graduated pump.fun tokens were pushed into quote mode.** Coverage
     counts post-graduation bars; if the token's PumpSwap pool has
     enough USD-unpriced trades, coverage < 95% -> price_unit = quote ->
     its PumpSwap bars are then DROPPED (that pull has no quote-unit
     prices) -> the token "vanishes at graduation" again. Explains
     phases = curve only in Query B.
- **Fixes for both bugs (Jack approved threshold, 2026-10-03).** Dust
  sizing: pump.fun curve trades < $1 21.4%, < $0.10 8.9%, < $0.01 6.2%
  (p05 $0.000135, median $7.44); LaunchLab < $0.01 1.7% (median $33.70).
  $1 would drop real small trades; $0.01 (~67k lamports, rounding error
  < 0.01%) only drops the broken ones. In silver_price_bars:
  `MIN_TRADE_USD = 0.01`; pump.fun curve trades quoted in WSOL must have
  price_quote within `PUMP_CURVE_PRICE_SOL = (2.0e-8, 6.0e-7)` (padded
  physical curve range); `OUTLIER_BAND = 1000` per-token median band on
  trade-bar prices and on post-grad bar closes (out-of-band open/high/low
  -> missing, fall back to close). Unit rule: `USD_QUOTES` (the quote
  denylist majors) -> always usd; coverage >= 95% still also qualifies.
  3 new Spark tests. Rebuild needed: 01b -> 02 -> 03 -> 04 (token_trades
  unchanged).
- **Pump.fun curve price range disabled (2026-10-03).** Check before 02:
  WSOL-quoted pump.fun curve trades -- not graduated 90.4M trades, dust
  6.3%, below 2e-8 SOL 15.4%, above 6e-7 0.2%, p01 1.4e-10, p50 4.9e-8,
  p99 3.0e-7; graduated 16.1M, dust 6.0%, below 25.5%, above 0.1%, p01
  6.2e-11, p50 1.3e-7, p99 4.0e-7. A quarter of graduated tokens' trades
  sit 100-1000x below the assumed launch price -- far too many to be dust,
  so the "physical curve range" assumption is wrong for a real slice of
  tokens (different curve parameters/decimals, or Dune unit differences).
  `PUMP_CURVE_PRICE_SOL = None`; dust ($0.01) and the 1000x per-token
  band remain. Open question logged: what are those low-priced trades?
  01b check #2 with the range ON had shown curve-only 1,047,405 (was
  1,182,968), postgrad-only 16,382 (was 13,950) -- graduates losing all
  curve trades, which is how this surfaced.
- **01b after the dust fixes, explained (2026-10-03).** Tokens by phase:
  curve+postgrad 20,621 (was 18,626), curve-only 1,061,515, postgrad-only
  16,238, amm-only 44,567. Breakdown of pump.fun tokens with trades:
  "no bars at all" 115,501 -- ALL dust-only (median 2 trades): spam, correctly
  removed. "has curve" 1,033,353 -- none dust-only. "postgrad only" 15,626 --
  only 1,961 dust-only, median 1 curve trade in the window: tokens launched
  before 2026-08-14 that graduated in-window (we see only their final curve
  buy, which shares the graduation minute with the first post-grad bar and
  loses the same-minute tie). Previously some counted as curve+postgrad with
  a fake launch_time; excluding them from entry candidates is correct.
- **04 after dust fixes (2026-10-03).** Ranking signal unchanged:
  Spearman 0.29 / 0.30 / 0.31 / 0.28 / 0.21, deciles monotone (d1 ~ -22%
  -> d10 -1.1% to -3.7%; top decile still negative after costs).
  Baselines at 10x: A -7.2% to -8.3%, B -8.8% to -10.1%. Model EV +0.1 /
  -4.7 / -4.4 / +2.7 / +1.3%; top-1% +1.3 / -4.6 / -3.6 / +2.3 / +5.3%.
  Stress: top 1% = 169-289% of profit, median -3.5% to -5.1%, ex-top-1%
  -4.5% to -12.5%. pump.fun/quote segment gone (unit fix worked).
  Segments: **pumpdotfun/usd (trusted) EV -1.26% vs base -8.14%** (51,328
  trades, 8,666 tokens; per fold +0.2/-4.8/-4.9/+4.7/+0.1) -- the
  earlier +0.44% was partly dust; the model lifts ~7 points but is
  negative in absolute terms. raydium_launchlab/quote +1.36% (untrusted,
  platform fees unmodelled). **raydium_launchlab/usd +3.98% vs base
  -3.61%** (6,252 trades, 1,120 tokens; per fold -2.1/+9.0/+7.7/+7.3/
  +3.6) -- the only segment positive in 4 of 5 folds, but small (one
  month, ~1.1k tokens) and LaunchLab fees are a floor: third-party
  platforms (StonkFun, letsbonk.fun, VOIDEX) add their own fees on top
  of Raydium's 1%. Conclusion: no edge in the trusted pump.fun segment; a
  possible edge in SOL-quoted LaunchLab that can't be trusted until
  per-platform fees are verified and modelled.
- **LaunchLab platform fees verified and modelled (2026-10-03).** On-chain
  LaunchLab curve cost per side = protocol `trade_fee_rate` (GlobalConfig,
  2500 millionths = 0.25%, confirmed via Dune create_config) + the launch's
  PlatformConfig `fee_rate` + `creator_fee_rate` (millionths; caps 5% and
  0.5%). The 182 configs (Dune, saved as
  `data/reference/launchlab_platform_configs.csv`): StonkFun 121 of 131 at
  1.0% + 0.5% creator; letsbonk.fun mostly 1.0% + 0-0.5%; Raydium 0.75%
  early, 1.0% since Oct 2025; VOIDEX 0.01% + 0.01%. So a typical LaunchLab
  curve side costs 1.25-1.75%, not the 1.0% modelled before -- the
  launchlab segment EVs above were overstated by ~0.5-1.5 points per round
  trip. Changes: 00f job `pools_launchlab_v2` re-pulls LaunchLab pools with
  `platform_config` (replaces the old rows); notebook 04 joins token ->
  config -> fee and passes it to `venue_fee_bps` (curve rows only; CPMM
  post-grad stays 0.25%); fallback for unknown configs raised 1.00% ->
  1.25%; segment tables now split by platform_name. Expected
  launchlab/usd EV after this: roughly +2.5% to +3.5% (estimate, pending
  the re-run).
- **04 with per-platform LaunchLab fees (2026-10-03).** 98.4% of LaunchLab
  rows matched a config. Actual mean curve fee by rows: StonkFun 125 bps
  (most volume is on its 1.0% + 0% creator config, despite most *configs*
  being 1.0% + 0.5%), letsbonk.fun 125, Raydium 100 (old 0.75% config),
  VOIDEX 27. Unmatched (Pairz, WTFO, tren.ch, GoBolt, Stocks Inu...) use the
  1.25% fallback -- their configs weren't in the 182-row pull. pump.fun
  results identical (fee change only touches LaunchLab rows).
  raydium_launchlab/usd still ~+4% EV over ~6.1k trades (vs base ~-4%),
  but it is NOT one consistent edge: StonkFun/usd (4,429 trades, 742
  tokens) is -34% / +8.9% / +3.2% in folds 2-4 (it only exists from
  Sept), median trade -6%, win rate 31% -- tail-driven; the earlier
  fold 0/1 positives came from Raydium/letsbonk.fun, different platforms.
  Every other platform has too few trades to read. StonkFun/quote (xStocks
  etc.; quote moves little in 15 min, so roughly USD-like) +1.3% over 51.6k
  trades vs base -2.6%, folds -11.9 / +1.2 / +1.4%. Conclusion: the fee
  correction didn't erase the LaunchLab result, but it rests on ~3 weeks
  of one platform and on tails. Next: stress-test the LaunchLab segments
  alone (ex-top-1%, first signal per token) before trusting any of it.
- **Per-segment stress tests: the LaunchLab "edge" is a few tokens
  (2026-10-03).** 10x cap, pooled over folds:
  launchlab/usd EV +4.1% but ex-top-1% -4.0%, top 1% of trades = 195% of
  profit, per-token EV -7.0%, only 19% of tokens profitable, ONE token =
  41% of profit (57% within StonkFun/usd, 68% within non-StonkFun/usd).
  ex-top-1% is negative in every fold for every segment. StonkFun/quote:
  +1.3% -> ex-top-1% -4.2%, per-token -17.8%, 13% of tokens profitable.
  pump.fun/usd: -1.3% -> ex-top-1% -9.6%, per-token -2.7%.
  first-per-token stays positive pooled for launchlab/usd (+4.1%) but
  fold 1's +42% is 62 trades; folds 3-4 (where the volume is) +1.5% /
  +3.3%. Verdict: no segment shows an edge we can distinguish from luck
  on one month of data. Dropping the top 1% is harsh for any memecoin
  strategy (returns ARE tail-driven), but one token carrying 40-70% of a
  segment's profit is not a repeatable result. The model ranks
  (Spearman 0.2-0.3) but its best picks don't clear ~3% round-trip costs
  at a fixed 15-minute hold.
- **Cost estimate for more months of data (2026-10-03).** Jack agreed the
  ex-top-1% test is harsh for a tail-driven strategy; the open question
  is whether winners repeat, which needs more months. Estimates (rates
  from measured runs: 00c ~780 credits/M trade rows, 00f PumpSwap ~22.7
  credits/6h chunk, CPMM ~790/month): repeating the trade-level 00c pull
  for pump.fun + LaunchLab is ~75-90k credits per month (~30-36 free
  accounts, or ~$1.3k on Dune Plus), almost all of it pump.fun's ~108M
  curve trades. But nothing downstream uses trade rows: Gold reads only
  token_price_1m (1-minute bars), and Bronze has no trader_id anyway. So
  the proposed route is to aggregate curve trades into 1-minute bars on
  Dune (same pattern as 00f), which can also add per-bar trader stats
  (n_traders, new wallets, top-buyer share) that trade-level Bronze never
  had. Cost unknown until a 1-day cost test (~$1). Preferred window:
  forward from 2026-09-13 (current LaunchLab regime: any-quote pools since
  09-07, StonkFun), not backward.
- **00g curve-bars cost test written (2026-10-03).** New
  `src/ingestion/curve_bars.py`: Dune SQL that turns pump.fun and LaunchLab
  curve trades into 1m bars (same OHLCV + quote-unit columns as
  postgrad_bars) plus trader stats per bar: volume_buy_q, volume_sell_q,
  max_trade_q, top_trader_share (largest wallet's share of the bar's
  volume). Dust (<$0.01) is dropped on Dune since bars can't be
  un-aggregated. pump.fun: pool_id = base mint (one curve per mint), quote
  = WSOL. LaunchLab: base/quote from stg_swaps, joined on both mints (not
  on project_program_id). Chunked on minute boundaries so every stat is
  chunk-safe; "new wallet" counts are NOT included because they'd need
  history across chunks. Notebook 00g: (1) trades/day since 09-13,
  (2) pump.fun 1-day bars summary-only (compute cost), (3) LaunchLab
  1-day summary, (4) pump.fun 1-hour full export (export cost),
  (5) validation hour on 09-01 vs Silver token_price_1m. Not yet run.
- **00g results (2026-10-03): curve bars on Dune validated and cheap.**
  Dune has full days through 2026-10-02 (20-day forward window from
  09-13): pump.fun 74.5M curve trades (3.7M/day, same as before),
  LaunchLab 11.5M (573k/day, ~5x the Aug-Sep month). Credits: pump.fun
  1-day bars summary-only 5.8 (189,798 bars, 41,725 tokens, 2.62M
  non-dust trades); LaunchLab 1-day 21.8 (81,443 bars, 11,111 tokens,
  731k trades, 37% without USD price -- quote prices cover them); 1 hour
  of pump.fun bars exported 8.6 for 7,073 bars (~1,200 credits/M exported
  bars -- export, not compute, is the cost). Validation hour 2026-09-01
  12:00 vs Silver token_price_1m curve bars: 9,077 vs 9,040 bars, 2,753
  vs 2,731 tokens, 140,733 vs 139,476 trades (all within 1%; Silver keeps
  one pool per token and drops out-of-band prices). Estimated full
  20-day pull: pump.fun ~4.7k + LaunchLab ~2.4k + post-grad (PumpSwap
  ~1.8k, CPMM ~1.5-3k) = ~10.5-12k credits (~$525-600 pay-as-you-go,
  inside one month of Plus, ~5 free-tier allowances), vs ~67k for raw
  trades.
- **Forward window 2026-09-13 to 2026-10-03 wired through (2026-10-04).**
  - New `notebooks/00h_bronze_curve_bars_dune.py` -> `bronze.curve_bars_1m`
    (+ `_curve_bars_progress`): 100 chunks (pump.fun 6h, LaunchLab daily,
    interleaved by day), same resumable/402-salvage rules as 00f,
    `MAX_CHUNKS = 2` for the first run.
  - 00f now loops over `WINDOWS` (window 1 = old, all done; window 2 =
    forward). Fixed a trap: `REPLACES_VENUE` deletes now apply to window 1
    only -- run on window 2 they would have wiped all of window 1's
    LaunchLab pools / CPMM bars. A LaunchLab pool trading in both windows
    gets two pool rows; a new cell dedupes pool_metadata to one row per
    (venue, pool_id) (earliest created_at). CPMM pools graduated late in
    window 1 get a late anchor in window 2, so some bars repeat; Silver's
    dedupe_bars keeps one.
  - Silver `token_price_1m`: `build_token_price_1m` takes optional curve
    bars (phase 'curve', same unit/outlier rules as post-grad bars) and
    gains flow columns volume_buy_q, volume_sell_q, max_trade_q,
    top_trader_share. Trade-built bars fill all but top_trader_share;
    post-grad bars have none yet. Caveat for features: these are null
    for most of window 1, so any feature built on them must not let the
    model learn "null = older data".
  - 02/03 `CURVE_DATA_END` -> 2026-10-03.
  Run order: 00f (window 2) -> 00h -> 01b -> 02 -> 03 -> 04.
- **00f window-2 CPMM job ran a fresh 2,500-credit key dry (2026-10-04).**
  0 rows; logged `partial`. My estimate (0.6-1.5k) scaled window 1's 790
  credits by days and LaunchLab volume, but window 1's query was itself
  inefficient: `SELECT t.*` over the wide `dex_solana.trades` and a join
  on `m.mint IN (bought, sold)`, which can't be a hash join. Window 2 has
  ~5x more LaunchLab mints, so the join cost grew much faster than the
  volume. Suspected cause, not confirmed. Rewrote the CPMM query: narrow
  column list, two equi-joins via UNION ALL (same rows). Same OR-join fix
  for launchlab_curve_bars_sql (00g cell 3 cost 3.8x pump.fun per day).
  Next: 00g cell 6 measures the CPMM query on one day before re-running.
- **Window-2 CPMM split into per-day steps (2026-10-04).** The rewritten
  whole-window CPMM query (narrow columns, equi-joins; 12 credits for a
  1-day test) still failed on the 20-day window with Dune's
  "exceeded the user defined maximum amount of resources" after ~14 min,
  and the key was then empty (the next PumpSwap chunk got a 402). Lesson:
  a 1-day cost does not scale linearly to one big query on the free tier
  -- big queries hit the per-query resource cap and still bill. Rule from
  now on: every Dune pull is chunked to <= 1 day. Window 2 CPMM is now:
  (1) `cpmm_first_trades` x20 days -> `bronze._cpmm_first_trades` (each
  LaunchLab mint's CPMM pools + first trade that day; mints from a 7-day
  stg_swaps lookback); (2) `bars_cpmm_v3` x20 graduation days: graduation
  = earliest first trade across all days, mints with window-1 CPMM bars
  skipped, bars pulled for that day's pools via a VALUES list over
  [graduation, +24h). Step 2 refuses to run until all step-1 days are
  done. `MAX_CPMM_BAR_DAYS = 1` for the first run. Window 1 is untouched.
- **Split CPMM run 1 (2026-10-04): 131 credits** for all 20
  `cpmm_first_trades` days (645-1,431 pools/day, 10-25 s each) + the first
  `bars_cpmm_v3` day (2026-09-13: 39,115 bars -- the heaviest day, it
  absorbs pools that graduated before the window but have no window-1
  bars). Can't split the 131 per step from one delta; ~1,200 credits/M
  exported bars puts the first bars day at ~50. Remaining 19 bars days
  estimated ~400-950 (upper bound = 19 x day 1).
- **Window-2 CPMM done (2026-10-04).** Run 2: 743 credits for the other
  19 `bars_cpmm_v3` days (838,580 bars; 14k-85k/day, busiest 09-17 to
  09-24) + 1 PumpSwap chunk (46,150 bars) = ~840 credits/M exported rows.
  Window-2 CPMM total: 877,695 bars for 20 days vs 418,593 for window 1's
  30 days (~3x per day, consistent with LaunchLab's ~5x curve volume).
  Whole split CPMM pull: 874 credits (vs one failed whole-window query
  that burned a full 2,500-credit key). Remaining estimates at 840-1,200
  credits/M rows: PumpSwap 79 chunks x ~46k bars ~2-3k; 00h ~5.4M bars
  ~5-7k. Total remaining ~7-10k.
- **00f window-2 PumpSwap run 1 (2026-10-05):** 53 chunks done
  (2026-09-13 06:00 to 09-26 12:00), ~2.10M bars (22k-74k per 6h chunk,
  avg ~40k) on the key's remaining ~1,600 credits -> ~770 credits/M rows.
  Chunk 2026-09-26 12:00-18:00 logged `partial` (10k rows salvaged; it is
  re-fetched whole next run, Silver dedupes). Remaining: 26 chunks, ~1M
  bars, ~800 credits.
- **00h first run (2026-10-06):** 59.7 credits for the first two pump.fun
  6h chunks of 2026-09-13 (56,911 bars, 14,438 tokens, 685k trades, 0%
  unpriced) -> ~1,050 credits/M bars, ~30/chunk on a light day (09-13 had
  the fewest trades, 2.5M). Est. ~200 credits per average pump.fun day +
  ~100 per LaunchLab day (unmeasured) -> ~5-6k for all 20 days.
  (MAX_CHUNKS=2 covered two pump.fun chunks, not one per venue as the
  notebook said -- comment fixed.)
- **00h complete and reconciled (2026-10-07).** 100/100 chunks success
  (80 pump.fun, 20 LaunchLab; 3 partial salvages = 40k duplicate bars,
  Silver dedupes). Sept 13-14 looked low (70-73% of raw trades in bars vs
  86-92% other days). Dune dust count explains it exactly: dust (<$0.01)
  was 30.4% / 26.9% of pump.fun trades on 09-13/14 vs 9-14% on 09-15..20,
  and raw minus dust equals the bars' trade count to the trade on every
  day 09-13..09-19. 09-20 is +70k (2.7%) over -- duplicate bars from a
  partial chunk, removed by Silver's dedupe.
- **01b with the forward window (2026-10-07).** Curve series is seamless
  across 2026-09-13 (pump.fun 150k bars on 09-12 -> 134k on 09-13, a
  dust-heavy day; LaunchLab 63k -> 50k); trader stats present on 100% of
  curve bars from 09-13, 0 before, as designed. Silver within ~1% of 00h's
  raw bar counts (dedupe + outlier band). Tokens by phase: curve+postgrad
  34,268 (was 20,621), curve-only 1,829,254 (was 1,061,515), postgrad-only
  28,392 (was 16,238; PumpSwap pools for non-curve or pre-window tokens,
  excluded from Gold), amm-only 44,563. Curve tokens: 1.80M usd + 64k
  quote. Graduation rate ~1.8%. nonpositive_close = high_below_low = 0.
- **Gold rebuilt on 50 days (2026-10-08).** entry_candidates 13,425,917 rows
  / 1,863,070 tokens (latest T 2026-10-03 23:24). exit_candidates 8,256,088
  rows / 424,636 tokens. Exit gap checked by launch window: old window
  232,090 of 1,082,128 tokens (21.4%; 56% of tokens whose last bar is
  15+ min after launch), new window 192,546 of 781,394 (24.6%; 64%) -- no
  window-specific bug. (The v2 figure of 413k exit tokens predates the
  activity filter.) The 287,593 tokens that "lived" 15+ min but have no
  exit row: median 1 bar after minute 15, 82% have <= 2 bars after it,
  98.8% have no bar between minutes 15 and 60 -- a launch burst plus a
  stray later trade, never tradeable after the first minutes. An exit row
  needs a trade within 2 min of T and a price within 30 min before T-15m,
  so excluding them is correct.
- **04 on 50 days (2026-10-08).** Ranking unchanged (Spearman 0.22-0.29,
  monotone deciles, best decile -2.5% to -3.8% after costs). Model EV
  -7.0/-1.0/-1.1/+0.2/-1.3% vs buy-everything -7.3% to -8.1%. Pooled
  stress: ex-top-1% -5.8% to -14.9% every fold; top 1% = 2-9x of profit;
  only the cap-100 EV is positive. Liquidity: the two zero-USD-volume
  quartiles are ~breakeven, the highest-liquidity quartile is worst
  (-4.0%). Segments: pumpdotfun/usd -2.6% (base -8.2%; 34.6k trades,
  7.5k tokens; folds -7.3/-4.2/+2.2/-0.5/+5.7). StonkFun/quote -0.8%
  (base -3.9%; 63k trades, 6.4k tokens). **StonkFun/usd -2.0%** (base
  -5.5%; 17.5k trades, 2.7k tokens; folds 2-4 all negative) -- the +4%
  on 30 days did not hold. Raydium/usd -4.0%, VOIDEX -0.8%. Only
  standout: **letsbonk.fun/usd +12.1%** (base -2.2%; 387 trades, 117
  tokens; folds +6.6/+15.7/-3.5/+22.8/+53.6) -- small, untested by the
  stress checks, added as its own segment in the per-segment stress cell.
- **Per-segment stress on 50 days (2026-10-08): no segment survives.**
  Pooled 10x: pumpdotfun/usd -2.6% (ex-top-1% -10.8%, per-token -4.3%,
  14% of tokens profitable); launchlab/usd -1.9% (ex-top-1% -7.8%,
  per-token -15.5%); StonkFun/usd -2.0% (ex-top-1% -7.9%, per-token
  -16.0%); StonkFun/quote -0.8% (ex-top-1% -5.9%, per-token -15.7%).
  ex-top-1% is negative in every fold for every segment. First signal per
  token is near zero (-0.3% to -6.7%). Only pumpdotfun's per-token EV is
  positive in folds 2-4 (+6.2/+1.6/+1.9%), but its ex-top-1% is -7% to
  -8% in those folds. Verdict: with a fixed 15-minute hold, nothing clears
  ~3% round-trip costs; the 30-day LaunchLab result did not repeat.
  letsbonk.fun/usd (+12.1%, 117 tokens) not yet stress-tested.
- **letsbonk.fun/usd stress (2026-10-08):** 387 trades / 117 tokens, EV
  +12.1%, ex-top-1% **+1.8%** (the only segment still positive), first
  signal per token +21.3%, per-token +5.0%, 20% of tokens profitable, top
  1% = 85% of profit, **one token = 51% of profit**. Treated as a lead,
  not evidence: 117 tokens is too few (rough standard error on the
  per-token mean is several points, so +5% is within noise), and half the
  profit is one token. Plan: hold it out as a pre-registered hypothesis
  and check it on data after 2026-10-03, untouched by any tuning.
- **Barrier label added (2026-10-08, Jack chose it).** "Does the price hit
  2x before -50%, within N?" for N = 1h / 6h / 24h, from each later bar's
  high/low (gold_input now passes high/low). Columns y_up_/r_up_/t_up_{N}m
  in entry_candidates; spec in label_spec.md. Conservative: a bar
  touching both barriers counts as down; up realizes exactly +100%; a gap
  through the stop realizes the bar's close. 8 new tests (43 pass);
  numpy-vectorised, one scan serves all horizons (busy 3,000-bar token:
  256 rows in 0.16 s). Next:
  re-run 02, check the barrier stats cell, then a classifier in 04.
- **Barrier base rates (02, 2026-10-09).** Hit 2x before -50%, by venue x
  phase at T (1h / 6h / 24h): pump.fun curve 5.0 / 6.0 / 6.8% (10.7M
  rows), mean rule return before costs -3.4 / -3.7 / -3.7%, median
  resolution = the full horizon (most rows time out: quiet curve tokens
  keep their price, so the stop rarely triggers). pump.fun post-grad 11.3
  / 16.0 / 28.3%, mean -7.3 / -9.8 / -9.2%. LaunchLab curve 5.3 / 6.9 /
  8.1%, mean -4.1 / -6.0 / -6.4%. LaunchLab post-grad 12.8 / 21.7 /
  34.4%, mean -3.4 / -3.4 / +1.0%. The 24h horizon is censored for 31-39%
  of post-grad rows and censoring only removes unresolved rows, so its
  hit rate/mean are biased up -- 04b uses 1h and 6h only. Buying
  everything loses before costs in every cell; the model must find a
  slice with a much higher hit rate.
- **04b written (2026-10-09):** `src/models/barrier_model.py`
  (LGBMClassifier on entry_model's features + minutes_since_launch +
  is_postgrad_at_t; censored rows excluded) and notebook 04b: per fold x
  horizon AUC, base hit/net, top 1/5/10% hit rate + net after per-row
  costs; last-fold decile calibration; top-5% picks by segment and fold
  with per-token concentration. tests/test_barrier_model.py (CI only).
- **04b barrier model results (2026-10-09).** AUC 0.79-0.81 (1h) and
  0.78-0.80 (6h) in every fold, well calibrated (decile mean_p ~= hit
  rate). Top 1% hit 2x first 26-32% vs base 5-7%. But net after costs is
  negative everywhere and gets WORSE with predicted probability: last fold
  decile 1 net -3.6%, decile 10 -14.0% (1h) / -12.6% (6h). Top 5% net -7%
  to -14% per fold; per-token -30% to -35%. Reason: high-probability
  tokens are the volatile ones -- when they miss they hit the -50% stop,
  while low-probability tokens time out with small losses. With misses at
  ~-50% the rule needs ~34% hit rate + costs to break even; the top 1%
  reaches 26-32%. Interpretation: the model has learned WHICH tokens will
  move a lot, not WHICH WAY. 2x and 0.5x are equal distances in log
  terms, so a 25-30% "up first" rate among movers means strong downward
  drift, and the features (trailing volume/trades/price change, age,
  phase) carry little directional information. Near-breakeven pockets
  (letsbonk.fun, StonkFun quote post-grad at 6h -4%) are small.
  Next: (1) direction diagnostic -- AUC of up-vs-down on rows that
  RESOLVED (excluding timeouts), to measure directional skill alone;
  (2) directional features: 00h's per-bar flow stats (buy/sell volume,
  largest trade, top-wallet share; 2026-09-13 onward only) and
  bonding-curve progress.
- **Direction diagnostic (2026-10-09): there IS directional signal.** On
  rows that resolved (16-21% of rows; timeouts dropped), predicting 2x-
  first vs stop-first: AUC 0.65-0.68 in every fold, both horizons; the
  model's top 10% went 2x-first 71-79% vs base 33-35%. Single features
  alone are weak (price change 0.46-0.48, i.e. slight mean-reversion; age
  0.53-0.55; buy/sell ratio and volume ~0.49-0.51) -- the signal is in
  combinations. Top importance: trailing volume, price change, age, buy
  count. So the binary classifier's failure was the target, not the
  features: up-vs-(down OR timeout) lumps a -50% stop with a small timeout
  loss and pushes the model to volatile tokens. Caveat: "among resolved
  rows" is not tradeable by itself (resolution isn't known at T). Added
  04b EV cell: (A) regressor on the bounded rule return r_up_{N}m, (B)
  3-class up/down/timeout model -> EV; scored on realized net with deciles,
  top 1/5%, "enter if EV > own cost", and per-token concentration.
- **04b EV models (2026-10-10).** (A) LightGBM regressor on the bounded
  rule return r_up: Spearman 0.33-0.38 every fold (best ranking so far;
  04's 15-min regressor was 0.22-0.29). Last-fold deciles monotone, but
  the top decile is ~ -cost (-3.4% at 1h, -3.2% at 6h) with a LOW hit
  rate (3-6%): the regressor's top picks are quiet tokens whose gross
  return is ~0, so it mostly learns "avoid losses", not "find winners".
  Top 1% at 6h: -2.6 / +3.2 / -0.5 / +3.3 / +0.7% by fold (~1-1.7k
  tokens each, bounded returns) -- the first top-1% that is positive in 3
  of 5 folds with many tokens, but ~+0.8% mean is within noise. "Enter if
  EV > own cost" at 6h: -2.6 / +2.9 / -1.2 / +1.1 / -0.5%. Top-5% pooled:
  -1.9% (6h), per-token -4.8%, 9.5% of tokens profitable. (B) 3-class EV
  is broken: Spearman -0.07 to -0.13 (inverse ranking) -- the fixed -50%
  / average-timeout payoffs don't match reality; dropped.
  **Segment lead: graduated StonkFun at 6h, model A top 5%: +1.6%, 48k
  trades, 1,708 tokens, 34% of tokens profitable, top token 9% of profit**
  -- broad, not one-token. Likely helped by cost: post-graduation CPMM
  fee is 0.25%/side vs ~1.25% on the curve, so the round trip is ~1.5%
  instead of ~3.5%. pump.fun graduated at 6h: -0.25%. Next: a model
  trained on post-graduation rows only (cheaper venue, 1.2M rows), and
  the flow features.
- **04b post-graduation-only cell added (2026-10-10):** model A trained
  and scored on is_postgrad_at_t rows only, 1h and 6h; per-fold Spearman,
  top 1/5/10% net, "enter if EV > own cost" (n, net, tokens), last-fold
  deciles + feature importance, top-5% concentration by venue/platform/
  unit and by fold. Note on fees: LaunchLab graduates (CPMM) pay 0.25%
  per side, pump.fun graduates (PumpSwap) still 1.20%.
- **Post-graduation-only model A (2026-10-10): first result that looks
  like an edge -- not yet proven.** 6h horizon, top 5% by predicted EV:
  pooled +3.0% net after costs (29.3k trades, 2,618 tokens, 36% of tokens
  profitable, top token 6.8% of profit); by fold -3.7 / +6.6 / +4.4 /
  +6.6 / +1.8% (4 of 5 positive). By segment: StonkFun usd graduates
  +8.9% (5.1k trades, 369 tokens, per-token +8.6%, 50% of tokens
  profitable, top token 7%); StonkFun quote +2.6% (per-token -1.0%);
  pump.fun graduates +1.6% (per-token -5.7%). 1h horizon: -5.1% pooled,
  negative every fold. **Caveats before trusting it:** (1) first-signal-
  per-token pooled -0.4% and per-token -2.5% -- much of the trade-level
  profit comes from repeated, overlapping entries on the same tokens
  (T is sampled every minute in the first hour); a bot holds one position
  per token, so this needs a non-overlapping backtest. (2) Selection: the
  post-grad slice was chosen after seeing it in the previous run, so the
  folds are not a clean out-of-sample test -- data after 2026-10-03 is.
  (3) Entry at the last close with 0.5%/side slippage; graduated pools,
  especially quote-priced StonkFun ones, may be thinner. (4) Quote-priced
  returns ignore the quote token's own move.
- **Non-overlapping backtest cell added to 04b (2026-10-10):** post-grad
  model A, 6h; one position per token, held to 2x / -50% / 6h, re-entry
  only on a new signal after exit. Cutoffs: 95th percentile of predicted
  EV on TRAINING rows (fixed before the test period -- the honest one)
  vs the test-period 95th percentile (the earlier, slightly peeking one).
  Reports trades, tokens, mean/median net, hit rate, per-token, share of
  profitable tokens, top-token share, total stakes won, max concurrent
  positions; by fold and segment. Logic checked on synthetic data.
- **Non-overlapping backtest (2026-10-10): the edge mostly survives.**
  Post-grad model A, 6h, one position per token, cutoff fixed from the
  TRAINING period (train_q95): 6,718 trades over 2,924 tokens, mean net
  **+2.3%** per trade after costs, +152 stakes total, median -21% (most
  trades lose; 27% hit 2x), top token 3.5% of profit, max 64 positions
  open at once. By fold -4.9 / +2.1 / +2.6 / +6.6 / -0.2% (3 of 5
  positive; fold-mean +1.2%). Per-token mean -2.3% -- the profit comes
  from re-entering tokens that keep working, while one-shot tokens lose
  on average. Segments: StonkFun usd graduates +6.3% (1,459 trades, 520
  tokens, per-token +3.4%, 46% of tokens profitable); StonkFun quote
  +2.6%; pump.fun graduates -0.3%. test_q95 cutoff similar (+1.7%).
  Rough significance: ~2 standard errors overall (clustering by token),
  fold-to-fold t ~1 -- suggestive, not established; and the post-grad /
  StonkFun slice was chosen after seeing it.
- **PRE-REGISTERED out-of-sample test (written 2026-10-10, before any
  data after 2026-10-03 is pulled or looked at).** Rule, frozen:
  1. Model: LGBMRegressor (random_state 17, 300 trees, lr 0.05) on
     r_up_360m, features = barrier_model.FEATURE_COLUMNS, trained on ALL
     post-graduation rows with T before 2026-10-03.
  2. Entry: post-graduation rows only; enter when predicted EV >= the
     95th percentile of predicted EV on that training set; one position
     per token; exit at 2x / -50% / 6h; re-entry only on a new signal
     after exit.
  3. Costs: per-row round trip as in 04/04b (venue + phase fees, 0.5%
     slippage per side), plus a stress case with 1.0% per side.
  4. Test data: T from 2026-10-03 through the last day with 24h of bars
     after it, never used for training or tuning.
  5. Pass = mean net per trade > 0 overall AND for StonkFun usd
     graduates, with >= 300 trades overall. Anything else = fail; no
     re-tuning on the test data.
- **Window 3 + out-of-sample cell set up (2026-10-11).** 00h WINDOW_END
  -> 2026-10-10 (MAX_CHUNKS None; 28 new chunks). 00f WINDOWS gains
  (10-03, 10-10); CPMM step 1/2 now loop over every window from 2 on;
  step 2's checks generalised (all step-1 days from window 2 through
  that day must be done; skip mints with CPMM bars before that day);
  MAX_PUMPSWAP_CHUNKS / MAX_CPMM_BAR_DAYS -> None (costs known). 02/03
  CURVE_DATA_END -> 2026-10-10. Estimated ~3-3.5k credits (00h ~2.1k,
  00f ~1.2k). New last cell in 04b runs the pre-registered rule exactly:
  reads ALL post-grad rows from Gold (no token sampling), trains once on
  T < 10-03, cutoff = train 95th pct, scores T >= 10-03 one position per
  token, base and 1%/side-slippage costs, by segment and day, and prints
  the pre-registered PASS/FAIL. Known small leak: training rows within
  6h before 10-03 have labels that end after it.
- **Social data architecture decided (2026-10-16, Jack): option C.**
  Social data gets its own Bronze -> Silver -> Gold pipeline that turns
  raw posts into a few point-in-time numeric features per (token, T)
  (mentions 1h/24h + growth, unique authors, bot share, follower-weighted
  reach, first-mention delay after launch, deployer/official posts).
  Those join the SAME LightGBM model as extra columns -- one model, so
  trees can learn social x on-chain interactions. Missing = no posts
  (NaN, not 0). Rules: only posts created before T; beware deleted
  posts missing from historical pulls (train/serve skew); kept only if a
  with/without walk-forward comparison shows out-of-sample lift. Order:
  on-chain flow features first (free, already in Silver from 09-13),
  social after, and both only after the pre-registered test result.

## 2026-10-16 — Flow features drafted (not run) + social data source research

**Flow features (code only; run after the pre-registered test).** `gold_features.flow_features` adds 8 point-in-time columns to entry_candidates: `flow_buy_share_15m`, `flow_max_trade_share_15m`, `flow_top_trader_share_15m` (volume-weighted mean of per-bar top-trader share), the curve-life versions `curve_buy_share`, `curve_max_trade_share`, `curve_top_trader_share`, `curve_top_trader_share_max`, and `minutes_since_graduation` (None until the first postgrad bar has closed). Missing stats → None, never 0. `barrier_model.FEATURE_COLUMNS` is left FROZEN; the new list is `FLOW_FEATURE_COLUMNS` / `FLOW_MODEL_FEATURE_COLUMNS`, and `prepare_barrier_features` takes `feature_columns`.
- Caveat 1: postgrad bars carry no flow stats (00f doesn't pull them), so post-grad rows -- Model A's whole training set -- only get the curve-life summary + minutes_since_graduation. The 15m flow columns are mostly NULL there. Adding buy/sell/top-trader stats to 00f's postgrad bars is a separate Dune cost decision.
- Caveat 2: `top_trader_share` exists only from 2026-09-13 (Dune curve bars); Silver trade bars before that have buy/sell/max but not it. NULL therefore partly encodes "older data". Compare flow vs frozen features on T ≥ 09-13 rows only.
- Needs a full 02 rerun (Gold schema change) -- after the pre-registered result is recorded.

**Social sources (option C), prices checked 2026-10-16, volume estimates unmeasured.**
- X API is pay-per-use since Feb 2026 (no free tier): $0.005/post read, counts endpoints $0.005 (recent) / $0.01 (all-archive) per request, 3M reads/month cap. All Counts gives per-minute mention counts for mint/cashtag queries -- the cleanest point-in-time signal. Backfill est. ~$2k for 70k graduated tokens (~3 requests each; bucket-per-request count unverified).
- Third-party scrapers (TwitterAPI.io $0.15/1k tweets, SocialData ~$0.20/1k, Apify $0.25-0.40/1k) give authors/followers/engagement with historical search. ToS risk. Backfill est. $3-9k; live ~$0.8-1.6k/month -- unmeasured, sample 500 tokens first.
- LunarCrush ($90-900/mo) and Santiment ($249/mo) target established coins; coverage of minutes-old mints unverified/unlikely.
- Free: pump.fun replies (unofficial API, timestamped, covers every pump.fun token), DexScreener socials/boosts (live snapshot only), Telegram call channels via Telethon (history from when joined; ban risk).
- Point-in-time: engagement/follower counts are as of scrape, not T; deleted posts vanish (biases rugs to look cleaner). Use counts + unique authors at T; treat engagement as lagged.
- Proposed order: (1) pump.fun replies backfill (free), (2) X All Counts on a 500-token sample to measure volume and cost, (3) decide on paid backfill only if (1)/(2) show out-of-sample lift.
- 2026-10-16: pump.fun replies backfill drafted -- `src/ingestion/pumpfun_replies.py` (+ 8 tests) and notebook 00i → `lawrence.bronze.pumpfun_replies`, resumable per mint. Graduated pump.fun tokens first, then a 50k hash sample of curve-only tokens. The API is unofficial; the notebook probes one token before bulk. pump.fun is blocked from both the Claude workspace and Jack's local VM, so the format is unverified until the Databricks probe runs.

## 2026-10-16 — Pre-registered out-of-sample test: FAIL

Rule frozen 2026-10-11 (see above): Model A, trained once on T < 2026-10-03, cutoff = train q95 of predicted EV (0.0656), one position per token, 2x / -50% / 6h, tested on T >= 2026-10-03.

| | base (0.5%/side slip) | stress (1%/side) |
|---|---|---|
| trades / tokens | 1,144 / 640 | same |
| mean net per trade | -0.00% (sum -0.02 stakes) | -1.0% (sum -11.5 stakes) |
| median net | -26% | -27% |
| hit 2x | 26% | 26% |
| pump.fun usd | +0.2% (806 trades) | -0.8% |
| StonkFun usd | -1.9% (148) | -2.9% |
| StonkFun quote | -1.3% (157) | -2.3% |
| tokens profitable | 38% | 38% |

Checks: trades >= 300 TRUE; overall net > 0 FALSE; StonkFun usd net > 0 FALSE -> **FAIL**. Days split 4 up / 4 down.

Reading:
- The in-sample +2.3% (non-overlapping, train_q95) did not hold out of sample; it decayed to break-even. No re-tuning on this window, per the rule.
- Selection still carries a direction signal: 26% hit 2x first vs a 16% base rate for post-grad pump.fun at 6h, and gross returns are well above the -9.8% unconditional mean. That's not enough to clear fees + slippage. The edge, if any, is smaller than costs.
- Rough noise: per-trade sd ~0.5 → SE ~1.5% on 1,144 trades, so +2.3% vs 0% isn't strongly distinguishable either way. The rule exists so this doesn't become a reason to keep going.
- **Not tradeable.** No live or paper capital on Model A as specified.
- The 10-03..10-10 window is now USED. It can be training/validation data for new work, but any new hypothesis (flow features, social) needs its own pre-registration and a fresh test window of data after 2026-10-10 that nobody has looked at.

## 2026-10-17 — pump.fun comments removed; 00i rewritten for metadata + callouts

The 00i probe returned `404 Cannot GET /replies/{mint}`, meaning the route itself is gone, not the token. Checked in a browser against pump.fun's own frontend JS: per-coin comment threads no longer exist, and the coin record has no reply count. Public, unauthenticated replacements on `frontend-api-v3.pump.fun`:
- `GET /coins-v2/{mint}`: coin record. Its `metadata_uri` (IPFS, immutable) holds the **launch-time** twitter/telegram/website/description, which are point-in-time safe. The record's own `twitter` may be edited later; it's stored as `current_twitter` and isn't a feature.
- `GET /callout/coin/{mint}?limit=100`: user "callouts" with `createdAt` and market cap at call. Engagement counts are as of the pull. **Max 100, ranked by `maxMultiplier` (post-hoc performance)**, so a coin returning 100 is truncated toward winners, which is look-ahead bias. Flagged `callouts_truncated`; those coins get no callout features. `maxMultiplier` is stored as `posthoc_max_multiplier` and is never a feature.

Code: `src/ingestion/pumpfun_social.py` (+ 8 tests) replaces `pumpfun_replies.py`. Notebook `00i_bronze_pumpfun_social.py` writes `bronze.pumpfun_coins`, `bronze.pumpfun_callouts` and `_pumpfun_social_progress`. IPFS via pump.fun's Pinata gateway, then ipfs.io, with 2 retries each. Expectation: weak signal (most coins likely have zero callouts); the metadata features are cheap and clean.
- 2026-10-17, 00i first 200 graduated pump.fun coins: all fetched, 3.4 mints/s.
  - Launch metadata loaded for 82%; the misses are IPFS gateway failures, so a retry cell was added.
  - 70.5% of coins have ≥1 callout (median 7, p90 88), and 8.5% are truncated at 100.
  - Launch socials: X 40.5%, Telegram 12.5%, website 30.5%.
  - Callout timing: 45% before graduation and 80% within 6h of graduation; 3,115 distinct callers across 4,371 callouts (mostly one-off).
  - Timely enough to matter for the 6h post-grad decision. Truncation (and its absence) is post-hoc and is never a feature.

## 2026-10-17 — Scoring contract v1 for shadow mode (request R1 from the product chat)
- `contracts/v1/CONTRACT.md` + `src/serving/contract_v1.py` (pure pandas; 7 equivalence tests against Gold: features, grid, selection, outcome, costs) + `notebooks/04c_export_contract_v1.py` (trains the frozen Model A spec on all resolved post-grad rows, saves booster + sha256 manifest + PSI baselines).
- v1 = the spec that failed on 2026-10-16. It ships for plumbing, skew measurement and a second independent pre-registered test, not because it's expected to pass.
- Authoritative shadow labels = the pipeline re-run over the shadow window, not live-resolved outcomes.
- Proposed pre-registration (CONTRACT.md §11): 21-day fixed window, extension to 35 days if < 300 trades, PASS = ≥300 trades AND net > 0 at base AND at stress costs. Pending Jack's approval at the freeze commit.

## 2026-10-17 — R2 answers; Silver look-ahead found; v1 selection fix
- **Contract fix:** v1 candidates must satisfy `as_of + 6h ≤ first postgrad bar + 24h`. Training only has 24h of post-graduation bars and censors later rows; the first contract draft would have scored rows the model never saw. Added to `select_row` and tested.
- **Look-ahead in Silver** (training side, minor but real), found while specifying live bars:
  1. The outlier-band median uses the token's full history.
  2. The canonical post-graduation pool is the busiest over 24h.
  3. `price_unit` uses full-history USD coverage (non-major quotes only).
  - Live uses point-in-time versions (`serving/bars_v1.py` with `band_mode="live"`, the migration pool, and `price_unit_live`).
  - The parity fixture (00j) measures the gap. If it isn't negligible, fix Silver to point-in-time for v2 (it would change training data, so not in v1).
- `src/serving/bars_v1.py` is the reference builder from raw swaps (4 synthetic tests); `tests/test_parity_fixture.py` runs once 00j's fixture is committed.
- 2026-10-17 (R3): Gold keeps a post-grad row past grad + 18h only when its 6h barrier resolved before the 24h data edge. That's **selection on the outcome**. The failed 10-03 test's train and test sets both included these rows. 04c now drops them, so v1's training universe = live selection (CONTRACT §4 rule 7); its cutoff is computed after the fix. Re-checking the failed test's result without them is low-value (the rule was frozen as run), but note it as a known flaw of that test.
- 2026-10-17, 00i metadata misses (6,191 coins, 18%, 0 recovered on retry): a 300-coin sample shows ~60% point at non-IPFS hosts. `metadata.j7tracker.io` (172) is a third-party launch tool; the rest are throwaway domains, now 404. Most of the others are unpinned or gateway-"blocked" IPFS content. So missing metadata is post-hoc and correlated with coin fate: `has_launch_metadata` is NEVER a feature. The URI **host**, though, is on-chain at creation, so it's launch-time-safe and a candidate bundler/launch-tool fingerprint. Added `metadata_uri`/`metadata_host` to coin rows and a backfill cell → `bronze.pumpfun_coin_uris`.
- 2026-10-17 (R4): **v1 = the Model A *spec*, not the Model A *model*.**
  - **Same as Model A:** algorithm and params (LGBMRegressor, seed 17, 300 trees, lr 0.05), features (`barrier_model.FEATURE_COLUMNS`), target (`r_up_360m`), cutoff rule (train p95), entry/exit and costs.
  - **Different:** the booster is RETRAINED on (a) all data through 2026-10-10, not < 10-03, and (b) only rows satisfying CONTRACT §4 rule 7 (≤ 18h after graduation), which removes the outcome-selected rows past 18h.
  - So v1's shadow test is a test of "the Model A spec trained on a corrected universe", not a re-run of the model that failed.
- 2026-10-17 (R4): startup tolerance set to 1e-9, not 1e-12.
  - Prediction = a fixed-order sum of 300 leaf values in IEEE doubles, so x86 and arm64 should agree exactly given identical inputs and the same lightgbm version, but that's unverified across builds.
  - Any real fault (wrong file, version or frame) shows up at ≫1e-6, and a 1e-9 score difference can't flip a decision except within 1e-9 of the cutoff.
  - Reference platform: x86-64 Linux.
- 2026-10-17, metadata URI hosts (35,102 graduated pump.fun coins, `bronze.pumpfun_coin_uris`):
  - **Standard path:** ipfs.io 76% (94% still resolve).
  - **Third-party, ~24%:** launch/deployer tools j7tracker 3,459 (service dead, 0% resolve), uxento 1,256, rapidlaunch 509, vortexdeployer 187, usepaid 243, agencypad 160, launchblitz 107; Pinata gateways; throwaway domains.
  - **Resolution:** depends on the host/tool, not the coin. Still post-hoc, so never a feature.
  - **Planned features (social pass):**
    - `metadata_host_group`: ipfs.io / launch-tool hosts with ≥100 coins in the TRAINING period / other_pinata / other_custom. Groups frozen from training-period counts; unseen hosts → other_custom.
    - `is_launch_tool`.
  - **Coverage gap:** graduated coins only. Curve-only coins need URIs from Dune (on-chain metadata) before this can be an entry feature before graduation.
  - **Open:** 35,102 distinct mints vs the expected ~34.9k from the pulls. Check for duplicates or format variants before building features.
- 2026-10-17, parity fixture (00j, 20 tokens, 43k+ swaps):
  - `serving/bars_v1.build_bars` reproduces Silver for 2,731 of 2,734 bars, with identical closes, identical n_trades and volume within 1e-15. The 3 misses are Silver bars on 10-07, past the fixture's swap window (window edge, not a builder gap); the test now compares only bars inside the swap window.
  - Live band mode gave identical bars on all 20 tokens.
  - All 12 pump.fun graduates' canonical (busiest) pool = the earliest PumpSwap pool.
  - Dune run cost: ~29 credits for the 1st chunk + prices.
- 2026-10-17, **v1 shadow pre-registration APPROVED** (CONTRACT.md §11), replacing the draft "≥300 trades AND mean net > 0".
  - Jack's point: outcomes matter more than trade count (2 good trades/day would be a winning strategy).
  - Agreed, with a correction: the trade minimum exists to separate edge from luck, not to reward volume. At a per-trade sd of ~0.5, 42 trades give a ±8-point SE.
  - **Final rule:** 21-day fixed window; ≥100 non-overlapping trades (extend once to 35 days, else inconclusive). PASS = 90% one-sided lower bound on mean net/trade > 0 at base costs (block bootstrap over token-days, 10k draws, seed 17) AND mean net > 0 at stress costs.
  - Exploratory, non-judged: top-2-per-day variant on the same window, for a possible future selective-rule pre-registration.
- 2026-10-17, **04c v1 export:**
  - Rule 7 dropped 15,034 of 1,336,584 labelled rows (1.1%).
  - Trained on 1,321,550 rows / 37,823 tokens, 2026-08-14 → 2026-10-10 17:38.
  - Cutoff (train p95) 0.0470, down from the failed model's 0.0656.
  - Expected ~137 trades/day (in-sample replay).
  - sha256 `1082b6879d86b97c916755a58db2d6d6b32da7e7f2345b3f785dd9547fc4a2dd`; lightgbm 4.7.0, pandas 2.3.3, numpy 2.3.4, scikit-learn 1.7.2.
- 2026-10-17: frozen evaluator `src/backtest/shadow_eval_v1.py` + 7 tests (CONTRACT §11). It goes in the freeze commit with manifest.json and must not change after SHADOW_START.
- 2026-10-18, **notebook 05 gate results** (rules set beforehand, above):
  - **Check A:** 35,102 coins = all graduated pump.fun tokens except 1 (the single missing progress row). No duplicates. My earlier ~34.9k "expected" figure was an arithmetic slip, not a data issue.
  - **Check B FAILED → callout features DROPPED from v2.** At top-50% truncation, the share of a coin's first 20 callouts kept had mean 0.855, p10 0.55 (gate ≥ 0.95 / ≥ 0.85) over 3,435 fully-observed coins. At 75% it would be borderline (0.958 / 0.85), but the gate was fixed at 50%. The API ranking keeps late winners, not early callers. Callouts collected LIVE by the runtime poller are unbiased, so a callout family can return in a later version trained on live-collected data.
  - **Check C FAILED → launch socials DROPPED.** Within ipfs.io, coins whose metadata didn't resolve hit 2x first 29.4% vs 22.9% (mean r −2.5% vs −13.6%). Unresolved metadata tracks better outcomes, so the NaN pattern would leak.
  - **Kept for v2:** `meta_host_group`, `meta_is_launch_tool` (on-chain at creation). Raw 2x-first rates by host range 17–33% (e.g. j7tracker 26.7%, usepaid 27.6%, googleapis 17.3% vs ipfs.io 23.3%). That's descriptive only; v2's test decides whether they help.
- 2026-10-18: `notebooks/04d_v2_dev.py` compares V1 / V2a (+flow) / V2b (+flow + host) on post-grad rows from 2026-09-13 to 10-10 (rule 7 applied), using v1's spec and rule. Walk-forward over 4 weekly folds, each trained on all earlier rows. **Pre-set decision:** take the variant with the highest pooled mean net per trade only if it beats V1 in ≥3 of 4 folds; otherwise v2 = no feature change. Development only: the 10-03..10-10 week is a dev fold now, and the shadow window is untouched.
- 2026-10-18, **04d result → v2 = V2b (v1 features + 8 flow + metadata host)**, by the pre-set rule (highest pooled mean net per trade AND beats V1 in ≥3/4 folds). Post-grad rows 09-13..10-10, walk-forward, v1 spec and rule, base costs:

  | variant | trades | pooled mean net | 90% lower bound | folds > V1 | median Spearman |
  |---|---|---|---|---|---|
  | V1 | 4,240 | −2.16% | −3.38% | – | 0.233 |
  | V2a (+flow) | 1,965 | −1.11% | −2.90% | 3 | 0.250 |
  | V2b (+flow+host) | 2,440 | +0.11% | −1.45% | 3 | 0.246 |

  - **Reading:** V2b improves on V1 by ~2.3 points per trade in development and is roughly break-even. Its lower bound is still < 0, so **it would FAIL the v1-style pre-registration on this data.** There's no demonstrated edge yet; the shadow window decides.
  - Selection among 3 variants on data that includes the already-used 10-03 week means the +0.11% is optimistic.
  - **Top gains:** trailing_price_change_pct, then curve_buy_share, curve_max_trade_share, curve_top_trader_share, minutes_since_graduation, curve_top_trader_share_max, meta_host_group. The flow_*_15m columns are small (mostly NaN post-grad).
  - meta_host_group has small categories (107–191 coins), so overfitting risk is real. It survived walk-forward, but watch its share of gain.
- **Training universe for v2:** post-grad rows ≥ 2026-09-13 only. Before that, top_trader_share is NaN by data era, which would teach "NaN = August".
- **Serving consequences for contract v2 (to build):**
  1. Live bars need per-bar flow stats (buy/sell quote volume, max trade, top-trader share), so `bars_v1` → `bars_v2`, plus a fixture regenerated with flow columns.
  2. The host feature needs the on-chain metadata URI at launch, plus the frozen host→group map.
  3. LaunchLab tokens had NO host in training, so live must also give NaN for LaunchLab even if a URI is available.
- 2026-10-18: **Runtime spend deferred; testing moves to a pre-registered Dune holdout** (`contracts/v2/PREREG.md`). Window 10-11 → 11-01 (21 days, never pulled), v1 + v2 both frozen beforehand. PASS = 95% lower bound > 0 at base AND mean > 0 at stress (α 0.05 each, Bonferroni for 2 models). Freeze = the commit with v2's artifacts + PREREG + `holdout_eval_2026_10.py` + notebook 06, before any window-4 pull. Live shadow stays the gate before real money.

## 2026-10-18 — v2 + Dune holdout FROZEN at commit e8f36b2 (branch `social`)
- Frozen: `contracts/v2/` (model sha256 `3e6fdf53…288a`, cutoff 0.0685, 667,514 training rows 09-13 → 10-10 17:38, 15 host groups), `src/models/v2_spec.py`, `src/backtest/holdout_eval_2026_10.py`, `notebooks/06_holdout_eval_2026_10.py`, and `contracts/v2/PREREG.md`.
- **No data after 2026-10-10 has been pulled.** Holdout window 2026-10-11 → 11-01. Pull after 2026-11-02 per PREREG, then run 06 once.
- Models under test: v1 (cd6a098) and v2 (e8f36b2); α = 0.05 each.
- **Pre-registration amendment 1 (2026-10-22, before any holdout data).** Adds "v2-R1" (frozen v2 model,
  cutoff = train p98 instead of p95) as a third holdout test, the best dev rule from v3 part B. Three tests →
  alpha 0.10/3 for all of v1, v2, v2-R1 (stricter than the original 0.05). Frozen files untouched; new
  `holdout_eval_2026_10_amend1.py` (3 tests), `notebooks/04i` (cutoff + reproduction checks), and an official
  verdict cell in notebook 06. See `contracts/v2/PREREG_AMENDMENT_1.md`.
- **Decision core started (2026-10-22, spec `docs/decision_agent_spec.md`).** `src/decision/`: types, `RiskGate`,
  `PaperBroker`, `Engine` (per-token state machine; exits re-use the contract's own label logic so live exits =
  training labels), `ContractV1` adapter (hash check + `verify_startup`), replay harness + `parity()`.
  7 unit tests (`tests/test_decision_engine.py`). Replay parity on real bars: `notebooks/07_decision_replay_v1.py`.
  Defaults per spec §5 (paper $100, ≤10 open, ≤30 entries/day) pending Jack's confirmation; v1 used for the
  replay because it is the only contract with live serving code.
- **Replay parity run 1 (2026-10-24): 10/11 trades matched exactly (max return diff 0.0), 1 timing mismatch.**
  The harness only called the engine at the token's own bar closes; a candidate at a minute with no bar close
  (allowed: a bar within the prior 2 min suffices) was missed and the engine entered 30 min later. Fix:
  `replay` ticks every minute per token (+6h tail); spec §4 now requires the live runtime to do the same.
  New test; 8/8 pass. Re-run 07.
- **Holdout pull started as pre-registered (2026-11-22), option A (no amendment).** 00h WINDOW_END -> 11-02,
  00f window 4 = (10-10, 11-02), 02 CURVE_DATA_END -> 11-02. Measured: 00h 96.9 credits for 2 pump.fun 6h
  chunks (~48/chunk, ~2x October); 00f 81.6 credits for metadata + 1 PumpSwap chunk + 1 CPMM day. Estimated
  remaining ~9.7k credits (00h ~6.4k, 00f ~3.3k). Cheaper options (curve bars for graduates only; shorter
  window) were considered and not taken, so the test runs exactly as pre-registered.
- **Holdout pull complete (2026-11-22).** 00h: 200/200 pump.fun 6h chunks + 50/50 LaunchLab days `success`
  (window 09-13 → 11-02). 00f: 320/320 PumpSwap chunks, 50/50 CPMM days, 50/50 CPMM first-trade days, 4+4 pool
  metadata jobs `success` (windows 1-4). Earlier `failed`/`partial` rows were later re-run whole. Total Dune
  credits for the pull: 6,676.6 measured (estimate was ~9.9k). Next: 00i (graduated + URI backfill), 01b, 02, 06.

## 2026-11-22 — Pre-registered Dune holdout 2026-10-11 → 11-01: all three models FAIL

Official verdicts (notebook 06, Amendment 1 cell; alpha = 0.10/3 per model; 0 excluded days):

| | v1 | v2 | v2-R1 |
|---|---|---|---|
| trades / tokens | 3,383 / 1,774 | 2,640 / 1,338 | 867 / 480 |
| mean net, base | -5.28% | -4.47% | -5.46% |
| lower bound (96.7%), base | -7.31% | -6.46% | -8.96% |
| mean net, stress | -6.28% | -5.47% | -6.46% |
| median net, base | -43.7% | -18.0% | -21.0% |
| hit 2x | 24.5% | 19.6% | 20.0% |
| status | FAIL | FAIL | FAIL |

Reading:
- Not close: every lower bound is below -6%, every mean below -4%. The development edge (v2-R1 +2.93%) did not
  survive unseen data; v2-R1 traded ~41/day vs ~22/day in development and hit 2x on 20% vs 27%.
- Reported, not judged: the exploratory top-2-per-day variant shows +10.3% for v2/v2-R1 on 42 trades. It ranks
  within the whole UTC day, the same look-ahead as rule R7, so it is not a tradeable result.
- Per PREREG.md: no live spend. Record and stop. Any new model is a new pre-registration on a later window.
- Dune credits for the pull: 6,676.6 (measured).
- **Replay parity root cause found (2026-11-22).** The one mismatch (token Hmuv…, backtest entry 10-05 06:25,
  engine 06:55) was not a feature or score difference (debug: identical scores, no feature diffs, no duplicate
  bars). The engine's previous position on that token timed out AT 06:25; the backtest's one-position rule frees
  the token at its exit time, but the engine returned after closing and skipped assessing that minute. Fix:
  `Engine.on_bar_close` falls through to assessment when the open position closes in the same tick. New test;
  9/9 engine tests pass. Re-run 07 (expect 11/11).
- **v5 set up (2026-11-22), plan in `docs/v5_plan.md` (written before any result).** Target = E3 net return,
  history 09-13 → 11-01, +11 market-context features. New: `features/market_context.py` (2 tests),
  `simulate_schedule` returns `exit_ts` (17 ladder tests), notebooks 02d (labels + context tables) and 04j
  (B0 vs v5, 6 folds, pre-declared freeze rule).
