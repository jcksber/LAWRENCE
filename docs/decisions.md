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
