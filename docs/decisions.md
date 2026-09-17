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
