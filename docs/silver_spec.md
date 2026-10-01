# Silver spec (v1) — `lawrence.silver.token_trades`

Silver's job: turn raw two-sided Bronze swaps into a clean, deduplicated,
one-row-per-(trade, token-of-interest) price/activity/cost series that Gold
windows-aggregates for features and looks up for the entry/exit labels in
`docs/label_spec.md`.

## Why dedup here, not in Bronze

The Bronze job re-pulls a 20-minute lookback every 15 minutes (deliberate
overlap so a slow run doesn't lose data) -- so the same trade (`tx_hash`)
can land in Bronze more than once across runs. That's fine; Silver dedupes,
not Bronze.

**Dedup key is `(swap_id, token_id)`** (changed 2026-09-22, see
`docs/decisions.md`). `swap_id = sha2(concat_ws('|', dex_venue, tx_hash,
buy_token_id, sell_token_id, buy_amount, sell_amount), 256)`, computed on
the two-sided Bronze swap. The old key `(dex_venue, tx_hash, token_id)`
collapsed legitimate multiple swaps of the same token inside one
transaction (router multi-hops, bundled launch buys) -- ~3.8M real rows in
the Dune backfill. `token_id` still disambiguates the two rows a
non-quote/non-quote swap produces.

## Why only one side of each trade

A swap is always token A <-> token B. For a new/candidate token traded
against SOL/WETH/USDC, we need *that* token's USD price, not raw two-sided
amounts. v1 approach: an explicit small per-chain denylist of quote/
reference assets (SOL, WSOL, USDC, USDT, WETH, ETH, ...) -- whichever side
of a trade isn't on that list is "the token of interest" for that row,
applied per side rather than per trade:

- one side on the denylist, one not (the common case): one row
- both sides on the denylist (e.g. a USDC/WSOL swap): zero rows -- nothing
  to score
- neither side on the denylist (two candidate tokens traded directly
  against each other): two rows

Implemented in `src/features/silver_token_trades.py::to_silver_rows()`
(pure Python, no PySpark -- unit-tested in
`tests/test_silver_token_trades.py`, runs in CI); orchestrated by
`notebooks/01_silver_token_trades.py`, which reads a bounded Bronze
lookback and MERGEs into Silver on the dedup key above so overlapping runs
are a no-op, not a duplicate. Wired into `resources/lawrence_job.yml` as a
task that runs right after Bronze each cycle.

## Schema

| column | notes |
|---|---|
| `token_id` | traded contract/mint address (the non-quote side) |
| `chain_family`, `network`, `dex_venue`, `dex_protocol` | carried from Bronze |
| `trade_time` | `TimestampType`, parsed from Bronze's string `block_time` (two formats: `... UTC` and ISO `...Z`) |
| `side` | `buy` or `sell`, from the token-of-interest's perspective |
| `price_usd` | token's USD price in this trade, normalized regardless of which raw side it was on; NULL when Dune has no USD price |
| `price_quote` | token's price in QUOTE-token units (`quote_amount / amount`); present even when Dune can't USD-price the trade (added 2026-09-30) |
| `amount_usd` | trade size, for volume/liquidity features; NULL when Dune has no USD price (row kept if `price_quote` exists) |
| `quote_amount` | amount of the quote token on the other side of the trade |
| `tx_fee_usd` | total transaction cost in USD, normalized cross-chain -- a proxy for competition/MEV heat at that moment, and needed for realistic backtest cost simulation |
| `chain_native_fee` | fee in the chain's native currency, as a decimal (SOL for Solana, ETH for EVM) -- **not** raw base units (lamports/wei); kept alongside `tx_fee_usd` for anyone doing chain-specific analysis |
| `tx_hash` | source transaction |
| `swap_id` | content hash of the two-sided swap; dedup key with `token_id` |
| `quote_token_id` | what it traded against (SOL, USDC, etc.) -- reference only, not modeled |

## Fee fields — resolved, now live in Bronze

Both chains' fee fields were verified against live Bitquery responses
(not guessed) before implementation:

- **Solana**: `Transaction { Fee FeeInUSD }` is a plain sibling field under
  `Transaction`, directly on the existing `Solana { DEXTrades }` root --
  no join needed. One bundled total; Solana doesn't split base/priority fee
  the way EVM does.
- **EVM**: no direct `Transaction.Fee` field exists under `DEXTrades`.
  Getting fees requires a join: `joinTransactions(join: inner,
  Transaction_Hash: Transaction_Hash) { Fee { SenderFee SenderFeeInUSD } }`,
  confirmed working on the existing `EVM(network: ...) { DEXTrades }` root
  (no query-root migration to `DEXTradeByTokens` needed). Also one bundled
  total, same as Solana -- no base/priority-fee split available this way.

Both `Fee`/`SenderFee` values are native-currency **decimals** (SOL / ETH),
confirmed from real responses (e.g. `0.000017540` SOL, `0.000001414...` ETH)
-- not raw base units (lamports/wei). That's why the Bronze/Silver column is
named `chain_native_fee`, not `chain_native_fee_raw`.

Implemented in `src/ingestion/bitquery_client.py` (query changes) and
`notebooks/00_bronze_dex_ingestion.py` (`tx_fee_native`/`tx_fee_usd` in
`_normalize`). Silver's job is still just to carry these two columns
through unchanged (per the schema above).

## `source` column: `bitquery` vs `dune`

Bronze rows carry a `source` column. Live rows (`00_bronze_dex_ingestion.py`)
are `source = 'bitquery'` and have real fee data. Historical backfill rows
(`00c_bronze_backfill_dune.py`) are `source = 'dune'` and always have
`tx_fee_native`/`tx_fee_usd` = NULL -- neither Dune table Bronze reads from
has a usable on-chain transaction fee (confirmed 2026-09-15, see
`docs/decisions.md` and `src/ingestion/dune_client.py`'s module docstring).
Anything downstream computing on these two Silver columns should expect
nulls for `dune`-sourced rows and handle them accordingly, not assume every
row has real fee data.
