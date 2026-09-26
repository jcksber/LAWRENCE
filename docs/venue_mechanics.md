# Venue mechanics — how pools are created and how Dune records them

Researched 2026-09-22, before Silver's token-of-interest logic is finalized.
Every rule in Silver/Gold/labels depends on these facts; re-check this doc
before adding a venue. Sources at the bottom -- primary ones are the
programs' own docs and Dune's spellbook SQL (the code that *builds*
`dex_solana.trades` / `dex.trades`), not blog summaries.

## Vocabulary: base vs quote, and where WSOL fits

Every pool/curve is created with exactly two mints fixed at creation:
- **base** = the token being launched / the thing we score
- **quote** = what it's priced and paid in (SOL, USDC, ...)

WSOL (`So11111111111111111111111111111111111111112`) is the SPL-token
wrapper around native SOL (1:1). AMMs (Raydium AMM v4, CPMM, PumpSwap)
hold SOL in a WSOL token vault, so WSOL genuinely is the pool's quote
mint. The **pump.fun bonding curve does not** -- it holds native lamports
directly on the bonding-curve PDA; Dune *labels* that side with the WSOL
mint by hardcoding it (`pumpdotfun_solana_base_trades.sql`). Same economic
asset, so for us "WSOL" = "SOL" on every Solana venue.

`dex_solana.trades` does NOT carry base/quote. It carries
`token_bought_*` / `token_sold_*` from the **trader's** perspective, so
which side is the base flips with trade direction. Recovering base/quote
needs either a denylist heuristic (what Silver v1 does) or the pool's
creation record (base/quote mints per pool) -- see "Implications".

## Per-venue

### pump.fun bonding curve — Dune `project = 'pumpdotfun'`
- Program `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`. `create` mints a
  new SPL token (1B supply, 6 decimals) and a bonding-curve PDA
  (`["bonding-curve", mint]`).
- Constant-product curve on virtual reserves: starts at 30 virtual SOL /
  1.073B virtual tokens; 793.1M tokens are actually sellable on the curve.
- Quote is **always SOL**. Denylist rule is correct here (confirmed:
  WSOL is on all 108,124,447 Bronze rows).
- Fees ~1.25% total per trade (protocol + creator). Dune `fee_tier` = 0.01.
- **Graduation**: when the 793.1M curve tokens are sold (~85 SOL raised),
  the curve is closed and liquidity migrates atomically to **PumpSwap**
  (`pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA`), LP burned. Only ~0.6% of
  tokens graduate; median time-to-graduation ~4.4 min (Sep 2025 sample).
- **Dune's `pumpdotfun` = curve trades only.** Post-graduation trades are
  `project = 'pumpswap'`, which Bronze does not ingest.
- Dune decodes the program's swap event, which also carries
  `sol_reserves`/`token_reserves` (curve state) and `user`; the base spell
  exposes those, `dex_solana.trades` exposes `trader_id` and
  `project_program_id` (= the bonding-curve account / pool id).

### Raydium LaunchLab — Dune `project = 'raydium_launchlab'`
- Program `LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj`. Each launch is a
  `PoolState` with fixed `base_mint` (new token, full supply pre-minted
  into base_vault) and `quote_mint`. Curve types: constant-product
  (default), fixed-price, linear. Graduates at ~80% of supply sold
  (`quote_reserve_target`).
- Third-party **platforms** (Bonk.fun, StonkFun, ...) register a
  `PlatformConfig` and brand their own launches on the same program.
- **Quote mint is not fixed.** Historically SOL / USDC / RAY / USD1;
  since **2026-09-07** "any token pair". StonkFun (first partner) pairs
  launches against tokenized stocks (SPYx, NVDAx, QQQx), pre-IPO tokens
  (OPENAI, ANTHROPIC PreStocks), crypto (ZEC, WBTC, HYPE), SOL, **or
  another StonkFun coin**. That's why only ~8.6% of our LaunchLab rows
  involve WSOL; top counterparties are ZEC, SPYx, OPENAI PreStock.
- Graduation: `migrate_to_amm` -> Raydium AMM v4 (legacy) or
  `migrate_to_cpswap` -> Raydium **CPMM**. CPMM-only for launches created
  on/after **2026-08-17**. Dune has CPMM/CLMM as separate `raydium`
  versions -- not in Bronze.
- Dune's `raydium_launchlab_v1.base_trades` *does* know `base_token_mint`,
  `quote_token_mint`, `pool_id`, `platform_name`; the unified
  `dex_solana.trades` drops all but `project_program_id` (= pool id).

### Raydium AMM v4 — Dune `project = 'raydium', version = 4`
- Legacy constant-product AMM, WSOL vaults. Historically the pump.fun
  migration target (until PumpSwap, Mar 2025) and BIA's venue.
- In Aug-Sep 2026 it is **not a launch venue**: pump.fun graduates to
  PumpSwap, LaunchLab to CPMM. Our v4 data is mostly established pairs --
  SOL/USDC alone is ~5.3M of ~8.1M txs (quote/quote -> 0 Silver rows,
  which is why Silver v4 is much smaller than Bronze v4).

### Uniswap v3 on Base — Dune `dex.trades`, `project='uniswap', version='3'`
- Factory `createPool(tokenA, tokenB, fee)`; token0/token1 sorted by
  address (no base/quote notion at all). Concentrated liquidity.
- **Not where Base launches happen in 2026**: Clanker v4 and Zora deploy
  to **Uniswap v4** pools (hooks, WETH default pair); Virtuals agent
  tokens pair against VIRTUAL. Our v3 data is mostly established tokens
  (WETH, USDC, cbBTC, VIRTUAL, ...). Fine for chain-agnostic plumbing,
  weak for "new launch" training data.

## How Dune builds the columns we use (`dex_solana_trades.sql`)
- Unique key: `(tx_id, outer_instruction_index, inner_instruction_index,
  tx_index, block_month)` -- we never selected the indices, hence the
  content-hash `swap_id` workaround in Silver.
- `amount_usd`: ONE value per trade = bought-side notional if the bought
  token is on Dune's trusted-token list, else sold-side, else bought-side,
  each x minute-level `prices.usd`. NULL when neither mint has a price
  (common for exotic quotes; those rows are dropped by Silver). Our
  `to_bronze_row` copies it into both `buy_amount_usd` and
  `sell_amount_usd`, so Silver's `price_usd = amount_usd / amount` is a
  valid per-token USD price for either side.
- `trade_source` = `'direct'` or the outer program that CPI'd the swap
  (aggregator/bot router) -- a bot-vs-human signal. Not selected.
- `trader_id` = signer / event user. Not selected.

## Implications (see docs/decisions.md, 2026-09-22)
1. **Right-censoring at graduation.** Bronze has pumpdotfun + LaunchLab
   curve trades but not PumpSwap / CPMM. A token that graduates -- the
   rare winners -- stops trading *in our data* at the exact moment it
   succeeds, which looks identical to a token that died. Any
   fixed-horizon return label is biased toward the losers. Must ingest
   the post-graduation venues (at minimum for mints that launched on our
   curves) before labels are trustworthy.
2. **The quote denylist is the wrong abstraction for launchpads.**
   It's correct for pump.fun (always SOL) and acceptable for mature AMMs,
   but LaunchLab's quote is per-pool and can be any token, including
   another launched coin. Correct rule: token-of-interest = the pool's
   **base mint from its creation record**. Cheap to get: one row per
   pool (LaunchLab pool -> base/quote/platform; PumpSwap
   `pumpswap_solana.pools` -> baseMint/quoteMint), not a re-pull of trades.
3. **Launch time is needed and not in Bronze.** Gold's point-in-time
   features ("N minutes since launch") need the create timestamp; the
   first trade in our window is wrong for anything launched before
   2026-08-14 (left-censoring).
4. **Columns worth adding to future Dune pulls**: `project_program_id`
   (pool), `trader_id`, `trade_source`, `outer_instruction_index`,
   `inner_instruction_index`, `tx_index`. Published work on pump.fun
   (arXiv 2602.14860) finds trade intensity and bot-vs-human share are
   the strongest graduation predictors -- both need trader/trade_source.
5. **Venue list needs revisiting for the goal** ("score new launches"):
   raydium_v4 and uniswap_v3_base are not launch venues in 2026. The
   launch venues are pump.fun curve -> PumpSwap, LaunchLab -> CPMM on
   Solana; Uniswap v4 (Clanker/Zora) on Base.

## Sources
- pump.fun program docs: https://github.com/pump-fun/pump-public-docs/blob/main/docs/PUMP_PROGRAM_README.md
- pump.fun bonding curve: https://pump.fun/docs/bonding-curve
- Raydium LaunchLab: https://docs.raydium.io/products/launchlab , /accounts.md , /bonding-curve.md
- LaunchLab CPMM-only graduation (2026-08-17): https://solanacompass.com/news/raydium-launchlab-mandates-cpmm-only-graduation-and-locks-creator-lp-at-token-migration
- LaunchLab any-pair (2026-09-07): https://crypto.news/raydium-launchlab-adds-support-for-any-token-pair-on-solana/
- StonkFun on LaunchLab: https://docs.bitquery.io/docs/blockchain/Solana/stonkfun-api/
- LaunchLab migrate instructions: https://docs.bitquery.io/docs/blockchain/Solana/launchpad-raydium/
- Clanker v4 on Uniswap v4: https://pool.fans/clank
- Dune spellbook (dex_solana_trades.sql, pumpdotfun_solana_base_trades.sql,
  raydium_launchlab_v1_*.sql, pumpswap_solana_pools.sql):
  https://github.com/duneanalytics/spellbook/tree/main/dbt_subprojects/solana/models/_sector/dex
- Pump.fun graduation study: https://arxiv.org/html/2602.14860v1
- Mint identities: ZEC https://solscan.io/token/A7bdiYdS5GjqGFtxf17ppRHtDKPkkRqbKtR27dxvQXaS ,
  SPYx https://solscan.io/token/XsoCS1TfEyfFhfvj8EtZ528L3CaKBDBRqRapnBbDF2W ,
  OPENAI PreStock https://solscan.io/token/PreweJYECqtQwBtpxHL171nL2K6umo692gTm7Q3rpgF
