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
