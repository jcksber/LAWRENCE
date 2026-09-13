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
