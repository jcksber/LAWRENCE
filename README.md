# LAWRENCE

An ML model that scores new DEX token launches for expected risk/return, trained on historical on-chain DEX/liquidity-pool data (and, from Phase 2 onward, crypto social data). Built on Databricks with a Bronze/Silver/Gold medallion architecture.

**Chain- and platform-agnostic by design.** LAWRENCE isn't a Solana bot — it's built around a `Venue` abstraction (chain + DEX protocol) so a chain or DEX is a config entry, not a rewrite. It started with Solana/Raydium (matching BIA's target, for a clean before/after comparison) but ships from day one with a second venue on a different chain to prove the abstraction actually holds. Speed tricks specific to one chain (e.g. Solana's Jito bundle tips) are treated as an optional, venue-specific execution detail in later phases — never a design assumption baked into the core pipeline.

**Relationship to BIA:** LAWRENCE is a separate, from-scratch project. [`BIA`](https://github.com/jcksber/BIA) (a TypeScript Raydium sniping bot with no ML/signal beyond "this pool is new") is studied only as historical context — a documented baseline strategy to understand and beat. No BIA code is imported here.

Full feasibility writeup and phased plan: see `docs/game_plan.md` (copied in from the original planning doc).

## Current phase

**Phase 1 — DEX-only Bronze ingestion, multi-venue.** `src/ingestion` currently has two live venues: Solana/Raydium and Base/Uniswap-v3, unified into one Bronze schema (see `notebooks/00_bronze_dex_ingestion.py`). Add more chains by adding `Venue` entries, not new code. Social data (Phase 2, via LunarCrush or Santiment) is deliberately deferred until the on-chain pipeline and a first baseline model are validated.

## Target & label (v1)

**Regression** — realized return at a fixed exit rule:

```
return = (price_at_exit - price_at_entry) / price_at_entry
entry  = first fillable price after pool creation
exit   = price at T + HOLD_MINUTES  (placeholder: 60 min — tune once real
         price-decay distributions are visible in Silver/Gold data)
```

See `docs/label_spec.md` for the full point-in-time-correctness rules (this is the part most trading-bot projects get wrong — features must never see data from after the label's exit time).

## Repo layout

```
lawrence/
  src/
    ingestion/    # Bronze: chain-agnostic Venue abstraction + Bitquery adapter
                   # (add a chain/DEX via config, not new code)
    features/     # Silver + Gold: cleaning, joins, feature engineering
    models/       # baseline + ML model training and evaluation
    backtest/     # walk-forward backtest harness (fees, slippage, latency-aware)
  notebooks/       # Databricks notebooks that orchestrate the src/ modules
  resources/       # Databricks Asset Bundle job/pipeline definitions
  tests/           # unit tests (esp. for label point-in-time correctness)
  docs/            # game plan, label spec, decisions log
  databricks.yml   # Asset Bundle root config
```

## Success metric (not a guarantee)

Positive expected value per trade after fees/slippage, validated via walk-forward
backtest and live paper trading before any real capital is used — tracked via
Sharpe/Sortino, hit rate, average win/loss, and max drawdown. See `docs/game_plan.md`
§2 for why "guaranteed" returns aren't the target.

## Setup

```bash
pip install -r requirements.txt
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

(Mirrors the CLI auth / bundle workflow from `blockchain-data-engineering-project`.)
