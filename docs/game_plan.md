# LAWRENCE: Feasibility & Game Plan
### An ML-driven evolution of BIA — DEX + social-data trading model

**Scope note:** LAWRENCE is a clean, separate build — none of BIA's TypeScript code carries over. BIA's *history* (what it does, how it decides, where it falls short) is studied purely as context: a documented baseline to understand and ultimately beat, not a codebase to extend.

**Update — chain/platform-agnostic, not Solana-centric:** the sections below were written while grounding the idea in BIA's own venue (Solana/Raydium) for a clean before/after comparison. That's no longer the framing. LAWRENCE targets a chain- and platform-agnostic design from Phase 1 onward — Solana/Raydium is one `Venue` among several (a second, on a different chain, is already in the Phase 1 scaffold), and Solana-specific execution mechanics like Jito bundle tips are explicitly *not* a design focus. Read the DEX-venue-specific details below (§3.1, §6) as "true of the first venue we tried," not as the target architecture.

---

## 0. TL;DR

- **What's feasible:** A Databricks medallion pipeline that ingests Solana DEX/liquidity-pool data + crypto social data, engineers features, and trains a model that scores new token launches for expected risk/return. This is a legitimate, portfolio-worthy data engineering + ML project, and it builds directly on your blockchain-data-engineering project and Databricks Associate cert.
- **What's not feasible:** "Guarantee an average profit of more than 50%" on memecoin/launchpad trades. No model — yours, a hedge fund's, or otherwise — can *guarantee* a return in an adversarial, thin-liquidity, largely-scam-saturated market. Below I explain why, and propose a metric you actually can chase and prove out (expected value per trade, win rate, max drawdown, Sharpe/Sortino) instead of a guarantee.
- **Biggest practical blocker:** Twitter/X's official API no longer has a usable middle tier for historical data (it's 7-day pay-per-use or ~$42K+/mo Enterprise). You'll want a crypto-specific social data vendor (LunarCrush, Santiment) instead of building on raw X API access.
- **BIA today:** Your existing bot is a pure latency sniper — it watches Raydium for new-pool-creation logs and buys/sells on fixed rules (buy amount, Jito tip, ~10 sell retries at 500ms). It has zero signal beyond "this pool is new." LAWRENCE's whole value-add is replacing "buy everything new" with "buy the subset of new things that a trained model says are worth the risk."

---

## 1. What BIA actually does today (grounding)

I pulled your repo (`jcksber/BIA`, package name `ts-bia`, "Solana trading bot using Raydium LP V4"). Confirmed from `package.json` and `src/app.ts`:

- **Stack:** TypeScript, `@solana/web3.js`, `@raydium-io/raydium-sdk`, `@coral-xyz/anchor`, `jito-ts` (for bundled, tip-prioritized transaction landing), `tweetnacl` (wallet signing), `ws`/`express`.
- **Strategy:** Watches Raydium fee-account logs for `init_pc_amount` (a brand-new pool being created). On detection, buys immediately with a fixed `BUY_AMOUNT` of SOL, paying a Jito tip (`TIP_LAMPORTS`) to jump the queue. Sells by polling the resulting token balance every 500ms for up to ~10 attempts.
- **No fundamental or social signal at all.** It doesn't know if the token is a rug, who the deployer is, whether liquidity is locked, or anything about the project. It's a race — "be one of the first buyers of anything new" — which is a latency game against other bots, not a prediction game.
- Your own code notes mention wanting distributed nodes (Frankfurt/NYC/Amsterdam/Tokyo) because a single machine is "too slow" — confirming this is fundamentally a speed arms race today, not a data-driven one.

**LAWRENCE's job**, as I understand your intent, is to turn this from "buy everything, fast" into "buy the small subset of things a model believes are good bets, using historical DEX + liquidity + social data as training signal." To be explicit: this means understanding BIA's strategy and its blind spots well enough to design against them — not importing or building on top of its actual code. LAWRENCE gets its own repo, its own stack choices (whatever Python/Databricks tooling fits an ML pipeline best, not necessarily TypeScript), and no dependency on `ts-bia`.

---

## 2. Why "guarantee >50% average profit" isn't the right target

I want to be straight with you on this rather than nod along, because it'll save you months of chasing something that isn't buildable:

- **No one can guarantee a return in a market they don't control.** Crypto launchpad tokens are adversarial: other bots, insiders, and the token deployer are actively trying to extract value from anyone who buys early. A "guarantee" implies the model has solved an adversarial, non-stationary system — that's not a data engineering or ML problem, it's a claim no legitimate quant fund makes either (they target risk-adjusted edge, not guarantees).
- **Base rates are brutal.** Public data on Solana memecoin/launchpad tokens consistently shows the large majority go to zero or near-zero within days (rug pulls, abandoned liquidity, failed momentum). Any backtest that shows a smooth 50%+ average return is almost certainly suffering from survivorship bias (only looking at tokens that *did* pump), look-ahead bias (using information not available at decision time), or is overfit to a specific historical window that won't repeat.
- **"Average" profit is a misleading target in fat-tailed, high-variance outcomes.** A handful of 50x winners can make an "average return" look great while most trades lose money — which is a real dynamic in this space, but "average profit" without also constraining variance/drawdown/hit-rate is not something you can respons­ibly optimize for or trust.
- **What's actually achievable and provable:** a model with a *positive expected value per trade* after fees/slippage, validated out-of-sample and in paper trading, with known win rate, average win/loss size, max drawdown, and a hard risk budget. That's a real, gradeable data science project. "Guaranteed 50%+" is not — and no one you'd want to work for (including Databricks) would expect that framing from you.

**Recommendation:** Reframe LAWRENCE's success metric from "guarantee 50%+ average profit" to something like: *"Achieve positive expected value per trade with Sharpe > X and max drawdown < Y%, validated via walk-forward backtest and live paper trading before any real capital is used."* I'll build the plan below around that framing — it's the version of this project that's both real and a much stronger portfolio piece.

---

## 3. Data feasibility, source by source

### 3.1 DEX / liquidity pool data — feasible, several good options
| Source | What it gives you | Notes |
|---|---|---|
| **Bitquery Solana DEX API** | Real-time + historical (archive) DEX trades, OHLCV candles (1s–1d), pool/liquidity snapshots, trader addresses | Plans start ~$49/mo; real-time window ~30 days, older data via `archive` dataset; daily OHLC back to June 2024 |
| **GeckoTerminal / DexScreener APIs** | Live prices, pool metadata, new-pool feeds across most Solana DEXs incl. Raydium, Orca, pump.fun-style launches | Good for near-real-time signal generation, thinner on deep history |
| **Birdeye, Defined.fi** | Token/pool analytics, holder data, real-time charts | Defined.fi is more of a trading terminal; check API terms before scraping vs. official endpoints |
| **Direct Solana RPC / Geyser plugins / your own validator feed** | Ground-truth, lowest latency, no vendor limits | Highest engineering effort — this is closer to what BIA already does (raw log subscriptions) |
| **Your existing Bronze/Silver/Gold pipeline** | You already built BTC/ETH ingestion on Databricks | Solana wasn't included (AWS Public Blockchain doesn't have it, BigQuery's free Solana set is frozen) — you'll need Bitquery/RPC/Geyser as the new Bronze source for Solana specifically |

A note on the specific venues you named: **ponsfamily.com/launchpad** appears to run on a very new chain/launchpad ecosystem (multiple "is it legit" articles are already circulating about it), and **Defined.fi** is a solid, more established trading-data platform. Very new/niche launchpads are exactly where API support and historical depth are worst and rug-pull rates are highest — I'd treat them as a *later* target once LAWRENCE is validated on higher-volume, better-instrumented venues (Raydium, pump.fun, Orca), not the first data source.

### 3.2 Twitter/X data — the real blocker, but solvable differently
As of 2026, X's official API has no usable middle ground for this use case:
- **Pay-per-use (default):** ~$0.005/tweet read, **7-day** historical window only — not enough to build training data tied to token launches from months ago.
- **Enterprise:** full-archive search, but **$42K+/month**, direct sales only.
- The old $200/mo Basic and $5,000/mo Pro tiers (which used to bridge this gap) are closed to new signups.

**Practical path:** don't build on raw X API. Use a **crypto-native social data vendor** instead — **LunarCrush** or **Santiment** are the two most established; both already aggregate X/Twitter (and Telegram/Reddit/Discord in some cases) mentions, sentiment scores, and social-volume time series specifically for crypto tickers/contracts, with historical backfill included in their paid tiers. This is very likely cheaper *and* higher-quality than trying to reconstruct sentiment from raw tweets yourself, since they've already solved bot/spam filtering, which is a serious problem in crypto Twitter.

If you specifically need contract-address-level, pre-launch chatter (e.g., "someone tweeted about a project a day before launch," which is the scenario you described), that's a narrower, keyword/entity-driven search problem — LunarCrush and Santiment both support ticker/topic tracking, but you may need to supplement with your own lightweight scraper for very fresh/low-follower accounts they haven't indexed yet. Treat that as a stretch goal, not a Phase 1 dependency.

### 3.3 Labels — the hardest part, and worth deciding early
You need a clear, leak-free definition of "good trade" before you can train anything:
- Binary: token is up >X% at T+N minutes/hours vs. did not rug/go to ~0
- Regression: realized return if bought at pool creation and sold at a fixed rule (e.g., T+1hr, or first liquidity-pull event)
- Multi-class: rug / dead / modest gain / big winner

Whichever you pick, **the label must only use information available at decision time** for features (no future liquidity, no future social volume) — this is the single most common way these projects quietly cheat in backtest and then fail live.

---

## 4. Feasibility of the medallion architecture on Databricks

This part is genuinely straightforward for you given what you've already built:

- **Bronze:** raw ingested data, append-only — Solana DEX trade/pool events (Bitquery/RPC), raw social posts/sentiment scores (LunarCrush/Santiment), one table per source, minimal transformation.
- **Silver:** cleaned, deduplicated, timestamp-aligned, joined by token mint address + time window. This is where you solve clock-skew between on-chain events and social posts, dedupe re-orgs, and filter obvious spam/bot accounts.
- **Gold:** feature tables at the token/launch level — pool age, initial liquidity, liquidity depth over time, holder concentration, deployer wallet history (has this wallet rugged before?), social velocity (mentions/min in first N minutes), sentiment score, and the label from §3.3. This is your model-ready table, same shape as the `whale_transactions` Gold table you already built for the anomaly-detection project.

You already have the CI/CD (Asset Bundles + GitHub Actions), job orchestration, and data-quality-gating *patterns* from the blockchain-de-project — LAWRENCE reuses that Databricks approach (not any BIA code) with a new Bronze source and a new modeling notebook, rather than needing new platform skills. It's fair to think of LAWRENCE as a sibling project to blockchain-de-project, not a branch of BIA.

---

## 5. Modeling & risk realities to design for

- **Class imbalance:** big winners are rare; a naive model will just learn "everything is a loser," which is technically high-accuracy and useless. Plan for precision/recall tuning, not raw accuracy.
- **Non-stationarity:** scammer tactics, bot competition, and market regime all shift month to month. A model trained on Q1 data can degrade fast — you'll want scheduled retraining and drift monitoring, not a train-once model.
- **Backtest realism:** simulate actual slippage, DEX fees, Jito tips, and the fact that by the time your model decides, faster bots (including your own BIA logic) may have already moved the price. A backtest that ignores execution latency will look far better than live performance.
- **Walk-forward validation only:** never a single random train/test split on time-series token data — use rolling/expanding windows so you're always testing on data the model couldn't have seen.
- **Baseline first:** before any ML, build a simple rules-based baseline *within LAWRENCE's own codebase* — one baseline that mirrors BIA's actual rule ("buy everything new") purely as a benchmark to beat, and a slightly smarter one ("buy new pools with liquidity > X and no prior rugs from this deployer"). This is a fresh, small implementation for backtesting purposes, not a fork of BIA. LAWRENCE only has to beat that baseline, not an imaginary 50%+ guarantee.

---

## 6. Phased execution plan

**Phase 0 — Scope & guardrails (1 week)**
Pick the target venues (start with Raydium + pump.fun-style launches, established and well-instrumented), define the label, define the *real* success metric (EV/trade, Sharpe, max drawdown — not "guarantee"), and decide the risk budget (max % of capital per trade, daily loss limit, kill switch conditions) before writing any pipeline code.

**Phase 1 — Data ingestion (Bronze), 2–3 weeks**
Stand up Solana DEX ingestion (Bitquery and/or direct RPC/Geyser) as a new Bronze source alongside your existing BTC/ETH pipeline pattern. Stand up a social-data Bronze source via LunarCrush or Santiment. Get both landing reliably on a schedule before touching modeling.

- **Not yet started, but confirmed essential (2026-09-17):** pool-liquidity-state
  (reserve/depth over time) and token holder/dev-allocation snapshot ingestion.
  DEX trade data alone (what's landing today) can train LAWRENCE to be a
  competent chart-based day trader, but Larry -- the trader LAWRENCE is
  modeled on -- makes surface-level use of liquidity/allocation signal
  (volume is already covered by the trade log; % supply held by
  devs/founders and real liquidity depth are not). Without this data,
  LAWRENCE caps out below that benchmark and can't do real rug-clamping
  (see `gold_spec.md`'s KNOWN GAP) or the holder-concentration/deployer-
  wallet-history features already listed above. Deliberately deferred
  behind an interim trade-log-only rug heuristic for now -- tracked here so
  it isn't lost, not scheduled yet.

**Phase 2 — Silver/Gold + labeling, 2–3 weeks**
Join on-chain and social data by token + time, build the Gold feature table, and implement the label definition from §3.3 with strict point-in-time correctness. This is the highest-leverage phase for your career pivot story — "built a leak-free, point-in-time feature store for a fast-moving adversarial market" is a strong data engineering narrative on its own, independent of trading outcomes.

**Phase 3 — Baseline + first model, 2–4 weeks**
Build the rules-based baseline, then a first ML model (start simple — gradient boosted trees on tabular features before anything fancier). Walk-forward backtest both, with realistic fees/slippage/latency assumptions.

- **In progress (2026-09-17):** entry side only for this pass (see
  decisions.md) — `src/backtest/baselines.py` (two baselines),
  `src/backtest/walk_forward.py` (expanding-window folds + trade
  simulation), `src/models/entry_model.py` (LightGBM regressor on
  `y_entry`), orchestrated by `notebooks/04_entry_baseline_and_model.py`
  with MLflow logging. Exit's baseline+model pass is a follow-up, not
  forgotten — see decisions.md for why entry went first.

**Phase 4 — Paper trading / shadow mode, 3–4 weeks minimum**
Run LAWRENCE live, generating buy/sell signals against real-time data, but with **no real capital**. Compare live signal quality to backtest expectations — this is where most trading-bot projects discover their backtest was lying to them, and you want to find that out for free.

**Phase 5 — Small-capital pilot, ongoing**
Only after Phase 4 holds up: deploy with a small, capped amount of real capital, strict per-trade and daily loss limits, and automatic kill switches. Treat this as validation, not income.

**Phase 6 — Iterate**
Scheduled retraining, drift monitoring, and only then consider expanding to newer/riskier venues like Pons-style launchpads once the core system is proven on better-instrumented ones.

---

## 7. How this fits your pivot

This project slots in well after you wrap up the blockchain-de-project write-up: it reuses your Bronze/Silver/Gold + CI/CD scaffolding, adds a second real-time streaming source (good prep for Databricks ML Associate topics like feature stores and model lifecycle), and gives you a much stronger interview story than a "guaranteed profit bot" would — "I built a leak-free ML pipeline for an adversarial, high-noise financial market and rigorously proved out (or honestly disproved) its edge before risking capital" is exactly the kind of judgment a Data Scientist role is testing for.

---

## 8. Caveats worth stating plainly

- I'm not a financial advisor, and nothing here is financial advice — this is a data engineering/ML project plan, not a recommendation to trade.
- Automated trading against other bots on thin-liquidity launchpads carries real capital-loss risk even with a good model; only ever risk money you're fully prepared to lose, and start Phase 5 small.
- Scraping X/social data outside official APIs/vendors can violate platform ToS — stick to licensed vendors (LunarCrush/Santiment) or the official API for anything beyond light, personal-scale research.
