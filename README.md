# LAWRENCE

A research system for trading newly launched Solana tokens (pump.fun and Raydium LaunchLab), built on
Databricks as a Bronze/Silver/Gold lakehouse — and built so it can't fool itself.

**Status (2026-10-22):** research phase complete. Three frozen models (v1, v2, v2-R1) wait on a
pre-registered 21-day holdout (2026-10-11 → 11-01), scored once after 2026-11-02. Best development result:
**+2.9% net per trade, 90% lower bound −0.6%** — not yet a proven edge, and the repo says so.

## At a glance

| | |
|---|---|
| Data | Dune Analytics SQL (Solana DEX trades, pump.fun events) → Delta tables on Unity Catalog |
| Scale | Gold: 7.5M entry candidates over 1.15M tokens; 1.59M launches; 602k post-crash hourly bars |
| Model | LightGBM regression on a triple-barrier return, walk-forward validated |
| Serving | Frozen, hash-checked contracts with training/serving parity tests |
| Rigor | Pre-registered holdout, a fixed experiment budget, every result logged — wins and failures |
| Tests | 216 unit tests across 25 files |
| Cost | Dune free-tier credits; no paid data |

## Architecture

```
Dune Analytics (SQL over API, per-day resumable jobs, cost-tested before every pull)
   │
Bronze   raw append-only Delta: DEX trades, curve + post-graduation 1m bars, pool metadata,
   │      launches, early buyers, creator trades, social metadata
Silver   token_price_1m — 1-minute OHLCV per token across curve and post-graduation phases
   │
Gold     entry_candidates — point-in-time features + triple-barrier labels
   │      (new features only ever added as new tables; frozen columns never edited)
   │
Research walk-forward LightGBM, 4 weekly folds, variant logs, block bootstrap
   │  freeze
Contract sha256-checked model + manifest + startup re-score check + parity fixture
   │
Holdout  pre-registered window, run once → PASS / FAIL / INCONCLUSIVE / INVALID
```

## The method, briefly

**Label: triple barrier.** From the entry price, exit at the first of: the price doubles, the price halves,
or 6 hours pass.

```math
r = \begin{cases}
+1 & \text{if } 2p_T \text{ is reached before } 0.5\,p_T \\
-0.5 & \text{if } 0.5\,p_T \text{ is reached first (bar close if it gapped through)} \\
\dfrac{p_{T+\tau}}{p_T} - 1 & \text{otherwise}, \quad \tau = 360 \text{ min}
\end{cases}
```

Capping the upside at 2x matters: an earlier model looked profitable only because a handful of 100x tokens
were counted (the top 1% of trades produced 251–408% of profit).

**Costs**, verified per venue: fee f = 125 bps (pump.fun curve), 120 bps (PumpSwap), 25 bps (Raydium CPMM);
slippage s = 50 bps (base) or 100 bps (stress).

```math
c = 2 \cdot \frac{f_{\text{venue}} + s}{10^4}, \qquad \text{net} = r - c
```

**Decision rule.** Trade when the predicted return clears a fixed percentile of the model's own training
predictions (98th for v2-R1, 95th for v2), with one open position per token.

```math
\text{trade at } (i, T) \iff \hat{r}_{i,T} \ge Q_{0.98}\big(\{\hat{r}_j\}_{j \in \text{train}}\big) \ \wedge\ \text{no open position in } i
```

**Uncertainty.** Trades on one token in one day move together, so the lower bound resamples token-days
(10,000 draws). Block b holds n_b trades with summed net S_b:

```math
\bar{x}^{*} = \frac{\sum_{b \in B^{*}} S_b}{\sum_{b \in B^{*}} n_b}, \qquad
LB_{\alpha} = Q_{\alpha}\big(\bar{x}^{*}_1, \dots, \bar{x}^{*}_{10000}\big)
```

Three models share the holdout, so each is tested at alpha = 0.10 / 3 (Bonferroni).

## What was found

Full table: [`docs/experiments.md`](docs/experiments.md). Highlights:

- **Buying everything loses in every window** tested: −7% to −11% net per trade.
- **The model ranks tokens well** (monotone deciles), but the top of the ranking only reaches break-even to
  about +3% after 2.5–4.5% round-trip costs.
- **The first 15 minutes are a latency game.** Filling one 1-minute bar later turned −0.7% into −4.5%;
  that edge belongs to slot-level snipers.
- **Days-later "post-crash" entries lose in every drawdown × rebound bucket.**
- **Exits beyond sell-all-at-2x** (ladders, half-at-2x-rest-at-10x) didn't help: only 7% of 2x trades
  went on to 10x.
- **Serial creators who already graduated a coin are a red flag** (stop-hit rate 25–29% vs 8–11%).

## Mistakes caught (and how)

| Bug | Symptom | Caught by |
|---|---|---|
| Fat-tail illusion | +4% to +9.5% per trade | Stress tests: capped returns, ex-top-1%, median |
| Daily top-k look-ahead (R7) | +8.0%, 4/4 folds | Ranking used later candidates; the causal version collapsed |
| Missing training horizon | Live model could score unseen rows | Contract rule: score only ≤ 18h after graduation |
| Missing create instruction | 248 launches/day instead of ~30k | Cross-check against the CreateEvent table |
| Survivorship | Early-buyer data exists only for graduates | Excluded from pre-graduation models |
| Stale entry price | Early-curve "signal" | Delayed-entry re-pricing |

## Built with an AI coding agent

Developed solo with Claude as a pair programmer through MCP tools (a scoped shell and file access on the repo,
a browser for read-only research, shared project docs). The agent drafted code, SQL and tests and kept the logs;
I set constraints, ran every notebook, approved every paid data pull, reviewed and committed. Guardrails:
pass/fail rules written before each experiment, a one-day cost test before every pull, and a standing
instruction for the agent to push back rather than agree.

## Repo layout

```
src/
  ingestion/   Dune SQL builders + client (resumable, holdout-capped, tested)
  features/    Silver/Gold feature + label code (point-in-time)
  models/      model specs (v1 barrier model, v2_spec)
  backtest/    walk-forward, block bootstrap, holdout evaluators, exit simulator
  serving/     frozen v1 contract: features, bars, outcome, costs
contracts/     v1/, v2/ — frozen models, manifests, PREREG.md, PREREG_AMENDMENT_1.md
notebooks/     active pipeline (00c → 02, 05), contracts (04c, 04e, 04i), holdout (06)
notebooks/archive/   cost tests and finished experiments, indexed in its README
docs/          decisions.md (2,600+ lines), experiments.md, variant logs, specs, interfaces.md
tests/         216 unit tests
```

## Reproduce

```bash
pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider --assert=plain
databricks bundle validate -t dev && databricks bundle deploy -t dev
```

Notebooks run in order on Databricks (00c → 01 → 01b → 02 → 05 → 04c/04e/04i → 06). The Dune API key lives
in the `lawrence` secret scope.

## Not financial advice

This is a research project. Nothing here is a recommendation to trade, and no strategy here has been shown
to make money.
