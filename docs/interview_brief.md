# LAWRENCE — Interview Project Brief

Oct 20, 2026 · @jack

## The 30-second pitch

LAWRENCE is a trading-research system I built on Databricks that decides whether to buy newly launched Solana tokens (pump.fun and Raydium LaunchLab), and it is built so it cannot fool itself.

- **What it does:** ingests on-chain trades from Dune into a Bronze/Silver/Gold lakehouse, builds point-in-time features, trains a LightGBM model to predict a 6-hour trade outcome after real fees, and serves it through a frozen, hash-checked contract.

- **Why it is hard:** most launches die within minutes, returns are dominated by a few extreme tokens, fees run 2.5–3.5% per round trip, and it is very easy to leak the future into a backtest.

- **What I am proud of:** the research discipline. Every model is pre-registered, every experiment is logged good or bad, I caught and fixed my own look-ahead bugs, and the final verdict comes from a 21-day holdout no one has looked at.

- **How I built it:** as a solo engineer pairing with an AI coding agent (Claude) through MCP tools, with specs, contracts and tests as the guardrails.

The honest headline: the best development result is about +3% net per trade with a lower bound still below zero, so I have not claimed an edge. Showing that the system refuses to overclaim is the point.

## Architecture

LAWRENCE is a Databricks medallion lakehouse: every layer can be rebuilt from raw on-chain trades, and only a frozen, hash-checked contract reaches evaluation.

```
Dune Analytics: Solana DEX trades and pump.fun events
  (SQL over API, per-day resumable jobs, one-day cost test before every paid pull)
        |
Bronze: raw, append-only Delta tables
  (DEX trades, post-graduation 1m bars, pool metadata, 1.59M launches, early buyers, creator trades)
        |
Silver: token_price_1m
  (1-minute OHLCV per token across curve and post-graduation phases, deduplicated)
        |
Gold: entry_candidates, 7.5M rows over 1.15M tokens
  (point-in-time features and triple-barrier labels; new features only as additive tables)
        |
Research: walk-forward LightGBM  --freeze-->  Frozen contract
  (4 weekly folds, variant log,                (sha256-checked model, manifest,
   block bootstrap)                             parity tests)
                                                     |
Pre-registered holdout, Oct 11 to Nov 1, then shadow trading
  (pass: 95% lower bound > 0 at base costs and mean > 0 at stress costs)
```

LAWRENCE architecture · Dune to holdout

Each Dune pull is chunked by day and logged in a progress table, so a run that hits a credit limit resumes where it stopped instead of paying twice.

## Modeling and the math

The model predicts the after-fee return of one well-defined trade, and the decision rule only acts on the top 2% of scores.

**1. The label: a triple barrier.** At assessment time T, with entry price p_T, the trade exits at the first of three events within 6 hours: price doubles (take profit), price halves (stop), or time runs out.

```math
r = \begin{cases}
+1 & \text{if } 2p_T \text{ is reached before } 0.5\,p_T \\
-0.5 & \text{if } 0.5\,p_T \text{ is reached first (bar close if it gapped through)} \\
\dfrac{p_{T+\tau}}{p_T} - 1 & \text{otherwise}, \quad \tau = 360 \text{ min}
\end{cases}
```

Capping the upside at 2x matters: earlier models looked profitable only because a handful of 100x tokens were counted. The barrier caps that tail, so the target is what a real position can capture.

**2. Costs, per venue.** Every trade pays a fee f and slippage s on both sides:

```math
c = 2 \cdot \frac{f_{\text{venue}} + s}{10^4}, \qquad \text{net} = r - c
```

Fees were verified from venue docs: pump.fun bonding curve 125 bps, PumpSwap after graduation 120 bps, Raydium CPMM 25 bps. Slippage is 50 bps base and 100 bps stress, so a curve round trip costs 3.5% (4.5% stressed).

**3. Model and decision rule.** LightGBM regression on r (300 trees, learning rate 0.05, fixed seed 17). A trade is taken only when the score clears the 98th percentile of that fold's own training scores, with at most one open position per token.

```math
\text{trade at } (i, T) \iff \hat{r}_{i,T} \ge Q_{0.98}\big(\{\hat{r}_j\}_{j \in \text{train}}\big) \ \wedge\ \text{no open position in } i
```

**4. Walk-forward validation.** Four weekly folds over 2026-09-13 to 10-10. Each fold trains only on rows strictly before its test week, so the model never sees its own future.

**5. Uncertainty: a block bootstrap lower bound.** Trades on the same token and day are correlated, so I resample token-days, not trades (10,000 draws). With block b holding n_b trades summing to S_b:

```math
\bar{x}^{*} = \frac{\sum_{b \in B^{*}} S_b}{\sum_{b \in B^{*}} n_b}, \qquad
LB_{\alpha} = Q_{\alpha}\big(\bar{x}^{*}_1, \dots, \bar{x}^{*}_{10000}\big)
```

**6. Multiple testing.** Two models enter the holdout, so each gets its own one-sided α = 0.05 (Bonferroni keeps the family-wide false-positive rate at or below 10%). Within development, I cap the number of variants (about 10 per version) because every extra try raises the chance a lucky result passes.

## Research rigor: how it avoids fooling itself

The most valuable thing I built is a process that kills attractive but fake results before they cost money.

**Pre-registered holdout.** Before any test data existed, I froze two models (v1 and v2) and wrote down the pass rule in contracts/v2/PREREG.md. No data after 2026-10-10 may be pulled or looked at until the holdout pull after Nov 2.

- Window: 2026-10-11 to 11-01 (21 days).

- PASS: 95% one-sided lower bound \> 0 at base costs AND mean \> 0 at stress costs.

- Fewer than 100 trades = INCONCLUSIVE; more than 20% excluded days = INVALID.

**A variant budget and log.** Each version gets about 10 experiments. Every one is declared with its adoption rule before it runs and logged in docs/v3_variant_log.md, including the failures.

**Bugs I caught in my own work** (each is a good interview story):

| Bug                              | Symptom                                         | Fix                                                                                                                 |
|----------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Fat-tail illusion (Oct 2)        | Model showed +4% to +9.5% per trade             | Stress tests showed the top 1% of trades produced 251–408% of profit; added the 2x barrier cap                      |
| Daily top-k look-ahead (rule R7) | +8.0% per trade, 4/4 folds positive             | It ranked against the whole day, including later candidates; the causal version (R9, prior-24h threshold) collapsed |
| Missing training horizon         | Live model could score rows it never trained on | Added rule 7: only score up to 18h after graduation, matching training                                              |
| Data source coverage             | Only 248 launches/day found                     | A second create instruction existed; the event table had all 30,783                                                 |
| Survivorship in features         | Early-buyer stats only exist for graduates      | Excluded them from the pre-graduation model, since they leak "this token graduates"                                 |

**Negative results, reported as-is.** v1 failed its first out-of-sample test. v2 is break-even in development (+0.11% per trade, lower bound −1.45%). Adding creator and early-buyer features (variant 10) raised the mean from +2.93% to +3.16% but improved only 2 of 4 folds, so the pre-declared rule rejected it.

## Specs, contracts and tests

A model is only "shipped" when a frozen contract proves the live code computes exactly what training computed.

- **Frozen contracts.** contracts/v1/ and contracts/v2/ each hold a CONTRACT.md or PREREG.md spec, a manifest.json (feature list, training window, cutoff, categories) and the model file. The model's sha256 is in the manifest, so load_model refuses a model that was changed after the freeze.

- **Startup self-check.** verify_startup re-scores a saved sample of training rows and must match the offline predictions within 1e-9. On Databricks the difference was 0.0.

- **Training/serving parity.** bars_v1.build_bars rebuilds 1-minute bars from raw trades the way the live service would. On a frozen fixture it reproduces Silver on 2,731 of 2,734 bars. The 3 misses fall outside the swap window, and the test documents why.

- **One serving module.** contract_v1.py holds the feature computation, row selection (including the 18h horizon rule), the barrier outcome and the cost function, so the backtest and the live path call the same code.

- **Additive-only schema rule.** New Gold features go in new columns or tables, never edits to frozen v1/v2 features or the label, with a guard test.

- **Tests.** pytest suites per module (contract, bars, parity fixture, shadow evaluation, causal rules, creator-flow SQL). One test asserts that the generated Dune SQL never reads a trade at or after the holdout start.

- **Written decisions.** docs/decisions.md (2,600+ lines) records every choice with its date and reason, and docs/interfaces.md is the spec between the model and the separate web-app ("trade room") project.

## How I use AI to code

I treat the AI agent as a fast, careful junior engineer: it writes the code, and I own the decisions, the runs and the commits.

**Division of labor**

| Me                                                                                   | Claude (AI agent)                                                     |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Set goals and constraints (e.g. "no data after 10-10", "additive Gold columns only") | Drafts notebooks, modules, SQL and tests directly in my repo          |
| Run notebooks on Databricks and watch credits                                        | Reads my pasted outputs and diagnoses errors                          |
| Review diffs, commit and push to git branches                                        | Keeps the decision and variant logs current                           |
| Approve spending (Dune credits) and anything irreversible                            | Proposes options with cost estimates, then pushes back when I'm wrong |

**The loop**

1.  I state the goal and constraints.

2.  The agent writes the files on my machine and runs the unit tests it can run locally.

3.  I sync the branch to Databricks and run the notebook.

4.  I paste the output back; the agent interprets it, fixes bugs and updates the logs.

5.  I commit when a step is done.

**Guardrails that make it trustworthy**

- Standing instruction: correct me rather than agree with me, and never frame results as guaranteed returns.

- Pass/fail rules are written before an experiment runs, so neither of us can move the goalposts after seeing results.

- Cheap first: every paid data pull starts with a one-day cost test (e.g. 28 credits per day measured before a ~1,000-credit pull).

- Small, reviewable changes, one notebook cell at a time, with tests that encode the rules (like the holdout-cutoff SQL test).

**What I learned:** the agent is excellent at volume and recall (2,600 lines of decision history stay consistent), but its biggest value was catching problems I would have shipped, like the R7 look-ahead and the missing create table.

## MCPs and tooling

MCP (Model Context Protocol) is the standard way an AI agent gets tools; I use it so the agent can work on my real repo and real data sources with my approval on each risky step.

| MCP / tool                                 | What the agent uses it for                                                           | My control                                                    |
|--------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Device bridge (shell + files on my laptop) | Edits notebooks and modules in my git repo, runs pytest, searches code               | Folder-scoped; deletes need my approval; I do the git commits |
| Built-in browser                           | Read-only research, e.g. the X accounts my co-user follows and wallet-tracker sites  | I sign in myself; each site load is approved                  |
| Claude Projects + Docs                     | Shared project docs, such as the handoff prompt for the web-app chat, and this brief | Docs are visible to me and editable                           |
| Memory                                     | Keeps my preferences and project context across sessions                             | Scoped to the project                                         |

**Not MCP, but part of the toolchain:** Dune Analytics SQL over its API (called from Databricks notebooks, with resumable per-day jobs and a progress table), Databricks with Unity Catalog and Delta tables, LightGBM, pandas and PySpark, git branches per version (social, v3).

**Two agents, one interface spec.** A second AI chat owns the product side (a web "trade room" on Isomux). The two never share code; they talk through numbered requests (R1–R4 so far) in docs/interfaces.md, the same way two teams would.

## Results, lessons and likely questions

No edge is proven yet, and the system is built to say so; the real verdict lands after the Nov 2 holdout pull.

| Model / rule (dev, 4 folds)         | Trades | Mean net per trade | 95% lower bound | Status                  |
|-------------------------------------|--------|--------------------|-----------------|-------------------------|
| v2, original rule (top 5%)          | 2,440  | +0.11%             | below 0         | Frozen, in holdout      |
| v2, rule R1 (top 2%)                | 553    | +2.93%             | −1.54%          | Best dev result         |
| v2 + creator / early-buyer features | 536    | +3.16%             | −1.46%          | Rejected (2/4 folds)    |
| R7 daily top-5 (look-ahead)         | 124    | +8.04%             | not valid       | Discarded as non-causal |

**What's next:** a v4 that changes the universe instead of adding features: enter on the bonding curve 1–15 minutes after launch, with creator history computed for all 1.59M launches. It must beat a strict pre-declared bar, including a one-bar-delayed entry, before it joins the holdout.

**Lessons**

- In fat-tailed markets the mean lies: always check the median, results without the top tokens, and capped returns.

- Leakage hides in rules, not just features: a ranking "per day" can quietly use the future.

- Fees decide everything at this scale: 3.5% round trip wipes out most raw signal.

- Write the pass rule down before the experiment, and keep a budget of tries.

**Likely interview questions**

- **Why not just report the +8% result?** It used information not available at decision time. The causal version collapsed, which proved it.

- **Why a block bootstrap?** Trades on one token in one day move together, so treating them as independent would overstate confidence.

- **How do you know live matches training?** A hash-checked model, a startup re-scoring check (diff 0.0) and a parity fixture that rebuilds bars from raw trades.

- **What did AI change about how you work?** I spend my time on design, review and judgment; the agent handles volume and is told to challenge me. Specs and pre-declared rules keep both of us honest.

- **What would you do with more time or money?** Wallet-level smart-money features, longer post-graduation price history, and a live shadow run before risking capital.
