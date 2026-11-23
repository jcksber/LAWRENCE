# v5 plan — written 2026-11-22, BEFORE any v5 result

After the holdout (all FAIL) and the E3 exit exploration (holdout −5.4% → −0.7% with E3, not a test), v5 asks
one question: **do the entries improve if the model learns the exit we would actually use, with more history
and the state of the whole market?**

## What changes vs v2
1. **Target = E3 net return** (`gold.e3_labels_v5`, notebook 02d): sell half of what's held at 2x/4x/6x/8x of
   entry, the rest at 10x; stop −20% from entry; no time limit (data ends 24h after graduation); net of the
   venue's base costs.
2. **More history:** 09-13 → 11-01 (49 days; flow features only exist from 09-13).
3. **+11 market-context features** (`features.market_context`, hourly, only completed hours before T):
   launches and graduations per 1/6/24h, graduation rate, share of recent graduates that hit 2x within 6h,
   post-graduation volume and trades.

## Development comparison (notebook 04j) — the only two models run
- **B0:** v2 features, old target (`r_up_360m`), top 2% of train predictions, judged on E3 net.
- **v5:** as above, same entry rule, judged on E3 net.
- 6 weekly walk-forward folds (test weeks 09-20 → 11-01); a fold trains only on rows whose E3 outcome was
  known before it starts; one position per token until its E3 exit.
- Caveat stated up front: these days include the already-judged holdout and have been looked at many times.
  This step can only REJECT v5 or promote it to a forward test; it cannot prove an edge.

## Decision rule (fixed now)
Freeze v5 for a forward paper test **only if all hold**: v5 pooled mean net > B0's, v5 95% lower bound > B0's,
v5 beats B0 in ≥ 4 of 6 folds, and v5 pooled mean net > 0. Otherwise stop; no feature or threshold tuning.

## If frozen: the forward test
A frozen v5 contract run by the paper engine on live data from the freeze date, pre-registered like PREREG.md
(window length, minimum trades, PASS = lower bound > 0 at base costs and mean > 0 at stress). No real money.
