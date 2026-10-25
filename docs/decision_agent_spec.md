# Decision agent — spec v0.1 (2026-10-22, draft for Jack's review)

LAWRENCE's third part: the component that turns model scores into trades and manages them. This spec
covers the **decision core** (model side, built now) and the **runtime + Isomux agent** (product side,
requested via `docs/interfaces.md` R5). No real money until the gates in §8 pass.

## 1. Goals and non-goals
- **Goal:** one deterministic, tested engine that, given live 1-minute bars, makes exactly the decisions the
  backtest would make (same features, same cutoff, same one-position-per-token rule, same exits), and
  records everything needed to measure live vs. backtest (latency, slippage, fill parity).
- **Goal:** model-agnostic. The engine takes a `ScoringContract` (v1 today; v2 / v2-R1 / a future social
  model later) and a `Broker` (paper today; live later).
- **Non-goal:** an LLM in the trade loop. LLMs supervise, explain and can pause; they never decide or sign.
- **Non-goal (for now):** sub-second execution. LAWRENCE trades on closed 1-minute bars; the early-curve
  results (v4-1) showed that game belongs to slot-level snipers.

## 2. Ownership (per `docs/interfaces.md`)
| Piece | Owner | Location |
|---|---|---|
| Decision core: state machine, risk gate, order intents, audit records | Model side | `src/decision/` |
| Scoring contracts (feature code, frozen model, cutoff) | Model side | `src/serving/`, `contracts/` |
| Replay harness + parity test (engine vs. backtest on Silver bars) | Model side | `src/decision/replay.py`, `tests/` |
| Paper broker (simulated fills with costs) | Model side | `src/decision/brokers.py` |
| Live market-data feed, bar building in real time, scheduling | Product side | `runtime/` |
| Live broker (wallet, swaps), key management | Product side, **later** | `runtime/` |
| Isomux "trader" agent, dashboards, controls | Product side | `rooms/`, `app/` |

## 3. Interfaces (Python, model side)
```python
class ScoringContract(Protocol):
    version: str                      # "v1", "v2-R1", ...
    cutoff: float                     # frozen at training
    def assess(self, token: TokenState, bars: pd.DataFrame, as_of: datetime) -> Assessment | None: ...
    # Assessment = {score, features (dict), row_ok: bool, reason_if_not}

class Broker(Protocol):
    def quote(self, token_id: str, side: str, notional_usd: float) -> Quote: ...       # expected fill price
    def submit(self, intent: OrderIntent) -> Fill: ...                                  # paper: simulated
    def mark(self, token_id: str) -> float | None: ...                                 # last price

class Engine:
    def __init__(self, contract: ScoringContract, broker: Broker, risk: RiskConfig, clock: Clock, log: AuditLog): ...
    def on_bar_close(self, token_id: str, bars: pd.DataFrame, now: datetime) -> list[Event]: ...
    def on_tick(self, now: datetime) -> list[Event]: ...   # exits (barriers, 6h timeout), risk checks
    def pause(self, reason: str) / resume(self) / kill(self, reason: str)
```
`ContractV1` wraps `serving.contract_v1` (`select_row`, `compute_features`, `to_model_frame`, `load_model`,
`verify_startup`). v2 needs a v2 live contract (flow stats from live trades + the frozen metadata-host map);
that is a separate model-side task, started only if v2 or v2-R1 passes the holdout.

## 4. Per-token state machine
```
WATCHING ──(score ≥ cutoff, risk gate OK)──> ENTERING ──fill──> OPEN
   ▲                                              │no fill          │ high ≥ 2× entry  → EXIT_TARGET
   │                                              ▼                 │ low ≤ 0.5× entry → EXIT_STOP
   └────────────── COOLDOWN <──── CLOSED <── EXITING <──────────────┘ 6h elapsed      → EXIT_TIMEOUT
```
- Assess only at contract assessment times (closed bars, rule 7: ≤ 18h after graduation).
- One open position per token (as in training); a token re-enters WATCHING only after its trade resolves.
- Exits use the same barrier logic as the label (`resolve_outcome`): a bar touching both levels = stop.
- Every transition writes an audit record (§6).

## 5. Risk gate (checked before every entry; any failure = no order, logged with the reason)
| Rule | Default (paper) | Why |
|---|---|---|
| Position size | fixed $ notional per trade (paper: $100) | no compounding until edge is shown |
| Max open positions | 10 | caps correlated exposure (R1 opened ~22/day in dev) |
| Max entries per day | 30 | guards against a scoring bug firing on everything |
| Daily loss limit | −10 × position size → pause until next UTC day | a bad day can't become a bad week |
| Quote sanity | skip if quoted slippage > 2× the contract's assumed slippage | the model's EV assumed 50 bps/side |
| Data freshness | skip if the token's last bar is > 2 min old or the feed lags > 60 s | stale prices = fake signals |
| Startup check | `verify_startup` must pass or the engine won't start | live model = frozen model |
| Kill switch | `kill()` closes all positions at market and stops | human / supervisor override |

## 6. Audit log (one row per event; also the shadow log in `interfaces.md`)
`event_id, ts, token_id, event (assess|skip|enter|fill|exit|pause|kill), contract_version, cutoff, score,
features (json), reason, quote_px, fill_px, slippage_bps, latency_ms (bar close → decision → fill),
position_id, barrier_outcome, r_gross, r_net`. Resolved trades become fresh out-of-sample rows.

## 7. Replay harness + parity test (the first deliverable)
Feed Silver's historical 1-minute bars through the engine minute by minute, with `PaperBroker` filling at the
contract's assumed price. **Pass = the engine's trades equal the backtest's trades** (same tokens, entry
times, exits, returns) on a fixed historical window, e.g. the v2-R1 dev trades. This proves the live code
path makes the research decisions before any live data is involved.

## 8. Gates before real money (all required, in order)
1. Replay parity passes (§7).
2. A model passes the pre-registered holdout (`PREREG.md` + amendment 1, after 2026-11-02).
3. Live paper shadow for a pre-registered window passes (`CONTRACT.md` §11 style), with measured
   slippage and latency inside the assumptions.
4. Jack's explicit written go-ahead, starting at the smallest size.

## 9. The Isomux supervisor agent (product side; request R5)
An LLM agent in the office that **reads** the engine's state and audit log and **can call** only
`pause`, `resume`, `kill` and report endpoints. Jobs: explain each trade in plain language; flag
anomalies (slippage drift, feed gaps, losing streaks, risk-gate trips); a daily report (trades, P&L vs.
backtest expectation, parity drift); answer "why did/didn't we buy X?". It never places or sizes orders.

## 10. Build order (model side)
1. `src/decision/` types + state machine + risk gate + `PaperBroker` + audit log, with unit tests.
2. `ContractV1` adapter + replay harness + parity test on Silver bars (free; Databricks notebook).
3. R5 request to the product side (runtime feed + Isomux agent).
4. v2 live contract — only if v2 / v2-R1 passes the holdout.

## Open questions for Jack
- Paper position size and limits in §5: fine as defaults?
- Which contract to run in paper mode first: v1 (has a live contract today) or wait for the holdout winner?
