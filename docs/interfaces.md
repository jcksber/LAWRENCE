# Interfaces: model side ↔ product side

Two workstreams share this repo:
- **Model** (features, labels, models, backtests): `src/features`, `src/models`, `src/backtest`, `src/ingestion`, `notebooks/`, `docs/decisions.md`, `docs/label_spec.md`.
- **Product** (web app, Isomux trade room, live runtime, shadow mode): its own top-level dirs (e.g. `app/`, `runtime/`, `rooms/`).

The product side consumes the scoring contract below and never edits model-side files. It asks for changes under "Requests to model side".

## Scoring contract
See `contracts/v1/CONTRACT.md` (supersedes the table below).

### Summary (v1, NOT approved for real money)

| Item | Current value | Source |
|---|---|---|
| Status | **Shadow only.** Model A failed its pre-registered OOS test (2026-10-16). | docs/decisions.md |
| Model | LGBMRegressor(random_state=17, n_estimators=300, learning_rate=0.05) on `r_up_360m` | notebooks/04b, last cell |
| Features | `models.barrier_model.FEATURE_COLUMNS` (frozen) | src/models/barrier_model.py |
| Feature code | `features.gold_features.build_entry_row` (point-in-time; bars with trade_time ≤ T only) | src/features/gold_features.py |
| Universe | post-graduation rows (`is_postgrad_at_t`), pump.fun + Raydium LaunchLab | |
| Entry | predicted EV ≥ the cutoff frozen at training (train 95th pct; 0.0656 for the 10-03 model); one position per token | |
| Exit | first of +100% / −50% / 6h timeout | src/features/gold_features.py `barrier_outcome` |
| Costs | `backtest.walk_forward.venue_fee_bps` + 50 bps/side slippage | src/backtest/walk_forward.py |

Live features must reuse the Gold code above, not a reimplementation.

## Shadow log → model side
Shadow trades must be loggable as fresh out-of-sample rows: `token_id`, `assessment_time`, every feature in the contract, predicted EV, the cutoff used, the model version, and the realized barrier outcome (`y_up_360m`, `r_up_360m`, `t_up_360m`) once resolved.

## Requests to model side
- **R1 (2026-10-17): scoring contract for shadow mode** → answered in `contracts/v1/CONTRACT.md` (code `src/serving/contract_v1.py`; artifact via `notebooks/04c_export_contract_v1.py`). §11 pre-registration approved 2026-10-17 (outcome-based: 90% lower bound on mean net > 0).
- **R2 (2026-10-17): bar parity + live ingestion** → answered in CONTRACT.md §9 (final log + `shadow_skips_daily`), §12–§16; reference builder `src/serving/bars_v1.py`; fixture via `notebooks/00j_parity_fixture.py` (pending Jack's run). Also fixed a v1 selection bug (§4 rule 7: 18h post-graduation limit).
- **R3 (2026-10-17): R2 follow-ups** → quote USD prices + both band-mode bar outputs added to 00j; 18h rule applied to 04c training rows (cutoff/baselines on the same universe); startup sample + `verify_startup()`. All pending Jack's 00j / 04c runs.

## Status note (2026-10-18)
The live shadow window (CONTRACT v1 §11) is **on hold**: Jack isn't funding the runtime until a model passes the pre-registered **Dune holdout 2026-10-11 → 2026-11-01** (`contracts/v2/PREREG.md`), which tests v1 and v2 on unseen history without any runtime. `SHADOW_START` isn't set. Contract v1 stays frozen and valid. If v2 passes, a v2 live contract (bars with flow stats + launch-time metadata host) will follow as a new request.
