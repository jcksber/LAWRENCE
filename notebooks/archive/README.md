# Archived notebooks

Moved here 2026-10-22 to cut noise. Nothing was deleted: every result these produced is logged
(see `docs/experiments.md`), and they still run from this folder (relative paths updated to `../../`).
The active pipeline, contracts and holdout notebooks stay in `notebooks/`.

| Notebook | What it was | Result / where logged |
|---|---|---|
| 00_bronze_dex_ingestion | First DEX ingestion (pre-Dune backfill) | Superseded by 00c |
| 00d_dune_cost_test | Dune credit test, trade backfill | decisions.md |
| 00e_dune_graduation_cost_test | Dune credit test, graduation venues | decisions.md |
| 00g_dune_curve_bars_cost_test | Dune credit test, curve bars | decisions.md |
| 00k_creator_cost_test | Dune credit test, creator / early-buyer pull (found the missing create table) | v3_variant_log.md |
| 00l_bronze_creator_flow | Bronze: 1.59M launches, early buyers, creator trades (~1k credits) | v3_variant_log.md |
| 00m_postcrash_cost_test | Dune credit test, hourly post-crash bars | v4_variant_log.md |
| 00n_bronze_postcrash_bars | Bronze: hourly bars, pools aged 24h–14d (~460 credits) | v4_variant_log.md |
| 02c_gold_postcrash_candidates | Gold: post-crash candidates + 72h labels | v4_variant_log.md |
| 03_gold_exit_candidates | Gold exit table (exit model never built) | decisions.md |
| 04_entry_baseline_and_model | First entry model + stress tests (tail-driven, no edge) | decisions.md (2026-10-02) |
| 04b_barrier_model | v1 development: barrier label model | decisions.md |
| 04d_v2_dev | v2 development (chose V2b) | decisions.md |
| 04f_v3_dev_rules | v3: rules R0–R9, variant 10 (C features), exits E1/E2 | v3_variant_log.md — all rejected except R1 → holdout |
| 04g_v4_early_dev | v4 early track (1–15 min on the curve) | v4_variant_log.md — rejected, latency |
| 04h_v4_postcrash_dev | v4 post-crash track (24h–14d) | v4_variant_log.md — rejected |
| 05b_creator_features | Gold: creator + early-buyer features (graduates) | v3_variant_log.md |
| 05c_creator_history_v4 | Gold: creator history for all launches | v4_variant_log.md |

Gold/Bronze tables these created still exist in Unity Catalog (`bronze.pump_*`, `bronze.postcrash_bars_1h`,
`gold.creator_*`, `gold.postcrash_candidates_v4`); none feed a frozen model.
