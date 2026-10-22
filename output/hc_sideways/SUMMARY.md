# Ringkasan eksperimen HC_SIDEWAYS (dihasilkan otomatis)

- problem: `configs/problem.json` (synthetic-2026; synthetic/demo instance (bukan data resmi))
- config: `{"max_sideways": 100, "max_iterations": 1000, "record_snapshots": true}`
- final objective: mean=1070.33, std=17.01, min=1053, max=1087
- runtime search (s): mean=13.0045, std=0.4904

| algorithm | seed | initial_objective | final_objective | best_found_objective | final_total_weight | final_loaded_count | iterations | runtime_seconds | stuck_event_count | local_optimum_check_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| hc_sideways | 3201 | 521 | 1053 | 1053 | 497 | 15 | 179 | 13.5511 | None | 0.0 |
| hc_sideways | 3202 | 601 | 1071 | 1071 | 498 | 16 | 200 | 12.8593 | None | 0.0 |
| hc_sideways | 3203 | 642 | 1087 | 1087 | 497 | 17 | 191 | 12.6032 | None | 0.0 |
