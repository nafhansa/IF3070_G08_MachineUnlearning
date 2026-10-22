# Ringkasan eksperimen HC_RANDOM_RESTART (dihasilkan otomatis)

- problem: `configs/problem.json` (synthetic-2026; synthetic/demo instance (bukan data resmi))
- config: `{"max_restarts": 100, "iterations_per_restart": 1000, "record_snapshots": true}`
- final objective: mean=1072.33, std=29.54, min=1042, max=1101
- runtime search (s): mean=58.1992, std=1.3127

| algorithm | seed | initial_objective | final_objective | best_found_objective | final_total_weight | final_loaded_count | iterations | runtime_seconds | stuck_event_count | local_optimum_check_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| hc_random_restart | 3301 | 583 | 1042 | 1042 | 498 | 16 | 1000 | 56.7095 | None | 0.0 |
| hc_random_restart | 3302 | 739 | 1101 | 1101 | 491 | 16 | 1049 | 58.7016 | None | 0.0 |
| hc_random_restart | 3303 | 582 | 1074 | 1074 | 482 | 16 | 1032 | 59.1865 | None | 0.0 |
