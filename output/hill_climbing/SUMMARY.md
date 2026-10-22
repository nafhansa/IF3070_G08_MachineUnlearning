# Ringkasan eksperimen HC (dihasilkan otomatis)

- problem: `configs/problem.json` (synthetic-2026; synthetic/demo instance (bukan data resmi))
- config: `{"max_iterations": 2000, "record_snapshots": true}`
- final objective: mean=901.33, std=59.07, min=844, max=962
- runtime search (s): mean=0.2766, std=0.0198

| algorithm | seed | initial_objective | final_objective | best_found_objective | final_total_weight | final_loaded_count | iterations | runtime_seconds | stuck_event_count | local_optimum_check_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| hc | 3001 | 663 | 898 | 898 | 492 | 16 | 2000 | 0.2598 | None | 0.0 |
| hc | 3002 | 698 | 962 | 962 | 486 | 15 | 2000 | 0.2715 | None | 0.0 |
| hc | 3003 | 691 | 844 | 844 | 485 | 15 | 2000 | 0.2984 | None | 0.0 |
