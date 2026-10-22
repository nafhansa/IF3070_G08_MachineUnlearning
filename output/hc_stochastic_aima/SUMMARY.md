# Ringkasan eksperimen HC_STOCHASTIC_AIMA (dihasilkan otomatis)

- problem: `configs/problem.json` (synthetic-2026; synthetic/demo instance (bukan data resmi))
- config: `{"max_iterations": 1000, "selection": "proportional", "record_snapshots": true}`
- final objective: mean=875.00, std=81.46, min=799, max=961
- runtime search (s): mean=0.7523, std=0.2090

| algorithm | seed | initial_objective | final_objective | best_found_objective | final_total_weight | final_loaded_count | iterations | runtime_seconds | stuck_event_count | local_optimum_check_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| hc_stochastic_aima | 3401 | 720 | 865 | 865 | 494 | 16 | 7 | 0.5528 | None | 0.0 |
| hc_stochastic_aima | 3402 | 642 | 799 | 799 | 496 | 16 | 9 | 0.7345 | None | 0.0 |
| hc_stochastic_aima | 3403 | 592 | 961 | 961 | 478 | 16 | 13 | 0.9697 | None | 0.0 |
