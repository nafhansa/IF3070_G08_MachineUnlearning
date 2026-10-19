"""stochastic hill climbing (bagian Nafhan).
Lihat src/result.py (RunResult)
Config: max_iterations; opsional: record_snapshots.
History per entry: iteration, objective, candidate_objective, accepted, improved, move, reason, current_state.
"""

from __future__ import annotations

from src.result import RunResult


def hill_climbing(problem, config: dict, seed: int) -> RunResult:
    """satu random.Random(seed) per run, hanya pakai src.core dan src.search_space."""
    raise NotImplementedError("TODO: implementasi stochastic hill climbing")
