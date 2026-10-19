"""genetic algorithm (bagian Faris).
Lihat src/result.py (RunResult). config wajib: population_size, generations;
Osional: crossover_rate, mutation_rate, elite_count, repair_tries, record_snapshots.
History per generasi: generation, best_fitness, average_fitness, worst_fitness, best_so_far_fitness, ...
Hard constraint dijaga lewat repair/feasibility check (src.core.is_feasible), bukan penalty.
"""

from __future__ import annotations

from src.result import RunResult


def genetic_algorithm(problem, config: dict, seed: int) -> RunResult:
    """satu random.Random(seed) per run, hanya pakai src.core dan src.search_space."""
    raise NotImplementedError("TODO: implementasi genetic algorithm")
