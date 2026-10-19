"""simulated annealing (bagian Kade).
Lihat src/result.py (RunResult)
Config: max_iterations, initial_temperature,cooling_rate, min_temperature
Opsional: detect_stuck, record_snapshots.
Spek meminta plot e^(dE/T) vs iterasi dan frekuensi stuck di local optimum.
"""

from __future__ import annotations

from src.result import RunResult


def simulated_annealing(problem, config: dict, seed: int) -> RunResult:
    """satu random.Random(seed) per run, hanya pakai src.core dan src.search_space."""
    raise NotImplementedError("TODO: implementasi simulated annealing")
