"""genetic algorithm sesuai salindia IF3070 (Materi03 slide 43-49, pseudocode AIMA).

alur per generasi:
    roulette-wheel selection (2 parent, with replacement)
    -> one-point crossover (peluang crossover_rate, satu child per pasangan)
    -> mutation memakai move problem swap/relocate/rotate (peluang mutation_rate per child)
    -> repair + feasibility check sebelum child masuk populasi
elite_count individu terbaik disalin tanpa mutasi (elitism, AIMA 4th ed. sec. 4.1.4).

representasi: satu individual = satu State lengkap; gene ke-i = Placement kendaraan ke-i
pada urutan problem.vehicles yang tetap. fitness = objective = total ShippingFee loaded.
"""

from __future__ import annotations

import random
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import accumulate
from statistics import mean

from src.core import UNLOADED, Placement, capacity_ok, footprint, inside_deck, is_feasible, objective, overlaps, problem_to_dict, rect
from src.result import RunResult, snapshot
from src.search_space import random_feasible_state, random_move


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    generations: int  # banyak iterasi; generasi 0 = populasi awal, lalu 1..generations
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2  # peluang per child (bukan per gene)
    elite_count: int = 1
    repair_tries: int = 20  # percobaan anchor random per kendaraan saat repair
    record_snapshots: bool = True  # simpan best individual tiap generasi di history

    def __post_init__(self):
        if not (isinstance(self.population_size, int) and self.population_size >= 1):
            raise ValueError("population_size harus integer >= 1")
        if not (isinstance(self.generations, int) and self.generations >= 0):
            raise ValueError("generations harus integer >= 0")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} harus di [0, 1]")
        if not 0 <= self.elite_count <= self.population_size:
            raise ValueError("elite_count harus di [0, population_size]")
        if self.repair_tries < 0:
            raise ValueError("repair_tries harus >= 0")


# --- operator ------------------------------------------------------------------

def roulette_index(cumulative, rng) -> int:
    """fitness-proportional selection (slide 46) dari daftar fitness kumulatif.

    titik r di [0, total) jatuh ke interval [cum[i-1], cum[i]); individu fitness 0 punya
    interval kosong sehingga tidak pernah terpilih. jika total 0, fallback uniform.
    """
    total = cumulative[-1]
    if total <= 0:
        return rng.randrange(len(cumulative))
    return bisect_right(cumulative, rng.random() * total)


def one_point_crossover(parent1, parent2, rng):
    """child = prefix parent1 + suffix parent2 dengan cut internal (slide 47)."""
    n = len(parent1)
    if n < 2:
        return tuple(parent1)
    cut = rng.randint(1, n - 1)
    return tuple(parent1[:cut]) + tuple(parent2[cut:])


def mutate(state, problem, rng):
    """satu move random (swap/relocate/rotate) tanpa cek feasibility; repair menyusul."""
    move = random_move(state, problem, rng)
    return move[1] if move else state


def repair(state, problem, rng, tries: int = 20):
    """jadikan state feasible secara bounded. return (state, status) dengan status
    "ok" (sudah feasible), "repaired", atau "fallback" (diganti random feasible state).

    kendaraan loaded diproses urut fee/weight menurun (tie: indeks), sehingga saat berat
    berlebih yang dilepas adalah yang rasionya paling rendah. tiap kendaraan dicoba:
    posisi semula -> rotate di anchor sama -> `tries` anchor/orientasi random -> unload.
    """
    if is_feasible(state, problem):
        return state, "ok"
    ship, vehicles = problem.ship, problem.vehicles
    fixed = [UNLOADED] * len(state)
    placed = []
    weight = 0

    def ratio(i):
        v = vehicles[i]
        return v.shipping_fee / v.weight if v.weight > 0 else float("inf")

    for i in sorted((i for i, p in enumerate(state) if p.loaded), key=lambda i: (-ratio(i), i)):
        v, p = vehicles[i], state[i]
        if not capacity_ok(weight + v.weight, ship):
            continue
        candidates = [p]
        if v.width != v.length:
            candidates.append(Placement(True, p.x, p.y, "V" if p.orientation == "H" else "H"))
        for _ in range(tries):
            o = "H" if v.width == v.length else rng.choice(("H", "V"))
            w, l = footprint(v, Placement(True, 0, 0, o))
            if w <= ship.width and l <= ship.length:
                candidates.append(Placement(True, rng.randint(0, ship.width - w), rng.randint(0, ship.length - l), o))
        for cand in candidates:
            r = rect(v, cand)
            if inside_deck(r, ship) and not any(overlaps(r, other) for other in placed):
                fixed[i] = cand
                placed.append(r)
                weight += v.weight
                break
    fixed = tuple(fixed)
    if is_feasible(fixed, problem):
        return fixed, "repaired"
    # secara konstruksi tidak seharusnya terjadi; tetap dijaga agar child invalid tidak lolos
    return random_feasible_state(problem, rng), "fallback"


def next_generation(population, fitnesses, problem, cfg: GAConfig, rng, counters: Counter):
    """bentuk populasi baru berukuran tepat len(population). counters dicatat untuk metrik."""
    n = len(population)
    ranked = sorted(range(n), key=lambda i: -fitnesses[i])  # sort stabil: tie -> indeks kecil
    new = [population[i] for i in ranked[: cfg.elite_count]]  # elite tidak dimutasi
    cumulative = list(accumulate(fitnesses))
    while len(new) < n:
        x = population[roulette_index(cumulative, rng)]
        y = population[roulette_index(cumulative, rng)]
        if rng.random() < cfg.crossover_rate:
            child = one_point_crossover(x, y, rng)
            counters["crossovers"] += 1
        else:
            child = x  # state immutable, jadi "salinan" aman tanpa aliasing
        if rng.random() < cfg.mutation_rate:
            child = mutate(child, problem, rng)
            counters["mutations"] += 1
        # repair diperlukan karena crossover/mutasi bisa membuat overlap atau kelebihan berat
        child, status = repair(child, problem, rng, cfg.repair_tries)
        counters[status] += 1
        new.append(child)
    return new


# --- run -----------------------------------------------------------------------

def genetic_algorithm(problem, config, seed: int) -> RunResult:
    """jalankan GA dengan satu rng ber-seed. runtime hanya mencakup search (tanpa serialisasi)."""
    cfg = config if isinstance(config, GAConfig) else GAConfig(**config)
    rng = random.Random(seed)
    counters = Counter()
    raw_history = []
    best_so_far, best_so_far_fit = None, None

    start = time.perf_counter()
    population = [random_feasible_state(problem, rng) for _ in range(cfg.population_size)]
    fitnesses = [objective(s, problem) for s in population]
    for generation in range(cfg.generations + 1):
        if generation > 0:
            population = next_generation(population, fitnesses, problem, cfg, rng, counters)
            # fitness selalu dihitung ulang dari state baru, jadi tidak ada cache yang basi
            fitnesses = [objective(s, problem) for s in population]
        b = max(range(len(population)), key=fitnesses.__getitem__)
        if best_so_far_fit is None or fitnesses[b] > best_so_far_fit:
            best_so_far, best_so_far_fit = population[b], fitnesses[b]
        raw_history.append((generation, fitnesses[b], mean(fitnesses), min(fitnesses), best_so_far_fit,
                            len(set(population)), population[b]))
    runtime = time.perf_counter() - start

    history = [
        {
            "generation": g,
            "best_fitness": best,
            "average_fitness": avg,
            "worst_fitness": worst,
            "best_so_far_fitness": so_far,
            "unique_individuals": unique,
            "best_individual": snapshot(state, problem) if cfg.record_snapshots else None,
        }
        for g, best, avg, worst, so_far, unique, state in raw_history
    ]
    initial = raw_history[0][-1]  # best individual populasi awal sebagai initial state representatif
    metrics = dict(counters)
    metrics.update(
        population_size=cfg.population_size,
        generations=cfg.generations,
        fitness_evaluations=cfg.population_size * (cfg.generations + 1),
        final_population_feasible=all(is_feasible(s, problem) for s in population),
        final_generation_best_fitness=raw_history[-1][1],
    )
    return RunResult(
        algorithm="ga",
        seed=seed,
        config=asdict(cfg),
        problem=problem_to_dict(problem),
        initial_state=snapshot(initial, problem),
        final_state=snapshot(best_so_far, problem),
        initial_objective=objective(initial, problem),
        final_objective=best_so_far_fit,
        runtime_seconds=runtime,
        history=history,
        metrics=metrics,
    )
