import json
import random
from collections import Counter
from itertools import accumulate

import pytest

from src.algorithms.genetic_algorithm import (
    GAConfig,
    genetic_algorithm,
    mutate,
    next_generation,
    one_point_crossover,
    repair,
    roulette_index,
)
from src.baseline import exact_optimum
from src.core import Problem, Ship, UNLOADED, Vehicle, is_feasible, objective, state_from_dict
from src.result import RunResult
from src.search_space import random_feasible_state
from tests.conftest import at, make_state


class FixedRandom:
    """rng palsu untuk test deterministik: random() dan randint() mengembalikan nilai yang diantrikan."""

    def __init__(self, randoms=(), ints=()):
        self.randoms, self.ints = list(randoms), list(ints)

    def random(self):
        return self.randoms.pop(0)

    def randint(self, a, b):
        value = self.ints.pop(0)
        assert a <= value <= b
        return value

    def randrange(self, n):
        return self.ints.pop(0)


# --- roulette wheel ----------------------------------------------------------

def test_roulette_intervals_follow_slide_example():
    # slide 46: fitness 24, 23, 20, 11 (total 78) -> interval kumulatif [0,24) [24,47) [47,67) [67,78)
    cumulative = list(accumulate([24, 23, 20, 11]))
    picks = [roulette_index(cumulative, FixedRandom([r / 78])) for r in (0, 23.9, 24, 46.9, 47, 66.9, 67, 77.9)]
    assert picks == [0, 0, 1, 1, 2, 2, 3, 3]


def test_roulette_never_picks_zero_fitness_when_others_positive():
    cumulative = list(accumulate([0, 5, 0]))
    rng = random.Random(0)
    assert {roulette_index(cumulative, rng) for _ in range(500)} == {1}


def test_roulette_all_zero_falls_back_to_uniform():
    cumulative = list(accumulate([0, 0, 0]))
    rng = random.Random(0)
    counts = Counter(roulette_index(cumulative, rng) for _ in range(3000))
    assert set(counts) == {0, 1, 2}
    assert all(800 < c < 1200 for c in counts.values())


def test_roulette_distribution_is_fitness_proportional():
    fitness = [24, 23, 20, 11]
    cumulative = list(accumulate(fitness))
    rng = random.Random(1)
    n = 20000
    counts = Counter(roulette_index(cumulative, rng) for _ in range(n))
    for i, f in enumerate(fitness):
        assert abs(counts[i] / n - f / 78) < 0.015


# --- crossover -----------------------------------------------------------------

def test_one_point_crossover_takes_prefix_and_suffix():
    p1, p2 = tuple("AAAAA"), tuple("BBBBB")
    assert one_point_crossover(p1, p2, FixedRandom(ints=[2])) == tuple("AABBB")
    # cut selalu internal: randint(1, n-1) sehingga child bukan clone parent
    assert one_point_crossover(p1, p2, FixedRandom(ints=[1])) == tuple("ABBBB")
    assert one_point_crossover(p1, p2, FixedRandom(ints=[4])) == tuple("AAAAB")


def test_crossover_short_chromosomes_are_safe():
    assert one_point_crossover((), (), FixedRandom()) == ()
    assert one_point_crossover(("A",), ("B",), FixedRandom()) == ("A",)


def test_crossover_keeps_length_and_vehicle_mapping(tiny_problem):
    rng = random.Random(4)
    for _ in range(50):
        p1, p2 = random_feasible_state(tiny_problem, rng), random_feasible_state(tiny_problem, rng)
        child = one_point_crossover(p1, p2, rng)
        assert len(child) == len(tiny_problem.vehicles)
        # gene ke-i selalu berasal dari gene ke-i salah satu parent
        assert all(c in (a, b) for c, a, b in zip(child, p1, p2))


# --- mutation & repair ---------------------------------------------------------

def test_mutate_changes_child_without_touching_parent(tiny_problem):
    rng = random.Random(2)
    parent = make_state(tiny_problem, A=at(0, 0), C=at(2, 0))
    before = tuple(parent)
    for _ in range(50):
        assert mutate(parent, tiny_problem, rng) != parent
    assert parent == before


@pytest.mark.parametrize(
    "placements",
    [
        {"A": at(0, 0), "B": at(1, 0)},  # overlap
        {"A": at(3, 0)},  # keluar deck
        {"A": at(0, 0), "B": at(2, 0), "D": at(0, 0)},  # overlap + overweight
        {"B": at(0, 0), "D": at(2, 0)},  # overweight saja (berat 11)
    ],
)
def test_repair_returns_feasible_state(tiny_problem, placements):
    state = make_state(tiny_problem, **placements)
    assert not is_feasible(state, tiny_problem)
    fixed, status = repair(state, tiny_problem, random.Random(0))
    assert is_feasible(fixed, tiny_problem)
    assert status in ("repaired", "fallback")


def test_repair_keeps_feasible_state_unchanged(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0), C=at(2, 0))
    assert repair(state, tiny_problem, random.Random(0)) == (state, "ok")


def test_repair_drops_lowest_fee_per_weight_on_overweight(tiny_problem):
    # B (80/4=20) vs D (200/7=28.6): D dipertahankan, B dibuang
    state = make_state(tiny_problem, B=at(0, 0), D=at(2, 0))
    fixed, _ = repair(state, tiny_problem, random.Random(0))
    assert fixed[tiny_problem.index("D")] == at(2, 0)
    assert fixed[tiny_problem.index("B")] == UNLOADED


def test_repair_relocates_before_unloading():
    # dua kendaraan 1x1 bertumpuk di deck lega: repair harus memindah, bukan membuang
    problem = Problem(Ship(5, 5, 100), (Vehicle("P", 1, 1, 10, 1, 0), Vehicle("Q", 1, 1, 10, 1, 0)))
    fixed, status = repair(make_state(problem, P=at(0, 0), Q=at(0, 0)), problem, random.Random(0))
    assert status == "repaired" and all(p.loaded for p in fixed)


# --- satu generasi -------------------------------------------------------------

def population_of(problem, n, seed):
    rng = random.Random(seed)
    return [random_feasible_state(problem, rng) for _ in range(n)]


@pytest.mark.parametrize("n,elite", [(7, 1), (7, 2), (4, 0), (1, 1)])
def test_next_generation_has_exact_size_and_is_feasible(tiny_problem, n, elite):
    pop = population_of(tiny_problem, n, seed=n)
    cfg = GAConfig(population_size=n, generations=1, elite_count=elite)
    counters = Counter()
    new = next_generation(pop, [objective(s, tiny_problem) for s in pop], tiny_problem, cfg, random.Random(0), counters)
    assert len(new) == n
    assert all(is_feasible(s, tiny_problem) for s in new)


def test_elite_is_copied_unmutated(tiny_problem):
    pop = population_of(tiny_problem, 6, seed=3)
    fits = [objective(s, tiny_problem) for s in pop]
    cfg = GAConfig(population_size=6, generations=1, mutation_rate=1.0, elite_count=1)
    new = next_generation(pop, fits, tiny_problem, cfg, random.Random(0), Counter())
    assert new[0] == pop[fits.index(max(fits))]


def test_mutation_rate_is_per_child(tiny_problem):
    pop = population_of(tiny_problem, 9, seed=1)
    fits = [objective(s, tiny_problem) for s in pop]
    for rate, expected in ((0.0, 0), (1.0, 8)):
        counters = Counter()
        cfg = GAConfig(population_size=9, generations=1, crossover_rate=0.0, mutation_rate=rate, elite_count=1)
        new = next_generation(pop, fits, tiny_problem, cfg, random.Random(5), counters)
        assert counters["mutations"] == expected
        if rate == 0.0:
            # tanpa crossover dan mutasi, child = salinan parent terpilih (parent feasible -> repair "ok")
            assert all(child in pop for child in new)


# --- run lengkap ---------------------------------------------------------------

def run_tiny(problem, seed=11, **overrides):
    cfg = dict(population_size=9, generations=15)
    cfg.update(overrides)
    return genetic_algorithm(problem, cfg, seed)


def test_ga_history_is_consistent(tiny_problem):
    result = run_tiny(tiny_problem)
    history = result.history
    assert len(history) == 16  # generasi 0 (populasi awal) sampai 15
    assert [h["generation"] for h in history] == list(range(16))
    previous_best = -1
    for h in history:
        best_state = state_from_dict(h["best_individual"], tiny_problem)
        assert objective(best_state, tiny_problem) == h["best_fitness"]
        assert is_feasible(best_state, tiny_problem)
        assert h["worst_fitness"] <= h["average_fitness"] <= h["best_fitness"] <= h["best_so_far_fitness"]
        # elitism aktif -> best per generasi tidak pernah turun
        assert h["best_fitness"] >= previous_best
        previous_best = h["best_fitness"]


def test_ga_final_state_matches_recomputation(tiny_problem):
    result = run_tiny(tiny_problem)
    final = state_from_dict(result.final_state, tiny_problem)
    assert is_feasible(final, tiny_problem)
    assert objective(final, tiny_problem) == result.final_objective == result.final_state["objective"]
    assert result.final_objective == max(h["best_so_far_fitness"] for h in result.history)
    assert result.final_objective <= exact_optimum(tiny_problem)[0]
    initial = state_from_dict(result.initial_state, tiny_problem)
    assert objective(initial, tiny_problem) == result.history[0]["best_fitness"] == result.initial_objective
    assert result.metrics["final_population_feasible"] is True
    assert result.config["population_size"] == 9 and result.config["generations"] == 15


def test_best_so_far_is_separate_from_generation_best_without_elitism(tiny_problem):
    result = run_tiny(tiny_problem, elite_count=0, generations=30, mutation_rate=0.9)
    best = [h["best_fitness"] for h in result.history]
    so_far = [h["best_so_far_fitness"] for h in result.history]
    assert so_far == list(accumulate(best, max))


def test_ga_is_deterministic_for_same_seed(tiny_problem):
    a, b = run_tiny(tiny_problem, seed=3), run_tiny(tiny_problem, seed=3)
    assert a.history == b.history and a.final_state == b.final_state
    c = run_tiny(tiny_problem, seed=4)
    assert c.history != a.history


def test_ga_reaches_tiny_optimum_on_some_seed(tiny_problem):
    # kualitas heuristik, bukan invariant: cukup salah satu seed mencapai optimum exact
    finals = [run_tiny(tiny_problem, seed=s, generations=30).final_objective for s in range(5)]
    assert max(finals) == 230


def test_ga_handles_degenerate_inputs():
    empty = Problem(Ship(2, 2, 5), ())
    assert genetic_algorithm(empty, {"population_size": 3, "generations": 2}, 0).final_objective == 0
    single = Problem(Ship(2, 2, 5), (Vehicle("S", 1, 1, 4, 1, 0),))
    assert genetic_algorithm(single, {"population_size": 1, "generations": 3}, 0).final_objective == 4


@pytest.mark.parametrize(
    "bad",
    [
        {"population_size": 0, "generations": 1},
        {"population_size": 3, "generations": -1},
        {"population_size": 3, "generations": 1, "mutation_rate": 1.5},
        {"population_size": 3, "generations": 1, "elite_count": 4},
        {"population_size": 3, "generations": 1, "unknown": 1},
    ],
)
def test_invalid_config_is_rejected(tiny_problem, bad):
    with pytest.raises((ValueError, TypeError)):
        genetic_algorithm(tiny_problem, bad, 0)


# --- result schema ---------------------------------------------------------------

def test_run_result_json_round_trip(tmp_path, tiny_problem):
    result = run_tiny(tiny_problem)
    path = tmp_path / "run.json"
    result.save(path)
    raw = json.loads(path.read_text())
    for key in ("algorithm", "seed", "config", "problem", "initial_state", "final_state", "history", "runtime_seconds"):
        assert key in raw
    assert raw["algorithm"] == "ga"
    loaded = RunResult.load(path)
    assert loaded.to_dict() == result.to_dict()
