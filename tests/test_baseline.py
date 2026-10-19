from src.baseline import exact_optimum, random_baseline
from src.core import Problem, Ship, Vehicle, is_feasible, objective


def test_exact_oracle_finds_manual_optimum(tiny_problem):
    value, state = exact_optimum(tiny_problem)
    assert value == 230  # D + C, lihat komentar fixture
    assert is_feasible(state, tiny_problem)
    assert objective(state, tiny_problem) == value
    assert {v.id for v, p in zip(tiny_problem.vehicles, state) if p.loaded} == {"C", "D"}


def test_exact_fit_fills_whole_deck():
    problem = Problem(Ship(2, 2, 100), tuple(Vehicle(f"V{i}", 1, 1, 10, 1, 0) for i in range(5)))
    value, state = exact_optimum(problem)
    assert value == 40  # hanya 4 sel
    assert is_feasible(state, problem)


def test_rotate_only_fit_is_found():
    # kendaraan 1x3 hanya muat di deck 3x1 bila vertikal (footprint 3x1)
    problem = Problem(Ship(3, 1, 10), (Vehicle("L", 1, 3, 50, 1, 0),))
    value, state = exact_optimum(problem)
    assert value == 50 and state[0].orientation == "V"


def test_capacity_limits_even_when_area_fits():
    problem = Problem(Ship(5, 5, 10), (Vehicle("H1", 1, 1, 60, 8, 0), Vehicle("L1", 1, 1, 30, 5, 0), Vehicle("L2", 1, 1, 30, 5, 0)))
    value, _ = exact_optimum(problem)
    assert value == 60  # dua kendaraan ringan (berat 10) = 60, sama dengan H1 sendirian


def test_random_baseline_is_feasible_seeded_and_not_above_optimum(tiny_problem):
    a = random_baseline(tiny_problem, samples=30, seed=5)
    b = random_baseline(tiny_problem, samples=30, seed=5)
    assert a == b
    assert is_feasible(a["best_state"], tiny_problem)
    assert objective(a["best_state"], tiny_problem) == a["best_objective"]
    assert a["best_objective"] <= exact_optimum(tiny_problem)[0]
    assert min(a["objectives"]) <= a["mean_objective"] <= a["best_objective"]
    assert len(a["objectives"]) == 30
