"""test helper bersama hc_common."""

import random

import pytest

from src.algorithms.hc_common import check_integer_config, climb_to_peak, find_best_neighbors
from src.core import is_feasible, objective
from src.search_space import generate_feasible_neighbors, random_feasible_state


@pytest.mark.parametrize("bad_value", [True, False, -1, 0.5, "3", None])
def test_check_integer_config_rejects(bad_value):
    with pytest.raises(ValueError, match="batas harus integer >= 1"):
        check_integer_config("batas", bad_value, 1)


def test_check_integer_config_accepts():
    assert check_integer_config("batas", 5, 1) == 5
    assert check_integer_config("batas", 0, 0) == 0


def test_find_best_neighbors_matches_bruteforce(tiny_problem):
    state = random_feasible_state(tiny_problem, random.Random(3))
    best_value, tied, evaluated = find_best_neighbors(state, tiny_problem)
    values = [objective(n, tiny_problem) for _, n in generate_feasible_neighbors(state, tiny_problem)]
    assert evaluated == len(values)
    assert best_value == max(values)
    assert len(tied) == values.count(best_value)
    for _, neighbor in tied:
        assert objective(neighbor, tiny_problem) == best_value


@pytest.mark.parametrize("seed", range(10))
def test_climb_to_peak(tiny_problem, seed):
    rng = random.Random(seed)
    start = random_feasible_state(tiny_problem, rng)
    final_state, step_log, reason, evaluated = climb_to_peak(start, tiny_problem, rng, 50)
    values = [objective(start, tiny_problem)] + [entry["value"] for entry in step_log]
    assert values == sorted(values)
    assert is_feasible(final_state, tiny_problem)
    assert all(is_feasible(entry["state"], tiny_problem) for entry in step_log)
    assert evaluated > 0
    assert reason == "local_optimum"
    final_value = objective(final_state, tiny_problem)
    for _, neighbor in generate_feasible_neighbors(final_state, tiny_problem):
        assert objective(neighbor, tiny_problem) <= final_value


def test_climb_to_peak_max_steps(tiny_problem):
    rng = random.Random(1)
    start = random_feasible_state(tiny_problem, rng)
    final_state, step_log, reason, evaluated = climb_to_peak(start, tiny_problem, rng, 0)
    assert (final_state, step_log, reason, evaluated) == (start, [], "max_iterations", 0)
