"""test hc_sideways."""

import random

import pytest

from src.algorithms.hc_common import climb_to_peak
from src.algorithms.hc_sideways import hc_sideways
from src.core import is_feasible, objective, state_from_dict
from src.main import check_result
from src.search_space import random_feasible_state


def without_runtime(result):
    data = result.to_dict()
    data.pop("runtime_seconds")
    return data


def longest_sideways_run(history):
    longest, current = 0, 0
    for entry in history:
        if entry["sideways"]:
            current += 1
            longest = max(longest, current)
        elif entry["improved"]:
            current = 0
    return longest


@pytest.mark.parametrize("max_sideways", [0, 1, 3, 10])
@pytest.mark.parametrize("seed", range(5))
def test_consecutive_sideways_within_limit(tiny_problem, max_sideways, seed):
    result = hc_sideways(tiny_problem, {"max_sideways": max_sideways, "max_iterations": 200}, seed)
    assert longest_sideways_run(result.history) <= max_sideways
    assert result.metrics["max_consecutive_sideways"] == longest_sideways_run(result.history)
    assert result.metrics["max_consecutive_sideways"] <= max_sideways


@pytest.mark.parametrize("seed", range(10))
def test_zero_sideways_equals_climb_to_peak(tiny_problem, seed):
    result = hc_sideways(tiny_problem, {"max_sideways": 0, "max_iterations": 200}, seed)
    rng = random.Random(seed)
    start = random_feasible_state(tiny_problem, rng)
    final_state, _, _, _ = climb_to_peak(start, tiny_problem, rng, 200)
    assert result.final_objective == objective(final_state, tiny_problem)


@pytest.mark.parametrize("seed", range(10))
def test_sideways_total_matches_history(tiny_problem, seed):
    result = hc_sideways(tiny_problem, {"max_sideways": 5, "max_iterations": 200}, seed)
    flagged = sum(1 for entry in result.history if entry["sideways"])
    assert result.metrics["sideways_moves_total"] == flagged


def test_sideways_actually_happens(tiny_problem):
    totals = [hc_sideways(tiny_problem, {"max_sideways": 5, "max_iterations": 200}, s).metrics["sideways_moves_total"]
              for s in range(20)]
    assert max(totals) > 0


@pytest.mark.parametrize("seed", range(5))
def test_feasible_monotonic_and_validated(tiny_problem, seed):
    result = hc_sideways(tiny_problem, {"max_sideways": 5, "max_iterations": 100}, seed)
    values = [entry["objective"] for entry in result.history]
    assert values == sorted(values)
    for entry in result.history:
        if entry["current_state"] is not None:
            state = state_from_dict(entry["current_state"], tiny_problem)
            assert is_feasible(state, tiny_problem)
            assert objective(state, tiny_problem) == entry["objective"]
    assert check_result(result, tiny_problem) == []


def test_max_iterations_stops(tiny_problem):
    result = hc_sideways(tiny_problem, {"max_sideways": 1000, "max_iterations": 3}, 1)
    assert result.metrics["iterations"] <= 3
    if result.metrics["termination_reason"] == "max_iterations":
        assert result.metrics["iterations"] == 3


def test_same_seed_same_result(tiny_problem):
    config = {"max_sideways": 5, "max_iterations": 100}
    assert without_runtime(hc_sideways(tiny_problem, config, 4)) == without_runtime(hc_sideways(tiny_problem, config, 4))


@pytest.mark.parametrize("key,minimum", [("max_sideways", 0), ("max_iterations", 0)])
@pytest.mark.parametrize("bad_value", [True, -1, 1.5, "3", None])
def test_config_validation(tiny_problem, key, minimum, bad_value):
    config = {"max_sideways": 3, "max_iterations": 10, key: bad_value}
    with pytest.raises(ValueError, match=f"{key} harus integer >= {minimum}"):
        hc_sideways(tiny_problem, config, 1)
