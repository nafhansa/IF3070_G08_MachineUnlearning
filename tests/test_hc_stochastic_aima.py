"""test hc_stochastic_aima."""

import pytest

from src.algorithms.hc_stochastic_aima import hc_stochastic_aima
from src.core import is_feasible, objective, state_from_dict
from src.main import check_result
from src.search_space import generate_feasible_neighbors


def without_runtime(result):
    data = result.to_dict()
    data.pop("runtime_seconds")
    return data


@pytest.mark.parametrize("selection", ["proportional", "uniform"])
@pytest.mark.parametrize("seed", range(5))
def test_every_accepted_step_is_uphill(tiny_problem, selection, seed):
    result = hc_stochastic_aima(tiny_problem, {"max_iterations": 100, "selection": selection}, seed)
    previous = None
    for h in result.history:
        if h["accepted"]:
            assert h["objective"] > previous
            assert h["candidate_objective"] == h["objective"]
        previous = h["objective"]
    assert result.metrics["improvements"] == sum(1 for h in result.history if h["accepted"])
    assert check_result(result, tiny_problem) == []


@pytest.mark.parametrize("selection", ["proportional", "uniform"])
def test_local_optimum_is_real(tiny_problem, selection):
    result = hc_stochastic_aima(tiny_problem, {"max_iterations": 100, "selection": selection}, 3)
    assert result.metrics["termination_reason"] == "local_optimum"
    final_state = state_from_dict(result.final_state, tiny_problem)
    for _, neighbor in generate_feasible_neighbors(final_state, tiny_problem):
        assert objective(neighbor, tiny_problem) <= result.final_objective


def test_snapshots_feasible_and_consistent(tiny_problem):
    result = hc_stochastic_aima(tiny_problem, {"max_iterations": 50}, 2)
    assert result.config["selection"] == "proportional"
    for h in result.history:
        if h["current_state"] is not None:
            state = state_from_dict(h["current_state"], tiny_problem)
            assert is_feasible(state, tiny_problem)
            assert objective(state, tiny_problem) == h["objective"]


def test_max_iterations_stops(tiny_problem):
    result = hc_stochastic_aima(tiny_problem, {"max_iterations": 1}, 1)
    assert result.metrics["iterations"] <= 1


def test_same_seed_same_result(tiny_problem):
    config = {"max_iterations": 50, "selection": "proportional"}
    assert without_runtime(hc_stochastic_aima(tiny_problem, config, 5)) == \
        without_runtime(hc_stochastic_aima(tiny_problem, config, 5))


@pytest.mark.parametrize("bad_value", [True, -1, 1.5, "3", None])
def test_max_iterations_validation(tiny_problem, bad_value):
    with pytest.raises(ValueError, match="max_iterations harus integer >= 0"):
        hc_stochastic_aima(tiny_problem, {"max_iterations": bad_value}, 1)


@pytest.mark.parametrize("bad_value", [True, -1, 1.5, 3, None, "steepest"])
def test_selection_validation(tiny_problem, bad_value):
    with pytest.raises(ValueError, match="selection harus"):
        hc_stochastic_aima(tiny_problem, {"max_iterations": 5, "selection": bad_value}, 1)
