"""test hc_random_restart."""

import pytest

from src.algorithms.hc_random_restart import hc_random_restart
from src.core import is_feasible, objective, state_from_dict
from src.experiments import plot_run
from src.main import check_result

CONFIG = {"max_restarts": 5, "iterations_per_restart": 50}


def without_runtime(result):
    data = result.to_dict()
    data.pop("runtime_seconds")
    return data


@pytest.mark.parametrize("seed", range(5))
def test_restart_bookkeeping(tiny_problem, seed):
    result = hc_random_restart(tiny_problem, CONFIG, seed)
    m = result.metrics
    assert m["restarts"] == CONFIG["max_restarts"]
    assert {h["restart"] for h in result.history} == set(range(CONFIG["max_restarts"]))
    assert result.final_objective == max(m["restart_final_objectives"])
    assert result.final_objective == m["best_found_objective"]
    for name in ("iterations_per_restart", "restart_initial_objectives", "restart_final_objectives",
                 "restart_termination_reasons"):
        assert len(m[name]) == CONFIG["max_restarts"]
    assert m["total_iterations"] == sum(m["iterations_per_restart"])
    assert m["restart_final_objectives"][m["best_restart_index"]] == result.final_objective
    # seri mempertahankan restart paling awal
    assert m["best_restart_index"] == m["restart_final_objectives"].index(result.final_objective)
    assert result.initial_objective == m["restart_initial_objectives"][0]


@pytest.mark.parametrize("seed", range(5))
def test_global_iteration_continuous_and_feasible(tiny_problem, seed):
    result = hc_random_restart(tiny_problem, CONFIG, seed)
    assert [h["iteration"] for h in result.history] == list(range(len(result.history)))
    starts = [h for h in result.history if h["reason"] == "restart_initial"]
    assert len(starts) == CONFIG["max_restarts"]
    best = None
    for h in result.history:
        best = h["objective"] if best is None else max(best, h["objective"])
        assert h["best_so_far_objective"] == best
        if h["current_state"] is not None:
            state = state_from_dict(h["current_state"], tiny_problem)
            assert is_feasible(state, tiny_problem)
            assert objective(state, tiny_problem) == h["objective"]
    assert check_result(result, tiny_problem) == []


def test_per_restart_objective_non_decreasing(tiny_problem):
    result = hc_random_restart(tiny_problem, CONFIG, 2)
    for restart_index in range(CONFIG["max_restarts"]):
        values = [h["objective"] for h in result.history if h["restart"] == restart_index]
        assert values == sorted(values)


def test_iterations_per_restart_limit(tiny_problem):
    result = hc_random_restart(tiny_problem, {"max_restarts": 3, "iterations_per_restart": 1}, 1)
    assert all(count <= 1 for count in result.metrics["iterations_per_restart"])


def test_same_seed_same_result(tiny_problem):
    assert without_runtime(hc_random_restart(tiny_problem, CONFIG, 9)) == \
        without_runtime(hc_random_restart(tiny_problem, CONFIG, 9))


def test_plot_runs(tiny_problem, tmp_path):
    plot_run(hc_random_restart(tiny_problem, CONFIG, 1), tmp_path / "rr.png")
    assert (tmp_path / "rr.png").exists()


@pytest.mark.parametrize("key,minimum", [("max_restarts", 1), ("iterations_per_restart", 0)])
@pytest.mark.parametrize("bad_value", [True, -1, 1.5, "3", None])
def test_config_validation(tiny_problem, key, minimum, bad_value):
    with pytest.raises(ValueError, match=f"{key} harus integer >= {minimum}"):
        hc_random_restart(tiny_problem, {**CONFIG, key: bad_value}, 1)


def test_max_restarts_zero_rejected(tiny_problem):
    with pytest.raises(ValueError, match="max_restarts harus integer >= 1"):
        hc_random_restart(tiny_problem, {"max_restarts": 0, "iterations_per_restart": 5}, 1)
