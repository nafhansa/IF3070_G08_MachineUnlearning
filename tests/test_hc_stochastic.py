"""test hc_stochastic: golden regression, monotonic, feasibility, reproducibility, validasi config."""

import json
from pathlib import Path

import pytest

from src.algorithms.hc_stochastic import hc_stochastic
from src.core import load_problem, state_from_dict, is_feasible, objective

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_hc_stochastic.json").read_text())
PROBLEM_PATH = Path(__file__).parent.parent / "configs" / "problem.json"


def without_runtime(result):
    data = result.to_dict()
    data.pop("runtime_seconds")
    return data


@pytest.mark.parametrize("seed", ["3001", "3002", "3003"])
def test_golden_regression(seed):
    problem = load_problem(PROBLEM_PATH)
    result = hc_stochastic(problem, GOLDEN["config"], int(seed))
    # lewat JSON supaya tuple/list diperlakukan sama seperti saat golden disimpan
    actual = json.loads(json.dumps(without_runtime(result), sort_keys=True))
    assert actual == GOLDEN["runs"][seed]


def test_objective_never_decreases_and_snapshots_feasible(tiny_problem):
    result = hc_stochastic(tiny_problem, {"max_iterations": 200}, 1)
    values = [entry["objective"] for entry in result.history]
    assert values == sorted(values)
    for entry in result.history:
        if entry["current_state"] is not None:
            state = state_from_dict(entry["current_state"], tiny_problem)
            assert is_feasible(state, tiny_problem)
            assert objective(state, tiny_problem) == entry["objective"]


def test_same_seed_same_result(tiny_problem):
    first = hc_stochastic(tiny_problem, {"max_iterations": 100}, 7)
    second = hc_stochastic(tiny_problem, {"max_iterations": 100}, 7)
    assert without_runtime(first) == without_runtime(second)


@pytest.mark.parametrize("bad_value", [True, False, -1, 1.5, "10", None])
def test_config_validation(tiny_problem, bad_value):
    with pytest.raises(ValueError, match="max_iterations harus integer >= 0"):
        hc_stochastic(tiny_problem, {"max_iterations": bad_value}, 1)
