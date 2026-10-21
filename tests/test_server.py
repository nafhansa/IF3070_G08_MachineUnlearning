import json

import pytest

from src.core import load_problem
from src.main import check_result
from src.result import RunResult
from src.server import INSTANCES, PARAM_SPECS, create_app, parse_args


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def run(client, **body):
    return client.post("/api/run", data=json.dumps(body), content_type="application/json")


SMALL = {
    "hc": {"max_iterations": 30},
    "sa": {"max_iterations": 40, "initial_temperature": 20.0, "cooling_rate": 0.95, "min_temperature": 0.1,
           "detect_stuck": True},
    "ga": {"population_size": 6, "generations": 4, "crossover_rate": 0.9, "mutation_rate": 0.2, "elite_count": 1},
    "hc_sideways": {"max_sideways": 5, "max_iterations": 10},
    "hc_random_restart": {"max_restarts": 2, "iterations_per_restart": 5},
    "hc_stochastic_aima": {"max_iterations": 10, "selection": "uniform"},
}


def test_health_and_index(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and b"IF3070 Vehicle Loading Optimization" in page.data


def test_instances_are_whitelisted_presets(client):
    data = client.get("/api/instances").get_json()
    ids = {item["id"] for item in data}
    assert ids == set(INSTANCES) == {"standard", "tiny", "stress"}
    tiny = next(item for item in data if item["id"] == "tiny")
    assert tiny["exact_optimum"] == 230  # dibuktikan oracle brute force
    standard = next(item for item in data if item["id"] == "standard")
    assert standard["exact_optimum"] is None and standard["vehicles"] == 30


def test_algorithm_specs_cover_all_algorithms(client):
    specs = client.get("/api/algorithms").get_json()
    assert set(specs) == {"hc", "sa", "ga", "hc_sideways", "hc_random_restart", "hc_stochastic_aima"}
    for algo, params in specs.items():
        assert all({"key", "label", "type", "default"} <= set(p) for p in params), algo


@pytest.mark.parametrize("algorithm", list(SMALL))
@pytest.mark.parametrize("instance", ["tiny", "standard"])
def test_run_returns_valid_reproducible_result(client, algorithm, instance):
    first = run(client, algorithm=algorithm, instance=instance, seed=7, config=SMALL[algorithm])
    assert first.status_code == 200, first.get_json()
    result = RunResult(**first.get_json())
    assert result.algorithm == algorithm and result.seed == 7
    problem = load_problem(INSTANCES[instance]["path"])
    assert check_result(result, problem) == []  # objective dan feasibility dihitung ulang dari state
    again = RunResult(**run(client, algorithm=algorithm, instance=instance, seed=7, config=SMALL[algorithm]).get_json())
    assert again.history == result.history and again.final_state == result.final_state


def test_hc_variant_rejects_bad_choice_and_unknown_param(client):
    bad_choice = run(client, algorithm="hc_stochastic_aima", instance="tiny", seed=1, config={"selection": "x"})
    assert bad_choice.status_code == 400 and "selection" in bad_choice.get_json()["error"]
    unknown = run(client, algorithm="hc_sideways", instance="tiny", seed=1, config={"max_restarts": 2})
    assert unknown.status_code == 400 and "max_restarts" in unknown.get_json()["error"]


def test_run_uses_spec_defaults_when_config_omitted(client):
    resp = run(client, algorithm="hc", instance="tiny", seed=1)
    assert resp.status_code == 200
    default = next(p["default"] for p in PARAM_SPECS["hc"] if p["key"] == "max_iterations")
    assert resp.get_json()["config"]["max_iterations"] == default


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"algorithm": "tabu", "instance": "tiny", "seed": 1}, "algorithm"),
        ({"algorithm": "hc", "instance": "../configs/problem.json", "seed": 1}, "instance"),
        ({"algorithm": "hc", "instance": "tiny"}, "seed"),
        ({"algorithm": "hc", "instance": "tiny", "seed": "1"}, "seed"),
        ({"algorithm": "hc", "instance": "tiny", "seed": True}, "seed"),
        ({"algorithm": "hc", "instance": "tiny", "seed": -1}, "seed"),
        ({"algorithm": "hc", "instance": "tiny", "seed": 1, "config": {"max_iterations": 10 ** 9}}, "max_iterations"),
        ({"algorithm": "hc", "instance": "tiny", "seed": 1, "config": {"max_iterations": 2.5}}, "max_iterations"),
        ({"algorithm": "hc", "instance": "tiny", "seed": 1, "config": {"record_snapshots": False}}, "record_snapshots"),
        ({"algorithm": "sa", "instance": "tiny", "seed": 1, "config": {"cooling_rate": 1.0}}, "cooling_rate"),
        ({"algorithm": "sa", "instance": "tiny", "seed": 1, "config": {"initial_temperature": 0}}, "initial_temperature"),
        ({"algorithm": "ga", "instance": "tiny", "seed": 1, "config": {"population_size": 5000}}, "population_size"),
        ({"algorithm": "ga", "instance": "tiny", "seed": 1, "config": {"population_size": 4, "elite_count": 9}}, "elite"),
        ({"algorithm": "hc", "instance": "tiny", "seed": 1, "config": "x"}, "config"),
    ],
)
def test_run_rejects_invalid_requests(client, body, fragment):
    resp = run(client, **body)
    assert resp.status_code == 400
    assert fragment in resp.get_json()["error"]


def test_run_rejects_malformed_and_oversized_bodies(client):
    bad = client.post("/api/run", data="{not json", content_type="application/json")
    assert bad.status_code == 400 and "JSON" in bad.get_json()["error"]
    huge = client.post("/api/run", data=json.dumps({"algorithm": "hc", "pad": "x" * 100_000}),
                       content_type="application/json")
    assert huge.status_code == 413


def test_results_listing_and_safe_download(client):
    listing = client.get("/api/results").get_json()
    ids = {item["id"] for item in listing}
    assert "hill_climbing/runs/hc_seed3001.json" in ids
    assert "genetic_algorithm/runs/A_pop50_gen50_seed1001.json" in ids
    assert {item["algorithm"] for item in listing} >= {"hc", "sa", "ga"}
    ok = client.get("/api/results/hill_climbing/runs/hc_seed3001.json")
    assert ok.status_code == 200 and ok.get_json()["algorithm"] == "hc"
    for bad in ("../configs/problem.json", "hill_climbing/raw_results.csv", "..%2Fsrc%2Fcore.py", "nope.json"):
        assert client.get(f"/api/results/{bad}").status_code == 404, bad
    # file statis frontend juga tidak boleh keluar dari folder frontend/
    for bad in ("/../src/core.py", "/%2e%2e/src/core.py", "/..%2Fsrc%2Fcore.py"):
        assert client.get(bad).status_code == 404, bad


def test_server_binds_localhost_by_default():
    args = parse_args([])
    assert args.host == "127.0.0.1"
