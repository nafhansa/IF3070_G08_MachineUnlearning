import csv
import json

import pytest

from src.core import is_feasible, load_problem, objective, problem_to_dict, state_from_dict
from src.experiments import build_runs, make_synthetic_problem, run_experiments
from src.result import RunResult


def test_synthetic_problem_is_valid_and_deterministic():
    a = make_synthetic_problem(seed=42, n_vehicles=12)
    b = make_synthetic_problem(seed=42, n_vehicles=12)
    assert problem_to_dict(a) == problem_to_dict(b)
    assert len(a.vehicles) == 12
    assert a.metadata["source"].startswith("synthetic")
    assert make_synthetic_problem(seed=43, n_vehicles=12).vehicles != a.vehicles


def matrix_config(problem_path, out_dir):
    return {
        "problem": str(problem_path),
        "output_dir": str(out_dir),
        "ga_defaults": {"crossover_rate": 0.9, "mutation_rate": 0.2, "elite_count": 1},
        "experiment_a": {"population_size": 6, "generations": [2, 4, 6], "seeds": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
        "experiment_b": {"generations": 4, "population_sizes": [3, 6, 9], "seeds": [[11, 12, 13], [14, 15, 16], [17, 18, 19]]},
    }


def test_build_runs_follows_mandatory_matrix(tmp_path):
    runs = build_runs(matrix_config("p.json", tmp_path))
    assert len(runs) == 18
    a = [r for r in runs if r["experiment"] == "A"]
    b = [r for r in runs if r["experiment"] == "B"]
    assert len(a) == len(b) == 9
    assert {r["population_size"] for r in a} == {6} and sorted({r["generations"] for r in a}) == [2, 4, 6]
    assert {r["generations"] for r in b} == {4} and sorted({r["population_size"] for r in b}) == [3, 6, 9]
    assert len({r["seed"] for r in runs}) == 18, "seed tiap run harus unik"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["experiment_a"]["generations"].pop(),
        lambda c: c["experiment_b"]["seeds"][0].pop(),
        lambda c: c["experiment_b"]["seeds"].__setitem__(0, [1, 2, 3]),
    ],
)
def test_build_runs_rejects_incomplete_matrix(tmp_path, mutation):
    cfg = matrix_config("p.json", tmp_path)
    mutation(cfg)
    with pytest.raises(ValueError):
        build_runs(cfg)


def test_run_experiments_writes_all_outputs(tmp_path):
    problem_path = tmp_path / "problem.json"
    problem_path.write_text(json.dumps(problem_to_dict(make_synthetic_problem(seed=1, n_vehicles=8, ship=(6, 6, 120)))))
    out = tmp_path / "out"
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(matrix_config(problem_path, out)))

    rows = run_experiments(config_path)
    assert len(rows) == 18

    problem = load_problem(problem_path)
    with open(out / "raw_results.csv") as f:
        raw = list(csv.DictReader(f))
    assert len(raw) == 18
    for row in raw:
        result = RunResult.load(out / row["json_path"])
        final = state_from_dict(result.final_state, problem)
        assert is_feasible(final, problem)
        assert objective(final, problem) == float(row["final_objective"]) == result.final_objective
        assert int(row["population_size"]) == result.config["population_size"]
        assert int(row["generations"]) == result.config["generations"] == len(result.history) - 1
        assert result.seed == int(row["seed"])
        assert (out / row["plot_path"]).exists()

    with open(out / "summary.csv") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 6 and all(int(s["runs"]) == 3 for s in summary)
    assert (out / "experiment_A_convergence.png").exists()
    assert (out / "experiment_B_convergence.png").exists()
    assert (out / "SUMMARY.md").exists()


# CLI

def test_cli_generate_validate_ga_and_pending_algorithms(tmp_path, capsys):
    from src.main import main

    problem_path = tmp_path / "problem.json"
    assert main(["generate-instance", "--seed", "3", "--vehicles", "8", "--output", str(problem_path)]) == 0
    assert main(["validate", "--config", str(problem_path)]) == 0

    out_json = tmp_path / "ga.json"
    assert main(["ga", "--config", str(problem_path), "--seed", "5", "--population", "5",
                 "--generations", "3", "--output", str(out_json)]) == 0
    printed = capsys.readouterr().out
    for word in ("final objective", "feasible", "seed", "runtime", str(out_json)):
        assert word in printed
    assert out_json.exists() and out_json.with_suffix(".png").exists()

    # validasi ulang hasil run dari file JSON (recompute objective + feasibility)
    assert main(["validate", "--config", str(problem_path), "--result", str(out_json)]) == 0



def test_cli_validate_detects_tampered_result(tmp_path):
    from src.main import main

    problem_path = tmp_path / "problem.json"
    main(["generate-instance", "--seed", "3", "--vehicles", "8", "--output", str(problem_path)])
    out_json = tmp_path / "ga.json"
    main(["ga", "--config", str(problem_path), "--seed", "5", "--population", "5", "--generations", "3",
          "--output", str(out_json)])
    data = json.loads(out_json.read_text())
    data["final_objective"] += 1
    out_json.write_text(json.dumps(data))
    assert main(["validate", "--config", str(problem_path), "--result", str(out_json)]) == 1


def test_cli_validate_reports_malformed_result_without_crashing(tmp_path, capsys):
    from src.main import main

    problem_path = tmp_path / "problem.json"
    main(["generate-instance", "--seed", "3", "--vehicles", "8", "--output", str(problem_path)])
    out_json = tmp_path / "ga.json"
    main(["ga", "--config", str(problem_path), "--seed", "5", "--population", "5", "--generations", "3",
          "--output", str(out_json)])
    data = json.loads(out_json.read_text())
    data["final_state"]["placements"][0]["id"] = "TIDAK_ADA"
    out_json.write_text(json.dumps(data))
    assert main(["validate", "--config", str(problem_path), "--result", str(out_json)]) == 1
    assert "INVALID" in capsys.readouterr().out


# tahap 2: hill climbing lewat CLI dan runner eksperimen 

def make_problem_file(tmp_path, seed=3, vehicles=8):
    from src.main import main

    path = tmp_path / "problem.json"
    main(["generate-instance", "--seed", str(seed), "--vehicles", str(vehicles), "--output", str(path)])
    return path


def test_cli_hc_runs_and_validates(tmp_path, capsys):
    from src.main import main

    problem_path = make_problem_file(tmp_path)
    out_json = tmp_path / "hc.json"
    assert main(["hc", "--config", str(problem_path), "--seed", "5", "--iterations", "80",
                 "--output", str(out_json)]) == 0
    printed = capsys.readouterr().out
    for word in ("final objective", "feasible", "seed", "runtime", str(out_json)):
        assert word in printed
    assert out_json.exists() and out_json.with_suffix(".png").exists()
    assert main(["validate", "--config", str(problem_path), "--result", str(out_json)]) == 0

    data = json.loads(out_json.read_text())
    data["history"][-1]["objective"] += 1  # rusak satu angka history
    out_json.write_text(json.dumps(data))
    assert main(["validate", "--config", str(problem_path), "--result", str(out_json)]) == 1


def test_hc_experiment_writes_three_valid_runs(tmp_path):
    problem_path = make_problem_file(tmp_path)
    out = tmp_path / "hc_out"
    config_path = tmp_path / "hc_experiment.json"
    config_path.write_text(json.dumps({"algorithm": "hc", "problem": str(problem_path), "output_dir": str(out),
                                       "config": {"max_iterations": 60}, "seeds": [11, 12, 13]}))
    rows = run_experiments(config_path)
    assert len(rows) == 3
    problem = load_problem(problem_path)
    with open(out / "raw_results.csv") as f:
        raw = list(csv.DictReader(f))
    assert [int(r["seed"]) for r in raw] == [11, 12, 13]
    for row in raw:
        result = RunResult.load(out / row["json_path"])
        assert result.algorithm == "hc"
        final = state_from_dict(result.final_state, problem)
        assert is_feasible(final, problem) and objective(final, problem) == float(row["final_objective"])
        assert (out / row["plot_path"]).exists()
    assert (out / "SUMMARY.md").exists()


def test_local_search_experiment_requires_unique_seeds(tmp_path):
    problem_path = make_problem_file(tmp_path)
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"algorithm": "hc", "problem": str(problem_path), "output_dir": str(tmp_path),
                                       "config": {"max_iterations": 5}, "seeds": [1, 1, 2]}))
    with pytest.raises(ValueError):
        run_experiments(config_path)


# tahap 2: simulated annealing lewat CLI dan runner eksperimen 

SA_ARGS = ["--iterations", "150", "--t0", "50", "--alpha", "0.97", "--min-temp", "0.05"]


def test_cli_sa_runs_and_validates(tmp_path, capsys):
    from src.main import main

    problem_path = make_problem_file(tmp_path)
    out_json = tmp_path / "sa.json"
    assert main(["sa", "--config", str(problem_path), "--seed", "5", *SA_ARGS, "--output", str(out_json)]) == 0
    printed = capsys.readouterr().out
    for word in ("final objective", "best-found", "stuck", "feasible", str(out_json)):
        assert word in printed
    assert out_json.with_suffix(".png").exists()
    assert main(["validate", "--config", str(problem_path), "--result", str(out_json)]) == 0

    for key, tamper in (("stuck", lambda d: d["metrics"].__setitem__("stuck_event_count", 99)),
                        ("best", lambda d: d["history"][-1].__setitem__("best_so_far_objective", -1)),
                        ("prob", lambda d: d["history"][1].__setitem__("acceptance_probability", 1.5))):
        data = json.loads(out_json.read_text())
        tamper(data)
        bad = tmp_path / f"sa_{key}.json"
        bad.write_text(json.dumps(data))
        assert main(["validate", "--config", str(problem_path), "--result", str(bad)]) == 1, key


def test_sa_experiment_writes_three_valid_runs(tmp_path):
    problem_path = make_problem_file(tmp_path)
    out = tmp_path / "sa_out"
    config_path = tmp_path / "sa_experiment.json"
    config_path.write_text(json.dumps({"algorithm": "sa", "problem": str(problem_path), "output_dir": str(out),
                                       "config": {"max_iterations": 100, "initial_temperature": 50.0,
                                                  "cooling_rate": 0.97, "min_temperature": 0.05},
                                       "seeds": [21, 22, 23]}))
    rows = run_experiments(config_path)
    assert len(rows) == 3 and all(r["algorithm"] == "sa" for r in rows)
    for row in rows:
        result = RunResult.load(out / row["json_path"])
        assert row["best_found_objective"] == result.metrics["best_found_objective"] >= result.final_objective
        assert (out / row["plot_path"]).exists()


def test_cli_registry_has_no_pending_algorithms():
    from src.experiments import ALGORITHMS

    assert set(ALGORITHMS) == {"ga", "hc", "sa"}
