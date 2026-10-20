"""test registry HC_VARIANTS dan eksperimen varian lewat run_experiments + validator CLI."""

import csv
import json
from pathlib import Path

import pytest

from src.algorithms.hc_variants import HC_VARIANTS
from src.experiments import ALGORITHMS, run_experiments
from src.main import main
from src.result import RunResult

TINY = str(Path(__file__).parent.parent / "configs" / "presets" / "tiny.json")

VARIANT_CONFIGS = {
    "hc_sideways": {"max_sideways": 5, "max_iterations": 50},
    "hc_random_restart": {"max_restarts": 3, "iterations_per_restart": 20},
    "hc_stochastic_aima": {"max_iterations": 50, "selection": "proportional"},
}


def test_registry_keys():
    assert set(HC_VARIANTS) == {"hc_sideways", "hc_random_restart", "hc_stochastic_aima"}
    assert set(ALGORITHMS) == {"ga", "hc", "sa"}


@pytest.mark.parametrize("algorithm", sorted(VARIANT_CONFIGS))
def test_variant_experiment_writes_valid_runs(tmp_path, algorithm):
    out = tmp_path / algorithm
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({"algorithm": algorithm, "problem": TINY, "output_dir": str(out),
                                       "config": VARIANT_CONFIGS[algorithm], "seeds": [1, 2, 3]}))
    assert len(run_experiments(config_path)) == 3
    with open(out / "raw_results.csv") as f:
        raw = list(csv.DictReader(f))
    assert len(raw) == 3
    for row in raw:
        result = RunResult.load(out / row["json_path"])
        assert result.algorithm == algorithm
        assert (out / row["plot_path"]).exists()
        assert main(["validate", "--config", TINY, "--result", str(out / row["json_path"])]) == 0
