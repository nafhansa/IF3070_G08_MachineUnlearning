"""server lokal tipis: menyajikan frontend dan menjalankan solver python yang sama dengan CLI.

    python -m src.server            # http://127.0.0.1:8000

server bukan solver keempat: ia hanya memvalidasi request, memanggil SOLVERS[...] (ALGORITHMS dari
src.experiments + HC_VARIANTS), lalu mengembalikan RunResult apa adanya. tidak ada path dari user yang dibuka langsung;
instance dan file hasil hanya dari whitelist.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from src.algorithms.hc_variants import HC_VARIANTS
from src.baseline import exact_optimum
from src.core import load_problem
from src.experiments import ALGORITHMS

# solver yang boleh dipanggil dari browser: tiga algoritma utama + varian hill climbing
SOLVERS = {**ALGORITHMS, **HC_VARIANTS}

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
OUTPUT = ROOT / "output"

# instance yang boleh dipilih dari browser; exact optimum hanya untuk instance yang dibuktikan brute force
INSTANCES = {
    "standard": {"path": ROOT / "configs/problem.json", "label": "Standard: 30 kendaraan sintetis, deck 10x20"},
    "tiny": {"path": ROOT / "configs/presets/tiny.json", "label": "Tiny/Teaching: 4 kendaraan, optimum terbukti",
             "prove_optimum": True},
    "stress": {"path": ROOT / "configs/presets/stress.json", "label": "Stress: rotasi wajib + kapasitas ketat"},
}

# folder hasil run python yang boleh dibaca frontend (hanya file .json di dalamnya)
RESULT_DIRS = ("hill_climbing/runs", "hc_sideways/runs", "hc_random_restart/runs", "hc_stochastic_aima/runs",
               "simulated_annealing/runs", "genetic_algorithm/runs", "demo")

# satu-satunya sumber parameter + batas aman untuk browser. batas ini sengaja lebih kecil dari CLI agar
# satu request tidak membekukan UI; eksperimen resmi tetap lewat CLI tanpa batas ini.
PARAM_SPECS = {
    "hc": [
        {"key": "max_iterations", "label": "nmax (iterasi)", "type": "int", "min": 0, "max": 20000, "default": 2000},
    ],
    # varian HC memeriksa SEMUA neighbor tiap langkah (sekitar 0,05 s per langkah di instance standar),
    # jadi batas browsernya jauh lebih kecil dari hc biasa. CLI tetap tanpa batas ini.
    "hc_sideways": [
        {"key": "max_sideways", "label": "maks sideways beruntun", "type": "int", "min": 0, "max": 300,
         "default": 50},
        {"key": "max_iterations", "label": "iterasi maksimum", "type": "int", "min": 0, "max": 300, "default": 100},
    ],
    "hc_random_restart": [
        {"key": "max_restarts", "label": "jumlah restart", "type": "int", "min": 1, "max": 10, "default": 3},
        {"key": "iterations_per_restart", "label": "batas langkah per restart", "type": "int", "min": 0, "max": 50,
         "default": 30},
    ],
    "hc_stochastic_aima": [
        {"key": "max_iterations", "label": "iterasi maksimum", "type": "int", "min": 0, "max": 300, "default": 100},
        {"key": "selection", "label": "cara memilih uphill neighbor", "type": "choice",
         "options": ["proportional", "uniform"], "default": "proportional"},
    ],
    "sa": [
        {"key": "max_iterations", "label": "iterasi maksimum", "type": "int", "min": 0, "max": 6000, "default": 3000},
        {"key": "initial_temperature", "label": "T0 (suhu awal)", "type": "float", "min": 1e-6, "max": 1e6,
         "default": 50.0},
        {"key": "cooling_rate", "label": "alpha (cooling, 0<alpha<1)", "type": "float", "min": 0.5, "max": 0.9999,
         "default": 0.998},
        {"key": "min_temperature", "label": "T minimum (stop)", "type": "float", "min": 1e-6, "max": 1e6,
         "default": 0.05},
        {"key": "detect_stuck", "label": "hitung episode local optimum", "type": "bool", "default": True},
    ],
    "ga": [
        {"key": "population_size", "label": "jumlah populasi", "type": "int", "min": 1, "max": 200, "default": 50},
        {"key": "generations", "label": "banyak generasi", "type": "int", "min": 0, "max": 500, "default": 100},
        {"key": "crossover_rate", "label": "crossover rate", "type": "float", "min": 0.0, "max": 1.0, "default": 0.9},
        {"key": "mutation_rate", "label": "mutation rate (per child)", "type": "float", "min": 0.0, "max": 1.0,
         "default": 0.2},
        {"key": "elite_count", "label": "elite count", "type": "int", "min": 0, "max": 200, "default": 1},
    ],
}
MAX_SEED = 2**31 - 1


class BadRequest(ValueError):
    """request tidak valid; dikembalikan sebagai HTTP 400 dengan pesan yang bisa dibaca mahasiswa."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_run_request(body) -> tuple[str, str, int, dict]:
    """validasi body POST /api/run. return (algorithm, instance, seed, config lengkap)."""
    if not isinstance(body, dict):
        raise BadRequest("body harus objek JSON")
    algorithm = body.get("algorithm")
    if algorithm not in SOLVERS:
        raise BadRequest(f"algorithm harus salah satu dari {sorted(SOLVERS)}")
    instance = body.get("instance")
    if instance not in INSTANCES:
        raise BadRequest(f"instance harus salah satu dari {sorted(INSTANCES)}")
    seed = body.get("seed")
    if not _is_int(seed) or not 0 <= seed <= MAX_SEED:
        raise BadRequest(f"seed harus integer 0..{MAX_SEED}")
    given = body.get("config", {})
    if not isinstance(given, dict):
        raise BadRequest("config harus objek JSON")
    specs = {p["key"]: p for p in PARAM_SPECS[algorithm]}
    unknown = sorted(set(given) - set(specs))
    if unknown:
        raise BadRequest(f"parameter tidak didukung untuk {algorithm}: {unknown}")
    config = {}
    for key, spec in specs.items():
        value = given.get(key, spec["default"])
        if spec["type"] == "bool":
            if not isinstance(value, bool):
                raise BadRequest(f"{key} harus true/false")
        elif spec["type"] == "choice":
            if value not in spec["options"]:
                raise BadRequest(f"{key} harus salah satu dari {spec['options']}")
        else:
            ok_type = _is_int(value) if spec["type"] == "int" else _is_int(value) or isinstance(value, float)
            if not ok_type or not spec["min"] <= value <= spec["max"]:
                raise BadRequest(f"{key} harus {spec['type']} di [{spec['min']}, {spec['max']}]")
        config[key] = value
    if algorithm == "ga" and config["elite_count"] > config["population_size"]:
        raise BadRequest("elite_count tidak boleh lebih besar dari population_size")
    # snapshot selalu direkam untuk browser agar setiap frame bisa direplay
    config["record_snapshots"] = True
    return algorithm, instance, seed, config


@lru_cache(maxsize=None)
def _instance_info(instance_id: str) -> dict:
    entry = INSTANCES[instance_id]
    problem = load_problem(entry["path"])
    return {
        "id": instance_id,
        "label": entry["label"],
        "name": problem.name,
        "source": problem.metadata.get("source", "-"),
        "note": problem.metadata.get("note", ""),
        "vehicles": len(problem.vehicles),
        "ship": {"width": problem.ship.width, "length": problem.ship.length,
                 "max_capacity": problem.ship.max_capacity},
        "exact_optimum": exact_optimum(problem)[0] if entry.get("prove_optimum") else None,
    }


def _list_results() -> list[dict]:
    items = []
    for folder in RESULT_DIRS:
        for path in sorted((OUTPUT / folder).glob("*.json")):
            # algoritma diambil dari nama folder/file agar json besar (sampai ~6 MB) tidak perlu dibuka
            name = path.name
            algorithm = ("hc" if folder.startswith("hill") else "sa" if folder.startswith("simulated") else
                         "ga" if folder.startswith("genetic") else
                         folder.split("/")[0] if folder.startswith("hc_") else name.split("_")[0])
            items.append({"id": f"{folder}/{name}", "algorithm": algorithm, "label": f"{folder}/{name}"})
    return items


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # request run cukup beberapa ratus byte

    @app.errorhandler(BadRequest)
    def bad_request(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify(error="request terlalu besar"), 413

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND, "index.html")

    @app.get("/<path:name>")
    def frontend_file(name):
        # send_from_directory menolak path yang keluar dari folder frontend
        return send_from_directory(FRONTEND, name)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", algorithms=sorted(SOLVERS))

    @app.get("/api/instances")
    def instances():
        return jsonify([_instance_info(key) for key in INSTANCES])

    @app.get("/api/algorithms")
    def algorithms():
        return jsonify(PARAM_SPECS)

    @app.post("/api/run")
    def run():
        try:
            body = json.loads(request.get_data(as_text=True) or "null")
        except json.JSONDecodeError as exc:
            raise BadRequest(f"body bukan JSON valid: {exc.msg}") from exc
        algorithm, instance, seed, config = parse_run_request(body)
        problem = load_problem(INSTANCES[instance]["path"])
        # pemanggilan solver yang sama persis dengan CLI; error tak terduga dibiarkan jadi 500 + traceback di log
        result = SOLVERS[algorithm](problem, config, seed)
        return jsonify(result.to_dict())

    @app.get("/api/results")
    def results():
        return jsonify(_list_results())

    @app.get("/api/results/<path:result_id>")
    def result_file(result_id):
        allowed = {item["id"] for item in _list_results()}
        if result_id not in allowed:
            abort(404)
        folder, name = result_id.rsplit("/", 1)
        return send_from_directory(OUTPUT / folder, name, mimetype="application/json")

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.server", description="server demo lokal IF3070")
    parser.add_argument("--host", default="127.0.0.1", help="default hanya localhost")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(f"buka http://{args.host}:{args.port}/  (Ctrl+C untuk berhenti)")
    create_app().run(host=args.host, port=args.port, threaded=True)
