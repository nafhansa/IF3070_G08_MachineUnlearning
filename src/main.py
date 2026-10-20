"""CLI. jalankan dari root work/, contoh:

    python -m src.main validate --config configs/problem.json
    python -m src.main ga --config configs/problem.json --seed 101 --population 20 --generations 50
    python -m src.main hc --config configs/problem.json --seed 1 --iterations 2000
    python -m src.main sa --config configs/problem.json --seed 1 --iterations 5000 --t0 100 --alpha 0.997 --min-temp 0.05
    python -m src.main experiment --config configs/experiment.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from src.core import is_feasible, load_problem, objective, problem_to_dict, state_from_dict
from src.experiments import ALGORITHMS, make_synthetic_problem, plot_run, run_experiments
from src.result import RunResult
from src.search_space import random_feasible_state


# keluarga hill climbing: objective current tidak boleh turun
HC_FAMILY = ("hc", "hc_sideways", "hc_random_restart", "hc_stochastic_aima")


def check_result(result: RunResult, problem) -> list[str]:
    """hitung ulang objective/feasibility dari JSON hasil run secara independen dari angka tersimpan."""
    issues = []
    if result.problem["ship"] != problem_to_dict(problem)["ship"] or \
            result.problem["vehicles"] != problem_to_dict(problem)["vehicles"]:
        issues.append("problem pada result berbeda dengan --config")
        return issues
    for name, stored in (("initial_state", result.initial_objective), ("final_state", result.final_objective)):
        state = state_from_dict(getattr(result, name), problem)
        value = objective(state, problem)
        if not is_feasible(state, problem):
            issues.append(f"{name} tidak feasible")
        if value != stored or value != getattr(result, name)["objective"]:
            issues.append(f"{name}: objective tersimpan {stored} != dihitung ulang {value}")
    if result.algorithm == "ga":
        issues += _check_ga_history(result, problem)
    else:
        issues += _check_local_search_history(result, problem)
    return issues


def _check_ga_history(result, problem) -> list[str]:
    issues, so_far = [], None
    for h in result.history:
        if h.get("best_individual") is not None:
            state = state_from_dict(h["best_individual"], problem)
            if not is_feasible(state, problem) or objective(state, problem) != h["best_fitness"]:
                issues.append(f"generasi {h['generation']}: best individual tidak konsisten")
        if so_far is not None and h["best_so_far_fitness"] < so_far:
            issues.append(f"generasi {h['generation']}: best-so-far turun")
        so_far = h["best_so_far_fitness"]
    if len(result.history) != result.config["generations"] + 1:
        issues.append("panjang history != generations + 1")
    if so_far is not None and so_far != result.final_objective:
        issues.append("final_objective != best-so-far terakhir")
    return issues


def _check_local_search_history(result, problem) -> list[str]:
    """hc/sa: current_state null = sama dengan frame sebelumnya, jadi state direkonstruksi lalu objective
    tiap frame dihitung ulang. hc: objective tidak turun, panjang = nmax + 1. sa: lihat _check_sa_history."""
    issues, state, previous = [], None, None
    key = "objective" if result.algorithm in HC_FAMILY else "current_objective"
    for h in result.history:
        if h["current_state"] is not None:
            state = state_from_dict(h["current_state"], problem)
            if not is_feasible(state, problem):
                issues.append(f"iterasi {h['iteration']}: current state tidak feasible")
        value = h[key]
        if state is not None and objective(state, problem) != value:
            issues.append(f"iterasi {h['iteration']}: objective {value} != current state")
        # random restart: awal tiap restart adalah state baru, jadi pengecekan non-menurun mulai ulang
        if result.algorithm == "hc_random_restart" and h["reason"] == "restart_initial":
            previous = None
        if result.algorithm in HC_FAMILY and previous is not None and value < previous:
            issues.append(f"iterasi {h['iteration']}: objective HC turun")
        previous = value
    if result.algorithm == "hc" and len(result.history) != result.config["max_iterations"] + 1:
        issues.append("panjang history != max_iterations + 1")
    if result.algorithm == "hc_random_restart":
        return issues + _check_random_restart_history(result, problem)
    final = state_from_dict(result.final_state, problem)
    if state is not None and state != final:
        issues.append("current state terakhir di history != final_state")
    if previous != result.final_objective:
        issues.append("objective iterasi terakhir != final_objective")
    if result.algorithm == "sa":
        issues += _check_sa_history(result, problem)
    return issues


def _check_random_restart_history(result, problem) -> list[str]:
    """final_state adalah hasil terbaik antar restart (bukan frame terakhir): cek best-so-far monoton dan
    sama dengan max objective sejauh ini, jumlah restart, nomor iterasi global, dan final = max peak."""
    issues, m, best = [], result.metrics, None
    expected_iteration = 0
    for h in result.history:
        if h["iteration"] != expected_iteration:
            issues.append(f"iterasi {h['iteration']}: nomor iterasi global tidak berlanjut")
        expected_iteration = h["iteration"] + 1
        best = h["objective"] if best is None else max(best, h["objective"])
        if h["best_so_far_objective"] != best:
            issues.append(f"iterasi {h['iteration']}: best_so_far tidak sama dengan max objective sejauh ini")
    restarts = {h["restart"] for h in result.history}
    if len(restarts) != m["restarts"] or restarts != set(range(m["restarts"])):
        issues.append("jumlah restart di history != metrics.restarts")
    for name in ("iterations_per_restart", "restart_initial_objectives", "restart_final_objectives",
                 "restart_termination_reasons"):
        if len(m[name]) != m["restarts"]:
            issues.append(f"panjang metrics.{name} != restarts")
    if result.final_objective != max(m["restart_final_objectives"]):
        issues.append("final_objective != max(restart_final_objectives)")
    if result.final_objective != best:
        issues.append("final_objective != best-so-far terakhir")
    return issues


def _check_sa_history(result, problem) -> list[str]:
    """best-so-far monoton = max current sejauh ini, peluang di [0, 1] dan konsisten dengan dE/T/draw,
    jumlah episode stuck = jumlah flag stuck_event, best-found feasible dan cocok."""
    issues, m, best = [], result.metrics, None
    for h in result.history:
        best = h["current_objective"] if best is None else max(best, h["current_objective"])
        if h["best_so_far_objective"] != best:
            issues.append(f"iterasi {h['iteration']}: best_so_far tidak sama dengan max current")
        p, delta = h["acceptance_probability"], h["delta_e"]
        if p is None:
            continue
        if not 0 <= p <= 1:
            issues.append(f"iterasi {h['iteration']}: peluang {p} di luar [0, 1]")
        elif delta >= 0 and (p != 1 or not h["accepted"]):
            issues.append(f"iterasi {h['iteration']}: dE >= 0 harus diterima dengan p = 1")
        elif delta < 0 and h["accepted"] != (h["random_draw"] < p):
            issues.append(f"iterasi {h['iteration']}: keputusan tidak sesuai draw < p")
    if len(result.history) != m["iterations_run"] + 1:
        issues.append("panjang history != iterations_run + 1")
    if m["stuck_event_count"] is not None and m["stuck_event_count"] != sum(h["stuck_event"] for h in result.history):
        issues.append("stuck_event_count != jumlah stuck_event di history")
    best_state = state_from_dict(m["best_found_state"], problem)
    if not is_feasible(best_state, problem) or objective(best_state, problem) != m["best_found_objective"] \
            or m["best_found_objective"] != best:
        issues.append("best_found_state/objective tidak konsisten")
    return issues


def cmd_validate(args) -> int:
    try:
        problem = load_problem(args.config)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"problem INVALID: {exc}")
        return 1
    ship = problem.ship
    area = sum(v.width * v.length for v in problem.vehicles)
    weight = sum(v.weight for v in problem.vehicles)
    print(f"problem valid: {problem.name} ({problem.metadata.get('source', '-')})")
    print(f"  kendaraan={len(problem.vehicles)}  deck={ship.width}x{ship.length} (area {ship.width * ship.length})"
          f"  max_capacity={ship.max_capacity}")
    print(f"  total area kendaraan={area}  total berat kendaraan={weight}")
    sample = random_feasible_state(problem, random.Random(0))
    print(f"  contoh random feasible state (seed 0): feasible={is_feasible(sample, problem)}"
          f" objective={objective(sample, problem)}")
    if args.result:
        try:
            issues = check_result(RunResult.load(args.result), problem)
        except (ValueError, KeyError, TypeError) as exc:
            issues = [f"result rusak/tidak terbaca: {exc!r}"]
        for issue in issues:
            print(f"  RESULT ISSUE: {issue}")
        print(f"result {args.result}: {'INVALID' if issues else 'konsisten (objective & feasibility dihitung ulang)'}")
        return 1 if issues else 0
    return 0


def _run_and_report(algorithm, args, config, default_out) -> int:
    """jalankan satu algoritma dari registry, simpan JSON + PNG, cetak ringkasan."""
    problem = load_problem(args.config)
    result = ALGORITHMS[algorithm](problem, config, args.seed)
    out = Path(args.output or default_out)
    result.save(out)
    plot_run(result, out.with_suffix(".png"))
    final = result.final_state
    print(f"{algorithm.upper()} seed={result.seed} config={json.dumps(config)}")
    print(f"  initial objective={result.initial_objective}")
    print(f"  final objective={result.final_objective}  weight={final['total_weight']}/{problem.ship.max_capacity}"
          f"  loaded={final['loaded_count']}/{len(problem.vehicles)}  feasible={final['feasible']}")
    if algorithm == "sa":
        m = result.metrics
        print(f"  best-found objective={m['best_found_objective']} (iterasi {m['best_found_iteration']})"
              f"  stuck episodes={m['stuck_event_count']}  iterasi={m['iterations_run']} ({m['termination_reason']})")
    print(f"  runtime={result.runtime_seconds:.3f}s (search saja)")
    print(f"  output: {out} dan {out.with_suffix('.png')}")
    return 0


def cmd_ga(args) -> int:
    config = {"population_size": args.population, "generations": args.generations,
              "crossover_rate": args.crossover_rate, "mutation_rate": args.mutation_rate,
              "elite_count": args.elite}
    return _run_and_report("ga", args, config, f"output/genetic_algorithm/cli/ga_seed{args.seed}"
                                                 f"_pop{args.population}_gen{args.generations}.json")


def cmd_sa(args) -> int:
    config = {"max_iterations": args.iterations, "initial_temperature": args.t0, "cooling_rate": args.alpha,
              "min_temperature": args.min_temp, "detect_stuck": not args.no_stuck}
    return _run_and_report("sa", args, config, f"output/simulated_annealing/cli/sa_seed{args.seed}"
                                                 f"_it{args.iterations}.json")


def cmd_hc(args) -> int:
    return _run_and_report("hc", args, {"max_iterations": args.iterations},
                           f"output/hill_climbing/cli/hc_seed{args.seed}_it{args.iterations}.json")


def cmd_experiment(args) -> int:
    rows = run_experiments(args.config)
    out = json.loads(Path(args.config).read_text())["output_dir"]
    print(f"selesai: {len(rows)} run. output di {out}/ (lihat SUMMARY.md, raw_results.csv, runs/, plots/)")
    return 0


def cmd_generate(args) -> int:
    problem = make_synthetic_problem(args.seed, args.vehicles)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(problem_to_dict(problem), indent=2))
    print(f"instance sintetis disimpan ke {args.output} (seed={args.seed}, kendaraan={args.vehicles})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main", description="IF3070 Tubes 1 - local search HC, SA, GA")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validasi problem JSON (dan opsional hasil run)")
    p.add_argument("--config", required=True)
    p.add_argument("--result", help="JSON hasil run untuk dicek ulang")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("ga", help="jalankan satu run genetic algorithm")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--population", type=int, required=True)
    p.add_argument("--generations", type=int, required=True)
    p.add_argument("--crossover-rate", type=float, default=0.9)
    p.add_argument("--mutation-rate", type=float, default=0.2)
    p.add_argument("--elite", type=int, default=1)
    p.add_argument("--output", help="path JSON hasil (default output/genetic_algorithm/cli/...)")
    p.set_defaults(func=cmd_ga)

    p = sub.add_parser("hc", help="jalankan satu run stochastic hill climbing")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, required=True, help="nmax")
    p.add_argument("--output", help="path JSON hasil (default output/hill_climbing/cli/...)")
    p.set_defaults(func=cmd_hc)

    p = sub.add_parser("sa", help="jalankan satu run simulated annealing")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, required=True, help="batas iterasi maksimum")
    p.add_argument("--t0", type=float, required=True, help="suhu awal T0")
    p.add_argument("--alpha", type=float, required=True, help="cooling rate geometric, 0 < alpha < 1")
    p.add_argument("--min-temp", type=float, required=True, help="berhenti bila T < nilai ini")
    p.add_argument("--no-stuck", action="store_true", help="matikan deteksi local optimum (lebih cepat)")
    p.add_argument("--output", help="path JSON hasil (default output/simulated_annealing/cli/...)")
    p.set_defaults(func=cmd_sa)

    p = sub.add_parser("experiment", help="jalankan eksperimen dari config (GA 18 run, atau hc/sa 3 run)")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("generate-instance", help="buat instance sintetis ber-seed")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--vehicles", type=int, default=30)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_generate)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
