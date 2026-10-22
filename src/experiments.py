"""matriks eksperimen GA wajib (2 kelompok x 3 konfigurasi x 3 run = 18 run), ringkasan, dan plot.

semua angka di output berasal dari eksekusi genetic_algorithm; tidak ada nilai yang di-hardcode.
"""

from __future__ import annotations

import csv
import json
import random
import textwrap
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")  # render ke file tanpa display
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.algorithms.genetic_algorithm import genetic_algorithm  # noqa: E402
from src.algorithms.hc_stochastic import hc_stochastic  # noqa: E402
from src.algorithms.hc_variants import HC_VARIANTS  # noqa: E402
from src.algorithms.simulated_annealing import simulated_annealing  # noqa: E402
from src.baseline import random_baseline  # noqa: E402
from src.core import Problem, Ship, Vehicle, load_problem  # noqa: E402

# registry tunggal algoritma yang sudah ada; dipakai runner, CLI, dan server
ALGORITHMS = {"ga": genetic_algorithm, "hc": hc_stochastic, "sa": simulated_annealing}


# instance sintetis 

def make_synthetic_problem(seed: int, n_vehicles: int = 30, ship=(10, 20, 500)) -> Problem:
    """instance demo ber-seed (BUKAN dataset resmi). default kapal mengikuti contoh spek:
    deck 10x20, MaxCapacity 500. total area dan berat sengaja melebihi kapasitas kapal
    supaya constraint geometri maupun berat sama-sama aktif."""
    rng = random.Random(seed)
    vehicles = []
    for k in range(1, n_vehicles + 1):
        w = rng.randint(1, 3)
        l = rng.randint(max(w, 2), 6)
        fee = w * l * rng.randint(3, 6) + rng.randint(0, 15)
        vehicles.append(Vehicle(f"V{k:02d}", w, l, fee, rng.randint(10, 60), rng.randint(1, 7)))
    params = {"seed": seed, "n_vehicles": n_vehicles, "ship": list(ship),
              "width": [1, 3], "length": ["max(width,2)", 6], "fee": "w*l*randint(3,6)+randint(0,15)",
              "weight": [10, 60], "eta": [1, 7]}
    return Problem(Ship(*ship), tuple(vehicles), name=f"synthetic-{seed}",
                   metadata={"source": "synthetic/demo instance (bukan data resmi)", "generator": params})


# matriks run 

def _check_group(name, values, seeds):
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError(f"{name}: butuh tepat 3 variasi parameter berbeda, dapat {values}")
    if len(seeds) != 3 or any(len(s) != 3 for s in seeds):
        raise ValueError(f"{name}: butuh 3 seed untuk tiap konfigurasi (3x3)")


def build_runs(config: dict) -> list[dict]:
    """terjemahkan experiment.json menjadi 18 spesifikasi run, lengkap dengan seed unik."""
    a, b = config["experiment_a"], config["experiment_b"]
    _check_group("experiment_a.generations", a["generations"], a["seeds"])
    _check_group("experiment_b.population_sizes", b["population_sizes"], b["seeds"])
    runs = []
    for gens, seeds in zip(a["generations"], a["seeds"]):
        runs += [dict(experiment="A", population_size=a["population_size"], generations=gens, seed=s) for s in seeds]
    for pop, seeds in zip(b["population_sizes"], b["seeds"]):
        runs += [dict(experiment="B", population_size=pop, generations=b["generations"], seed=s) for s in seeds]
    if len({r["seed"] for r in runs}) != len(runs):
        raise ValueError("seed antar run harus unik agar tiap run independen")
    for r in runs:
        r["config_label"] = f"pop{r['population_size']}_gen{r['generations']}"
        r["label"] = f"{r['experiment']}_{r['config_label']}_seed{r['seed']}"
    return runs


# plot 

def draw_deck(ax, state: dict, problem: dict, title: str) -> None:
    """gambar deck: x = kolom (kanan), y = baris (bawah), (0,0) pojok kiri atas."""
    ship = problem["ship"]
    W, L = ship["width"], ship["length"]
    cmap = plt.get_cmap("tab20")
    outside = []
    for k, p in enumerate(state["placements"]):
        if not p["loaded"]:
            outside.append(p["id"])
            continue
        ax.add_patch(Rectangle((p["x"], p["y"]), p["width"], p["length"], facecolor=cmap(k % 20),
                               edgecolor="black", linewidth=1.2, alpha=0.85))
        ax.text(p["x"] + p["width"] / 2, p["y"] + p["length"] / 2, f"{p['id']}\n{p['orientation']}",
                ha="center", va="center", fontsize=7)
    ax.set_xlim(0, W)
    ax.set_ylim(L, 0)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(L + 1))
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.4, color="0.8")
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nobjective={state['objective']}  weight={state['total_weight']}/{ship['max_capacity']}"
                 f"  loaded={state['loaded_count']}/{len(state['placements'])}", fontsize=9)
    ax.set_xlabel(textwrap.fill("di luar kapal: " + (", ".join(outside) or "-"), 45), fontsize=7)


def plot_convergence(ax, history, label="", color=None, legend=True):
    gens = [h["generation"] for h in history]
    ax.plot(gens, [h["best_fitness"] for h in history], color=color, label=f"max {label}".strip())
    ax.plot(gens, [h["average_fitness"] for h in history], color=color, linestyle="--",
            label=f"average {label}".strip())
    ax.set_xlabel("generasi (iterasi)")
    ax.set_ylabel("objective (total ShippingFee)")
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=7)


def plot_hc_curve(ax, history):
    """objective current HC per iterasi; titik hijau = iterasi saat candidate diterima."""
    its = [h["iteration"] for h in history]
    ax.plot(its, [h["objective"] for h in history], color="C0", label="current objective")
    acc = [h for h in history if h["accepted"]]
    ax.scatter([h["iteration"] for h in acc], [h["objective"] for h in acc], color="C2", s=14, zorder=3,
               label="improvement diterima")
    # random restart: garis putus-putus di awal tiap restart dan kurva best-so-far
    if history and "best_so_far_objective" in history[0]:
        for h in history:
            if h["reason"] == "restart_initial" and h["iteration"] > 0:
                ax.axvline(h["iteration"], color="0.6", linestyle="--", linewidth=0.8)
        ax.plot(its, [h["best_so_far_objective"] for h in history], color="C3", label="best-so-far")
    ax.set_xlabel("iterasi")
    ax.set_ylabel("objective (total ShippingFee)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def plot_sa_run(result, path) -> None:
    """SA: deck initial / terminal / best-found, kurva objective, dan kurva peluang penerimaan e^(dE/T)."""
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
    m, h, cfg = result.metrics, result.history, result.config
    draw_deck(fig.add_subplot(grid[0, 0]), result.initial_state, result.problem, "initial (random)")
    draw_deck(fig.add_subplot(grid[0, 1]), result.final_state, result.problem, "terminal (current terakhir)")
    draw_deck(fig.add_subplot(grid[0, 2]), m["best_found_state"], result.problem,
              f"best-found (iterasi {m['best_found_iteration']})")
    its = [x["iteration"] for x in h]
    ax = fig.add_subplot(grid[1, 0:2])
    ax.plot(its, [x["current_objective"] for x in h], color="C0", linewidth=0.8, label="current objective")
    ax.plot(its, [x["best_so_far_objective"] for x in h], color="C3", label="best-so-far")
    events = [x for x in h if x["stuck_event"]]
    ax.scatter([x["iteration"] for x in events], [x["current_objective"] for x in events], marker="v", color="k",
               s=18, zorder=3, label="masuk local optimum")
    ax.set_xlabel("iterasi")
    ax.set_ylabel("objective (total ShippingFee)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"SA seed={result.seed} T0={cfg['initial_temperature']} alpha={cfg['cooling_rate']} "
                 f"minT={cfg['min_temperature']}  iterasi={m['iterations_run']} ({m['termination_reason']})  "
                 f"stuck={m['stuck_event_count']}  runtime={result.runtime_seconds:.2f}s", fontsize=9)
    ax = fig.add_subplot(grid[1, 2])
    # e^(dE/T) hanya bermakna untuk candidate yang ada; dE >= 0 ditampilkan sebagai 1
    pts = [x for x in h if x["acceptance_probability"] is not None]
    ax.scatter([x["iteration"] for x in pts], [x["acceptance_probability"] for x in pts], s=3, color="C1",
               label="peluang terima e^(dE/T) (1 bila dE >= 0)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("iterasi")
    ax.set_ylabel("acceptance probability")
    ax.grid(True, alpha=0.3)
    twin = ax.twinx()
    temps = [x for x in h if x["temperature"] is not None]
    twin.plot([x["iteration"] for x in temps], [x["temperature"] for x in temps], color="C4", linewidth=1,
              label="temperature T")
    twin.set_ylabel("T", color="C4")
    ax.legend(fontsize=7, loc="upper right")
    twin.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_run(result, path) -> None:
    """satu PNG per run: initial state, final state, dan kurva objective sesuai algoritma."""
    if result.algorithm == "sa":
        return plot_sa_run(result, path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    cfg = result.config
    if result.algorithm == "ga":
        draw_deck(axes[0], result.initial_state, result.problem, "initial (best populasi awal)")
        draw_deck(axes[1], result.final_state, result.problem, "final (best-found)")
        plot_convergence(axes[2], result.history)
        title = f"GA seed={result.seed} pop={cfg['population_size']} gen={cfg['generations']}"
    else:
        draw_deck(axes[0], result.initial_state, result.problem, "initial (random)")
        draw_deck(axes[1], result.final_state, result.problem, "final (current terakhir)")
        plot_hc_curve(axes[2], result.history)
        names = {"hc": "Stochastic HC", "hc_sideways": "Sideways HC", "hc_random_restart": "Random Restart HC",
                 "hc_stochastic_aima": "Stochastic HC (AIMA)"}
        title = f"{names.get(result.algorithm, result.algorithm)} seed={result.seed}"
        if result.algorithm == "hc_random_restart":
            title += f" restarts={cfg['max_restarts']} langkah/restart={cfg['iterations_per_restart']}"
        else:
            title += f" max_iterations={cfg['max_iterations']}"
        if result.algorithm == "hc_sideways":
            title += f" max_sideways={cfg['max_sideways']}"
    axes[2].set_title(f"{title}\nruntime={result.runtime_seconds:.2f}s", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_group(experiment, grouped, path) -> None:
    """kurva tiga run untuk tiap konfigurasi dalam satu kelompok eksperimen."""
    fig, axes = plt.subplots(1, len(grouped), figsize=(6 * len(grouped), 4.5), sharey=True)
    for ax, (config_label, results) in zip(axes, grouped.items()):
        for k, result in enumerate(results):
            plot_convergence(ax, result.history, f"seed {result.seed}", color=f"C{k}")
        ax.set_title(f"Eksperimen {experiment}: {config_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# runner 

def _summary_rows(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["config_label"]), []).append(r)
    out = []
    for (experiment, label), rs in groups.items():
        finals = [r["final_objective"] for r in rs]
        times = [r["runtime_seconds"] for r in rs]
        out.append({
            "experiment": experiment,
            "config_label": label,
            "population_size": rs[0]["population_size"],
            "generations": rs[0]["generations"],
            "runs": len(rs),
            "mean_final_objective": round(mean(finals), 3),
            "std_final_objective": round(stdev(finals), 3) if len(finals) > 1 else 0.0,
            "min_final_objective": min(finals),
            "max_final_objective": max(finals),
            "mean_initial_objective": round(mean(r["initial_objective"] for r in rs), 3),
            "mean_runtime_seconds": round(mean(times), 4),
            "std_runtime_seconds": round(stdev(times), 4) if len(times) > 1 else 0.0,
        })
    return out


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows, keys):
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def run_repeated(config: dict) -> list[dict]:
    """eksperimen hc/sa dan varian hc: satu konfigurasi dijalankan untuk tiap seed (spek: minimal 3 run)."""
    algorithm, seeds = config["algorithm"], config["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("butuh minimal 3 seed unik agar run independen")
    problem = load_problem(config["problem"])
    out = Path(config["output_dir"])
    solver = ALGORITHMS.get(algorithm) or HC_VARIANTS[algorithm]
    rows = []
    for seed in seeds:
        result = solver(problem, config["config"], seed)
        json_path, plot_path = Path("runs") / f"{algorithm}_seed{seed}.json", Path("plots") / f"{algorithm}_seed{seed}.png"
        result.save(out / json_path)
        plot_run(result, out / plot_path)
        rows.append({
            "algorithm": algorithm,
            "seed": seed,
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
            "best_found_objective": result.metrics.get("best_found_objective", result.final_objective),
            "final_total_weight": result.final_state["total_weight"],
            "final_loaded_count": result.final_state["loaded_count"],
            "iterations": len(result.history) - 1,
            "runtime_seconds": round(result.runtime_seconds, 4),
            # khusus sa: jumlah episode stuck dan porsi runtime untuk deteksi local optimum (sudah termasuk runtime)
            "stuck_event_count": result.metrics.get("stuck_event_count"),
            "local_optimum_check_seconds": round(result.metrics.get("local_optimum_check_seconds", 0.0), 4),
            "json_path": str(json_path),
            "plot_path": str(plot_path),
        })
        print(f"{algorithm} seed={seed}: final={result.final_objective} runtime={result.runtime_seconds:.2f}s",
              flush=True)
    _write_csv(out / "raw_results.csv", rows)
    finals = [r["final_objective"] for r in rows]
    times = [r["runtime_seconds"] for r in rows]
    (out / "SUMMARY.md").write_text(
        f"# Ringkasan eksperimen {algorithm.upper()} (dihasilkan otomatis)\n\n"
        f"- problem: `{config['problem']}` ({problem.name}; {problem.metadata.get('source', '-')})\n"
        f"- config: `{json.dumps(config['config'])}`\n"
        f"- final objective: mean={mean(finals):.2f}, std={stdev(finals):.2f}, min={min(finals)}, max={max(finals)}\n"
        f"- runtime search (s): mean={mean(times):.4f}, std={stdev(times):.4f}\n\n"
        f"{_markdown_table(rows, list(rows[0])[:11])}\n"
    )
    return rows


def run_experiments(config_path) -> list[dict]:
    """jalankan eksperimen dari file config: matriks GA 18 run, atau repeated run hc/sa
    bila config punya key "algorithm"."""
    config = json.loads(Path(config_path).read_text())
    if "algorithm" in config:
        return run_repeated(config)
    problem = load_problem(config["problem"])
    out = Path(config["output_dir"])
    runs = build_runs(config)
    defaults = config.get("ga_defaults", {})
    rows, grouped = [], {"A": {}, "B": {}}

    for k, run in enumerate(runs, 1):
        result = genetic_algorithm(problem, {**defaults, "population_size": run["population_size"],
                                             "generations": run["generations"]}, run["seed"])
        json_path, plot_path = Path("runs") / f"{run['label']}.json", Path("plots") / f"{run['label']}.png"
        result.save(out / json_path)
        plot_run(result, out / plot_path)
        grouped[run["experiment"]].setdefault(run["config_label"], []).append(result)
        rows.append({
            "experiment": run["experiment"],
            "config_label": run["config_label"],
            "population_size": run["population_size"],
            "generations": run["generations"],
            "seed": run["seed"],
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
            "final_total_weight": result.final_state["total_weight"],
            "final_loaded_count": result.final_state["loaded_count"],
            "runtime_seconds": round(result.runtime_seconds, 4),
            "json_path": str(json_path),
            "plot_path": str(plot_path),
        })
        print(f"[{k:2d}/{len(runs)}] {run['label']}: final={result.final_objective} "
              f"runtime={result.runtime_seconds:.2f}s", flush=True)

    summary = _summary_rows(rows)
    _write_csv(out / "raw_results.csv", rows)
    _write_csv(out / "summary.csv", summary)
    for experiment, by_config in grouped.items():
        plot_group(experiment, by_config, out / f"experiment_{experiment}_convergence.png")

    baseline_cfg = config.get("baseline", {"samples": 1000, "seed": 0})
    base = random_baseline(problem, baseline_cfg["samples"], baseline_cfg["seed"])
    keys = ["experiment", "config_label", "runs", "mean_final_objective", "std_final_objective",
            "min_final_objective", "max_final_objective", "mean_initial_objective", "mean_runtime_seconds"]
    (out / "SUMMARY.md").write_text(
        f"# Ringkasan eksperimen GA (dihasilkan otomatis)\n\n"
        f"- problem: `{config['problem']}` ({problem.name}, {len(problem.vehicles)} kendaraan, "
        f"deck {problem.ship.width}x{problem.ship.length}, MaxCapacity {problem.ship.max_capacity})\n"
        f"- sumber data: {problem.metadata.get('source', '-')}\n"
        f"- parameter GA tetap: `{json.dumps(defaults)}`\n"
        f"- total run: {len(rows)}; runtime = waktu search saja (tanpa serialisasi/plot)\n"
        f"- referensi random feasible baseline ({base['samples']} sampel, seed {base['seed']}): "
        f"best={base['best_objective']}, mean={base['mean_objective']:.1f}\n\n"
        f"## Ringkasan per konfigurasi\n\n{_markdown_table(summary, keys)}\n\n"
        f"## Raw per run\n\n{_markdown_table(rows, list(rows[0])[:10])}\n"
    )
    return rows
