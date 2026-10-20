"""Hill climbing dengan random restart (Salindia IF3070 Materi03 hlm. 28).

Hasil dikemas dalam RunResult (lihat src/result.py).

Tiap restart memakai steepest ascent murni (climb_to_peak di hc_common.py) dari state awal
acak yang baru. Seluruh restart memakai satu random_generator yang sama dan terus dilanjutkan.

Config yang diterima:
    max_restarts          : integer >= 1, jumlah restart (default 10)
    iterations_per_restart: integer >= 0, batas langkah tiap restart (default 1000)
    record_snapshots      : opsional, default True

Key tiap entri history:
    iteration (nomor global, berlanjut lintas restart), objective, candidate_objective,
    accepted, improved, move, reason, current_state,
    restart (indeks restart mulai dari 0), best_so_far_objective
Awal tiap restart dicatat dengan reason "restart_initial".

Pseudocode:
    best_state <- kosong
    ulangi max_restarts kali:
        initial_state <- state awal acak yang baru
        peak_state <- climb_to_peak(initial_state, iterations_per_restart)
        jika nilai(peak_state) > nilai(best_state): best_state <- peak_state
    kembalikan best_state

Keputusan desain: hasil restart hanya menggantikan best_state kalau STRICTLY lebih besar,
jadi kalau seri maka yang lebih awal dipertahankan.

Urutan Pengerjaan:
1. Membaca dan memeriksa config
2. Mendefinisikan variabel awal yang akan dipakai
3. Logika algoritma (inti pencarian)
4. Mengubah catatan iterasi (iteration_log) menjadi history
5. Meringkas metrics hasil iterasi dan algoritma
6. Mengemas semua hasil akhir menjadi RunResult
"""

import random
import time

from src.algorithms.hc_common import check_integer_config, climb_to_peak
from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import random_feasible_state


def hc_random_restart(problem, config, seed):
    # 1. Membaca dan memeriksa config
    max_restarts = check_integer_config("max_restarts", config.get("max_restarts", 10), 1)
    iterations_per_restart_limit = check_integer_config(
        "iterations_per_restart", config.get("iterations_per_restart", 1000), 0)

    # Kalau tidak diisi, snapshot disimpan (default True)
    save_snapshots = config.get("record_snapshots", True)

    # 2. Mendefinisikan variabel awal yang akan dipakai
    # Satu generator acak untuk seluruh run supaya hasil bisa diulang dengan seed yang sama
    random_generator = random.Random(seed)

    iterations_per_restart = []  # jumlah langkah yang benar-benar dijalankan tiap restart
    restart_initial_objectives = []  # nilai state awal tiap restart
    restart_final_objectives = []  # nilai peak yang dicapai tiap restart
    restart_termination_reasons = []  # alasan berhenti tiap restart
    neighbors_evaluated = 0  # jumlah neighbor yang dinilai objective-nya, dari semua restart

    best_state = None  # state terbaik di antara semua restart
    best_value = None
    best_restart_index = 0
    initial_state = None  # state awal restart pertama, dilaporkan sebagai initial_state

    # Catatan tiap iterasi, nanti diubah menjadi history pada bagian 4
    iteration_log = []
    global_iteration = 0  # nomor iterasi yang berlanjut terus lintas restart

    # 3. Logika algoritma (inti pencarian), hanya bagian ini yang dihitung waktunya
    start_time = time.perf_counter()

    for restart_index in range(max_restarts):
        # State awal baru untuk restart ini, dijamin feasible
        restart_state = random_feasible_state(problem, random_generator)
        restart_value = objective(restart_state, problem)
        if restart_index == 0:
            initial_state = restart_state

        # Nilai terbaik yang diketahui SETELAH state awal restart ini ikut dipertimbangkan
        # (best_so_far dicatat di tiap entri supaya kurvanya bisa digambar)
        if best_value is None or restart_value > best_value:
            best_so_far = restart_value
        else:
            best_so_far = best_value

        iteration_log.append({
            "iteration": global_iteration,
            "restart": restart_index,
            "state": restart_state,
            "value": restart_value,
            "neighbor_value": None,
            "accepted": False,
            "move": None,
            "reason": "restart_initial",
            "best_so_far": best_so_far,
        })

        # Naik sampai peak atau sampai batas langkah per restart
        peak_state, step_log, termination_reason, evaluated_now = climb_to_peak(
            restart_state, problem, random_generator, iterations_per_restart_limit)
        neighbors_evaluated += evaluated_now
        peak_value = objective(peak_state, problem)

        # Pindahkan catatan langkah ke iteration_log dengan nomor iterasi global
        for step_entry in step_log:
            global_iteration += 1
            if step_entry["value"] > best_so_far:
                best_so_far = step_entry["value"]
            iteration_log.append({
                "iteration": global_iteration,
                "restart": restart_index,
                "state": step_entry["state"],
                "value": step_entry["value"],
                "neighbor_value": step_entry["neighbor_value"],
                "accepted": step_entry["accepted"],
                "move": step_entry["move"],
                "reason": step_entry["reason"],
                "best_so_far": best_so_far,
            })
        # Entri restart_initial berikutnya memakai nomor global yang baru
        global_iteration += 1

        iterations_per_restart.append(len(step_log))
        restart_initial_objectives.append(restart_value)
        restart_final_objectives.append(peak_value)
        restart_termination_reasons.append(termination_reason)

        # Hasil terbaik hanya diganti kalau STRICTLY lebih besar (seri mempertahankan yang awal)
        if best_value is None or peak_value > best_value:
            best_state = peak_state
            best_value = peak_value
            best_restart_index = restart_index

    runtime_seconds = time.perf_counter() - start_time

    # 4. Mengubah catatan iterasi (iteration_log) menjadi history
    history = []
    for entry in iteration_log:
        # Snapshot ukurannya besar, jadi hanya disimpan saat state berubah:
        # awal restart (state baru) dan langkah yang diterima
        state_changed = entry["reason"] == "restart_initial" or entry["accepted"]
        if save_snapshots and state_changed:
            state_snapshot = snapshot(entry["state"], problem)
        else:
            state_snapshot = None  # None artinya "sama dengan entri sebelumnya"

        history.append({
            "iteration": entry["iteration"],
            "objective": entry["value"],
            "candidate_objective": entry["neighbor_value"],
            "accepted": entry["accepted"],
            "improved": entry["accepted"],  # Di steepest ascent, diterima berarti membaik
            "move": list(entry["move"]) if entry["move"] else None,
            "reason": entry["reason"],
            "current_state": state_snapshot,
            "restart": entry["restart"],
            "best_so_far_objective": entry["best_so_far"],
        })

    # 5. Meringkas metrics hasil iterasi dan algoritma
    # objective() dipanggil 1 kali per state awal restart, lalu 1 kali per neighbor yang dinilai
    objective_evaluations = max_restarts + neighbors_evaluated

    metrics = {
        "restarts": max_restarts,
        "max_restarts": max_restarts,
        "iterations_per_restart_limit": iterations_per_restart_limit,
        "iterations_per_restart": iterations_per_restart,
        "total_iterations": sum(iterations_per_restart),
        "restart_initial_objectives": restart_initial_objectives,
        "restart_final_objectives": restart_final_objectives,
        "restart_termination_reasons": restart_termination_reasons,
        "best_restart_index": best_restart_index,
        "best_found_objective": best_value,
        "neighbors_evaluated": neighbors_evaluated,
        "objective_evaluations": objective_evaluations,
    }

    # 6. Mengemas semua hasil akhir menjadi RunResult
    return RunResult(
        algorithm="hc_random_restart",
        seed=seed,
        config={
            "max_restarts": max_restarts,
            "iterations_per_restart": iterations_per_restart_limit,
            "record_snapshots": save_snapshots,
        },
        problem=problem_to_dict(problem),
        initial_state=snapshot(initial_state, problem),
        final_state=snapshot(best_state, problem),
        initial_objective=restart_initial_objectives[0],
        final_objective=best_value,
        runtime_seconds=runtime_seconds,
        history=history,
        metrics=metrics,
    )
