"""Stochastic hill climbing (versi Salindia IF3070 Materi03 hlm. 29).

Catatan istilah: kata "stochastic" di file ini mengikuti salindia, yaitu mengambil SATU
neighbor acak per iterasi. AIMA memakai definisi yang berbeda (memilih acak di antara
uphill moves), dan versi itu ada di hc_stochastic_aima.py.

Hasil dikemas dalam RunResult (lihat src/result.py).

Config yang diterima:
    max_iterations   : integer >= 0, jumlah iterasi
    record_snapshots : opsional, default True

Key tiap entri history:
    iteration, objective, candidate_objective, accepted, improved, move, reason, current_state

Pseudocode:
    current_state <- state awal acak
    ulangi sebanyak max_iterations kali:
        neighbor_state <- SATU neighbor acak dari current_state
        jika nilai(neighbor_state) > nilai(current_state):
            current_state <- neighbor_state

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

from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import describe_move, random_feasible_neighbor, random_feasible_state


def hc_stochastic(problem, config, seed):
    # 1. Membaca dan memeriksa config
    max_iterations = config["max_iterations"]

    # True/False di Python dianggap angka juga (bool turunan int), jadi harus ditolak manual
    is_valid_number = isinstance(max_iterations, int) and not isinstance(max_iterations, bool)
    if not is_valid_number or max_iterations < 0:
        raise ValueError("max_iterations harus integer >= 0")

    # Kalau tidak diisi, snapshot disimpan (default True)
    save_snapshots = config.get("record_snapshots", True)

    # 2. Mendefinisikan variabel awal yang akan dipakai
    # Satu generator acak untuk seluruh run supaya hasil bisa diulang dengan seed yang sama
    random_generator = random.Random(seed)

    # Penghitung: berapa kali tiap hasil keputusan terjadi
    count_improved = 0  # neighbor lebih baik dan diterima
    count_equal = 0  # neighbor bernilai sama dan ditolak
    count_worse = 0  # neighbor lebih buruk dan ditolak
    count_no_neighbor = 0  # iterasi yang tidak menemukan neighbor feasible
    last_improvement_iteration = 0  # iterasi terakhir saat current_state membaik

    # Catatan tiap iterasi, nanti diubah menjadi history pada bagian 4
    iteration_log = []

    # 3. Logika algoritma (inti pencarian), hanya bagian ini yang dihitung waktunya
    start_time = time.perf_counter()

    # State awal dipilih acak dan dijamin feasible
    current_state = random_feasible_state(problem, random_generator)
    current_value = objective(current_state, problem)

    # Disimpan untuk dilaporkan di hasil akhir
    initial_state = current_state
    initial_value = current_value

    # Iterasi 0 mencatat state awal, belum ada neighbor yang dinilai
    iteration_log.append({
        "iteration": 0,
        "state": current_state,
        "value": current_value,
        "neighbor_value": None,
        "accepted": False,
        "move": None,
        "reason": "initial",
    })

    for iteration in range(1, max_iterations + 1):
        # Ambil SATU neighbor acak, hasilnya None kalau tidak ketemu yang feasible
        neighbor_state = random_feasible_neighbor(current_state, problem, random_generator)

        # Kasus 1: tidak ada neighbor feasible, state tidak berubah
        if neighbor_state is None:
            count_no_neighbor += 1
            iteration_log.append({
                "iteration": iteration,
                "state": current_state,
                "value": current_value,
                "neighbor_value": None,
                "accepted": False,
                "move": None,
                "reason": "no_feasible_candidate",
            })
            continue  # lanjut ke iterasi berikutnya

        neighbor_value = objective(neighbor_state, problem)
        move = describe_move(current_state, neighbor_state, problem)

        # Kasus 2: neighbor lebih baik, pindah ke neighbor
        if neighbor_value > current_value:
            current_state = neighbor_state
            current_value = neighbor_value
            count_improved += 1
            last_improvement_iteration = iteration
            accepted = True
            reason = "improved"

        # Kasus 3: nilainya sama, tidak pindah
        elif neighbor_value == current_value:
            count_equal += 1
            accepted = False
            reason = "equal"

        # Kasus 4: neighbor lebih buruk, tidak pindah
        else:
            count_worse += 1
            accepted = False
            reason = "worse"

        # Yang dicatat adalah state SETELAH keputusan diambil
        iteration_log.append({
            "iteration": iteration,
            "state": current_state,
            "value": current_value,
            "neighbor_value": neighbor_value,
            "accepted": accepted,
            "move": move,
            "reason": reason,
        })

    runtime_seconds = time.perf_counter() - start_time

    # 4. Mengubah catatan iterasi (iteration_log) menjadi history
    history = []
    for entry in iteration_log:
        # Snapshot ukurannya besar, jadi hanya disimpan di iterasi 0 dan saat state berubah
        state_changed = entry["iteration"] == 0 or entry["accepted"]
        if save_snapshots and state_changed:
            state_snapshot = snapshot(entry["state"], problem)
        else:
            state_snapshot = None  # None artinya "sama dengan entri sebelumnya"

        history.append({
            "iteration": entry["iteration"],
            "objective": entry["value"],
            "candidate_objective": entry["neighbor_value"],
            "accepted": entry["accepted"],
            "improved": entry["accepted"],  # Di hill climbing, diterima berarti membaik
            "move": list(entry["move"]) if entry["move"] else None,
            "reason": entry["reason"],
            "current_state": state_snapshot,
        })

    # 5. Meringkas metrics hasil iterasi dan algoritma
    # objective() dipanggil 1 kali untuk state awal, lalu 1 kali per neighbor yang dinilai
    # (iterasi tanpa neighbor tidak memanggil objective)
    objective_evaluations = 1 + max_iterations - count_no_neighbor

    metrics = {
        "improvements": count_improved,
        "rejected_equal": count_equal,
        "rejected_worse": count_worse,
        "no_candidate": count_no_neighbor,
        "iterations": max_iterations,
        "last_improvement_iteration": last_improvement_iteration,
        "objective_evaluations": objective_evaluations,
    }

    # 6. Mengemas semua hasil akhir menjadi RunResult
    return RunResult(
        algorithm="hc",
        seed=seed,
        config={"max_iterations": max_iterations, "record_snapshots": save_snapshots},
        problem=problem_to_dict(problem),
        initial_state=snapshot(initial_state, problem),
        final_state=snapshot(current_state, problem),
        initial_objective=initial_value,
        final_objective=current_value,
        runtime_seconds=runtime_seconds,
        history=history,
        metrics=metrics,
    )
