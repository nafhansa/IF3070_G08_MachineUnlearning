"""Hill climbing dengan sideways move (Salindia IF3070 Materi03 hlm. 27).

Hasil dikemas dalam RunResult (lihat src/result.py).

Config yang diterima:
    max_sideways     : integer >= 0, batas sideways move BERUTURUTAN (default 100)
    max_iterations   : integer >= 0, batas pengaman jumlah iterasi (default 1000)
    record_snapshots : opsional, default True

Key tiap entri history:
    iteration, objective, candidate_objective, accepted, improved, move, reason,
    current_state, sideways (True kalau langkah itu adalah sideways move)

Keputusan desain: batas max_sideways dihitung sebagai sideways move yang BERURUTAN, bukan
total. Salindia hlm. 27 dan AIMA membatasi sideways move yang terjadi berturut-turut dan
counter-nya kembali ke 0 setelah ada langkah yang membaik. Spek menulis "banyak sideways
move yang dilakukan sudah mencapai maksimum", yang bisa dibaca sebagai total, jadi tafsiran
ini sengaja ditulis eksplisit. Jumlah total tetap dicatat di metrics (sideways_moves_total).

Pseudocode:
    current_state <- state awal acak
    consecutive_sideways <- 0
    ulangi sampai berhenti:
        best_neighbor_value <- nilai neighbor terbaik dari current_state
        jika best_neighbor_value > nilai(current_state):
            pindah ke salah satu neighbor terbaik (acak), consecutive_sideways <- 0
        jika best_neighbor_value == nilai(current_state):
            jika consecutive_sideways >= max_sideways: berhenti ("max_sideways")
            kalau belum: pindah ke salah satu neighbor seri (acak), consecutive_sideways + 1
        jika best_neighbor_value < nilai(current_state) atau tidak ada neighbor:
            berhenti ("local_optimum")
        jika jumlah iterasi mencapai max_iterations: berhenti ("max_iterations")

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

from src.algorithms.hc_common import check_integer_config, find_best_neighbors
from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import random_feasible_state


def hc_sideways(problem, config, seed):
    # 1. Membaca dan memeriksa config
    max_sideways = check_integer_config("max_sideways", config.get("max_sideways", 100), 0)
    max_iterations = check_integer_config("max_iterations", config.get("max_iterations", 1000), 0)

    # Kalau tidak diisi, snapshot disimpan (default True)
    save_snapshots = config.get("record_snapshots", True)

    # 2. Mendefinisikan variabel awal yang akan dipakai
    # Satu generator acak untuk seluruh run supaya hasil bisa diulang dengan seed yang sama
    random_generator = random.Random(seed)

    # Penghitung: berapa kali tiap hasil keputusan terjadi
    count_improved = 0  # langkah yang membuat nilai naik
    total_sideways = 0  # seluruh sideways move selama run
    consecutive_sideways = 0  # sideways move beruntun sejak langkah terakhir yang membaik
    max_consecutive_sideways = 0  # rekor consecutive_sideways tertinggi selama run
    neighbors_evaluated = 0  # jumlah neighbor yang dinilai objective-nya
    last_improvement_iteration = 0  # iterasi terakhir saat nilai current naik
    termination_reason = "max_iterations"  # alasan berhenti, diganti kalau loop di-break

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
        "sideways": False,
        "move": None,
        "reason": "initial",
    })

    for iteration in range(1, max_iterations + 1):
        # Periksa SEMUA neighbor, lalu ambil nilai terbaik dan semua neighbor yang seri
        best_neighbor_value, tied_best_neighbors, evaluated_now = find_best_neighbors(current_state, problem)
        neighbors_evaluated += evaluated_now

        # Kasus 1: tidak ada neighbor feasible atau semua neighbor lebih buruk, sudah di peak
        if best_neighbor_value is None or best_neighbor_value < current_value:
            iteration_log.append({
                "iteration": iteration,
                "state": current_state,
                "value": current_value,
                "neighbor_value": best_neighbor_value,
                "accepted": False,
                "sideways": False,
                "move": None,
                "reason": "local_optimum",
            })
            termination_reason = "local_optimum"
            break

        # Kasus 2: neighbor terbaik lebih baik, pindah dan counter sideways direset
        if best_neighbor_value > current_value:
            move, neighbor_state = random_generator.choice(tied_best_neighbors)
            current_state = neighbor_state
            current_value = best_neighbor_value
            consecutive_sideways = 0
            count_improved += 1
            last_improvement_iteration = iteration
            is_sideways = False
            reason = "improved"

        # Kasus 3: neighbor terbaik bernilai sama (plateau), boleh sideways selama belum mencapai batas
        else:
            if consecutive_sideways >= max_sideways:
                iteration_log.append({
                    "iteration": iteration,
                    "state": current_state,
                    "value": current_value,
                    "neighbor_value": best_neighbor_value,
                    "accepted": False,
                    "sideways": False,
                    "move": None,
                    "reason": "max_sideways",
                })
                termination_reason = "max_sideways"
                break

            move, neighbor_state = random_generator.choice(tied_best_neighbors)
            current_state = neighbor_state
            consecutive_sideways += 1
            total_sideways += 1
            if consecutive_sideways > max_consecutive_sideways:
                max_consecutive_sideways = consecutive_sideways
            is_sideways = True
            reason = "sideways"

        # Yang dicatat adalah state SETELAH keputusan diambil
        iteration_log.append({
            "iteration": iteration,
            "state": current_state,
            "value": current_value,
            "neighbor_value": best_neighbor_value,
            "accepted": True,
            "sideways": is_sideways,
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
            "improved": entry["accepted"] and not entry["sideways"],  # sideways diterima tetapi tidak membaik
            "move": list(entry["move"]) if entry["move"] else None,
            "reason": entry["reason"],
            "current_state": state_snapshot,
            "sideways": entry["sideways"],
        })

    # 5. Meringkas metrics hasil iterasi dan algoritma
    # objective() dipanggil 1 kali untuk state awal, lalu 1 kali per neighbor yang dinilai
    objective_evaluations = 1 + neighbors_evaluated

    metrics = {
        "iterations": iteration_log[-1]["iteration"],  # iterasi terakhir yang dijalankan
        "improvements": count_improved,
        "sideways_moves_total": total_sideways,
        "max_consecutive_sideways": max_consecutive_sideways,
        "termination_reason": termination_reason,
        "neighbors_evaluated": neighbors_evaluated,
        "objective_evaluations": objective_evaluations,
        "last_improvement_iteration": last_improvement_iteration,
    }

    # 6. Mengemas semua hasil akhir menjadi RunResult
    return RunResult(
        algorithm="hc_sideways",
        seed=seed,
        config={
            "max_sideways": max_sideways,
            "max_iterations": max_iterations,
            "record_snapshots": save_snapshots,
        },
        problem=problem_to_dict(problem),
        initial_state=snapshot(initial_state, problem),
        final_state=snapshot(current_state, problem),
        initial_objective=initial_value,
        final_objective=current_value,
        runtime_seconds=runtime_seconds,
        history=history,
        metrics=metrics,
    )
