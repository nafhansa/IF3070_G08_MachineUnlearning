"""Stochastic hill climbing versi AIMA (Russell & Norvig, 4th ed., Sec. 4.1).

Hasil dikemas dalam RunResult (lihat src/result.py).

AIMA mendefinisikan stochastic hill climbing sebagai: memilih secara acak di antara uphill
moves, dan peluang pemilihan dapat bergantung pada kecuramannya. Ini BERBEDA dari
hc_stochastic.py. Versi salindia (Materi03 hlm. 29) mengambil satu neighbor acak lalu
menerimanya hanya kalau lebih baik, sedangkan versi AIMA mengumpulkan SEMUA uphill neighbor
lebih dulu lalu memilih satu di antaranya. Keduanya dipertahankan karena istilah yang sama
dipakai dengan arti berbeda di salindia dan di buku, sehingga hasilnya bisa dibandingkan.

Config yang diterima:
    max_iterations   : integer >= 0, batas pengaman jumlah iterasi (default 1000)
    selection        : "proportional" (default) atau "uniform"
                       proportional: peluang sebanding dengan selisih nilai (kecuraman)
                       uniform     : semua uphill neighbor berpeluang sama
    record_snapshots : opsional, default True

Key tiap entri history:
    iteration, objective, candidate_objective, accepted, improved, move, reason,
    current_state, uphill_count (jumlah uphill neighbor yang tersedia pada iterasi itu)

Keputusan desain (mode proportional): bobot tiap uphill neighbor adalah
nilai(neighbor) - nilai(current). Bobot selalu > 0 karena hanya neighbor dengan nilai lebih
besar yang dikumpulkan, jadi tidak ada bobot nol atau negatif.

Pseudocode:
    current_state <- state awal acak
    ulangi sampai berhenti:
        uphill_neighbors <- semua neighbor feasible dengan nilai > nilai(current_state)
        jika uphill_neighbors kosong: berhenti ("local_optimum")
        pilih satu neighbor dari uphill_neighbors sesuai selection, lalu pindah
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

from src.algorithms.hc_common import check_integer_config
from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import generate_feasible_neighbors, random_feasible_state

SELECTION_MODES = ("proportional", "uniform")


def hc_stochastic_aima(problem, config, seed):
    # 1. Membaca dan memeriksa config
    max_iterations = check_integer_config("max_iterations", config.get("max_iterations", 1000), 0)

    selection = config.get("selection", "proportional")
    if selection not in SELECTION_MODES:
        raise ValueError("selection harus 'proportional' atau 'uniform'")

    # Kalau tidak diisi, snapshot disimpan (default True)
    save_snapshots = config.get("record_snapshots", True)

    # 2. Mendefinisikan variabel awal yang akan dipakai
    # Satu generator acak untuk seluruh run supaya hasil bisa diulang dengan seed yang sama
    random_generator = random.Random(seed)

    count_improved = 0  # langkah uphill yang diambil
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
        "uphill_count": 0,
        "move": None,
        "reason": "initial",
    })

    for iteration in range(1, max_iterations + 1):
        # Kumpulkan SEMUA uphill neighbor (nilai lebih besar dari current)
        uphill_moves = []
        uphill_states = []
        uphill_values = []
        for move, neighbor_state in generate_feasible_neighbors(current_state, problem):
            neighbor_value = objective(neighbor_state, problem)
            neighbors_evaluated += 1
            if neighbor_value > current_value:
                uphill_moves.append(move)
                uphill_states.append(neighbor_state)
                uphill_values.append(neighbor_value)

        # Kasus 1: tidak ada uphill move, sudah di local optimum
        if len(uphill_states) == 0:
            iteration_log.append({
                "iteration": iteration,
                "state": current_state,
                "value": current_value,
                "neighbor_value": None,
                "accepted": False,
                "uphill_count": 0,
                "move": None,
                "reason": "local_optimum",
            })
            termination_reason = "local_optimum"
            break

        # Kasus 2: ada uphill move, pilih satu sesuai mode selection
        if selection == "proportional":
            # Bobot = kecuraman (selisih nilai), makin curam makin besar peluangnya
            weights = []
            for uphill_value in uphill_values:
                weights.append(uphill_value - current_value)
            chosen_index = random_generator.choices(range(len(uphill_states)), weights=weights)[0]
        else:
            chosen_index = random_generator.randrange(len(uphill_states))

        move = uphill_moves[chosen_index]
        current_state = uphill_states[chosen_index]
        current_value = uphill_values[chosen_index]
        count_improved += 1
        last_improvement_iteration = iteration

        # Yang dicatat adalah state SETELAH keputusan diambil
        iteration_log.append({
            "iteration": iteration,
            "state": current_state,
            "value": current_value,
            "neighbor_value": current_value,
            "accepted": True,
            "uphill_count": len(uphill_states),
            "move": move,
            "reason": "improved",
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
            "improved": entry["accepted"],  # Di varian ini, diterima berarti membaik
            "move": list(entry["move"]) if entry["move"] else None,
            "reason": entry["reason"],
            "current_state": state_snapshot,
            "uphill_count": entry["uphill_count"],
        })

    # 5. Meringkas metrics hasil iterasi dan algoritma
    # objective() dipanggil 1 kali untuk state awal, lalu 1 kali per neighbor yang dinilai
    objective_evaluations = 1 + neighbors_evaluated

    metrics = {
        "iterations": iteration_log[-1]["iteration"],  # iterasi terakhir yang dijalankan
        "improvements": count_improved,
        "termination_reason": termination_reason,
        "selection": selection,
        "neighbors_evaluated": neighbors_evaluated,
        "objective_evaluations": objective_evaluations,
        "last_improvement_iteration": last_improvement_iteration,
    }

    # 6. Mengemas semua hasil akhir menjadi RunResult
    return RunResult(
        algorithm="hc_stochastic_aima",
        seed=seed,
        config={
            "max_iterations": max_iterations,
            "selection": selection,
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
