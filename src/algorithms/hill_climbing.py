"""stochastic hill climbing (bagian Nafhan).
Lihat src/result.py (RunResult)
Config: max_iterations; opsional: record_snapshots.
History per entry: iteration, objective, candidate_objective, accepted, improved, move, reason, current_state.

Algoritma:
    current <- state awal acak
    looping sebanyak nmax kali:
        cand <- SATU neighbor secara acak dari yang current
        jika nilai(cand) > nilai(current): current <- cand

Urutan Pengerjaan:
1. Reading and checking available configs
2. Mendefinisikan variabel - variabel awal yang akan digunakan
3. Logika algoritma untuk mencari neighbor
4. Convert current CATETAN menjadi sebuah HISTORY
5. Meringkas metriks hasil iterasi dan algoritma
6. Covering semua hasil akhir
"""

import random
import time

from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import describe_move, random_feasible_neighbor, random_feasible_state

def hill_climbing(problem, config, seed):
    # 1. read and ngecek config
    max_iteration = config["max_iterations"]

    # True/False di Python dianggapnya angka juga, jadi harus manually ditolak
    is_valid_number = isinstance(max_iterations, int) and not isinstance(max_iterations,bool)
    is not is_valid_number or max_iterations < 0:
        raise valueError("max_iterations harus integer >= 0")
    
    # Kalau tidak diisi, snapshot disimpan (default True)
    save_snapshots = config.get("record_snapshots", True)

    # 2. Siapkan variable awal
    random_generator = random.Random(seed) # Satu generator acak itu untuk seluruh run

    # Logic counter: berapa kali tiap hasil itu berjalan terjadi
    count_improved = 0
    count_equal = 0
    count_worse = 0
    count_no_neighbor = 0
    last_improvement_iteration = 0

    # Logs untuk mencatat tiap iterasinya
    # Nanti bakal diubah jadi history
    iteration_log = []

    # 3. Logic Neighbor Finder -> cuma ini yang memang diitung waktunya
    start_time = time.perf_counter()

    # FIRST STATE dipilih secara acak
    current_state = random_feasible_state(problem, random_generator)
    current_value = objective(current_state, problem)

    # Simpen FIRST STATE buat dilaporin ke hasil akhir
    initial_state = current_state
    initial_value = current_value

    # nambahin logs dari hasil iterasinya 
    iteration_log.append({
        "itereation": 0,
        "state": current_state,
        "value": current_value,
        "neighbor_value": None,
        "accepted": False,
        "move": None, 
        "reason": "initial",
    })

    for iteration in range(1, max_iterations +1):
        # Mengambil SATU tetangga yang memang acak, hasilnya bisa NONE kalau ngga ketemu
        neighbor_state = random_feasible_neighbor(current_state, problem, random_generator)

        # Kasus 1: tidak ada tetangga yang memang valid
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
            continue # ini akan lanjut ke iterasi2 berikutnya

        neighbor_value = objective(neighbor_state, problem)
        move = describe_move(current_state, neighbor_state, problem)

        # Kasus 2: tetangga yang lebih bagus, maka yaudah pindahlah ya
        if neighbor_value > current_value:
            current_state = neighbor_state
            current_value = neighbor_value
            count_improved += 1
            last_improvement_iteration = iteration
            accepted = True
            reasion = "improved"

        # Kasus 3: nilainya sama, tidak pindah
        elif neighbor_value == current_value:
            count_equal += 1
            accepted = False
            reasion = "equal"

        # Kasus 4: tetangganya lebih jelek, tidak pindah
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

    # 4. Mengubah catetan menjadi sebuah History
    history = []
    for entry in iteration_log:
        # snapshot itu cukup besar, jadi cuma bisa simpan di iterasi 0 dan saat state berubah saja
        state_changed = entry["iteration"] == 0 or entry["accepted"]
        if save_snapshots and state_changed:
            state_snapshot = snapshot(entry["state"], problem)
        else:
            state_snapshot = None  # None artinya "sama dengan sebelumnya"

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


    # 5. Ringkasan angka (metrics) 
    # Jumlah objective() yang dipanggil = 1 (state awal) + 1 per tetangga yang dinilai
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

    #  6. Cover semuanya jadi hasil akhir
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
