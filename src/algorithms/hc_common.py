"""Helper bersama untuk semua varian hill climbing (bukan varian algoritma).

Isi:
    check_integer_config : memeriksa config integer dengan batas minimum
    find_best_neighbors  : mencari nilai neighbor terbaik beserta semua neighbor yang seri
    climb_to_peak        : steepest ascent murni sampai peak (dipakai hc_random_restart)
"""

from src.core import objective
from src.search_space import generate_feasible_neighbors


def check_integer_config(name, value, minimum):
    # True/False di Python dianggap angka juga (bool turunan int), jadi harus ditolak manual
    is_valid_number = isinstance(value, int) and not isinstance(value, bool)
    if not is_valid_number or value < minimum:
        raise ValueError(f"{name} harus integer >= {minimum}")
    return value


def find_best_neighbors(state, problem):
    """Cari neighbor feasible dengan nilai objective tertinggi.

    Return (best_neighbor_value, tied_best_neighbors, neighbors_evaluated).
    tied_best_neighbors adalah daftar pasangan (move, neighbor) yang nilainya sama dengan yang
    terbaik. neighbors_evaluated adalah jumlah neighbor yang dinilai (untuk metrics). Kalau
    tidak ada neighbor feasible, return (None, [], 0). Memilih satu di antara yang seri dilakukan oleh pemanggil
    memakai random_generator (Salindia hlm. 16), supaya seluruh run memakai satu generator.
    """
    best_neighbor_value = None
    tied_best_neighbors = []
    neighbors_evaluated = 0

    for move, neighbor_state in generate_feasible_neighbors(state, problem):
        neighbor_value = objective(neighbor_state, problem)
        neighbors_evaluated += 1

        # Kasus 1: lebih baik dari yang terbaik sejauh ini, daftar seri dimulai ulang
        if best_neighbor_value is None or neighbor_value > best_neighbor_value:
            best_neighbor_value = neighbor_value
            tied_best_neighbors = [(move, neighbor_state)]

        # Kasus 2: sama dengan yang terbaik, masuk daftar seri
        elif neighbor_value == best_neighbor_value:
            tied_best_neighbors.append((move, neighbor_state))

    return best_neighbor_value, tied_best_neighbors, neighbors_evaluated


def climb_to_peak(start_state, problem, random_generator, max_steps):
    """Steepest ascent murni dari start_state.

    Tiap langkah memeriksa SEMUA neighbor. Berhenti dengan "local_optimum" kalau tidak ada
    neighbor atau neighbor terbaik tidak lebih baik dari current (Salindia hlm. 22: peak
    termasuk flat, jadi nilai sama tidak dipakai untuk pindah). Berhenti dengan
    "max_iterations" kalau jumlah langkah sudah mencapai max_steps.

    Return (final_state, step_log, termination_reason, neighbors_evaluated).
    step_log berformat sama dengan iteration_log varian lain, nomor iterasinya dimulai
    dari 1 (entri state awal dicatat oleh pemanggil).
    """
    current_state = start_state
    current_value = objective(current_state, problem)
    step_log = []
    neighbors_evaluated = 0

    for step in range(1, max_steps + 1):
        best_neighbor_value, tied_best_neighbors, step_evaluated = find_best_neighbors(current_state, problem)
        neighbors_evaluated += step_evaluated

        # Kasus 1: tidak ada neighbor atau tidak ada yang lebih baik, sudah di peak
        if best_neighbor_value is None or best_neighbor_value <= current_value:
            step_log.append({
                "iteration": step,
                "state": current_state,
                "value": current_value,
                "neighbor_value": best_neighbor_value,
                "accepted": False,
                "move": None,
                "reason": "local_optimum",
            })
            return current_state, step_log, "local_optimum", neighbors_evaluated

        # Kasus 2: ada neighbor lebih baik, pilih acak di antara yang seri terbaik
        move, neighbor_state = random_generator.choice(tied_best_neighbors)
        current_state = neighbor_state
        current_value = best_neighbor_value
        step_log.append({
            "iteration": step,
            "state": current_state,
            "value": current_value,
            "neighbor_value": best_neighbor_value,
            "accepted": True,
            "move": move,
            "reason": "improved",
        })

    return current_state, step_log, "max_iterations", neighbors_evaluated
