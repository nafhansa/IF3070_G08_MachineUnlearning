
# final akan save  state terakhir, best menyimpan state terbaik selama pencarian

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass

from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import (
    describe_move,
    generate_feasible_neighbors,
    random_feasible_neighbor,
    random_feasible_state,
)


@dataclass(frozen=True)
class SAConfig:
    max_iterations: int
    initial_temperature: float
    cooling_rate: float
    min_temperature: float
    detect_stuck: bool = True
    record_snapshots: bool = True

    # otomatis dipanggil setelah config dibuat untuk mengecek input
    def __post_init__(self):
        # bool dicek terpisah karena di python bool juga termasuk int
        if isinstance(self.max_iterations, bool):
            raise ValueError("max_iterations harus integer >= 0")
        if not isinstance(self.max_iterations, int):
            raise ValueError("max_iterations harus integer >= 0")
        if self.max_iterations < 0:
            raise ValueError("max_iterations harus integer >= 0")

        # suhu harus positif, cant null or takhingga
        if not math.isfinite(self.initial_temperature):
            raise ValueError("initial_temperature harus angka berhingga > 0")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature harus angka berhingga > 0")

        if not math.isfinite(self.cooling_rate):
            raise ValueError("cooling_rate harus di (0, 1) agar suhu turun")
        if self.cooling_rate <= 0 or self.cooling_rate >= 1:
            raise ValueError("cooling_rate harus di (0, 1) agar suhu turun")

        if not math.isfinite(self.min_temperature):
            raise ValueError("min_temperature harus angka berhingga > 0")
        if self.min_temperature <= 0:
            raise ValueError("min_temperature harus angka berhingga > 0")


# schedule Temp buat turun bertahap with iteration agar sesuai jg dengan konsep eksplorasiny
# iterasi pertama pakai suhu awal, lalu turun sesuai cooling rate
def temperature_at(cfg: SAConfig, iteration: int) -> float:
    return cfg.initial_temperature * cfg.cooling_rate ** (iteration - 1)


# delta positif berarti candiatenya lebih baik karena objective dimaksimalkan
# candidate yg lebih buruk punya peluang diterima sebesar exp(delta / temperature)
def acceptance_probability(delta: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature harus > 0 sebelum menghitung e^(dE/T)")

    if delta >= 0:
        return 1.0

    return math.exp(delta / temperature)


# cukup temukan satu neighbor lebih baik untuk membuktikan belum local optimum
def is_local_optimum(state, problem) -> bool:
    current_value = objective(state, problem)

    for move, neighbor in generate_feasible_neighbors(state, problem):
        neighbor_value = objective(neighbor, problem)

        if neighbor_value > current_value:
            return False

    return True


def simulated_annealing(problem, config, seed: int) -> RunResult:
    if isinstance(config, SAConfig):
        settings = config
    else:
        # config dari dictionary diubah menjadi objek SAConfig
        settings = SAConfig(**config)

    rng = random.Random(seed)

    counters = {
        "accepted_better": 0,
        "accepted_equal": 0,
        "accepted_worse": 0,
        "rejected_worse": 0,
        "no_candidate": 0,
        "local_optimum_checks": 0,
        "local_optimum_cache_hits": 0,
    }

    stuck_cache = {}
    check_time = 0.0

    # save hasil cek supaya state yg sama gaperlu dicek lg 
    def check_stuck(state):
        nonlocal check_time  #update check_time

        if not settings.detect_stuck:
            return None

        if state in stuck_cache:
            counters["local_optimum_cache_hits"] += 1
            return stuck_cache[state]

        check_start = time.perf_counter()

        stuck_cache[state] = is_local_optimum(state, problem)

        check_time += time.perf_counter() - check_start
        counters["local_optimum_checks"] += 1

        return stuck_cache[state]

    # ini buat initial state random dan mulai pengukuran runtime
    # raw menyimpan satu dictionary per iterasi sebelum dibuat menjadi history
    raw = []
    start = time.perf_counter()

    current = random_feasible_state(problem, rng)
    current_value = objective(current, problem)

    best = current
    best_value = current_value
    best_iteration = 0

    is_stuck = check_stuck(current)
    initial_stuck = is_stuck

    # initial state yg sudah local optimum dihitung sebagai first eps
    stuck_events = 0
    if is_stuck:
        stuck_events = 1

    escapes = 0

    raw.append({
        "iteration": 0,
        "state": current,
        "candidate_value": None,
        "delta": None,
        "temperature": None,
        "probability": None,
        "draw": None,
        "accepted": False,
        "move": None,
        "reason": "initial",
        "best_value": best_value,
        "stuck": is_stuck,
        "stuck_event": bool(is_stuck),
    })

    termination = "max_iterations"
    last_temperature = None

    # itung suhu trs coba satu random neighbor pada setiap iterasi
    for iteration in range(1, settings.max_iterations + 1):
        temperature = temperature_at(settings, iteration)

        if temperature < settings.min_temperature:
            termination = "min_temperature"
            break

        last_temperature = temperature
        candidate = random_feasible_neighbor(current, problem, rng)

        # proses searching neighbor bisa fail IF batas percobaan random habis
        # condition ini belum ngasih valid reason bahwa current berada di local optimum
        if candidate is None:
            counters["no_candidate"] += 1

            raw.append({
                "iteration": iteration,
                "state": current,
                "candidate_value": None,
                "delta": None,
                "temperature": temperature,
                "probability": None,
                "draw": None,
                "accepted": False,
                "move": None,
                "reason": "no_feasible_candidate",
                "best_value": best_value,
                "stuck": is_stuck,
                "stuck_event": False,
            })

            continue

        candidate_value = objective(candidate, problem)
        delta = candidate_value - current_value
        probability = acceptance_probability(delta, temperature)
        draw = None

        # candidate yg lebih baik atau sama selalu diterima
        # angka random cmn diambil kalau candidate lebih buruk
        if delta > 0:
            accepted = True
            reason = "better"
            counters["accepted_better"] += 1

        elif delta == 0:
            accepted = True
            reason = "equal"
            counters["accepted_equal"] += 1

        else:
            draw = rng.random()

            if draw < probability:
                accepted = True
                reason = "worse_accepted"
                counters["accepted_worse"] += 1
            else:
                accepted = False
                reason = "worse_rejected"
                counters["rejected_worse"] += 1

        move = describe_move(current, candidate, problem)
        stuck_event = False

        # kalau diterima,  current di move ke candidate
        # best di update cmn pas nemuin nilai yg lebih tinggi
        if accepted:
            current = candidate
            current_value = candidate_value

            if current_value > best_value:
                best = current
                best_value = current_value
                best_iteration = iteration

            # episode diitung saat baru masuk local optimum
            # pindah dari satu state stuck ke state stuck lain in 1 eps
            new_stuck = check_stuck(current)

            if new_stuck and not is_stuck:
                stuck_events += 1
                stuck_event = True

            elif is_stuck and not new_stuck:
                escapes += 1

            is_stuck = new_stuck

        raw.append({
            "iteration": iteration,
            "state": current,
            "candidate_value": candidate_value,
            "delta": delta,
            "temperature": temperature,
            "probability": probability,
            "draw": draw,
            "accepted": accepted,
            "move": move,
            "reason": reason,
            "best_value": best_value,
            "stuck": is_stuck,
            "stuck_event": stuck_event,
        })

    runtime = time.perf_counter() - start

    # history untuk grafik dan replay setelah timer stop
    history = []

    for item in raw:
        current_state = None
        if settings.record_snapshots:
            if item["iteration"] == 0 or item["accepted"]:
                current_state = snapshot(item["state"], problem)

        move_data = None
        if item["move"]:
            move_data = list(item["move"])

        history.append({
            "iteration": item["iteration"],
            "current_objective": objective(item["state"], problem),
            "best_so_far_objective": item["best_value"],
            "candidate_objective": item["candidate_value"],
            "delta_e": item["delta"],
            "temperature": item["temperature"],
            "acceptance_probability": item["probability"],
            "random_draw": item["draw"],
            "accepted": item["accepted"],
            "move": move_data,
            "reason": item["reason"],
            "stuck": item["stuck"],
            "stuck_event": item["stuck_event"],
            "current_state": current_state,
        })

    # none berarti detection mati, bukan zero accident 
    total_stuck = None
    total_escapes = None

    if settings.detect_stuck:
        total_stuck = stuck_events
        total_escapes = escapes

    # summary
    initial = raw[0]["state"]

    metrics = {
        "accepted_better": counters["accepted_better"],
        "accepted_equal": counters["accepted_equal"],
        "accepted_worse": counters["accepted_worse"],
        "rejected_worse": counters["rejected_worse"],
        "no_candidate": counters["no_candidate"],
        "iterations_run": len(raw) - 1,
        # menghitung evaluasi initial dan kandidat, di luar cek local optimum
        "objective_evaluations": len(raw) - counters["no_candidate"],
        "termination_reason": termination,
        "final_temperature": last_temperature,
        "best_found_objective": best_value,
        "best_found_iteration": best_iteration,
        "best_found_state": snapshot(best, problem),
        "stuck_event_count": total_stuck,
        "escape_count": total_escapes,
        "initial_stuck": initial_stuck,
        "local_optimum_checks": counters["local_optimum_checks"],
        "local_optimum_cache_hits": counters["local_optimum_cache_hits"],
        "local_optimum_check_seconds": check_time,
    }
    
    return RunResult(
        algorithm="sa",
        seed=seed,
        config=asdict(settings),
        problem=problem_to_dict(problem),
        initial_state=snapshot(initial, problem),
        final_state=snapshot(current, problem),
        initial_objective=objective(initial, problem),
        final_objective=current_value,
        runtime_seconds=runtime,
        history=history,
        metrics=metrics,
    )