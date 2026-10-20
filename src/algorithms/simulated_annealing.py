import math
import random
import time

from src.core import objective, problem_to_dict
from src.result import RunResult, snapshot
from src.search_space import (
    describe_move,
    generate_feasible_neighbors,
    random_feasible_neighbor,
    random_feasible_state,
)

def temperature_at(initial_temperature, cooling_rate, iteration):
    return initial_temperature * cooling_rate ** (iteration - 1)

def acceptance_probability(delta, temperature):
    # delta positif berarti candidate lebih baik karena objective dimaksimalkan
    if delta >= 0:
        return 1.0
    return math.exp(delta / temperature)

def is_local_optimum(state, problem):
    # satu neighbor yang lebih baik berarti state belum local optimum
    current_value = objective(state, problem)
    for _, neighbor in generate_feasible_neighbors(state, problem):
        if objective(neighbor, problem) > current_value:
            return False
    return True

def make_log(iteration, state, value, best_value, candidate_value=None,
             delta=None, temperature=None, probability=None, random_draw=None,
             accepted=False, move=None, reason=""):
    return {
        "iteration": iteration, "state": state, "value": value,
        "best_value": best_value, "candidate_value": candidate_value,
        "delta": delta, "temperature": temperature,
        "probability": probability, "random_draw": random_draw,
        "accepted": accepted, "move": move, "reason": reason,
    }

def check_stuck_history(iteration_log, problem, detect_stuck):
    # pengecekan stuck dilakukan setelah timer searching berhenti
    if not detect_stuck:
        for entry in iteration_log:
            entry["stuck"] = None
            entry["stuck_event"] = False
        return None, None, None, 0, 0, 0.0

    cache = {}
    stuck_events = 0
    escape_count = 0
    check_count = 0
    cache_hits = 0
    check_seconds = 0.0
    previous_stuck = False

    for entry in iteration_log:
        state_changed = entry["iteration"] == 0 or entry["accepted"]

        if state_changed:
            state = entry["state"]

            if state in cache:
                current_stuck = cache[state]
                cache_hits += 1
            else:
                check_start = time.perf_counter()
                current_stuck = is_local_optimum(state, problem)
                check_seconds += time.perf_counter() - check_start
                cache[state] = current_stuck
                check_count += 1
        else:
            current_stuck = previous_stuck

        stuck_event = current_stuck and not previous_stuck

        if stuck_event:
            stuck_events += 1
        elif previous_stuck and not current_stuck:
            escape_count += 1

        entry["stuck"] = current_stuck
        entry["stuck_event"] = stuck_event
        previous_stuck = current_stuck

    initial_stuck = iteration_log[0]["stuck"]

    return (
        stuck_events, escape_count, initial_stuck,
        check_count, cache_hits, check_seconds,
    )

def simulated_annealing(problem, config, seed):
    # 1. read and cek config
    max_iterations = config["max_iterations"]
    initial_temperature = config["initial_temperature"]
    cooling_rate = config["cooling_rate"]
    min_temperature = config["min_temperature"]
    detect_stuck = config.get("detect_stuck", True)
    save_snapshots = config.get("record_snapshots", True)

    valid_iterations = (
        isinstance(max_iterations, int)
        and not isinstance(max_iterations, bool)
    )

    if not valid_iterations or max_iterations < 0:
        raise ValueError("max_iterations harus integer >= 0")

    def valid_number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    if not valid_number(initial_temperature) or initial_temperature <= 0:
        raise ValueError("initial_temperature harus angka berhingga > 0")

    if not valid_number(cooling_rate) or not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate harus berada di antara 0 dan 1")

    if not valid_number(min_temperature) or min_temperature <= 0:
        raise ValueError("min_temperature harus angka berhingga > 0")

    if not isinstance(detect_stuck, bool) or not isinstance(save_snapshots, bool):
        raise ValueError("detect_stuck dan record_snapshots harus berupa boolean")

    # 2. siapin variable awal
    random_generator = random.Random(seed)

    counter = {
        "accepted_better": 0,
        "accepted_equal": 0,
        "accepted_worse": 0,
        "rejected_worse": 0,
        "no_candidate": 0,
    }

    iteration_log = []
    start_time = time.perf_counter()

    current_state = random_feasible_state(problem, random_generator)
    current_value = objective(current_state, problem)

    initial_state = current_state
    initial_value = current_value

    # current menyimpan state sekarang, best menyimpan hasil terbaik
    best_state = current_state
    best_value = current_value
    best_iteration = 0

    iteration_log.append(make_log(
        0, current_state, current_value, best_value,
        reason="initial",
    ))

    termination_reason = "max_iterations"
    final_temperature = None

    # 3. logic simulated annealing
    for iteration in range(1, max_iterations + 1):
        temperature = temperature_at(
            initial_temperature, cooling_rate, iteration
        )

        if temperature < min_temperature:
            termination_reason = "min_temperature"
            break

        final_temperature = temperature

        candidate_state = random_feasible_neighbor(
            current_state, problem, random_generator
        )

        # none berarti percobaan random habis, bukan pasti local optimum
        if candidate_state is None:
            counter["no_candidate"] += 1

            iteration_log.append(make_log(
                iteration, current_state, current_value, best_value,
                temperature=temperature,
                reason="no_feasible_candidate",
            ))
            continue

        candidate_value = objective(candidate_state, problem)
        delta = candidate_value - current_value
        probability = acceptance_probability(delta, temperature)
        random_draw = None

        if delta > 0:
            accepted = True
            reason = "better"
            counter["accepted_better"] += 1

        elif delta == 0:
            accepted = True
            reason = "equal"
            counter["accepted_equal"] += 1

        else:
            random_draw = random_generator.random()
            accepted = random_draw < probability
            reason = "worse_accepted" if accepted else "worse_rejected"

            if accepted:
                counter["accepted_worse"] += 1
            else:
                counter["rejected_worse"] += 1

        move = describe_move(current_state, candidate_state, problem)

        # current pindah kalau candidate diterima
        if accepted:
            current_state = candidate_state
            current_value = candidate_value

            # best cuma berubah kalau nilainya lebih tinggi
            if current_value > best_value:
                best_state = current_state
                best_value = current_value
                best_iteration = iteration

        iteration_log.append(make_log(
            iteration, current_state, current_value, best_value,
            candidate_value, delta, temperature, probability,
            random_draw, accepted, move, reason,
        ))

    runtime_seconds = time.perf_counter() - start_time

    # 4. cek stuck di luar runtime searching
    stuck_result = check_stuck_history(
        iteration_log, problem, detect_stuck
    )

    (
        stuck_events,
        escape_count,
        initial_stuck,
        check_count,
        cache_hits,
        check_seconds,
    ) = stuck_result

    # 5. convert ke history
    history = []

    for entry in iteration_log:
        state_changed = (
            entry["iteration"] == 0
            or entry["accepted"]
        )

        if save_snapshots and state_changed:
            state_snapshot = snapshot(entry["state"], problem)
        else:
            state_snapshot = None

        history.append({
            "iteration": entry["iteration"],
            "current_objective": entry["value"],
            "best_so_far_objective": entry["best_value"],
            "candidate_objective": entry["candidate_value"],
            "delta_e": entry["delta"],
            "temperature": entry["temperature"],
            "acceptance_probability": entry["probability"],
            "random_draw": entry["random_draw"],
            "accepted": entry["accepted"],
            "move": list(entry["move"]) if entry["move"] else None,
            "reason": entry["reason"],
            "stuck": entry["stuck"],
            "stuck_event": entry["stuck_event"],
            "current_state": state_snapshot,
        })

    # 6. summary
    iterations_run = len(iteration_log) - 1

    metrics = {
        **counter,
        "iterations_run": iterations_run,
        "objective_evaluations": (
            1 + iterations_run - counter["no_candidate"]
        ),
        "termination_reason": termination_reason,
        "final_temperature": final_temperature,
        "best_found_objective": best_value,
        "best_found_iteration": best_iteration,
        "best_found_state": snapshot(best_state, problem),
        "stuck_event_count": stuck_events,
        "escape_count": escape_count,
        "initial_stuck": initial_stuck,
        "local_optimum_checks": check_count,
        "local_optimum_cache_hits": cache_hits,
        "local_optimum_check_seconds": check_seconds,
    }

    used_config = {
        "max_iterations": max_iterations,
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "min_temperature": min_temperature,
        "detect_stuck": detect_stuck,
        "record_snapshots": save_snapshots,
    }

    # 7. cover semua hasil jadi RunResult
    return RunResult(
        algorithm="sa",
        seed=seed,
        config=used_config,
        problem=problem_to_dict(problem),
        initial_state=snapshot(initial_state, problem),
        final_state=snapshot(current_state, problem),
        initial_objective=initial_value,
        final_objective=current_value,
        runtime_seconds=runtime_seconds,
        history=history,
        metrics=metrics,
    )