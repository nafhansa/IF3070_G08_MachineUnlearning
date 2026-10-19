"""baseline correctness: random feasible baseline dan exact oracle untuk instance tiny.

ini BUKAN algoritma mandatory; hanya patokan. exact_optimum memakai geometri berbasis sel
sendiri (bukan rect/overlaps dari core) supaya bisa dipakai sebagai cross-check independen.
"""

from __future__ import annotations

import random
from statistics import mean

from src.core import UNLOADED, Placement, objective
from src.search_space import random_feasible_state


def random_baseline(problem, samples: int, seed: int) -> dict:
    """ambil `samples` random feasible state dengan satu rng ber-seed, catat yang terbaik."""
    rng = random.Random(seed)
    states = [random_feasible_state(problem, rng) for _ in range(samples)]
    values = [objective(s, problem) for s in states]
    best = max(range(samples), key=values.__getitem__)
    return {
        "seed": seed,
        "samples": samples,
        "objectives": values,
        "mean_objective": mean(values),
        "best_objective": values[best],
        "best_state": states[best],
    }


def exact_optimum(problem):
    """brute force backtracking semua placement (unloaded + setiap anchor/orientasi).

    hanya untuk instance sangat kecil (mis. <= 5 kendaraan di deck <= 4x4).
    return (nilai optimum, state optimum).
    """
    W, L, cap = problem.ship.width, problem.ship.length, problem.ship.max_capacity
    options = []
    for v in problem.vehicles:
        opts = [(UNLOADED, frozenset())]
        dims = {(v.width, v.length, "H"), (v.length, v.width, "V")}
        if v.width == v.length:
            dims = {(v.width, v.length, "H")}
        for w, l, o in sorted(dims, key=lambda d: d[2]):
            for x in range(W - w + 1):
                for y in range(L - l + 1):
                    cells = frozenset((cx, cy) for cx in range(x, x + w) for cy in range(y, y + l))
                    opts.append((Placement(True, x, y, o), cells))
        options.append(opts)

    best = [-1, None]
    chosen = []

    def search(k, used, weight):
        if k == len(options):
            value = objective(chosen, problem)
            if value > best[0]:
                best[0], best[1] = value, tuple(chosen)
            return
        v = problem.vehicles[k]
        for placement, cells in options[k]:
            if placement.loaded and (weight + v.weight > cap + 1e-9 or used & cells):
                continue
            chosen.append(placement)
            search(k + 1, used | cells, weight + (v.weight if placement.loaded else 0))
            chosen.pop()

    search(0, frozenset(), 0)
    return best[0], best[1]
