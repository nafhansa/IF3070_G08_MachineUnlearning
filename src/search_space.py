"""search space bersama untuk GA sekarang dan HC/SA nanti.

definisi satu move (dipakai oleh random neighbor maupun neighborhood exhaustive):
- swap(i, j): tukar seluruh placement dua kendaraan (loaded<->loaded atau loaded<->unloaded)
- relocate(i, x, y, o): pindah ke anchor lain di deck. loaded mempertahankan orientasi;
  unloaded -> inside boleh memilih orientasi H/V; x=None berarti pindah ke luar kapal
- rotate(i): ganti orientasi kendaraan loaded di anchor yang sama

move yang tidak mengubah state (no-op) dikembalikan sebagai None.
kendaraan persegi selalu dinormalkan ke orientasi "H" supaya tidak ada state kembar.
"""

from __future__ import annotations

from src.core import ORIENTATIONS, UNLOADED, Placement, capacity_ok, footprint, inside_deck, overlaps, rect

MOVE_TYPES = ("swap", "relocate", "rotate")

# peluang relocate kendaraan loaded memilih "ke luar kapal" alih-alih anchor baru di deck
UNLOAD_PROB = 0.1


def _orientations(vehicle):
    return ("H",) if vehicle.width == vehicle.length else ORIENTATIONS


def _replace(state, changes: dict):
    return tuple(changes.get(i, p) for i, p in enumerate(state))


# move dasar (tidak mengecek feasibility) 

def _normalized(problem, i: int, p: Placement) -> Placement:
    # persegi selalu "H": footprint H dan V identik, jadi cukup satu representasi
    if p.loaded and p.orientation not in _orientations(problem.vehicles[i]):
        return Placement(True, p.x, p.y, "H")
    return p


def swap(state, problem, i: int, j: int):
    if i == j or state[i] == state[j]:
        return None
    cand = _replace(state, {i: _normalized(problem, i, state[j]), j: _normalized(problem, j, state[i])})
    return None if cand == state else cand


def relocate(state, problem, i: int, x, y, orientation=None):
    """pindah kendaraan i ke (x, y); x=None berarti unload. orientation None = pertahankan."""
    if x is None:
        new = UNLOADED
    else:
        new = _normalized(problem, i, Placement(True, x, y, orientation or state[i].orientation))
    if new == state[i]:
        return None
    return _replace(state, {i: new})


def rotate(state, problem, i: int):
    p = state[i]
    if not p.loaded or len(_orientations(problem.vehicles[i])) == 1:
        return None
    return _replace(state, {i: Placement(True, p.x, p.y, "V" if p.orientation == "H" else "H")})


def _fits(problem, state, changed) -> bool:
    """cek feasibility incremental. prasyarat: state sebelum move feasible, sehingga
    hanya kendaraan yang berubah yang perlu dicek terhadap batas deck dan kendaraan lain."""
    if not capacity_ok(sum(v.weight for v, p in zip(problem.vehicles, state) if p.loaded), problem.ship):
        return False
    for i in changed:
        if not state[i].loaded:
            continue
        r = rect(problem.vehicles[i], state[i])
        if not inside_deck(r, problem.ship):
            return False
        for j, q in enumerate(state):
            if j != i and q.loaded and overlaps(r, rect(problem.vehicles[j], q)):
                return False
    return True


# initial state 

def random_feasible_state(problem, rng, tries_per_vehicle: int = 20):
    """state awal random yang selalu feasible.

    urutan kendaraan diacak; tiap kendaraan dicoba diletakkan pada anchor/orientasi random
    sebanyak tries_per_vehicle kali, kalau tetap tidak muat (overlap/berat) dibiarkan unloaded.
    """
    ship = problem.ship
    state = [UNLOADED] * len(problem.vehicles)
    placed = []  # rectangle kendaraan yang sudah diletakkan
    weight = 0
    order = list(range(len(problem.vehicles)))
    rng.shuffle(order)
    for i in order:
        v = problem.vehicles[i]
        if not capacity_ok(weight + v.weight, ship):
            continue
        for _ in range(tries_per_vehicle):
            p = Placement(True, 0, 0, rng.choice(_orientations(v)))
            w, l = footprint(v, p)
            if w > ship.width or l > ship.length:
                continue
            p = Placement(True, rng.randint(0, ship.width - w), rng.randint(0, ship.length - l), p.orientation)
            r = rect(v, p)
            if not any(overlaps(r, other) for other in placed):
                state[i] = p
                placed.append(r)
                weight += v.weight
                break
    return tuple(state)


# random move / neighbor 

def _random_candidate(state, problem, rng):
    """satu percobaan move random. return (move, candidate, changed) atau None bila no-op."""
    n = len(state)
    kind = rng.choice(MOVE_TYPES)
    if kind == "swap":
        if n < 2:
            return None
        i, j = rng.sample(range(n), 2)
        cand = swap(state, problem, i, j)
        return cand and (("swap", problem.vehicles[i].id, problem.vehicles[j].id), cand, (i, j))
    if kind == "rotate":
        options = [i for i, p in enumerate(state) if p.loaded and len(_orientations(problem.vehicles[i])) == 2]
        if not options:
            return None
        i = rng.choice(options)
        return ("rotate", problem.vehicles[i].id), rotate(state, problem, i), (i,)
    # relocate
    if n == 0:
        return None
    i = rng.randrange(n)
    v, p = problem.vehicles[i], state[i]
    if p.loaded and rng.random() < UNLOAD_PROB:
        x = y = o = None
    else:
        o = p.orientation if p.loaded else rng.choice(_orientations(v))
        w, l = footprint(v, Placement(True, 0, 0, o))
        if w > problem.ship.width or l > problem.ship.length:
            return None
        # anchor diambil dari posisi yang pasti di dalam deck; anchor di luar pasti infeasible
        x, y = rng.randint(0, problem.ship.width - w), rng.randint(0, problem.ship.length - l)
    cand = relocate(state, problem, i, x, y, o)
    return cand and (("relocate", v.id, x, y, o), cand, (i,))


def random_move(state, problem, rng, max_tries: int = 100):
    """move random yang bukan no-op TANPA cek feasibility (dipakai mutasi GA sebelum repair).
    return (move, candidate) atau None bila semua percobaan no-op."""
    for _ in range(max_tries):
        found = _random_candidate(state, problem, rng)
        if found:
            return found[0], found[1]
    return None


def random_feasible_neighbor(state, problem, rng, max_tries: int = 200):
    """satu random successor yang feasible, atau None setelah max_tries percobaan.

    None BUKAN bukti local optimum; gunakan generate_feasible_neighbors untuk itu.
    prasyarat: state feasible.
    """
    for _ in range(max_tries):
        found = _random_candidate(state, problem, rng)
        if found and _fits(problem, found[1], found[2]):
            return found[1]
    return None


def generate_feasible_neighbors(state, problem):
    """lazy generator semua successor feasible satu-move dari state (prasyarat: state feasible).

    yield (move, neighbor). urutan deterministik: swap, rotate, relocate. duplikat dan
    state asal dibuang, sehingga cocok untuk cek local optimum SA di tahap berikutnya.
    """
    n = len(state)
    seen = {state}
    ship = problem.ship

    def emit(cand, changed):
        if cand is not None and cand not in seen and _fits(problem, cand, changed):
            seen.add(cand)
            return True
        return False

    for i in range(n):
        for j in range(i + 1, n):
            cand = swap(state, problem, i, j)
            if emit(cand, (i, j)):
                yield ("swap", problem.vehicles[i].id, problem.vehicles[j].id), cand
    for i in range(n):
        cand = rotate(state, problem, i)
        if emit(cand, (i,)):
            yield ("rotate", problem.vehicles[i].id), cand
    for i in range(n):
        v, p = problem.vehicles[i], state[i]
        cand = relocate(state, problem, i, None, None)
        if emit(cand, (i,)):
            yield ("relocate", v.id, None, None, None), cand
        for o in ((p.orientation,) if p.loaded else _orientations(v)):
            w, l = footprint(v, Placement(True, 0, 0, o))
            for x in range(ship.width - w + 1):
                for y in range(ship.length - l + 1):
                    cand = relocate(state, problem, i, x, y, o)
                    if emit(cand, (i,)):
                        yield ("relocate", v.id, x, y, o), cand


def describe_move(before, after, problem):
    """tebak move dari selisih dua state (tidak memakai rng), formatnya sama dengan
    generate_feasible_neighbors. dipakai untuk history HC/SA. return None bila state identik."""
    changed = [i for i, (p, q) in enumerate(zip(before, after)) if p != q]
    if not changed:
        return None
    if len(changed) == 2:
        i, j = changed
        return "swap", problem.vehicles[i].id, problem.vehicles[j].id
    i = changed[0]
    p, q = before[i], after[i]
    if p.loaded and q.loaded and (p.x, p.y) == (q.x, q.y):
        return "rotate", problem.vehicles[i].id
    if not q.loaded:
        return "relocate", problem.vehicles[i].id, None, None, None
    return "relocate", problem.vehicles[i].id, q.x, q.y, q.orientation
