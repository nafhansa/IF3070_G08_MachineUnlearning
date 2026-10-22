import random

from src.core import Placement, Problem, Ship, UNLOADED, Vehicle, is_feasible, objective
from src.search_space import (
    generate_feasible_neighbors,
    random_feasible_neighbor,
    random_feasible_state,
    random_move,
    relocate,
    rotate,
    swap,
)
from tests.conftest import at, make_state


def ids(problem, state):
    return {v.id: p for v, p in zip(problem.vehicles, state)}


# move dasar 

def test_swap_inside_outside_exchanges_slot(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0))
    a, c = tiny_problem.index("A"), tiny_problem.index("C")
    after = ids(tiny_problem, swap(state, tiny_problem, a, c))
    assert after["A"] == UNLOADED
    assert after["C"] == at(0, 0)
    # fee mengikuti kendaraan, bukan slot: sekarang hanya C (30) yang loaded
    assert objective(swap(state, tiny_problem, a, c), tiny_problem) == 30


def test_swap_inside_inside_exchanges_positions(tiny_problem):
    state = make_state(tiny_problem, B=at(0, 0), C=at(2, 0))
    after = swap(state, tiny_problem, tiny_problem.index("B"), tiny_problem.index("C"))
    assert ids(tiny_problem, after)["B"] == at(2, 0)
    assert ids(tiny_problem, after)["C"] == at(0, 0)
    assert is_feasible(after, tiny_problem)


def test_swap_noop_returns_none(tiny_problem):
    state = make_state(tiny_problem)
    assert swap(state, tiny_problem, 0, 1) is None  # dua-duanya unloaded
    assert swap(state, tiny_problem, 2, 2) is None


def test_relocate_variants(tiny_problem):
    c = tiny_problem.index("C")
    inside = make_state(tiny_problem, C=at(0, 0))
    assert ids(tiny_problem, relocate(inside, tiny_problem, c, 3, 1))["C"] == at(3, 1)  # inside -> inside
    assert ids(tiny_problem, relocate(inside, tiny_problem, c, None, None))["C"] == UNLOADED  # inside -> outside
    outside = make_state(tiny_problem)
    assert ids(tiny_problem, relocate(outside, tiny_problem, c, 1, 2, "V"))["C"] == at(1, 2, "V")  # outside -> inside
    assert relocate(inside, tiny_problem, c, 0, 0) is None  # posisi sama = no-op
    assert relocate(outside, tiny_problem, c, None, None) is None


def test_rotate_changes_orientation_not_vehicle(tiny_problem):
    a = tiny_problem.index("A")
    state = make_state(tiny_problem, A=at(1, 0, "V"))
    after = rotate(state, tiny_problem, a)
    assert after[a] == at(1, 0, "H")
    assert tiny_problem.vehicles[a].width == 2


def test_rotate_unloaded_or_square_is_noop():
    problem = Problem(Ship(3, 3, 10), (Vehicle("S", 2, 2, 5, 1, 0), Vehicle("R", 1, 2, 5, 1, 0)))
    assert rotate(make_state(problem, S=at(0, 0)), problem, 0) is None
    assert rotate(make_state(problem), problem, 1) is None


def test_moves_do_not_mutate_input(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0), C=at(2, 0))
    before = tuple(state)
    swap(state, tiny_problem, 0, 1)
    relocate(state, tiny_problem, 2, 3, 0)
    rotate(state, tiny_problem, 0)
    assert state == before


# neighborhood exhaustive 

def naive_neighbors(state, problem):
    """enumerasi independen: semua move tanpa cek incremental, lalu filter is_feasible penuh."""
    out = set()
    n = len(problem.vehicles)
    for i in range(n):
        for j in range(i + 1, n):
            out.add(swap(state, problem, i, j))
        out.add(rotate(state, problem, i))
        out.add(relocate(state, problem, i, None, None))
        for o in ("H", "V"):
            if state[i].loaded and o != state[i].orientation:
                continue
            for x in range(-1, problem.ship.width + 1):
                for y in range(-1, problem.ship.length + 1):
                    out.add(relocate(state, problem, i, x, y, o))
    out.discard(None)
    out.discard(state)
    return {s for s in out if is_feasible(s, problem)}


def test_exhaustive_neighbors_match_naive_enumeration(tiny_problem):
    for state in (
        make_state(tiny_problem),
        make_state(tiny_problem, A=at(0, 0), C=at(2, 0)),
        make_state(tiny_problem, A=at(0, 0), B=at(2, 0), C=at(2, 2, "V")),
    ):
        produced = [s for _, s in generate_feasible_neighbors(state, tiny_problem)]
        assert len(produced) == len(set(produced)), "neighbor duplikat"
        assert state not in produced
        assert all(is_feasible(s, tiny_problem) for s in produced)
        assert set(produced) == naive_neighbors(state, tiny_problem)


def test_exhaustive_neighbors_include_known_moves(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0), C=at(2, 0))
    produced = {s: m for m, s in generate_feasible_neighbors(state, tiny_problem)}
    a, b, c = (tiny_problem.index(k) for k in "ABC")
    assert swap(state, tiny_problem, c, b) in produced  # B mengambil slot C: berat 9, muat
    assert relocate(state, tiny_problem, c, 3, 1) in produced
    assert rotate(state, tiny_problem, c) in produced
    assert produced[rotate(state, tiny_problem, c)][0] == "rotate"
    # rotate A di (0,0) jadi 3x2 overlap dengan C, jadi tidak boleh muncul
    assert rotate(state, tiny_problem, a) not in produced


def test_neighbors_are_lazy(tiny_problem):
    gen = generate_feasible_neighbors(make_state(tiny_problem), tiny_problem)
    assert next(gen) is not None


# random 

def test_random_feasible_state_is_feasible_and_seeded(tiny_problem):
    states = [random_feasible_state(tiny_problem, random.Random(s)) for s in range(50)]
    assert all(is_feasible(s, tiny_problem) for s in states)
    assert len(set(states)) > 5, "initializer harus benar-benar random"
    assert random_feasible_state(tiny_problem, random.Random(7)) == states[7]


def test_random_feasible_neighbor_is_feasible_and_different(tiny_problem):
    rng = random.Random(3)
    state = random_feasible_state(tiny_problem, rng)
    for _ in range(200):
        nxt = random_feasible_neighbor(state, tiny_problem, rng)
        assert nxt is not None and nxt != state and is_feasible(nxt, tiny_problem)
        state = nxt


def test_random_neighbor_can_change_loaded_set(tiny_problem):
    rng = random.Random(0)
    state = random_feasible_state(tiny_problem, rng)
    loaded_sets = set()
    for _ in range(300):
        state = random_feasible_neighbor(state, tiny_problem, rng)
        loaded_sets.add(tuple(p.loaded for p in state))
    assert len(loaded_sets) > 3


def test_random_move_never_noop_and_may_be_infeasible(tiny_problem):
    rng = random.Random(1)
    state = make_state(tiny_problem, A=at(0, 0), B=at(2, 0))
    seen_infeasible = False
    for _ in range(200):
        move, cand = random_move(state, tiny_problem, rng)
        assert cand != state and move[0] in ("swap", "relocate", "rotate")
        seen_infeasible |= not is_feasible(cand, tiny_problem)
    assert seen_infeasible


def test_degenerate_problems_do_not_hang():
    rng = random.Random(0)
    empty = Problem(Ship(2, 2, 5), ())
    assert random_feasible_state(empty, rng) == ()
    assert random_feasible_neighbor((), empty, rng) is None
    assert random_move((), empty, rng) is None
    assert list(generate_feasible_neighbors((), empty)) == []

    # kendaraan lebih besar dari deck: hanya bisa unloaded
    too_big = Problem(Ship(2, 2, 5), (Vehicle("BIG", 3, 3, 9, 1, 0),))
    assert random_feasible_state(too_big, rng) == (UNLOADED,)
    assert random_feasible_neighbor((UNLOADED,), too_big, rng) is None
    assert list(generate_feasible_neighbors((UNLOADED,), too_big)) == []

    # deck penuh oleh satu kendaraan dan kapasitas nol untuk sisanya
    full = Problem(Ship(2, 2, 1), (Vehicle("F", 2, 2, 9, 1, 0), Vehicle("G", 1, 1, 9, 1, 0)))
    state = (Placement(True, 0, 0, "H"), UNLOADED)
    neighbors = [s for _, s in generate_feasible_neighbors(state, full)]
    assert all(is_feasible(s, full) for s in neighbors)
    assert random_feasible_neighbor(state, full, rng) in neighbors


def test_swap_normalizes_square_orientation():
    # regresi: persegi yang mengambil slot kendaraan "V" harus tetap "H" agar tidak ada state kembar
    problem = Problem(Ship(4, 4, 10), (Vehicle("R", 1, 2, 5, 1, 0), Vehicle("S", 2, 2, 5, 1, 0)))
    state = make_state(problem, R=at(0, 0, "V"))
    after = swap(state, problem, 0, 1)
    assert after[1] == at(0, 0, "H")
    assert after[0] == UNLOADED


# describe_move (tahap 2: nama move untuk history HC/SA) 

def test_describe_move_matches_generator_moves(tiny_problem):
    from src.search_space import describe_move

    for state in (
        make_state(tiny_problem),
        make_state(tiny_problem, A=at(0, 0), C=at(2, 0)),
        make_state(tiny_problem, A=at(0, 0), B=at(2, 0), C=at(2, 2, "V")),
    ):
        for move, neighbor in generate_feasible_neighbors(state, tiny_problem):
            assert describe_move(state, neighbor, tiny_problem) == move
        assert describe_move(state, state, tiny_problem) is None


def test_describe_move_does_not_use_rng(tiny_problem):
    from src.search_space import describe_move

    state = make_state(tiny_problem, A=at(0, 0))
    after = relocate(state, tiny_problem, tiny_problem.index("C"), 3, 0)
    rng = random.Random(9)
    before_state = rng.getstate()
    assert describe_move(state, after, tiny_problem) == ("relocate", "C", 3, 0, "H")
    assert rng.getstate() == before_state
