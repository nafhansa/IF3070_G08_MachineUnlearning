import json

import pytest

from src.core import (
    Placement,
    Problem,
    Ship,
    UNLOADED,
    Vehicle,
    constraint_violations,
    footprint,
    is_feasible,
    load_problem,
    objective,
    problem_from_dict,
    problem_to_dict,
    state_from_dict,
    state_to_dict,
    total_weight,
    validate_state,
)
from tests.conftest import at, make_state


# --- objective -------------------------------------------------------------

def test_objective_counts_only_loaded_fee(tiny_problem):
    # fixture sanity dari panduandebug: loaded 100 + 30, unloaded 80 diabaikan
    state = make_state(tiny_problem, A=at(0, 0), C=at(2, 0))
    assert objective(state, tiny_problem) == 130


def test_all_unloaded_is_feasible_with_zero_objective(tiny_problem):
    state = make_state(tiny_problem)
    assert is_feasible(state, tiny_problem)
    assert objective(state, tiny_problem) == 0
    assert total_weight(state, tiny_problem) == 0


def test_unloaded_vehicle_has_no_weight_or_footprint(tiny_problem):
    # D unloaded tidak boleh ikut berat maupun overlap
    state = make_state(tiny_problem, A=at(0, 0))
    assert total_weight(state, tiny_problem) == 5
    assert is_feasible(state, tiny_problem)


# --- geometri ---------------------------------------------------------------

def test_rotation_transposes_footprint_without_changing_vehicle(tiny_problem):
    a = tiny_problem.vehicles[0]
    assert footprint(a, at(0, 0, "H")) == (2, 3)
    assert footprint(a, at(0, 0, "V")) == (3, 2)
    assert (a.width, a.length) == (2, 3)


def test_exact_deck_edge_is_valid_and_one_cell_overflow_invalid(tiny_problem):
    # A (2x3) di x=2 tepat menyentuh tepi kanan deck lebar 4
    assert is_feasible(make_state(tiny_problem, A=at(2, 0)), tiny_problem)
    assert not is_feasible(make_state(tiny_problem, A=at(3, 0)), tiny_problem)
    assert not is_feasible(make_state(tiny_problem, B=at(0, 2)), tiny_problem)
    assert not is_feasible(make_state(tiny_problem, C=at(-1, 0)), tiny_problem)


def test_rotate_only_fit(tiny_problem):
    # A vertikal (3x2) tidak muat di deck tinggi 3 bila y=2, horizontal tingginya 3 sehingga harus y=0
    assert is_feasible(make_state(tiny_problem, A=at(1, 1, "V")), tiny_problem)
    assert not is_feasible(make_state(tiny_problem, A=at(1, 1, "H")), tiny_problem)


def test_adjacent_rectangles_do_not_overlap(tiny_problem):
    horizontal = make_state(tiny_problem, A=at(0, 0), B=at(2, 0))
    vertical = make_state(tiny_problem, B=at(0, 0), C=at(0, 2, "V"))
    corner = make_state(tiny_problem, B=at(0, 0), C=at(2, 2, "V"))
    for state in (horizontal, vertical, corner):
        assert is_feasible(state, tiny_problem)


def test_partial_and_full_overlap_are_invalid(tiny_problem):
    partial = make_state(tiny_problem, A=at(0, 0), B=at(1, 1))
    full = make_state(tiny_problem, B=at(0, 0), D=at(0, 0))
    for state in (partial, full):
        assert not is_feasible(state, tiny_problem)
        assert any("overlap" in v for v in constraint_violations(state, tiny_problem))


# --- kapasitas --------------------------------------------------------------

def test_capacity_exactly_at_limit_is_valid(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0), B=at(2, 0), C=at(2, 2, "V"))
    assert total_weight(state, tiny_problem) == 10
    assert is_feasible(state, tiny_problem)


def test_overweight_is_invalid_even_when_area_fits():
    # area muat tapi berat membatasi
    ship = Ship(4, 4, 10)
    problem = Problem(ship, (Vehicle("X", 1, 1, 10, 6, 0), Vehicle("Y", 1, 1, 10, 6, 0)))
    state = make_state(problem, X=at(0, 0), Y=at(3, 3))
    assert not is_feasible(state, problem)
    assert any("capacity" in v for v in constraint_violations(state, problem))


def test_float_capacity_uses_small_explicit_tolerance():
    ship = Ship(4, 4, 0.3)
    problem = Problem(ship, (Vehicle("X", 1, 1, 1, 0.1, 0), Vehicle("Y", 1, 1, 1, 0.2, 0)))
    # 0.1 + 0.2 = 0.30000000000000004 harus tetap dianggap tepat batas
    assert is_feasible(make_state(problem, X=at(0, 0), Y=at(1, 0)), problem)
    over = Problem(Ship(4, 4, 0.3), (Vehicle("X", 1, 1, 1, 0.31, 0),))
    assert not is_feasible(make_state(over, X=at(0, 0)), over)


# --- validasi state ---------------------------------------------------------

def test_state_with_wrong_length_is_rejected(tiny_problem):
    with pytest.raises(ValueError):
        validate_state((UNLOADED,) * 3, tiny_problem)
    assert not is_feasible((UNLOADED,) * 5, tiny_problem)


@pytest.mark.parametrize(
    "placement",
    [
        Placement(True, 1.0, 0, "H"),
        Placement(True, 0, True, "H"),
        Placement(True, None, 0, "H"),
        Placement(True, 0, 0, "X"),
        Placement(False, 0, 0, "H"),
    ],
)
def test_malformed_placement_is_rejected(tiny_problem, placement):
    state = make_state(tiny_problem, C=placement)
    with pytest.raises(ValueError):
        validate_state(state, tiny_problem)
    assert not is_feasible(state, tiny_problem)


# --- validasi problem -------------------------------------------------------

@pytest.mark.parametrize(
    "vehicle",
    [
        Vehicle("Z", 0, 1, 1, 1, 0),
        Vehicle("Z", 1, -2, 1, 1, 0),
        Vehicle("Z", 1.5, 1, 1, 1, 0),
        Vehicle("Z", 1, 1, -1, 1, 0),
        Vehicle("Z", 1, 1, 1, -1, 0),
        Vehicle("Z", 1, 1, float("nan"), 1, 0),
        Vehicle("Z", 1, 1, 1, 1, -1),
        Vehicle("", 1, 1, 1, 1, 0),
    ],
)
def test_invalid_vehicle_is_rejected(vehicle):
    with pytest.raises(ValueError):
        Problem(Ship(4, 4, 10), (vehicle,))


@pytest.mark.parametrize("ship", [Ship(0, 4, 10), Ship(4, 2.5, 10), Ship(4, 4, -1)])
def test_invalid_ship_is_rejected(ship):
    with pytest.raises(ValueError):
        Problem(ship, (Vehicle("A", 1, 1, 1, 1, 0),))


def test_duplicate_vehicle_id_is_rejected():
    with pytest.raises(ValueError):
        Problem(Ship(4, 4, 10), (Vehicle("A", 1, 1, 1, 1, 0), Vehicle("A", 1, 1, 2, 1, 0)))


# --- serialisasi ------------------------------------------------------------

def test_state_dict_round_trip(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0), C=at(2, 2, "V"))
    data = json.loads(json.dumps(state_to_dict(state, tiny_problem)))
    assert state_from_dict(data, tiny_problem) == state


def test_state_dict_rejects_missing_duplicate_and_unknown_ids(tiny_problem):
    good = state_to_dict(make_state(tiny_problem), tiny_problem)
    missing = {"placements": good["placements"][:-1]}
    duplicate = {"placements": good["placements"] + [good["placements"][0]]}
    unknown = {"placements": good["placements"][:-1] + [{"id": "Q", "loaded": False}]}
    for data in (missing, duplicate, unknown):
        with pytest.raises(ValueError):
            state_from_dict(data, tiny_problem)


def test_problem_json_round_trip(tmp_path, tiny_problem):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(problem_to_dict(tiny_problem)))
    loaded = load_problem(path)
    assert loaded.ship == tiny_problem.ship
    assert loaded.vehicles == tiny_problem.vehicles
    assert problem_from_dict(problem_to_dict(tiny_problem)).vehicles == tiny_problem.vehicles


def test_states_are_immutable(tiny_problem):
    state = make_state(tiny_problem, A=at(0, 0))
    with pytest.raises(AttributeError):
        state[0].x = 1
