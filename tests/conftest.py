"""fixture bersama untuk seluruh test."""

import pytest

from src.core import Placement, Problem, Ship, UNLOADED, Vehicle


def make_state(problem, **placements):
    """bangun state lengkap; vehicle yang tidak disebut dianggap unloaded."""
    return tuple(placements.get(v.id, UNLOADED) for v in problem.vehicles)


def at(x, y, orientation="H"):
    return Placement(True, x, y, orientation)


@pytest.fixture
def tiny_problem():
    # deck 4x3, kapasitas 10. optimum manual = D+C = 230 (berat 8); A+B+C hanya 210 walau berat
    # tepat 10, dan D tidak bisa digabung dengan A atau B karena berat (capacity stress)
    ship = Ship(width=4, length=3, max_capacity=10)
    vehicles = (
        Vehicle("A", 2, 3, 100, 5, 1),
        Vehicle("B", 2, 2, 80, 4, 2),
        Vehicle("C", 1, 2, 30, 1, 3),
        Vehicle("D", 2, 1, 200, 7, 1),
    )
    return Problem(ship, vehicles, name="tiny")
