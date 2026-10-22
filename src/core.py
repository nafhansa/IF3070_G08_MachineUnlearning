"""model problem bersama: entitas, geometri, constraint, objective, dan serialisasi.

konvensi koordinat:
- x = kolom (0..ship.width-1), y = baris (0..ship.length-1), (0, 0) = pojok kiri atas
- kendaraan loaded menempati rectangle half-open [x, x+w) x [y, y+l)
- orientasi "H": w = vehicle.width, l = vehicle.length; "V": keduanya ditukar
- state = tuple Placement dengan urutan sama seperti problem.vehicles (immutable)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ORIENTATIONS = ("H", "V")

# toleransi eksplisit untuk penjumlahan berat float (mis. 0.1 + 0.2); kelebihan nyata tetap ditolak
CAPACITY_EPS = 1e-9


@dataclass(frozen=True)
class Vehicle:
    """atribut statis kendaraan; tidak pernah berubah selama search."""

    id: str
    width: int
    length: int
    shipping_fee: float
    weight: float
    eta: float


@dataclass(frozen=True)
class Ship:
    width: int
    length: int
    max_capacity: float


@dataclass(frozen=True)
class Placement:
    """keputusan dinamis untuk satu kendaraan. unloaded selalu (False, None, None, "H")."""

    loaded: bool
    x: int | None = None
    y: int | None = None
    orientation: str = "H"


UNLOADED = Placement(False)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Problem:
    """satu kapal + kumpulan kendaraan; urutan vehicles menentukan urutan gene/state."""

    ship: Ship
    vehicles: tuple[Vehicle, ...]
    name: str = ""
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "vehicles", tuple(self.vehicles))
        s = self.ship
        if not (_is_int(s.width) and _is_int(s.length) and s.width > 0 and s.length > 0):
            raise ValueError(f"dimensi kapal harus integer positif: {s}")
        if not (_is_number(s.max_capacity) and s.max_capacity >= 0):
            raise ValueError(f"max_capacity harus angka >= 0: {s.max_capacity}")
        seen = set()
        for v in self.vehicles:
            if not isinstance(v.id, str) or not v.id:
                raise ValueError(f"id kendaraan harus string tidak kosong: {v!r}")
            if v.id in seen:
                raise ValueError(f"id kendaraan duplikat: {v.id}")
            seen.add(v.id)
            if not (_is_int(v.width) and _is_int(v.length) and v.width > 0 and v.length > 0):
                raise ValueError(f"dimensi kendaraan {v.id} harus integer positif")
            for name in ("shipping_fee", "weight", "eta"):
                value = getattr(v, name)
                if not (_is_number(value) and value >= 0):
                    raise ValueError(f"{name} kendaraan {v.id} harus angka >= 0: {value}")
        object.__setattr__(self, "_index", {v.id: i for i, v in enumerate(self.vehicles)})

    def index(self, vehicle_id: str) -> int:
        return self._index[vehicle_id]


# geometri 

def footprint(vehicle: Vehicle, placement: Placement) -> tuple[int, int]:
    """ukuran efektif (w, l) di deck sesuai orientasi; Vehicle asli tidak diubah."""
    if placement.orientation == "V":
        return vehicle.length, vehicle.width
    return vehicle.width, vehicle.length


def rect(vehicle: Vehicle, placement: Placement) -> tuple[int, int, int, int]:
    """rectangle half-open (x0, y0, x1, y1) untuk kendaraan loaded."""
    w, l = footprint(vehicle, placement)
    return placement.x, placement.y, placement.x + w, placement.y + l


def inside_deck(r, ship: Ship) -> bool:
    x0, y0, x1, y1 = r
    return 0 <= x0 and 0 <= y0 and x1 <= ship.width and y1 <= ship.length


def overlaps(a, b) -> bool:
    # half-open: rectangle yang hanya bersentuhan sisi/pojok tidak overlap
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# state, constraint, objective 

def validate_state(state, problem: Problem) -> None:
    """cek struktur state (bukan constraint): panjang, tipe, koordinat integer, orientasi."""
    if len(state) != len(problem.vehicles):
        raise ValueError(f"state harus berisi {len(problem.vehicles)} placement, dapat {len(state)}")
    for vehicle, p in zip(problem.vehicles, state):
        if not isinstance(p, Placement):
            raise ValueError(f"placement {vehicle.id} bukan Placement: {p!r}")
        if p.orientation not in ORIENTATIONS:
            raise ValueError(f"orientasi {vehicle.id} tidak valid: {p.orientation!r}")
        if p.loaded is True:
            if not (_is_int(p.x) and _is_int(p.y)):
                raise ValueError(f"koordinat {vehicle.id} harus integer: ({p.x!r}, {p.y!r})")
        elif p != UNLOADED:
            # unloaded harus sepenuhnya di luar kapal, jadi tidak boleh membawa koordinat
            raise ValueError(f"placement unloaded {vehicle.id} harus {UNLOADED}: {p}")


def total_weight(state, problem: Problem) -> float:
    return sum(v.weight for v, p in zip(problem.vehicles, state) if p.loaded)


def objective(state, problem: Problem) -> float:
    """objective mandatory (maximize): total ShippingFee kendaraan loaded."""
    return sum(v.shipping_fee for v, p in zip(problem.vehicles, state) if p.loaded)


def loaded_count(state) -> int:
    return sum(1 for p in state if p.loaded)


def capacity_ok(weight: float, ship: Ship) -> bool:
    return weight <= ship.max_capacity + CAPACITY_EPS


def constraint_violations(state, problem: Problem) -> list[str]:
    """daftar pelanggaran hard constraint; list kosong berarti feasible (struktur dianggap valid)."""
    violations = []
    loaded = [(v, rect(v, p)) for v, p in zip(problem.vehicles, state) if p.loaded]
    for v, r in loaded:
        if not inside_deck(r, problem.ship):
            violations.append(f"out of deck: {v.id} {r}")
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            if overlaps(loaded[i][1], loaded[j][1]):
                violations.append(f"overlap: {loaded[i][0].id} & {loaded[j][0].id}")
    weight = total_weight(state, problem)
    if not capacity_ok(weight, problem.ship):
        violations.append(f"capacity exceeded: {weight} > {problem.ship.max_capacity}")
    return violations


def is_feasible(state, problem: Problem) -> bool:
    try:
        validate_state(state, problem)
    except ValueError:
        return False
    return not constraint_violations(state, problem)


# serialisasi 

def problem_from_dict(data: dict) -> Problem:
    ship = data["ship"]
    vehicles = tuple(
        Vehicle(str(v["id"]), v["width"], v["length"], v["shipping_fee"], v["weight"], v["eta"])
        for v in data["vehicles"]
    )
    return Problem(
        Ship(ship["width"], ship["length"], ship["max_capacity"]),
        vehicles,
        name=data.get("name", ""),
        metadata=data.get("metadata", {}),
    )


def problem_to_dict(problem: Problem) -> dict:
    return {
        "name": problem.name,
        "metadata": problem.metadata,
        "ship": {
            "width": problem.ship.width,
            "length": problem.ship.length,
            "max_capacity": problem.ship.max_capacity,
        },
        "vehicles": [
            {
                "id": v.id,
                "width": v.width,
                "length": v.length,
                "shipping_fee": v.shipping_fee,
                "weight": v.weight,
                "eta": v.eta,
            }
            for v in problem.vehicles
        ],
    }


def load_problem(path) -> Problem:
    return problem_from_dict(json.loads(Path(path).read_text()))


def state_to_dict(state, problem: Problem) -> dict:
    return {
        "placements": [
            {"id": v.id, "loaded": p.loaded, "x": p.x, "y": p.y, "orientation": p.orientation}
            for v, p in zip(problem.vehicles, state)
        ]
    }


def state_from_dict(data: dict, problem: Problem) -> tuple:
    """kebalikan state_to_dict; menolak id hilang, duplikat, atau tidak dikenal."""
    by_id = {}
    for item in data["placements"]:
        vid = item["id"]
        if vid in by_id:
            raise ValueError(f"id duplikat dalam state: {vid}")
        if vid not in problem._index:
            raise ValueError(f"id tidak dikenal dalam state: {vid}")
        if item["loaded"]:
            by_id[vid] = Placement(True, item["x"], item["y"], item.get("orientation", "H"))
        else:
            by_id[vid] = UNLOADED
    missing = [v.id for v in problem.vehicles if v.id not in by_id]
    if missing:
        raise ValueError(f"state tidak memuat kendaraan: {missing}")
    state = tuple(by_id[v.id] for v in problem.vehicles)
    validate_state(state, problem)
    return state
