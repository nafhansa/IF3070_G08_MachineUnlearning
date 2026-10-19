"""hasil run yang bisa diserialisasi ke JSON; dipakai GA sekarang serta HC/SA/frontend nanti."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.core import footprint, is_feasible, loaded_count, objective, state_to_dict, total_weight

SCHEMA_VERSION = 1


def snapshot(state, problem) -> dict:
    """salinan state yang immutable-by-value (dict JSON) + nilai turunan untuk visualizer.

    width/length adalah ukuran efektif di deck sesuai orientasi (None bila unloaded).
    bisa dibaca kembali dengan core.state_from_dict (field tambahan diabaikan).
    """
    data = state_to_dict(state, problem)
    for item, v, p in zip(data["placements"], problem.vehicles, state):
        w, l = footprint(v, p) if p.loaded else (None, None)
        item["width"], item["length"] = w, l
    data.update(
        objective=objective(state, problem),
        total_weight=total_weight(state, problem),
        loaded_count=loaded_count(state),
        feasible=is_feasible(state, problem),
    )
    return data


@dataclass
class RunResult:
    """kontrak hasil run untuk semua algoritma (ga sekarang; hc/sa di tahap berikutnya)."""

    algorithm: str
    seed: int
    config: dict
    problem: dict
    initial_state: dict
    final_state: dict
    initial_objective: float
    final_objective: float
    runtime_seconds: float
    history: list
    metrics: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False: gagal keras bila ada NaN/inf, bukan diam-diam menulis JSON invalid
        path.write_text(json.dumps(self.to_dict(), allow_nan=False))

    @classmethod
    def load(cls, path) -> "RunResult":
        return cls(**json.loads(Path(path).read_text()))
