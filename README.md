# IF3070_G08_Machine-Unlearning
## Tugas Besar IF3070 Dasar Inteligensi Artifisial 2026/2027

### Milestone 1
- Forward Deployed Engineer (FDE) di PT Owen Intim Line

Optimasi penempatan kendaraan di deck kapal 2D: pilih kendaraan yang dimuat beserta posisi dan orientasinya untuk
**memaksimalkan total ShippingFee**. Hard constraint: tidak overlap, seluruh kendaraan di dalam deck, total berat
<= MaxCapacity.

Repo ini berisi **baseline** (kode bersama). Ketiga algoritma masih stub dan dikerjakan terpisah:

| algoritma | file | pemilik |
|---|---|---|
| Hill Climbing | `src/algorithms/hill_climbing.py` | Nafhan |
| Simulated Annealing | `src/algorithms/simulated_annealing.py` | Kade |
| Genetic Algorithm | `src/algorithms/genetic_algorithm.py` | Faris |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Struktur

```text
configs/        problem.json (instance standar), presets/{tiny,stress}.json, *_experiment.json
src/core.py         entitas, geometri, constraint, objective, serialisasi
src/search_space.py initial random, swap/relocate/rotate, neighbor random dan exhaustive
src/result.py       RunResult + snapshot (format hasil bersama)
src/baseline.py     random baseline + exact oracle (instance tiny)
src/experiments.py  registry ALGORITHMS, runner eksperimen, plot
src/main.py         CLI + validator hasil
src/algorithms/     stub HC, SA, GA (isi di sini)
tests/              test inti (core, search_space, baseline)
```

## Aturan kerja

- Tiap orang hanya mengubah file algoritmanya sendiri, di branch sendiri (`feat/hc`, `feat/sa`, `feat/ga`).
- Fungsi solver: `fn(problem, config, seed) -> RunResult` (format hasil: `RunResult` di `src/result.py`).
- Jangan ubah `core.py` dan `search_space.py` tanpa kesepakatan bertiga; jalankan `pytest` sebelum push.
- Validasi hasil run: `python -m src.main validate --config configs/problem.json --result <json>`.
