"""Registry varian hill climbing tambahan. Dipakai experiments.run_repeated.

ALGORITHMS di src/experiments.py sengaja tidak diubah (server dan test menegaskan isinya
persis ga, hc, sa), jadi varian lain didaftarkan di sini.
"""

from src.algorithms.hc_random_restart import hc_random_restart
from src.algorithms.hc_sideways import hc_sideways
from src.algorithms.hc_stochastic_aima import hc_stochastic_aima

HC_VARIANTS = {
    "hc_sideways": hc_sideways,
    "hc_random_restart": hc_random_restart,
    "hc_stochastic_aima": hc_stochastic_aima,
}
