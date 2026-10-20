"""pengaman frontend: test adapter js (node), tanpa innerHTML, preset merujuk file hasil nyata."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


@pytest.mark.skipif(shutil.which("node") is None, reason="node tidak terpasang")
def test_runmodel_node_tests_pass():
    proc = subprocess.run(["node", "--test", *map(str, sorted((FRONTEND / "tests").glob("*.test.js")))],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_frontend_never_uses_inner_html():
    # data dinamis (id kendaraan, metadata) wajib lewat textContent agar aman dari XSS
    for js in FRONTEND.glob("*.js"):
        assert not re.search(r"\.(innerHTML|outerHTML)\b|insertAdjacentHTML|document\.write", js.read_text()), js.name


def test_static_presets_point_to_real_results():
    urls = re.findall(r'url: "\.\./(output/[^"]+\.json)"', (FRONTEND / "app.js").read_text())
    assert urls, "preset statis tidak ditemukan"
    for rel in urls:
        assert (ROOT / rel).exists(), rel
