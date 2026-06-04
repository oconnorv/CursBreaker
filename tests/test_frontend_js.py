"""Run the stubbed-DOM Node harnesses under pytest so they execute in CI.

The actual assertions live in the .mjs files (loaded with Node's `vm` against a
fake DOM); this just shells out to `node` and fails if any harness exits non-zero.
Skips cleanly when `node` isn't on PATH (it is on CI's ubuntu image)."""
import shutil
import subprocess
from pathlib import Path

import pytest

JS_DIR = Path(__file__).parent / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize("harness", sorted(p.name for p in JS_DIR.glob("test_*.mjs")))
def test_js_harness(harness):
    proc = subprocess.run(
        ["node", str(JS_DIR / harness)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
