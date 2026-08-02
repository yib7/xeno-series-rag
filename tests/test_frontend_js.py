"""Run the frontend JS unit tests (Node's built-in runner) from inside pytest.

The answer-pane Markdown renderer lives in ``xeno_rag/web/static/render.js`` and is unit-tested with
``node --test`` (no npm install needed). Wrapping it here means the "full suite green" gate actually
covers the browser-side logic: the gap that let the "every number renders as 'undefined'" bug ship.
Skips cleanly when Node is unavailable.
"""

import glob
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed; skipping frontend JS tests")
def test_frontend_js_suite():
    files = sorted(glob.glob(os.path.join(REPO, "tests", "js", "*.test.mjs")))
    assert files, "no JS test files found under tests/js/"
    proc = subprocess.run(
        [NODE, "--test", *files],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "node --test failed:\n--- stdout ---\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr
        )
