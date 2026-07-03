"""Guards the sdist/wheel packaging manifest against silently dropping served static assets.

Regression cover for finding P1-1: the wheel packaged only ``static/*.html``, so ``render.js``
(loaded by ``index.html`` via ``<script src="/static/render.js">``) was absent from a non-editable
install and 404'd at runtime, meaning no answer ever rendered.

This is a *static* check: it reads the ``[tool.setuptools.package-data]`` globs from
``pyproject.toml`` and confirms every served asset on disk (``*.html`` / ``*.js``) under
``xeno_rag/web/static/`` is matched by one of those globs. No build is run, so it stays fast and has
no network / model dependency. Binary art under ``static/art/`` is copyrighted, gitignored, and never
shipped (see ``static/art/README.md``); it is intentionally excluded from the served-asset set here.
"""

import tomllib
from pathlib import Path

# tests/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_WEB_PKG_DIR = _REPO_ROOT / "xeno_rag" / "web"
_STATIC_DIR = _WEB_PKG_DIR / "static"


def _package_data_globs() -> list[str]:
    """The declared package-data glob list for the ``xeno_rag.web`` package."""
    with _PYPROJECT.open("rb") as fh:
        cfg = tomllib.load(fh)
    return cfg["tool"]["setuptools"]["package-data"]["xeno_rag.web"]


def _matched_by_package_data() -> set[Path]:
    """Files under ``xeno_rag/web/`` that the declared globs would package (absolute paths)."""
    matched: set[Path] = set()
    for pattern in _package_data_globs():
        for path in _WEB_PKG_DIR.glob(pattern):
            if path.is_file():
                matched.add(path.resolve())
    return matched


def _served_assets_on_disk() -> set[Path]:
    """Served static assets that must ship: ``*.html`` / ``*.js`` directly under ``static/``.

    Only the top level of ``static/`` is considered a served-asset set; ``static/art/`` holds
    copyrighted, gitignored binaries that are deliberately not packaged.
    """
    assets: set[Path] = set()
    for ext in ("*.html", "*.js"):
        for path in _STATIC_DIR.glob(ext):
            if path.is_file():
                assets.add(path.resolve())
    return assets


def test_all_served_static_assets_are_packaged():
    on_disk = _served_assets_on_disk()
    # Sanity: the static dir exists and has served assets to check (guards a silently-empty test).
    assert on_disk, f"no *.html/*.js served assets found under {_STATIC_DIR}"

    matched = _matched_by_package_data()
    missing = on_disk - matched
    rel_missing = sorted(str(p.relative_to(_REPO_ROOT)) for p in missing)
    assert not missing, (
        "served static assets on disk are not matched by any "
        "[tool.setuptools.package-data]['xeno_rag.web'] glob and would be dropped from the wheel: "
        f"{rel_missing}"
    )
