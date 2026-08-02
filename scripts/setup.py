"""One-shot setup: download the prebuilt vector store from the GitHub release so the app is runnable
without scraping the wiki (~35 min) or running the multi-hour embed.

Flow: download the ``vectorstore.zip`` release asset -> verify its sha256 -> extract into
``data/vectorstore/`` -> rebuild the BM25 lexical index from the extracted collection (so the index
always matches the shipped vectors, and the asset stays smaller). Idempotent: with a store already
present it is a no-op unless ``--force`` is passed.

The download uses a plain public HTTPS request to the release asset, or the GitHub CLI
(``gh release download``) when it is installed (handy for a progress bar). The repo is public, so no
auth is needed either way.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

# --- Release coordinates (keep in sync with the uploaded asset) ---
REPO = "yib7/xeno-series-rag"
TAG = "data-v2"                              # Qwen3-Embedding-0.6B store (data-v1 was bge-base, 768-dim)
ASSET = "xeno-rag-vectorstore.zip"
# Refreshed 2026-07-17: same vectors, corrected per-game `g_<game>` membership flags (~130 pages the
# pre-fix HTML parser mis-tagged). The checksum changed with the asset, so a v1.3.0 checkout pins the
# previous value and will report a mismatch against the current asset. Use v1.3.1 or later.
SHA256 = "6bb281f2827a311ebdeb7b005ade6b26ddbc045117cccde926a7dfabe78b8458"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS = os.path.join(REPO_ROOT, "data", "vectorstore")
CHROMA = os.path.join(VS, "chroma.sqlite3")


def _download(dest_dir: str) -> str:
    """Fetch the release asset into ``dest_dir`` and return its path. Use ``gh`` when present (it shows
    a download progress bar); otherwise a plain public HTTPS request (the repo is public, no auth)."""
    out = os.path.join(dest_dir, ASSET)
    if shutil.which("gh"):
        print(f"[setup] downloading {ASSET} from {REPO} @ {TAG} via gh ...", flush=True)
        subprocess.run(["gh", "release", "download", TAG, "--repo", REPO,
                        "--pattern", ASSET, "--dir", dest_dir], check=True)
    else:
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
        print(f"[setup] downloading {ASSET} via HTTPS:\n        {url}", flush=True)
        urllib.request.urlretrieve(url, out)
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract(zip_path: str):
    """Clear any existing store (so a stale collection dir can't linger beside the new one) and
    extract the asset's top-level ``chroma.sqlite3`` + HNSW collection dir into ``data/vectorstore``."""
    if os.path.isdir(VS):
        for name in os.listdir(VS):
            p = os.path.join(VS, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(VS, exist_ok=True)
    print(f"[setup] extracting into {VS} ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        # CPython's zipfile already strips ".." components on extractall (a "../evil.txt" member
        # lands sanitized inside VS, not escaping it) -- but with --skip-verify a tampered archive
        # should be rejected outright, not silently rewritten. Validate every member's resolved path
        # stays within VS and fail closed before extracting anything.
        vs_real = os.path.realpath(VS)
        for name in z.namelist():
            dest = os.path.realpath(os.path.join(VS, name))
            try:
                inside = os.path.commonpath([vs_real, dest]) == vs_real
            except ValueError:
                # commonpath raises ValueError when the paths share no common root at all (e.g. a
                # drive-absolute member like "D:/evil.txt" landing on a different drive than VS on
                # Windows). Paths on different drives can never be "inside" VS, so treat this the
                # same as an ordinary outside-VS rejection instead of letting ValueError leak out.
                inside = False
            if not inside:
                raise RuntimeError(
                    f"refusing to extract {zip_path!r}: member {name!r} resolves outside {VS}"
                )
        z.extractall(VS)


def _rebuild_bm25():
    """Build the SQLite-FTS5 BM25 index from the freshly extracted collection (no model needed)."""
    print("[setup] rebuilding BM25 lexical index from the collection ...", flush=True)
    from xeno_rag.config import load_config
    from xeno_rag import bm25_index
    n = bm25_index.run(load_config())
    print(f"[setup] BM25 built over {n} chunks", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Download the prebuilt vector store and rebuild BM25.")
    ap.add_argument("--force", action="store_true", help="re-download even if a store already exists")
    ap.add_argument("--skip-verify", action="store_true", help="skip the sha256 integrity check")
    args = ap.parse_args()

    if os.path.exists(CHROMA) and not args.force:
        print(f"[setup] {CHROMA} already exists: nothing to do (use --force to re-download).")
        return

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = _download(tmp)
        if not args.skip_verify:
            got = _sha256(zip_path)
            if got != SHA256:
                sys.exit(f"[setup] checksum mismatch!\n  expected {SHA256}\n  got      {got}\n"
                         "Re-download, or pass --skip-verify if you trust the file.")
            print("[setup] sha256 OK", flush=True)
        _extract(zip_path)

    _rebuild_bm25()
    print("\n[setup] done. Start the app with:\n"
          "  .venv\\Scripts\\python.exe -m uvicorn xeno_rag.web.app:app --port 8000")


if __name__ == "__main__":
    main()
