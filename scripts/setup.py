"""One-shot setup: download the prebuilt vector store from the GitHub release so the app is runnable
without scraping the wiki (~35 min) or running the multi-hour embed.

Flow: download the ``vectorstore.zip`` release asset -> verify its sha256 -> extract into
``data/vectorstore/`` -> rebuild the BM25 lexical index from the extracted collection (so the index
always matches the shipped vectors, and the asset stays smaller). Idempotent: with a store already
present it is a no-op unless ``--force`` is passed.

The download prefers the GitHub CLI (``gh release download``), which handles auth — **required while
the repo is private**. If ``gh`` is absent it falls back to the public asset URL (works only once the
repo/release is public).
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
REPO = "yib7/zohar-rag"
TAG = "data-v1"
ASSET = "zohar-rag-vectorstore.zip"
SHA256 = "54ce1075de33348b5a779c1e0372ec7895619215f019b5c06337cd4455c16938"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS = os.path.join(REPO_ROOT, "data", "vectorstore")
CHROMA = os.path.join(VS, "chroma.sqlite3")


def _download(dest_dir: str) -> str:
    """Fetch the release asset into ``dest_dir`` and return its path. Prefer ``gh`` (private-repo auth
    + a progress bar); fall back to the public browser_download_url."""
    out = os.path.join(dest_dir, ASSET)
    if shutil.which("gh"):
        print(f"[setup] downloading {ASSET} from {REPO} @ {TAG} via gh ...", flush=True)
        subprocess.run(["gh", "release", "download", TAG, "--repo", REPO,
                        "--pattern", ASSET, "--dir", dest_dir], check=True)
    else:
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
        print(f"[setup] 'gh' not found; trying public download:\n        {url}", flush=True)
        print("        (a private repo returns 404 here — install the GitHub CLI and run "
              "`gh auth login`)", flush=True)
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
        print(f"[setup] {CHROMA} already exists — nothing to do (use --force to re-download).")
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
