"""One-shot setup: download the prebuilt vector store from the GitHub release so the app is runnable
without scraping the wiki (~35 min) or running the multi-hour embed.

Flow: download the ``vectorstore.zip`` release asset, verify its sha256, extract into
``data/vectorstore/``, then rebuild the BM25 lexical index from the extracted collection (so the index
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
import socket
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

# --- Release coordinates (keep in sync with the uploaded asset) ---
REPO = "yib7/xeno-series-rag"
TAG = "data-v2"                              # Qwen3-Embedding-0.6B store
ASSET = "xeno-rag-vectorstore.zip"
# This asset carries corrected per-game `g_<game>` membership flags (~130 pages the earlier HTML parser
# mis-tagged) on the same vectors. A checkout older than v1.3.1 pins the previous checksum and reports a
# mismatch against it.
SHA256 = "6bb281f2827a311ebdeb7b005ade6b26ddbc045117cccde926a7dfabe78b8458"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS = os.path.join(REPO_ROOT, "data", "vectorstore")
CHROMA = os.path.join(VS, "chroma.sqlite3")
BM25 = os.path.join(VS, "bm25.sqlite3")      # config.yaml paths.bm25 default
DOWNLOAD_TIMEOUT_S = 60                      # per socket operation, not for the whole ~1 GB transfer


def _download_https(out: str) -> None:
    url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
    print(f"[setup] downloading {ASSET} via HTTPS:\n        {url}", flush=True)
    # urlretrieve has no timeout argument; without one a stalled connection hangs setup forever.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_S)
    try:
        urllib.request.urlretrieve(url, out)
    finally:
        socket.setdefaulttimeout(previous)


def _download(dest_dir: str) -> str:
    """Fetch the release asset into ``dest_dir`` and return its path. Uses ``gh`` when installed (it
    shows a progress bar). Falls back to a plain public HTTPS request, with no auth, when ``gh`` is
    missing or fails (it refuses to run until you log in, even for a public repo)."""
    out = os.path.join(dest_dir, ASSET)
    if shutil.which("gh"):
        print(f"[setup] downloading {ASSET} from {REPO} @ {TAG} via gh ...", flush=True)
        try:
            subprocess.run(["gh", "release", "download", TAG, "--repo", REPO,
                            "--pattern", ASSET, "--dir", dest_dir], check=True)
            return out
        except (subprocess.CalledProcessError, OSError):
            print("[setup] gh could not download (not logged in?); falling back to HTTPS.", flush=True)
    _download_https(out)
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract(zip_path: str):
    """Extract the asset's top-level ``chroma.sqlite3`` + HNSW collection dir into ``data/vectorstore``.

    Order matters for a re-run over a working store: the archive is opened and every member validated
    first, then extracted into a sibling ``.partial`` directory, and only a fully extracted tree
    replaces the old store (clearing it also stops a stale collection dir lingering beside the new
    one). A corrupt zip, a rejected member, or a full disk therefore leaves the previous store
    untouched instead of a half-deleted one."""
    partial = VS + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        # CPython's zipfile already strips ".." components on extractall (a "../evil.txt" member
        # lands sanitized inside the target instead of escaping it), but with --skip-verify a tampered
        # archive should be rejected outright, not silently rewritten. Check that every member's
        # resolved path stays within VS and fail closed before extracting anything.
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
        print(f"[setup] extracting into {VS} ...", flush=True)
        try:
            z.extractall(partial)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    if os.path.isdir(VS):
        shutil.rmtree(VS)
    os.makedirs(os.path.dirname(VS), exist_ok=True)
    os.replace(partial, VS)


def _rebuild_bm25():
    """Build the SQLite-FTS5 BM25 index from the freshly extracted collection (no model needed)."""
    print("[setup] rebuilding BM25 lexical index from the collection ...", flush=True)
    from xeno_rag import bm25_index
    from xeno_rag.config import load_config
    n = bm25_index.run(load_config())
    print(f"[setup] BM25 built over {n} chunks", flush=True)


def _store_state() -> str:
    """``complete`` (vectors and BM25 index both present), ``needs-bm25`` (vectors only: an earlier
    run died before the BM25 rebuild finished), or ``missing``."""
    if not os.path.exists(CHROMA):
        return "missing"
    return "complete" if os.path.exists(BM25) else "needs-bm25"


def main():
    ap = argparse.ArgumentParser(description="Download the prebuilt vector store and rebuild BM25.")
    ap.add_argument("--force", action="store_true", help="re-download even if a store already exists")
    ap.add_argument("--skip-verify", action="store_true", help="skip the sha256 integrity check")
    args = ap.parse_args()

    state = _store_state()
    if state == "complete" and not args.force:
        print(f"[setup] {CHROMA} already exists: nothing to do (use --force to re-download).")
        return

    if state == "missing" or args.force:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                zip_path = _download(tmp)
            except OSError as exc:   # URLError, HTTPError and socket timeouts are all OSError subclasses
                sys.exit(f"[setup] download failed: {exc}\n"
                         f"  Check your connection and re-run. To fetch it by hand, download {ASSET} from\n"
                         f"  https://github.com/{REPO}/releases/tag/{TAG} and extract it into {VS}.")
            if not args.skip_verify:
                got = _sha256(zip_path)
                if got != SHA256:
                    sys.exit(f"[setup] checksum mismatch!\n  expected {SHA256}\n  got      {got}\n"
                             "Re-download, or pass --skip-verify if you trust the file.")
                print("[setup] sha256 OK", flush=True)
            try:
                _extract(zip_path)
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                sys.exit(f"[setup] could not extract the archive: {exc}\n"
                         "  Any existing store was left in place. Re-run with --force to download again.")
    else:
        print(f"[setup] {CHROMA} exists but the BM25 index is missing (an earlier run was "
              "interrupted); rebuilding it.", flush=True)

    try:
        _rebuild_bm25()
    except Exception as exc:  # noqa: BLE001 (setup boundary: a readable message, not a traceback)
        sys.exit(f"[setup] BM25 rebuild failed: {exc}\n"
                 "  The vector store is in place; re-run `python -m scripts.setup` to retry that step.")
    print("\n[setup] done. Start the app (venv active) with:\n"
          "  python -m uvicorn xeno_rag.web.app:app --port 8000")


if __name__ == "__main__":
    main()
