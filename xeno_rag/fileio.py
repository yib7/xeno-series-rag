"""Small file helpers shared by the corpus-build steps."""

import os
from contextlib import contextmanager

from .errors import SetupError


@contextmanager
def atomic_text_writer(path: str):
    """Open ``path`` for UTF-8 text writing, but publish it only when the ``with`` block completes.

    Output goes to ``path + ".tmp"`` and is ``os.replace``d over ``path`` on success; on any exception
    the temp file is removed and the previous ``path`` is left byte-for-byte untouched. Every build
    step (parse, chunk, checkpoints) used to open its output with ``"w"`` up front, so a crash, a
    full disk, or a missing input discovered mid-stream left a truncated file that later steps
    treated as the real corpus."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def require_input(path: str, what: str, fix: str) -> str:
    """Return ``path`` if it is an existing, non-empty file; otherwise raise a ``SetupError`` that
    names ``what`` is missing and the command (``fix``) that produces it. Build steps call this on
    their input first, so a user who skipped an earlier step gets one actionable line instead of a
    FileNotFoundError traceback, and a destructive step never starts on missing input."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise SetupError(f"{what} {path} is missing or empty. Nothing was changed. Run `{fix}` first.")
    return path
