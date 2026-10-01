"""atomic_text_writer: publish on success, leave the previous file alone on failure."""

import pytest

from xeno_rag.fileio import atomic_text_writer


def test_success_replaces_the_file_and_leaves_no_temp(tmp_path):
    p = tmp_path / "sub" / "out.jsonl"
    with atomic_text_writer(str(p)) as f:
        f.write("new" + chr(10))
    assert p.read_text(encoding="utf-8") == "new" + chr(10)
    assert not (tmp_path / "sub" / "out.jsonl.tmp").exists()


def test_failure_keeps_the_previous_file_and_removes_the_temp(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError), atomic_text_writer(str(p)) as f:
        f.write("partial")
        raise RuntimeError("disk full")
    assert p.read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "out.jsonl.tmp").exists()


def test_writes_utf8_with_lf_endings(tmp_path):
    p = tmp_path / "out.txt"
    with atomic_text_writer(str(p)) as f:
        f.write("Rhéa" + chr(10) + "x" + chr(10))
    assert p.read_bytes() == "Rhéa".encode() + b"\nx\n"
