"""Durable, self-healing driver for the HTML fetch, built to survive Windows Modern Standby.

The fetch is a ~5h serial pull. This machine drops into Modern Standby (S0) when the user is away,
which suspends/kills background processes (a plain keep-awake flag does NOT defeat it). So instead of
trusting one long-lived process, a Windows Scheduled Task runs THIS script every 2 minutes with
MultipleInstancesPolicy=IgnoreNew: while an instance is mid-fetch the re-triggers are ignored, but the
moment standby kills the instance the next trigger resumes it from the per-batch checkpoint. Power
settings are also disabled (see scripts that create the task) so on AC it should never sleep at all;
the task is the backstop for whatever still slips through.

Each run does, in order: resume the main fetch; when that is done, run the second table-gap pass;
when BOTH are complete, restore the saved power settings and delete the scheduled task, leaving the
machine as it was found. A PID lockfile guarantees only one fetch runs even if triggers overlap.
"""

import json
import math
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
LOCK = os.path.join(REPO, "data", "raw", "fetch.lock")
POWER_BACKUP = os.path.join(REPO, "data", "raw", "power_backup.json")
TASK_NAME = "XenoRagHtmlFetch"

MAIN_TITLES = "data/raw/titles_stats_additional.jsonl"
MAIN_CKPT = "data/raw/html_extra_checkpoint.json"
TAB_TITLES = "data/raw/titles_stats_tables.jsonl"
TAB_CKPT = "data/raw/html_tables_checkpoint.json"

SUB_SLEEP = "238C9FA8-0AAD-41ED-83F4-97BE242C8F20"
PWR = {  # setting GUID -> backup-json key
    "29F6C1DB-86DA-48C5-9FDB-F2B67B1F44DA": "standby_ac",      # STANDBYIDLE
    "9D7815A6-7EE4-497E-8888-515A05F02364": "hibernate_ac",    # HIBERNATEIDLE
    "7BC4A2F9-D8FC-4469-B07B-33EB785AACA0": "unattend_ac",     # UNATTENDSLEEP
}


def _pid_alive(pid: int) -> bool:
    """True when ``pid`` is a live Python process. Checking the image name too matters: after a crash
    or a standby kill the lock can name a PID that Windows has since handed to an unrelated program,
    and a bare "is that PID alive?" would then block the fetch forever."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return True            # alive but unreadable: assume it is ours rather than start a second fetch
        return os.path.basename(buf.value).lower().startswith("python")
    finally:
        kernel32.CloseHandle(h)


def _try_create_lock() -> bool:
    """Create the lock file atomically (O_EXCL), so two overlapping triggers cannot both win."""
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return True


def _locked() -> bool:
    """True when another live fetch instance holds the lock; otherwise take the lock and return False.
    A lock naming a dead PID, or unreadable contents, is stale: remove it and take over."""
    if _try_create_lock():
        return False
    try:
        with open(LOCK, encoding="utf-8") as f:
            pid = int(f.read().strip())
        if _pid_alive(pid):
            return True
    except Exception:  # noqa: BLE001, S110 (corrupt/missing lock contents: treat as stale)
        pass
    try:
        os.remove(LOCK)
    except OSError:
        return True                # someone else just replaced or holds it: do not fight over it
    return not _try_create_lock()


def _batches_done(ckpt: str) -> int:
    if not os.path.exists(ckpt):
        return -1
    try:
        with open(ckpt, encoding="utf-8") as f:
            return json.load(f)["last_completed_batch"]
    except Exception:  # noqa: BLE001 (advisory check only: a bad checkpoint just means "start over")
        return -1


def _last_index(titles_path: str, batch_size: int = 100) -> int:
    with open(titles_path, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    return math.ceil(n / batch_size) - 1


def _restore_power_and_cleanup():
    if os.path.exists(POWER_BACKUP):
        try:
            # utf-8-sig: PowerShell's Out-File -Encoding utf8 writes a BOM that plain json.load rejects.
            with open(POWER_BACKUP, encoding="utf-8-sig") as f:
                saved = json.load(f)
            inv = {v: k for k, v in PWR.items()}
            for key, guid in inv.items():
                if key in saved:
                    subprocess.run(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", SUB_SLEEP, guid,
                                    str(saved[key])], shell=False, check=False)
            subprocess.run(["powercfg", "/setactive", "SCHEME_CURRENT"], shell=False, check=False)
            print("[durable] restored power settings", flush=True)
        except Exception as e:  # noqa: BLE001 (never let cleanup crash-loop; deleting the task matters more)
            print(f"[durable] power restore skipped ({e})", flush=True)
    subprocess.run(["schtasks", "/delete", "/tn", TASK_NAME, "/f"], shell=False, check=False)
    print("[durable] deleted scheduled task; fully done", flush=True)


def main():
    os.chdir(REPO)
    if _locked():
        print("[durable] another fetch instance holds the lock; exiting", flush=True)
        return
    try:
        try:
            main_last = _last_index(MAIN_TITLES)
            tab_last = _last_index(TAB_TITLES)
        except OSError as exc:
            # Without the title lists there is nothing to resume, and the scheduler would retry every
            # 2 minutes with a traceback each time. Say what is missing, once, and stop cleanly.
            print(f"[durable] cannot read the title lists ({exc}); run the harvest/title steps first.",
                  flush=True)
            return

        if _batches_done(MAIN_CKPT) < main_last:
            print(f"[durable] running main fetch (at batch {_batches_done(MAIN_CKPT)+1}/{main_last+1})",
                  flush=True)
            subprocess.run([PY, "-u", "-m", "scripts.fetch_html_extra"], shell=False, check=False)

        if _batches_done(MAIN_CKPT) >= main_last and _batches_done(TAB_CKPT) < tab_last:
            print("[durable] main complete; running table-gap second pass", flush=True)
            subprocess.run([PY, "-u", "-m", "scripts.fetch_html_extra",
                            TAB_TITLES, TAB_CKPT, "2000"], shell=False, check=False)

        if _batches_done(MAIN_CKPT) >= main_last and _batches_done(TAB_CKPT) >= tab_last:
            print("[durable] BOTH passes complete", flush=True)
            _restore_power_and_cleanup()
    finally:
        if os.path.exists(LOCK):
            try:
                with open(LOCK, encoding="utf-8") as f:
                    held_pid = int(f.read().strip())
                if held_pid == os.getpid():
                    os.remove(LOCK)
            except Exception:  # noqa: BLE001, S110 (best-effort lock cleanup, never fail the run over it)
                pass


if __name__ == "__main__":
    main()
