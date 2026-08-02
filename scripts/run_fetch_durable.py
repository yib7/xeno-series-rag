"""Durable, self-healing driver for the HTML fetch, built to survive Windows Modern Standby.

The fetch is a ~5h serial pull. This machine drops into Modern Standby (S0) when the user is away,
which suspends/kills background processes (a plain keep-awake flag does NOT defeat it). So instead of
trusting one long-lived process, a Windows Scheduled Task runs THIS script every 2 minutes with
MultipleInstancesPolicy=IgnoreNew: while an instance is mid-fetch the re-triggers are ignored, but the
moment standby kills the instance the next trigger resumes it from the per-batch checkpoint. Power
settings are also disabled (see scripts that create the task) so on AC it ideally never sleeps at all;
the task is the backstop for whatever still slips through.

Order each run: resume main fetch -> (when main is fully done) second table-gap pass -> when BOTH are
complete, restore the saved power settings and delete the scheduled task, so the machine is left
exactly as found. A PID lockfile guarantees only one fetch runs even if triggers overlap.
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
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    return False


def _locked() -> bool:
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK, encoding="utf-8").read().strip())
            if _pid_alive(pid):
                return True
        except Exception:
            pass
    open(LOCK, "w", encoding="utf-8").write(str(os.getpid()))
    return False


def _batches_done(ckpt: str) -> int:
    if not os.path.exists(ckpt):
        return -1
    try:
        return json.load(open(ckpt, encoding="utf-8"))["last_completed_batch"]
    except Exception:
        return -1


def _last_index(titles_path: str, batch_size: int = 100) -> int:
    n = sum(1 for line in open(titles_path, encoding="utf-8") if line.strip())
    return math.ceil(n / batch_size) - 1


def _restore_power_and_cleanup():
    if os.path.exists(POWER_BACKUP):
        try:
            # utf-8-sig: PowerShell's Out-File -Encoding utf8 writes a BOM that plain json.load rejects.
            saved = json.load(open(POWER_BACKUP, encoding="utf-8-sig"))
            inv = {v: k for k, v in PWR.items()}
            for key, guid in inv.items():
                if key in saved:
                    subprocess.run(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", SUB_SLEEP, guid,
                                    str(saved[key])], shell=False)
            subprocess.run(["powercfg", "/setactive", "SCHEME_CURRENT"], shell=False)
            print("[durable] restored power settings", flush=True)
        except Exception as e:  # noqa: BLE001 - never let cleanup crash-loop; deleting the task matters more
            print(f"[durable] power restore skipped ({e})", flush=True)
    subprocess.run(["schtasks", "/delete", "/tn", TASK_NAME, "/f"], shell=False)
    print("[durable] deleted scheduled task -> fully done", flush=True)


def main():
    os.chdir(REPO)
    if _locked():
        print("[durable] another fetch instance holds the lock; exiting", flush=True)
        return
    try:
        main_last = _last_index(MAIN_TITLES)
        tab_last = _last_index(TAB_TITLES)

        if _batches_done(MAIN_CKPT) < main_last:
            print(f"[durable] running main fetch (at batch {_batches_done(MAIN_CKPT)+1}/{main_last+1})",
                  flush=True)
            subprocess.run([PY, "-u", "-m", "scripts.fetch_html_extra"], shell=False)

        if _batches_done(MAIN_CKPT) >= main_last and _batches_done(TAB_CKPT) < tab_last:
            print("[durable] main complete -> running table-gap second pass", flush=True)
            subprocess.run([PY, "-u", "-m", "scripts.fetch_html_extra",
                            TAB_TITLES, TAB_CKPT, "2000"], shell=False)

        if _batches_done(MAIN_CKPT) >= main_last and _batches_done(TAB_CKPT) >= tab_last:
            print("[durable] BOTH passes complete", flush=True)
            _restore_power_and_cleanup()
    finally:
        if os.path.exists(LOCK):
            try:
                if int(open(LOCK, encoding="utf-8").read().strip()) == os.getpid():
                    os.remove(LOCK)
            except Exception:
                pass


if __name__ == "__main__":
    main()
