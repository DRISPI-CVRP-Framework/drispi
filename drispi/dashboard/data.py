"""File readers and parsers for the DRISPI monitor dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _snapshot_path_key(path: Path) -> tuple[int, int, int] | None:
    """Sort key: (iteration, phase_num, sub_index). sub: pre=0, plain=1, post=2."""
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "iter" or parts[2] != "phase":
        return None
    try:
        iteration = int(parts[1])
        phase_num = int(parts[3])
    except ValueError:
        return None
    tag = parts[4] if len(parts) > 4 else ""
    sub = {"pre": 0, "": 1, "post": 2}.get(tag, 1)
    return iteration, phase_num, sub


def get_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    """Read the first line of run.jsonl (type=\"init\")."""
    events = _read_jsonl(run_dir / "run.jsonl")
    for event in events:
        if event.get("type") == "init":
            return event
    return None


def get_latest_run_state(run_dir: Path) -> dict[str, Any] | None:
    """Read the last summary line from run.jsonl."""
    events = _read_jsonl(run_dir / "run.jsonl")
    for event in reversed(events):
        if event.get("type") == "summary":
            return event
    return None


def get_all_summary_lines(run_dir: Path) -> list[dict[str, Any]]:
    """All type=summary lines from run.jsonl."""
    return [e for e in _read_jsonl(run_dir / "run.jsonl") if e.get("type") == "summary"]


def get_latest_haos_roll(run_dir: Path) -> dict[str, Any] | None:
    """Latest haos_roll line (typically start-of-iteration, no levels)."""
    events = _read_jsonl(run_dir / "run.jsonl")
    for event in reversed(events):
        if event.get("type") == "haos_roll":
            return event
    return None


def get_haos_roll_for_iteration(run_dir: Path, iteration: int) -> dict[str, Any] | None:
    """HAOS roll logged at the start of ``iteration`` (no weight levels)."""
    for event in _read_jsonl(run_dir / "run.jsonl"):
        if (
            event.get("type") == "haos_roll"
            and event.get("iteration") == iteration
            and event.get("levels") is None
        ):
            return event
    return None


def get_latest_haos_roll_without_levels(run_dir: Path) -> dict[str, Any] | None:
    """Latest haos_roll line that does not include weight levels."""
    events = _read_jsonl(run_dir / "run.jsonl")
    for event in reversed(events):
        if event.get("type") == "haos_roll" and event.get("levels") is None:
            return event
    return None


def get_haos_weights(run_dir: Path) -> dict[str, Any] | None:
    """
    Latest haos_roll with full weight levels (end-of-iteration).
    Falls back to haos_weights_final.json when no in-run levels exist.
    """
    events = _read_jsonl(run_dir / "run.jsonl")
    for event in reversed(events):
        if event.get("type") == "haos_roll" and event.get("levels") is not None:
            return event

    final_path = run_dir / "haos_weights_final.json"
    if final_path.is_file():
        data = json.loads(final_path.read_text(encoding="utf-8"))
        levels = data.get("levels")
        if isinstance(levels, dict):
            roll = get_latest_haos_roll_without_levels(run_dir) or get_latest_haos_roll(run_dir) or {}
            return {
                "levels": levels,
                "k": roll.get("k"),
                "lambda_demand": roll.get("lambda_demand"),
                "paradigm": roll.get("paradigm"),
                "method": roll.get("method"),
                "solver": roll.get("solver"),
            }
    return None


def get_latest_phase_done(run_dir: Path) -> dict[str, Any] | None:
    """Latest type=phase_done line."""
    events = _read_jsonl(run_dir / "run.jsonl")
    for event in reversed(events):
        if event.get("type") == "phase_done":
            return event
    return None


def snapshot_sort_key_from_payload(snapshot: dict[str, Any]) -> tuple[int, int, int]:
    """Sort key aligned with on-disk snapshot ordering: (iter, phase, sub-stage)."""
    phase_num = int(snapshot.get("phase_num") or 0)
    stage = snapshot.get("bg_stage")
    sub = {"pre": 0, "post": 2}.get(str(stage) if stage else "", 1)
    return int(snapshot.get("iteration") or 0), phase_num, sub


def get_current_iteration_0(
    run_dir: Path,
    state: dict[str, Any] | None = None,
) -> int:
    """
    In-progress iteration (0-based).

    Uses the newest snapshot on disk. ``run.jsonl`` summaries are written only
    after an iteration finishes, so the latest summary lags during the next iter.
    """
    latest = get_latest_snapshot(run_dir)
    if latest is not None:
        return int(latest.get("iteration", 0))
    if state is None:
        state = get_latest_run_state(run_dir)
    if state is not None:
        return int(state.get("iteration", 0))
    return 0


def get_latest_snapshot_for_iteration(
    run_dir: Path,
    iteration: int,
) -> dict[str, Any] | None:
    """Newest phase snapshot for one iteration (merges list + global latest)."""
    snaps = list_iteration_snapshots(run_dir, iteration)
    latest = snaps[-1] if snaps else None
    global_latest = get_latest_snapshot(run_dir)
    if global_latest is not None and int(global_latest.get("iteration", -1)) == iteration:
        if latest is None or snapshot_sort_key_from_payload(global_latest) > snapshot_sort_key_from_payload(
            latest
        ):
            latest = global_latest
    return latest


def list_iteration_snapshots(run_dir: Path, iteration: int) -> list[dict[str, Any]]:
    """All phase snapshots for one iteration, ordered by phase_num then bg sub-stage."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        return []
    found: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for path in snap_dir.glob(f"iter_{iteration}_phase_*.json"):
        key = _snapshot_path_key(path)
        if key is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        found.append((key, payload))
    found.sort(key=lambda x: x[0])
    return [payload for _, payload in found]


def get_latest_snapshot(run_dir: Path) -> dict[str, Any] | None:
    """Snapshot with highest iteration and phase number."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        return None
    best_key: tuple[int, int, int] | None = None
    best_path: Path | None = None
    for path in snap_dir.glob("iter_*_phase_*.json"):
        key = _snapshot_path_key(path)
        if key is None:
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    if best_path is None:
        return None
    return json.loads(best_path.read_text(encoding="utf-8"))


def get_bg_snapshots_for_iteration(
    run_dir: Path,
    iteration: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (pre-perturbation, post-BG-AILS) snapshots for an iteration, if present."""
    pre: dict[str, Any] | None = None
    post: dict[str, Any] | None = None
    for snap in list_iteration_snapshots(run_dir, iteration):
        if snap.get("phase_name") != "bg_ails":
            continue
        stage = snap.get("bg_stage")
        if stage == "pre":
            pre = snap
        elif stage == "post":
            post = snap
        elif stage is None and post is None:
            post = snap
    return pre, post


def get_best_solution(run_dir: Path) -> dict[str, Any] | None:
    """Read best_solution.json if present."""
    path = run_dir / "best_solution.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_improve_events(run_dir: Path) -> list[dict[str, Any]]:
    """All type=improve lines from run.jsonl."""
    return [e for e in _read_jsonl(run_dir / "run.jsonl") if e.get("type") == "improve"]


def parse_first_summary_timestamp(run_dir: Path) -> str | None:
    """Extract HH:MM:SS from run.log for the first Summary line."""
    log_path = run_dir / "run.log"
    if not log_path.is_file():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " Summary " in line or "[ Summary ]" in line:
            ts = line.split("|", 1)[0].strip()
            if len(ts) >= 8 and ts[2] == ":":
                return ts
    return None
