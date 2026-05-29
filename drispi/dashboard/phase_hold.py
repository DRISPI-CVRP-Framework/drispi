"""Minimum dwell time for current-iteration phase snapshots."""

from __future__ import annotations

import copy
from typing import Any

MIN_CURRENT_PHASE_DISPLAY_S = 10.0


def snapshot_fingerprint(snap: dict[str, Any]) -> str:
    stage = snap.get("bg_stage") or "main"
    return (
        f"{snap.get('iteration')}|{snap.get('phase_num')}|{stage}|{snap.get('timestamp', '')}"
    )


def _empty_hold() -> dict[str, Any]:
    return {
        "iter": -1,
        "queue_fps": [],
        "snaps": {},
        "display_idx": -1,
        "since": 0.0,
    }


def resolve_current_iteration_display(
    all_snaps: list[dict[str, Any]],
    current_iter_0: int,
    hold: dict[str, Any] | None,
    now: float,
    *,
    min_display_s: float = MIN_CURRENT_PHASE_DISPLAY_S,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Pick which snapshot to show in the current-iteration panel.

    Queues phase snapshots in order; each stays visible for at least ``min_display_s``
    before advancing to the next queued phase.
    """
    state = copy.deepcopy(hold) if hold is not None else _empty_hold()

    if not all_snaps:
        return None, state

    if state["iter"] != current_iter_0:
        state = _empty_hold()
        state["iter"] = current_iter_0
        state["since"] = now

    for snap in all_snaps:
        fp = snapshot_fingerprint(snap)
        if fp not in state["snaps"]:
            state["snaps"][fp] = snap
            state["queue_fps"].append(fp)

    if not state["queue_fps"]:
        return None, state

    if state["display_idx"] < 0:
        state["display_idx"] = len(state["queue_fps"]) - 1
        state["since"] = now

    idx = int(state["display_idx"])
    if idx < len(state["queue_fps"]) - 1 and now - float(state["since"]) >= min_display_s:
        state["display_idx"] = idx + 1
        state["since"] = now
        idx += 1

    return state["snaps"][state["queue_fps"][idx]], state
