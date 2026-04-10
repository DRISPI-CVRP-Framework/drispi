from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class EventKind(str, Enum):
    ITER_START     = "iter_start"
    DECOMPOSED     = "decomposed"
    CLUSTER_SOLVED = "cluster_solved"
    LS1_DONE       = "ls1_done"
    SC_DONE        = "sc_done"
    LS2_DONE       = "ls2_done"
    BEST_UPDATED   = "best_updated"
    DONE           = "done"


@dataclass
class GUIEvent:
    kind: EventKind
    data: dict = field(default_factory=dict)
