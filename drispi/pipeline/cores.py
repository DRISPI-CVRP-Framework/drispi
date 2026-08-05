"""CPU-list parsing and cores-block resolution for the pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoresConfig:
    """Resolved core split for one pipeline slice.

    When ``from_cores_block`` is False, only ``dri`` (from ``n_workers``) is
    meaningful for subcluster solving; ``sp`` / ``cpu_list`` / pinning are unused.
    """

    total: int
    dri: int
    sp: int
    cpu_list: list[int] | None
    from_cores_block: bool

    @property
    def dri_cpus(self) -> list[int] | None:
        if self.cpu_list is None:
            return None
        return list(self.cpu_list[: self.dri])

    @property
    def sp_cpus(self) -> list[int] | None:
        if self.cpu_list is None:
            return None
        return list(self.cpu_list[self.dri : self.dri + self.sp])

    @property
    def idle_cpus(self) -> list[int]:
        if self.cpu_list is None:
            return []
        return list(self.cpu_list[self.dri + self.sp :])


def parse_cpu_list(spec: str) -> list[int]:
    """Parse ``0-7``, ``0,2,4``, or ``0-3,8-11`` into a sorted unique CPU id list."""
    text = spec.strip()
    if not text:
        raise ValueError("CPU list string is empty")
    cpus: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid CPU range {token!r}: end < start")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        raise ValueError(f"No CPUs parsed from {spec!r}")
    return sorted(cpus)


def resolve_cores(
    *,
    cores_total: int | None,
    cores_dri: int | None,
    cores_sp: int | None,
    cores_cpu_list: list[int] | None,
    n_workers: int,
    cli_cpus: str | None = None,
) -> CoresConfig:
    """
    Resolve the effective core split.

    If any of ``cores_total`` / ``cores_dri`` / ``cores_sp`` is set, the ``cores:``
    block is the source of truth (all three required). Otherwise legacy
    ``n_workers`` supplies the DRI worker count with no pinning.
    """
    cpu_list = list(cores_cpu_list) if cores_cpu_list is not None else None
    if cli_cpus is not None:
        cpu_list = parse_cpu_list(cli_cpus)

    block_fields = (cores_total, cores_dri, cores_sp)
    any_set = any(v is not None for v in block_fields)
    all_set = all(v is not None for v in block_fields)

    if any_set and not all_set:
        raise ValueError(
            "cores block requires all of cores.total, cores.dri, and cores.sp "
            f"(got total={cores_total!r}, dri={cores_dri!r}, sp={cores_sp!r})"
        )

    if all_set:
        assert cores_total is not None and cores_dri is not None and cores_sp is not None
        if cores_total < 1 or cores_dri < 1 or cores_sp < 0:
            raise ValueError(
                f"Invalid cores values: total={cores_total}, dri={cores_dri}, sp={cores_sp}"
            )
        if cores_dri + cores_sp > cores_total:
            raise AssertionError(
                f"cores.dri + cores.sp ({cores_dri}+{cores_sp}={cores_dri + cores_sp}) "
                f"exceeds cores.total ({cores_total})"
            )
        if cpu_list is not None and len(cpu_list) != cores_total:
            raise ValueError(
                f"cpu_list length {len(cpu_list)} does not match cores.total {cores_total}"
            )
        cfg = CoresConfig(
            total=cores_total,
            dri=cores_dri,
            sp=cores_sp,
            cpu_list=cpu_list,
            from_cores_block=True,
        )
        if cores_dri + cores_sp < cores_total:
            idle = cfg.idle_cpus
            idle_msg = (
                f"idle CPUs in list: {idle}"
                if idle
                else f"{cores_total - cores_dri - cores_sp} logical core(s) unused (no cpu_list)"
            )
            LOGGER.warning(
                "cores.dri + cores.sp (%d+%d=%d) < cores.total (%d); "
                "remainder left idle (never silently reassigned). %s",
                cores_dri,
                cores_sp,
                cores_dri + cores_sp,
                cores_total,
                idle_msg,
            )
        return cfg

    # Legacy: n_workers only
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if cli_cpus is not None:
        LOGGER.warning(
            "--cpus / cpu_list ignored because cores: block is absent "
            "(legacy n_workers path; no pinning)"
        )
    return CoresConfig(
        total=n_workers,
        dri=n_workers,
        sp=0,
        cpu_list=None,
        from_cores_block=False,
    )


def log_current_affinity(label: str) -> set[int]:
    """Log and return ``os.sched_getaffinity(0)`` for verification."""
    try:
        aff = os.sched_getaffinity(0)
    except AttributeError:
        LOGGER.warning("%s: sched_getaffinity not available on this platform", label)
        return set()
    LOGGER.info("%s: sched_getaffinity(0)=%s", label, sorted(aff))
    return set(aff)


def set_affinity(cpus: list[int], *, label: str) -> None:
    """Pin the current process to ``cpus`` and log the effective affinity."""
    if not cpus:
        raise ValueError("cannot set empty affinity mask")
    os.sched_setaffinity(0, set(cpus))
    log_current_affinity(label)
