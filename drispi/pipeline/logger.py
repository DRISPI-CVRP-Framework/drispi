"""Structured pipeline logging: terminal (color), run.log, and run.jsonl."""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from logging import Handler, LogRecord
from pathlib import Path
from typing import Any, TextIO

import colorlog

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.haos.haos import HAOSSelection
from drispi.haos.tag import HAOSTag
from drispi.pipeline.config import DRISPIConfig
from drispi.route_pool.pool import RoutePool
from drispi.utils.io import write_sol
from drispi.utils.metrics import gap_to_bks

# ANSI (terminal only)
_RESET = "\033[0m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_BOLD = "\033[1m"
_YELLOW = "\033[33m"

_TAG_INIT = "  INIT   "
_TAG_FINAL = "  FINAL  "
_TAG_HAOS = "HAOS ROLL"
_TAG_IMPROVE = " IMPROVE "
_TAG_NEW_BKS = " NEW BKS "
_TAG_SUMMARY = " Summary "

_HRULE = "════════════════════════════════════════"
_DASH = "────────────────────────────────────────"


def _format_tag(label: str) -> str:
    if len(label) > 9:
        return label[:9]
    return label.center(9)


def _format_phase_tag(phase_num: int, total_phases: int) -> str:
    return _format_tag(f"Phase {phase_num}/{total_phases}")


def _format_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _format_iteration(iteration: int | None) -> str:
    if iteration is None:
        return "i —"
    return f"i {iteration + 1}"


_METHOD_LOG_ALIASES: dict[str, str] = {
    "agglomerative_avg": "ac_avg",
    "agglomerative_complete": "ac_complete",
    "agglomerative_single": "ac_single",
}


def _format_method_short(method: str) -> str:
    return _METHOD_LOG_ALIASES.get(method, method)


def _format_line(iteration: int | None, tag: str, content: str) -> str:
    ts = _format_timestamp()
    it = _format_iteration(iteration)
    return f"{ts} | {it} | [{_format_tag(tag)}] {content}"


def _pct_gap(numerator_cost: float, denominator_cost: float) -> float:
    return (numerator_cost - denominator_cost) / denominator_cost * 100.0


def _format_pct_gap(numerator_cost: float, denominator_cost: float | None) -> tuple[str, float | None]:
    if denominator_cost is None or denominator_cost <= 0 or not math.isfinite(denominator_cost):
        return "N/A", None
    gap = _pct_gap(numerator_cost, denominator_cost)
    return _format_signed_pct(gap), gap


def _format_signed_pct(gap: float) -> str:
    sign = "+" if gap >= 0 else "−"
    return f"{sign}{abs(gap):.2f}%"


def _format_gap_to_bks(cost: float, bks_cost: float | None, instance_name: str) -> str:
    if bks_cost is None:
        return "N/A"
    gap = ((cost - bks_cost) / bks_cost) * 100.0
    return _format_signed_pct(gap)


def _gap_to_bks_value(cost: float, bks_cost: float | None, instance_name: str) -> float | None:
    if bks_cost is None:
        return None
    return gap_to_bks(cost, instance_name, {instance_name: bks_cost})


def _paradigm_short(paradigm: str) -> str:
    if paradigm == "vertex":
        return "vb"
    if paradigm == "route":
        return "rb"
    return paradigm


def _format_elapsed_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _config_to_json(config: DRISPIConfig) -> dict[str, Any]:
    raw = asdict(config)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif is_dataclass(value):
            out[key] = asdict(value)
        else:
            out[key] = value
    return out


class JsonlHandler(Handler):
    """Append one JSON object per log event."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def emit(self, record: LogRecord) -> None:
        event = getattr(record, "json_event", None)
        if event is None:
            return
        self._file.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        super().close()


class _ColorStyle:
    DEFAULT = "default"
    IMPROVE = "improve"
    NEW_BKS = "new_bks"
    HAOS_ROLL = "haos_roll"


class _TerminalHandler(colorlog.StreamHandler):
    """colorlog StreamHandler with per-event ANSI styling."""

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__(stream if stream is not None else sys.stderr)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: LogRecord) -> None:
        try:
            msg = record.getMessage()
            style = getattr(record, "color_style", _ColorStyle.DEFAULT)
            colored = _colorize(msg, style)
            self.stream.write(colored + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self.stream is not sys.stderr:
            self.stream.close()
        super().close()


def _colorize(message: str, style: str) -> str:
    if style == _ColorStyle.IMPROVE:
        return f"{_BLUE}{message}{_RESET}"
    if style == _ColorStyle.NEW_BKS:
        return f"{_GREEN}{_BOLD}{message}{_RESET}"
    if style == _ColorStyle.HAOS_ROLL:
        # Yellow-highlight operator segment: k=... through solver before P≈
        marker = "  P≈"
        idx = message.find(marker)
        if idx == -1:
            return message
        prefix = message[:idx]
        suffix = message[idx:]
        roll_idx = prefix.find("] ")
        if roll_idx == -1:
            return message
        head = prefix[: roll_idx + 2]
        ops = prefix[roll_idx + 2 :]
        return f"{head}{_YELLOW}{ops}{_RESET}{suffix}"
    return message


_HaosComboKey = tuple[str, str, int, float, str]


class PipelineLogger:
    """Three-target logger: colored terminal, plain run.log, structured run.jsonl."""

    def __init__(
        self,
        instance_name: str,
        output_dir: Path,
        bks: float | None = None,
        *,
        stream: TextIO | None = None,
    ) -> None:
        self._instance_name = instance_name
        self._bks_cost = bks
        self._run_dir = output_dir / instance_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(f"drispi.pipeline.{instance_name}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        self._terminal = _TerminalHandler(stream=stream)
        self._terminal.setFormatter(logging.Formatter("%(message)s"))

        file_handler = logging.FileHandler(self._run_dir / "run.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        jsonl_handler = JsonlHandler(self._run_dir / "run.jsonl")

        self._logger.addHandler(self._terminal)
        self._logger.addHandler(file_handler)
        self._logger.addHandler(jsonl_handler)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def _emit(
        self,
        iteration: int | None,
        tag: str,
        content: str,
        *,
        color_style: str = _ColorStyle.DEFAULT,
        json_event: dict[str, Any] | None = None,
    ) -> None:
        line = _format_line(iteration, tag, content)
        record = self._logger.makeRecord(
            self._logger.name,
            logging.INFO,
            "",
            0,
            line,
            (),
            None,
        )
        record.color_style = color_style  # type: ignore[attr-defined]
        record.json_event = json_event  # type: ignore[attr-defined]
        self._logger.handle(record)

    def log_init(self, config: DRISPIConfig, instance: CVRPInstance) -> None:
        bks_display = f"{self._bks_cost:.2f}" if self._bks_cost is not None else "N/A"
        lines = [
            _HRULE,
            (
                f"DRISPI  instance={instance.name}  n={instance.n_customers}  "
                f"capacity={instance.capacity}"
            ),
            (
                f"time_limit={config.time_limit}  max_no_improve={config.max_no_improve}  "
                f"n_workers={config.n_workers}  seed={config.seed}"
            ),
            (
                f"warmup={config.warmup_iterations}  sp_interval={config.sp_interval}  "
                f"min_coverage={config.min_coverage}  max_pool={config.max_pool_size}"
            ),
            (
                f"decay={config.haos_config.decay}  "
                f"bg_ails_omega={config.bg_ails_initial_omega}  "
                f"bg_time={config.bg_ails_time_limit}  "
                f"std_time={config.standard_improvement_time_limit}"
            ),
            f"BKS={bks_display}",
            _HRULE,
        ]
        for i, line in enumerate(lines):
            json_event = None
            if i == len(lines) - 1:
                json_event = {
                    "type": "init",
                    "instance": instance.name,
                    "n_customers": instance.n_customers,
                    "config": _config_to_json(config),
                    "bks": self._bks_cost,
                }
            self._emit(None, _TAG_INIT, line, json_event=json_event)

    def log_haos_roll(
        self,
        iteration: int,
        selection: HAOSSelection,
        is_spsc_iter: bool,
        joint_prob: float,
    ) -> None:
        p_short = _paradigm_short(selection.paradigm)
        method_short = _format_method_short(selection.method)
        content = (
            f"k={selection.k}  λ={selection.lambda_demand}  "
            f"{p_short}  {method_short}  {selection.solver}  P≈{joint_prob:.6f}"
        )
        if is_spsc_iter:
            content += "  SP/SC ✓"
        self._emit(
            iteration,
            _TAG_HAOS,
            content,
            color_style=_ColorStyle.HAOS_ROLL,
            json_event={
                "type": "haos_roll",
                "iteration": iteration,
                "k": selection.k,
                "lambda_demand": selection.lambda_demand,
                "paradigm": _paradigm_short(selection.paradigm),
                "method": selection.method,
                "solver": selection.solver,
                "joint_prob": joint_prob,
                "is_spsc": is_spsc_iter,
            },
        )

    def log_phase_done(
        self,
        iteration: int,
        phase_num: int,
        total_phases: int,
        phase_name: str,
        elapsed: float,
        budget: float | None,
        operator_info: dict[str, Any] | None = None,
    ) -> None:
        op = operator_info or {}
        elapsed_s = f"{elapsed:.2f}s"
        if budget is None:
            budget_s = "—"
            load_s = "—"
            load_pct: float | None = None
        else:
            budget_s = f"{budget:.2f}s"
            load_pct = (elapsed / budget * 100.0) if budget > 0 else 0.0
            load_s = f"{load_pct:.1f}%"

        parts = [phase_name, f"{elapsed_s} / {budget_s} / {load_s}"]

        if phase_name == "decompose":
            lam = op.get("lambda_demand")
            paradigm = op.get("paradigm")
            method = op.get("method")
            if lam is not None and paradigm is not None and method is not None:
                parts.append(f"λ={lam}")
                parts.append(
                    f"{_paradigm_short(str(paradigm))} {_format_method_short(str(method))}"
                )
        elif phase_name == "route":
            solver = op.get("solver")
            k = op.get("k")
            if solver is not None:
                parts.append(str(solver))
            if k is not None:
                parts.append(f"k={k}")
        elif phase_name == "sp_sc":
            mode = op.get("sc_or_sp")
            pool_size = op.get("pool_size")
            avg_coverage = op.get("avg_coverage")
            min_coverage = op.get("min_coverage")
            if mode is not None:
                parts.append(str(mode))
            if avg_coverage is not None and min_coverage is not None:
                parts.append(f"avg_coverage={float(avg_coverage):.2f}/{min_coverage}")
            if pool_size is not None:
                parts.append(f"pool={pool_size}")

        content = "  ".join(parts)
        tag = _format_phase_tag(phase_num, total_phases)
        json_event: dict[str, Any] = {
            "type": "phase_done",
            "iteration": iteration,
            "phase_num": phase_num,
            "total_phases": total_phases,
            "phase_name": phase_name,
            "elapsed": round(elapsed, 2),
            "budget": budget,
            "load_pct": round(load_pct, 1) if load_pct is not None else None,
            "operator_info": op or None,
        }
        self._emit(iteration, tag, content, json_event=json_event)

    def log_improve(self, iteration: int, cost: float, phase_name: str) -> None:
        gap = _format_gap_to_bks(cost, self._bks_cost, self._instance_name)
        content = f"{cost:.2f}  via {phase_name}  ·  Δ_BKS={gap}"
        gap_val = _gap_to_bks_value(cost, self._bks_cost, self._instance_name)
        self._emit(
            iteration,
            _TAG_IMPROVE,
            content,
            color_style=_ColorStyle.IMPROVE,
            json_event={
                "type": "improve",
                "iteration": iteration,
                "cost": cost,
                "phase_name": phase_name,
                "gap_to_bks": gap_val,
            },
        )

    def log_new_bks(
        self,
        iteration: int,
        cost: float,
        phase_name: str,
        routes: list[Route],
        instance: CVRPInstance,
    ) -> None:
        gap = _format_gap_to_bks(cost, self._bks_cost, self._instance_name)
        content = f"{cost:.2f}  via {phase_name}  ·  Δ_BKS={gap}"
        gap_val = _gap_to_bks_value(cost, self._bks_cost, self._instance_name)
        self._emit(
            iteration,
            _TAG_NEW_BKS,
            content,
            color_style=_ColorStyle.NEW_BKS,
            json_event={
                "type": "new_bks",
                "iteration": iteration,
                "cost": cost,
                "phase_name": phase_name,
                "gap_to_bks": gap_val,
            },
        )
        bks_path = self._run_dir / f"{self._instance_name}_BKS.sol"
        write_sol(routes, cost, bks_path)

    def log_summary(
        self,
        iteration: int,
        iter_cost: float,
        best_cost: float,
        last_cost: float,
        no_improve: int,
    ) -> None:
        del last_cost
        delta_s_star_s, delta_s_star = _format_pct_gap(iter_cost, best_cost)
        delta_iter_bks_s, delta_iter_bks = _format_pct_gap(iter_cost, self._bks_cost)
        delta_best_bks_s, delta_best_bks = _format_pct_gap(best_cost, self._bks_cost)

        content = (
            f"cost={iter_cost:.2f}  Δ_S*={delta_s_star_s}  "
            f"Δ_BKS={delta_iter_bks_s}  Δ_S*_BKS={delta_best_bks_s}  no_improve={no_improve}"
        )
        self._emit(
            iteration,
            _TAG_SUMMARY,
            content,
            json_event={
                "type": "summary",
                "iteration": iteration,
                "iter_cost": iter_cost,
                "best_cost": best_cost,
                "delta_to_best_pct": round(delta_s_star, 2) if delta_s_star is not None else None,
                "delta_to_bks_pct": round(delta_iter_bks, 2) if delta_iter_bks is not None else None,
                "delta_best_to_bks_pct": (
                    round(delta_best_bks, 2) if delta_best_bks is not None else None
                ),
                "no_improve": no_improve,
            },
        )

    def log_final(
        self,
        iterations: int,
        elapsed: float,
        best_cost: float,
        stopped_by_no_improve: bool,
        stopped_by_time: bool,
        best_solution: list[Route],
        pool: RoutePool,
        instance: CVRPInstance,
    ) -> None:
        del instance
        delta_bks_s = _format_gap_to_bks(best_cost, self._bks_cost, self._instance_name)

        self._emit(None, _TAG_FINAL, _HRULE)
        self._emit(
            None,
            _TAG_FINAL,
            (
                f"Run complete  iterations={iterations}  "
                f"elapsed={_format_elapsed_hms(elapsed)}  best={best_cost:.2f}"
            ),
        )
        self._emit(
            None,
            _TAG_FINAL,
            (
                f"Δ_BKS={delta_bks_s}  no_improve_stop={stopped_by_no_improve}  "
                f"time_stop={stopped_by_time}"
            ),
        )
        self._emit(None, _TAG_FINAL, "")
        self._write_haos_summary("BEST SOLUTION — HAOS TAG SUMMARY", best_solution, pool)
        self._emit(None, _TAG_FINAL, "")
        self._write_haos_summary("ROUTE POOL — HAOS TAG SUMMARY", None, pool)
        self._emit(
            None,
            _TAG_FINAL,
            _HRULE,
            json_event={
                "type": "final",
                "iterations": iterations,
                "elapsed": elapsed,
                "best_cost": best_cost,
                "stopped_by_no_improve": stopped_by_no_improve,
                "stopped_by_time": stopped_by_time,
            },
        )

    def _write_haos_summary(
        self,
        title: str,
        routes: list[Route] | None,
        pool: RoutePool,
    ) -> None:
        self._emit(None, _TAG_FINAL, f"[{title}]")
        self._emit(None, _TAG_FINAL, _DASH)
        non_improve, improve_count = self._collect_tag_stats(routes, pool)
        for line in self._format_combo_lines(non_improve):
            self._emit(None, _TAG_FINAL, f"  {line}")
        if improve_count > 0:
            self._emit(
                None,
                _TAG_FINAL,
                f"  {improve_count} routes  post-SP/SC improvement",
            )

    def _collect_tag_stats(
        self,
        routes: list[Route] | None,
        pool: RoutePool,
    ) -> tuple[Counter[_HaosComboKey | str], int]:
        """Return non-improvement combo counts and improvement-route count."""
        non_improve: Counter[_HaosComboKey | str] = Counter()
        improve_count = 0

        if routes is not None:
            for route in routes:
                tag = pool.get_haos_tag(route)
                if tag is None:
                    non_improve["[untagged]"] += 1
                elif tag.is_improvement_route:
                    improve_count += 1
                else:
                    non_improve[self._combo_key(tag)] += 1
            return non_improve, improve_count

        for entry in pool:
            tag = entry.haos_tag
            if tag is None:
                non_improve["[untagged]"] += 1
            elif tag.is_improvement_route:
                improve_count += 1
            else:
                non_improve[self._combo_key(tag)] += 1
        return non_improve, improve_count

    @staticmethod
    def _combo_key(tag: HAOSTag) -> _HaosComboKey:
        return (tag.paradigm, tag.method, tag.k, tag.lambda_demand, tag.solver)

    def _format_combo_lines(self, counts: Counter[_HaosComboKey | str]) -> list[str]:
        lines: list[str] = []
        for key, count in counts.most_common():
            if key == "[untagged]":
                lines.append(f"{count} routes  [untagged]")
            else:
                paradigm, method, k, lam, solver = key
                lines.append(
                    f"{count} routes  {_paradigm_short(paradigm)}  "
                    f"{_format_method_short(method)}  "
                    f"k={k}  λ={lam}  solver={solver}"
                )
        return lines

    def beats_bks(self, cost: float) -> bool:
        if self._bks_cost is None:
            return False
        return cost < self._bks_cost
