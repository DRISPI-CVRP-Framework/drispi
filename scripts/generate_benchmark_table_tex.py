#!/usr/bin/env python3
"""Generate a LaTeX results table for benchmark_50xl (Altendeitering-style)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".tmp_pypdf"))

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "artifacts/benchmarks/benchmark_50xl"
DRSCI_PDF = ROOT / "artifacts/comparison/20260331_Altendeitering+Bachmann_PS.pdf"
OUTPUT = BENCHMARK_DIR / "comparison" / "benchmark_table.tex"


def _fmt_cost(value: int) -> str:
    s = f"{value:,}".replace(",", "\\,")
    return s


def _fmt_gap(cost: float, reference: int) -> str:
    gap = (cost - reference) / reference * 100.0
    return f"{gap:+.2f}"


def _parse_report_table() -> dict[str, dict]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(DRSCI_PDF).pages)
    start = text.find("Table 1:")
    end = text.find("To contextualize")
    meta: dict[str, dict] = {}

    for line in text[start:end].splitlines():
        line = line.strip()
        match = re.match(r"^(\d+)\s+(XL-n\d+-k\d+)\s+(.+)$", line)
        if not match:
            continue
        inst = match.group(2)
        rest = match.group(3)
        gaps = list(re.finditer(r"\+\d+\.\d+", rest))
        if len(gaps) < 2:
            continue

        before_first_gap = rest[: gaps[0].start()].strip().split()
        route_size_idx = next(
            i for i in range(len(before_first_gap) - 1, -1, -1) if re.fullmatch(r"\d+\.\d+", before_first_gap[i])
        )
        attrs = before_first_gap[:route_size_idx]
        q_val = before_first_gap[route_size_idx - 1]
        dem = before_first_gap[route_size_idx - 2]
        dep = attrs[0]
        cust = " ".join(attrs[1 : route_size_idx - 2])
        r_val = before_first_gap[route_size_idx]
        drsci_cost = int("".join(before_first_gap[route_size_idx + 1 :]))

        after_first_gap = rest[gaps[0].end() :].strip()
        second_gap = re.search(r"\+\d+\.\d+", after_first_gap)
        initial_part = after_first_gap[: second_gap.start()].strip()
        final_part = after_first_gap[second_gap.end() :].strip()
        initial_bks = int("".join(initial_part.split()))
        final_bks = int("".join(final_part.split()))

        meta[inst] = {
            "num": int(match.group(1)),
            "dep": dep,
            "cust": cust.replace("–", "--"),
            "dem": dem.replace("–", "--"),
            "q": q_val,
            "r": r_val,
            "drsci": drsci_cost,
            "initial_bks": initial_bks,
            "final_bks": final_bks,
        }
    return meta


def _load_drispi_results() -> dict[str, int | None]:
    results: dict[str, int | None] = {}
    for run_jsonl in sorted(BENCHMARK_DIR.glob("*/run.jsonl")):
        stem = run_jsonl.parent.name
        match = re.match(r"^(X-n\d+-k\d+)_\d{4}_\d{4}$", stem)
        inst = f"XL{match.group(1)[1:]}" if match else stem

        lines = [json.loads(raw) for raw in run_jsonl.read_text().splitlines() if raw.strip()]
        final = [line for line in lines if line.get("type") == "final"]
        if final and isinstance(final[0].get("best_cost"), (int, float)):
            results[inst] = int(round(final[0]["best_cost"]))
        else:
            results[inst] = None
    return results


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def build_latex(rows: list[dict]) -> str:
    body_lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        if row["drispi"] is None:
            drispi_cells = "error & --"
        else:
            drispi_cells = f"{_fmt_cost(row['drispi'])} & {_fmt_gap(row['drispi'], row['initial_bks'])}"

        drsci_gap = _fmt_gap(row["drsci"], row["initial_bks"])
        init_gap = _fmt_gap(row["initial_bks"], row["final_bks"])

        body_lines.append(
            " & ".join(
                [
                    str(idx),
                    row["name"],
                    row["dep"],
                    _latex_escape(row["cust"]),
                    row["dem"],
                    row["q"],
                    row["r"],
                    drispi_cells,
                    f"{_fmt_cost(row['drsci'])} & {drsci_gap}",
                    f"{_fmt_cost(row['initial_bks'])} & {init_gap}",
                    _fmt_cost(row["final_bks"]),
                ]
            )
            + r" \\"
        )

    body = "\n".join(body_lines)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.6in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{caption}}

\captionsetup{{font=small, labelfont=bf}}

\begin{{document}}

\begin{{table}}[ht]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3.5pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\caption{{Detailed results on the 50-instance \texttt{{benchmark\_50xl}} subset. Failed runs are marked \emph{{error}}.}}
\label{{tab:benchmark50xl}}
\begin{{tabular}}{{@{{}}r l c c c r r rr rr rr r@{{}}}}
\toprule
 & & & & & & & \multicolumn{{2}}{{c}}{{DRISPI}} & \multicolumn{{2}}{{c}}{{DRSCI}} & \multicolumn{{2}}{{c}}{{Initial-BKS}} & Final-BKS \\
\cmidrule(lr){{8-9}} \cmidrule(lr){{10-11}} \cmidrule(lr){{12-13}}
\# & Name & Dep & Cust & Dem & $Q$ & $r$ & Cost & Gap (\%) & Cost & Gap (\%) & Cost & Gap (\%) & Cost \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}

\end{{document}}
"""


def main() -> None:
    meta = _parse_report_table()
    drispi = _load_drispi_results()

    rows: list[dict] = []
    for inst in sorted(drispi, key=lambda name: meta[name]["num"]):
        row = {"name": inst, "drispi": drispi[inst], **meta[inst]}
        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_latex(rows), encoding="utf-8")
    errors = sum(1 for row in rows if row["drispi"] is None)
    print(f"Wrote {len(rows)} rows ({errors} errors) to {OUTPUT}")


if __name__ == "__main__":
    main()
