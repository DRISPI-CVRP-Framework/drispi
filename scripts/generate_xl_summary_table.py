#!/usr/bin/env python3
"""Generate the LaTeX longtable for thesis Section 5.2.1 (XL instance summary).

Combines three sources:
  1. Instance characteristics (Dep, Cust, Dem, Q, r) and the Initial-BKS /
     Final-BKS costs reported in Table 1 of the DRSCI comparison paper
     (Altendeitering & Bachmann, 2026), which reproduces the values from
     Queiroga et al. (2026).
  2. DRISPI's best-of-3-seed cost and its gap to the official current BKS,
     from data/results/finalBenchmarkResults.csv.
  3. A cross-check of the Initial-BKS / Final-BKS values against the CSV's
     own bks_initial / bks_current columns (which were themselves read off
     the CVRPLib BKS Challenge page).

Writes thesis/tables/xl_summary_table.tex, a standalone longtable body meant
to be \\input from chapters/5_Study.tex inside a landscape environment.

Usage:
    python scripts/generate_xl_summary_table.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".tmp_pypdf"))

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DRSCI_PDF = ROOT / "artifacts/comparison/20260331_Altendeitering+Bachmann_PS.pdf"
CSV_PATH = ROOT / "data/results/finalBenchmarkResults_lagrange.csv"
OUTPUT = ROOT / "thesis/tables/xl_summary_table_lagrange.tex"

ROW_RE = re.compile(
    r"^(?P<num>\d+)\s+(?P<name>XL-n\d+-k\d+)\s+(?P<rest>.+)$"
)


def _parse_pdf_table() -> dict[str, dict]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(DRSCI_PDF).pages)
    start = text.find("Table 1:")
    end = text.find("To contextualize")
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate Table 1 in the DRSCI PDF text")

    meta: dict[str, dict] = {}
    for line in text[start:end].splitlines():
        line = line.strip()
        m = ROW_RE.match(line)
        if not m:
            continue
        rest = m.group("rest")

        # Two "+X.XX" gap markers separate DRSCI-cost | Initial-BKS | Final-BKS.
        gaps = list(re.finditer(r"\+\d+\.\d+", rest))
        if len(gaps) < 2:
            continue

        before_first_gap = rest[: gaps[0].start()].strip().split()
        # Walk back from the DRSCI cost block to find the "r" token (a bare
        # decimal not part of the cost group), which marks where the
        # characteristics end and the DRSCI cost digits begin.
        route_size_idx = next(
            i
            for i in range(len(before_first_gap) - 1, -1, -1)
            if re.fullmatch(r"\d+\.\d+", before_first_gap[i])
        )
        attrs = before_first_gap[:route_size_idx]
        q_val = before_first_gap[route_size_idx - 1]
        dem = before_first_gap[route_size_idx - 2]
        dep = attrs[0]
        cust = " ".join(attrs[1 : route_size_idx - 2])
        r_val = before_first_gap[route_size_idx]

        after_first_gap = rest[gaps[0].end() :].strip()
        second_gap = re.search(r"\+\d+\.\d+", after_first_gap)
        initial_part = after_first_gap[: second_gap.start()].strip()
        final_part = after_first_gap[second_gap.end() :].strip()
        initial_bks = int("".join(initial_part.split()))
        final_bks = int("".join(final_part.split()))

        meta[m.group("name")] = {
            "num": int(m.group("num")),
            "dep": dep,
            "cust": cust.replace("\u2013", "--"),
            "dem": dem.replace("\u2013", "--"),
            "q": q_val,
            "r": r_val,
            "pdf_initial_bks": initial_bks,
            "pdf_final_bks": final_bks,
        }
    return meta


def _load_csv_rows(csv_path: Path = CSV_PATH) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["instance"]] = row
    return rows


def _fmt_cost(value: int) -> str:
    # Braced comma grouping, matching the "1{,}000" convention already used
    # for large numbers elsewhere in the thesis (e.g. chapters/3_Problem.tex).
    return f"{value:,}".replace(",", "{,}")


def _fmt_gap(pct: float) -> str:
    return f"{pct:+.2f}"


def _latex_escape(text: str) -> str:
    return text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def build_rows(meta: dict[str, dict], csv_rows: dict[str, dict]) -> list[str]:
    missing = set(csv_rows) ^ set(meta)
    if missing:
        raise RuntimeError(f"Instance mismatch between PDF table and CSV: {missing}")

    body_lines: list[str] = []
    mismatches: list[str] = []
    for inst in sorted(csv_rows, key=lambda name: meta[name]["num"]):
        m = meta[inst]
        c = csv_rows[inst]

        bks_initial = int(round(float(c["bks_initial"])))
        bks_current = int(round(float(c["bks_current"])))
        if bks_initial != m["pdf_initial_bks"] or bks_current != m["pdf_final_bks"]:
            mismatches.append(
                f"{inst}: csv(initial={bks_initial}, current={bks_current}) "
                f"vs pdf(initial={m['pdf_initial_bks']}, final={m['pdf_final_bks']})"
            )

        drispi_cost = int(round(float(c["cost_best3"])))
        drispi_gap = float(c["gap_bks_best3"])
        initial_gap = (bks_initial - bks_current) / bks_current * 100.0

        body_lines.append(
            " & ".join(
                [
                    str(m["num"]),
                    inst,
                    m["dep"],
                    _latex_escape(m["cust"]),
                    _latex_escape(m["dem"]),
                    m["q"],
                    m["r"],
                    _fmt_cost(drispi_cost),
                    _fmt_gap(drispi_gap),
                    _fmt_cost(bks_initial),
                    _fmt_gap(initial_gap),
                    _fmt_cost(bks_current),
                ]
            )
            + r" \\"
        )

    if mismatches:
        raise RuntimeError(
            "CSV/PDF BKS mismatch on "
            + str(len(mismatches))
            + " instance(s):\n" + "\n".join(mismatches)
        )

    return body_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    meta = _parse_pdf_table()
    csv_rows = _load_csv_rows(args.csv)
    print(f"Parsed {len(meta)} rows from PDF, {len(csv_rows)} rows from CSV")

    body_lines = build_rows(meta, csv_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(body_lines)} rows to {args.output}")


if __name__ == "__main__":
    main()
