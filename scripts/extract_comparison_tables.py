#!/usr/bin/env python3
"""Extract XL benchmark comparison data from PDF tables into JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".tmp_pypdf"))
    from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
XL_REPORT = ROOT / "artifacts/comparison/xl_report.pdf"
DRSCI_REPORT = ROOT / "artifacts/comparison/20260331_Altendeitering+Bachmann_PS.pdf"
BKS_FILE = ROOT / "data/bks/xl-bks.json"
OUTPUT = ROOT / "artifacts/comparison/xl_heuristic_comparison.json"

HEURISTICS = [
    "AILS-II",
    "FILO",
    "FILO2",
    "KGLSXXL",
    "HGS-CVRP",
    "SISRs",
    "LKH-3",
    "OR-Tools",
]


def pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def to_int(value: float) -> int | float:
    return int(value) if value == int(value) else value


def format_metric(value: float | None) -> int | float | None:
    if value is None:
        return None
    rounded = round(value, 1)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def fix_merged_numbers(text: str) -> str:
    """Repair PDF extraction artifacts like '1,882,369.01,889,998' or
    '2,334,348.42,340,824', where pdftotext drops the space between a
    one-decimal "avg" value and the next comma-grouped integer. Matches any
    decimal point + single digit immediately followed by a comma-grouped
    number (>=1 comma), not just ones starting with "1,"."""
    return re.sub(r"\.(\d)(\d{1,3}(?:,\d{3})+)", r".\1 \2", text)


def parse_number_tokens(rest: str) -> list[float]:
    numbers: list[float] = []
    for token in re.findall(r"[\d,]+\.?\d*", rest):
        if token.startswith(","):
            token = "1" + token
        numbers.append(float(token.replace(",", "")))
    return numbers


def load_bks() -> dict[str, int]:
    raw = json.loads(BKS_FILE.read_text(encoding="utf-8"))
    bks = {name: cost for name, cost in raw.items() if name.startswith("XL-")}
    if len(bks) != 100:
        raise RuntimeError(f"Expected 100 XL instances in {BKS_FILE}, got {len(bks)}")
    return bks


def parse_table2_heuristics(xl_text: str) -> dict[str, dict[str, dict[str, int | float]]]:
    results: dict[str, dict[str, dict[str, int | float]]] = {}
    for line in xl_text.splitlines():
        line = line.strip()
        if not re.match(r"^XL-n\d+-k\d+\s", line):
            continue

        instance = re.match(r"^(XL-n\d+-k\d+)", line).group(1)
        rest = fix_merged_numbers(line[len(instance) :])
        has_lkh_missing = "–" in rest or re.search(r"\s-\s", rest)

        numbers = parse_number_tokens(rest)
        expected_pairs = 8
        if has_lkh_missing and len(numbers) == 14:
            pairs: list[tuple[float | None, float | None]] = []
            idx = 0
            for pair_idx in range(expected_pairs):
                if pair_idx == 6:
                    pairs.append((None, None))
                    continue
                pairs.append((numbers[idx], numbers[idx + 1]))
                idx += 2
        elif len(numbers) == expected_pairs * 2:
            pairs = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
        else:
            raise ValueError(f"{instance}: expected 16 numbers, got {len(numbers)}: {rest[:120]}")

        entry: dict[str, dict[str, int | float]] = {}
        for heuristic, (best, avg) in zip(HEURISTICS, pairs, strict=True):
            metric: dict[str, int | float] = {}
            if best is not None:
                metric["best"] = to_int(best)
            if avg is not None:
                metric["avg"] = format_metric(avg)
            if metric:
                entry[heuristic] = metric
        results[instance] = entry
    return results


def parse_drsci_table(drsci_text: str) -> dict[str, int]:
    start = drsci_text.find("Table 1:")
    end = drsci_text.find("To contextualize")
    results: dict[str, int] = {}

    for line in drsci_text[start:end].splitlines():
        line = line.strip()
        match = re.match(r"^(\d+)\s+(XL-n\d+-k\d+)\s+(.+)$", line)
        if not match:
            continue

        instance = match.group(2)
        rest = match.group(3)
        gaps = list(re.finditer(r"\+\d+\.\d+", rest))
        if not gaps:
            continue

        before_first_gap = rest[: gaps[0].start()].strip().split()
        route_size_idx = next(
            i for i in range(len(before_first_gap) - 1, -1, -1) if re.fullmatch(r"\d+\.\d+", before_first_gap[i])
        )
        drsci_cost = int("".join(before_first_gap[route_size_idx + 1 :]))
        results[instance] = drsci_cost

    return results


def build_comparison_json() -> dict[str, dict]:
    xl_text = pdf_text(XL_REPORT)
    drsci_text = pdf_text(DRSCI_REPORT)

    bks = load_bks()
    heuristics = parse_table2_heuristics(xl_text)
    drsci = parse_drsci_table(drsci_text)

    if len(heuristics) != 100 or len(drsci) != 100:
        raise RuntimeError(
            f"Expected 100 instances, got heuristics={len(heuristics)}, DRSCI={len(drsci)}"
        )

    missing = set(bks) - set(heuristics)
    if missing:
        raise RuntimeError(f"BKS instances missing heuristic results: {sorted(missing)[:5]}")

    comparison: dict[str, dict] = {}
    for instance in sorted(bks):
        entry = {
            "BKS": bks[instance],
            "heuristics": dict(heuristics[instance]),
        }
        entry["heuristics"]["DRSCI"] = {"best": drsci[instance], "avg": drsci[instance]}
        comparison[instance] = entry

    return comparison


def main() -> None:
    comparison = build_comparison_json()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(comparison)} instances to {OUTPUT}")


if __name__ == "__main__":
    main()
