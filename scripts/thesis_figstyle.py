"""Shared matplotlib style for BG-AILS ablation figures.

Text width is the KOMA ``scrreprt`` 12pt A4 ``\\textwidth`` from a local
``\\the\\textwidth`` measurement: 426.79135 pt. Do not pass a width to
``\\includegraphics``.
"""

from __future__ import annotations

import matplotlib as mpl

TEXTWIDTH_PT = 426.79135
OKABE_ITO = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def apply() -> None:
    mpl.use("pdf")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "LMRoman10", "Times"],
            "text.usetex": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.titlesize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_size(width_frac: float = 1.0, height_pt: float | None = None) -> tuple[float, float]:
    w = TEXTWIDTH_PT * width_frac / 72.27
    h = (height_pt / 72.27) if height_pt is not None else w * 0.75
    return w, h
