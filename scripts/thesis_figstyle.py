"""Shared matplotlib style for thesis figures.

Text width is the KOMA ``scrreprt`` 12pt A4 ``\\textwidth`` under
``geometry`` left=50mm, right=20mm on A4: 140 mm = 398.33858 pt.
Do not pass a width to ``\\includegraphics``.

Body text uses Times via ``\\usepackage{times}`` (Nimbus Roman on this
system). Figures register that OpenType family and use 10 pt for labels
and 9 pt for ticks, matching caption scale under a 12 pt document class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm

TEXTWIDTH_PT = 398.33858
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
# Same hue as the Okabe–Ito blue (#0072B2), mixed toward white. Shared by the
# immediate-reward bars (Figure 4.7) and the short-route line (Figure 4.9b),
# so that line stays apart from the black series and from the 4.9a bars.
BLUE_LIGHT = "#5BA3D9"
# One color per HAOS wheel, in wheel order: k, lambda_q, paradigm,
# vertex method, route method, subsolver. k is red so it does not repeat
# the vermillion used for the subsolver. lambda_q uses the Okabe–Ito yellow.
# The vertex method uses the Okabe–Ito reddish purple.
HAOS_WHEEL_COLORS = (
    "#E41A1C",
    "#F0E442",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#D55E00",
)

_NIMBUS_DIR = Path("/usr/share/fonts/opentype/urw-base35")
_NIMBUS_FILES = (
    "NimbusRoman-Regular.otf",
    "NimbusRoman-Bold.otf",
    "NimbusRoman-Italic.otf",
    "NimbusRoman-BoldItalic.otf",
)
_SERIF = [
    "Nimbus Roman",
    "Times New Roman",
    "Times",
    "TeX Gyre Termes",
    "Liberation Serif",
    "DejaVu Serif",
]


def _register_nimbus() -> None:
    for name in _NIMBUS_FILES:
        path = _NIMBUS_DIR / name
        if path.is_file():
            try:
                fm.fontManager.addfont(str(path))
            except (RuntimeError, OSError, ValueError):
                pass


def apply() -> None:
    mpl.use("pdf")
    _register_nimbus()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _SERIF,
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def fig_size(width_frac: float = 1.0, height_pt: float | None = None) -> tuple[float, float]:
    w = TEXTWIDTH_PT * width_frac / 72.27
    h = (height_pt / 72.27) if height_pt is not None else w * 0.75
    return w, h
