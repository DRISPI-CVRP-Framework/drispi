#!/usr/bin/env python3
"""Single source of truth for every number quoted in Section 4.2.

Inputs
------
  data/results/finalBenchmarkResults_lagrange.csv
  data/results/instance_chars.csv

r is the Table A.1 target average route length (r_tab), not (n-1)/K.
"gap" without qualification is the mean-of-3 percentage gap to the current BKS.
qbar is the mean customer demand, sum of customer demands divided by the
number of customers, read from the instance file. Q/r approximates it
(the generation scheme sets capacity from the target route length) but is
not the regressor: on unitary demand Q/r falls to about 0.9 while qbar is 1.

Run from the repository root:
    python scripts/section_4_2_numbers.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/results/finalBenchmarkResults_lagrange.csv"
CHARS = ROOT / "data/results/instance_chars.csv"
SEEDS = ["11", "22", "33"]


def hodges_lehmann(d):
    d = np.asarray(d, float)
    walsh = (d[:, None] + d[None, :])[np.triu_indices(len(d), 0)] / 2.0
    return float(np.median(walsh))


def holm(praw: dict) -> dict:
    keys = list(praw)
    p = np.array([praw[k] for k in keys])
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(running, 1.0)
    return dict(zip(keys, adj))


def mean_customer_demand(instance: str) -> float:
    """Mean demand over customers. Node 1 is the depot and is excluded."""
    path = ROOT / "data/instances/xl" / f"{instance}.vrp"
    demands: list[int] = []
    in_demand = False
    for line in path.read_text().splitlines():
        if line.startswith("DEMAND_SECTION"):
            in_demand = True
            continue
        if not in_demand:
            continue
        if line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
            break
        parts = line.split()
        if len(parts) >= 2:
            demands.append(int(parts[1]))
    if len(demands) < 2:
        raise ValueError(f"no customer demands in {path}")
    return float(np.mean(demands[1:]))


def load():
    df = pd.read_csv(CSV).merge(pd.read_csv(CHARS), on="instance", validate="1:1")
    df["r"] = df.r_tab
    df["g"] = df.gap_bks_mean3
    df["a"] = df.gap_bks_ails2_mean
    df["paired_diff"] = df.g - df.a
    df["qbar"] = df.instance.map(mean_customer_demand)
    df["long"] = df.r >= 50
    df["cust_grp"] = df.cust.str.replace(r"\(\d+\)", "", regex=True).str.strip()
    for src, dst in [("n", "zn"), ("r", "zr")]:
        df[dst] = (df[src] - df[src].mean()) / df[src].std(ddof=1)
    lq = np.log(df.qbar)
    df["zq"] = (lq - lq.mean()) / lq.std(ddof=1)
    for m in (30, 60, 90, 120):
        df[f"mean_{m}"] = df[[f"mark{m}_seed{s}" for s in SEEDS]].mean(axis=1)
        df[f"gap_{m}"] = 100 * (df[f"mean_{m}"] - df.bks_current) / df.bks_current
    df["mean_final"] = df[[f"cost_seed{s}" for s in SEEDS]].mean(axis=1)
    df["gap_final"] = 100 * (df.mean_final - df.bks_current) / df.bks_current
    df["iters_mean"] = df[[f"iters_seed{s}" for s in SEEDS]].mean(axis=1)
    return df


def sec_421(df):
    print("\n" + "=" * 76 + "\n 4.2.1  COMPARISON AGAINST THE MONOLITHIC SOLVERS\n" + "=" * 76)
    g = df.g
    print(
        f"mean-of-3 gap to current BKS  {g.mean():.4f} %   median {g.median():.4f}   "
        f"IQR [{g.quantile(.25):.4f}, {g.quantile(.75):.4f}]   p90 {g.quantile(.9):.4f}   max {g.max():.4f}"
    )
    print(f"best-of-3 gap to current BKS  {df.gap_bks_best3.mean():.4f} %")
    print(
        f"mean-of-3 gap to initial BKS  {df.gap_init_mean3.mean():.4f} %   "
        f"best-of-3 {df.gap_init_best3.mean():.4f} %"
    )
    print(f"AILS-II mean-of-60 to current BKS  {df.a.mean():.4f} %")
    print(f"n < 3400: mean-of-3 {df[df.n < 3400].g.mean():.4f}   n > 3400: {df[df.n > 3400].g.mean():.4f}")

    print("\n-- best-of-S ladder (why best-of-N is not the primary comparison) --")
    cost = df[[f"cost_seed{s}" for s in SEEDS]].values
    pairs = [np.minimum(cost[:, i], cost[:, j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    best2 = np.mean([(100 * (p - df.bks_current) / df.bks_current).mean() for p in pairs])
    print(f"  expected best-of-1 = mean-of-3 = {g.mean():.4f} %")
    print(f"  best-of-2 (mean over the three retained pairs) = {best2:.4f} %")
    print(f"  best-of-3 = {df.gap_bks_best3.mean():.4f} %")
    print(f"  marginal gain  1->2 {g.mean() - best2:+.4f} pp   2->3 {best2 - df.gap_bks_best3.mean():+.4f} pp")

    print("\n-- per-seed breakdown --")
    for s in SEEDS:
        print(
            f"  seed {s}: mean gap {df[f'gap_bks_seed{s}'].mean():.4f} %   "
            f"median {df[f'gap_bks_seed{s}'].median():.4f}   "
            f"iterations {df[f'iters_seed{s}'].sum():5d}   "
            f"mean wall {df[f'wall_s_seed{s}'].mean():.1f} s"
        )
    fr = stats.friedmanchisquare(*[df[f"gap_bks_seed{s}"] for s in SEEDS])
    print(f"  Friedman across the three seeds: chi2 = {fr.statistic:.3f}, df = 2, p = {fr.pvalue:.4f}")
    for a, b in [("11", "22"), ("11", "33"), ("22", "33")]:
        d = df[f"gap_bks_seed{a}"] - df[f"gap_bks_seed{b}"]
        print(
            f"    seed {a} - seed {b}: mean {d.mean():+.4f} pp, "
            f"Wilcoxon p = {stats.wilcoxon(d).pvalue:.3f}"
        )
    rms = np.sqrt((df.seed_sd_pp ** 2).mean())
    print(
        f"  per-instance seed SD: mean {df.seed_sd_pp.mean():.4f} pp, median {df.seed_sd_pp.median():.4f}, "
        f"RMS {rms:.4f}, max {df.seed_sd_pp.max():.4f} ({df.loc[df.seed_sd_pp.idxmax(), 'instance']})"
    )
    print(f"  realized 95% half-width of the grand mean: {1.96 * rms / np.sqrt(300):.5f} pp")
    print(
        f"  corr(seed SD, r) = {stats.pearsonr(df.r, df.seed_sd_pp)[0]:+.3f}   "
        f"corr(seed SD, n) = {stats.pearsonr(df.n, df.seed_sd_pp)[0]:+.3f}"
    )

    print("\n-- paired comparison against AILS-II (recomputed from the campaign CSV) --")
    d = df["paired_diff"]
    print(f"  mean paired difference {d.mean():+.4f} pp   Hodges-Lehmann {hodges_lehmann(d):+.4f} pp")
    print(f"  DRISPI ahead on {int((d < 0).sum())} of 100")
    for meth, kw in [
        ("approx, no cc", dict(method="approx", correction=False)),
        ("approx, cc", dict(method="approx", correction=True)),
    ]:
        print(f"  Wilcoxon two-sided ({meth}): p = {stats.wilcoxon(d, **kw).pvalue:.4g}")


def sec_422(df):
    print("\n" + "=" * 76 + "\n 4.2.2  PERFORMANCE BY INSTANCE CHARACTERISTICS\n" + "=" * 76)
    print(
        f"1 SD of n = {df.n.std(ddof=1):.1f} customers,  1 SD of r = {df.r.std(ddof=1):.1f} stops,  "
        f"1 SD of log qbar = {np.log(df.qbar).std(ddof=1):.3f}"
    )
    print(
        f"Pearson corr(n, r) = {stats.pearsonr(df.n, df.r)[0]:.3f};  "
        f"corr(log qbar, r) = {stats.pearsonr(np.log(df.qbar), df.r)[0]:.3f}"
    )

    print("\n-- raw-unit regression on n (reproduces the draft) --")
    for lab, sub in [("all 100", df), ("r < 50 (86)", df[~df.long])]:
        m = sm.OLS(sub.g, sm.add_constant(sub.n.astype(float))).fit()
        lo, hi = m.conf_int().loc["n"]
        print(
            f"  {lab:12s}: {m.params['n']:.3e} pp per customer  t = {m.tvalues['n']:.2f}  "
            f"p = {m.pvalues['n']:.5f}  CI [{lo:.3e}, {hi:.3e}]  R2 = {m.rsquared:.3f}"
        )

    print("\n-- standardized, one predictor at a time --")
    for v, lab in [("zn", "n"), ("zr", "r"), ("zq", "qbar")]:
        m = smf.ols(f"g ~ {v}", data=df).fit()
        print(
            f"  gap ~ {lab:5s}: {m.params[v]:+.4f} pp per SD  (t = {m.tvalues[v]:.2f}, "
            f"p = {m.pvalues[v]:.3g}, R2 = {m.rsquared:.3f})"
        )

    print("\n-- joint model, the three continuous drivers --")
    m = smf.ols("g ~ zn + zr + zq", data=df).fit()
    for k in ["zn", "zr", "zq"]:
        lo, hi = m.conf_int().loc[k]
        print(
            f"  {k}: {m.params[k]:+.4f} pp per SD  t = {m.tvalues[k]:.2f}  p = {m.pvalues[k]:.3g}  "
            f"CI [{lo:+.4f}, {hi:+.4f}]"
        )
    print(f"  R2 = {m.rsquared:.3f}")
    mj = smf.ols("g ~ zn + zr", data=df).fit()
    V = mj.cov_params()
    c = mj.params["zr"] - mj.params["zn"]
    se = np.sqrt(V.loc["zr", "zr"] + V.loc["zn", "zn"] - 2 * V.loc["zr", "zn"])
    t = c / se
    h = stats.t.ppf(0.975, mj.df_resid) * se
    print(
        f"  two-predictor contrast r - n: {c:+.4f} pp per SD, t = {t:.2f}, "
        f"p = {2 * (1 - stats.t.cdf(abs(t), mj.df_resid)):.4f}, CI [{c - h:+.4f}, {c + h:+.4f}]"
    )

    print("\n-- run-to-run spread responds to r only --")
    for v, lab in [("zn", "n"), ("zr", "r"), ("zq", "qbar")]:
        m2 = smf.ols(f"seed_sd_pp ~ {v}", data=df).fit()
        print(
            f"  seed SD ~ {lab:5s}: {m2.params[v]:+.4f} pp per SD "
            f"(p = {m2.pvalues[v]:.3g}, R2 = {m2.rsquared:.3f})"
        )
    ms = smf.ols("seed_sd_pp ~ zn + zr + zq", data=df).fit()
    print(
        "  joint: "
        + "  ".join(f"{k} = {ms.params[k]:+.4f} (p = {ms.pvalues[k]:.2g})" for k in ["zn", "zr", "zq"])
    )

    print("\n-- the long-route band --")
    lo_, hi_ = df[~df.long], df[df.long]
    print(f"  r >= 50: {len(hi_)} instances, mean gap {hi_.g.mean():.4f} % vs {lo_.g.mean():.4f} % for r < 50")
    print(f"  mean seed SD {hi_.seed_sd_pp.mean():.4f} pp vs {lo_.seed_sd_pp.mean():.4f} pp")
    print(f"  Mann-Whitney (long > short) p = {stats.mannwhitneyu(hi_.g, lo_.g, alternative='greater').pvalue:.4g}")

    print("\n-- categorical attributes --")
    raw = {}
    base = smf.ols("g ~ zn + zr", data=df).fit()
    base3 = smf.ols("g ~ zn + zr + zq", data=df).fit()
    for col, lab in [
        ("depot", "depot position"),
        ("cust_grp", "customer distribution"),
        ("demand", "demand distribution"),
    ]:
        samples = [v.to_numpy() for _, v in df.groupby(col).g]
        kw = stats.kruskal(*samples)
        raw[col] = kw.pvalue
        a2 = sm.stats.anova_lm(base, smf.ols(f"g ~ zn + zr + C({col})", data=df).fit())
        a3 = sm.stats.anova_lm(base3, smf.ols(f"g ~ zn + zr + zq + C({col})", data=df).fit())
        print(
            f"  {lab:22s} Kruskal-Wallis H = {kw.statistic:6.3f}, p = {kw.pvalue:.4f} | "
            f"partial F after n,r = {a2['F'].iloc[1]:.3f} (p = {a2['Pr(>F)'].iloc[1]:.4f}) | "
            f"after n,r,qbar = {a3['F'].iloc[1]:.3f} (p = {a3['Pr(>F)'].iloc[1]:.4f})"
        )
    for k, v in holm(raw).items():
        print(f"    Holm over the three: {k:9s} adjusted p = {v:.4f}")
    print("\n  gap and mean demand by demand class:")
    print(
        df.groupby("demand")
        .agg(N=("g", "size"), gap=("g", "mean"), qbar=("qbar", "mean"), r=("r", "mean"), n=("n", "mean"))
        .round(3)
        .to_string()
    )
    mid = df[~df.demand.isin(["U", "Q"])]
    print(
        f"  middle five demand classes alone (N = {len(mid)}): "
        f"Kruskal-Wallis p = {stats.kruskal(*[v.to_numpy() for _, v in mid.groupby('demand').g]).pvalue:.4f}"
    )

    print("\n-- instance hardness vs. what decomposition costs --")
    for target, lab in [("a", "AILS-II mean-of-60 gap"), ("paired_diff", "paired difference DRISPI - AILS-II")]:
        m3 = smf.ols(f"{target} ~ zn + zr + zq", data=df).fit()
        print(
            f"  {lab:34s} R2 = {m3.rsquared:.3f}  "
            + "  ".join(f"{k} = {m3.params[k]:+.4f} (p = {m3.pvalues[k]:.2g})" for k in ["zn", "zr", "zq"])
        )
    print(
        f"  full observed range in SD units: n {(df.n.max() - df.n.min()) / df.n.std(ddof=1):.2f}, "
        f"r {(df.r.max() - df.r.min()) / df.r.std(ddof=1):.2f}"
    )


def sec_424(df):
    print("\n" + "=" * 76 + "\n 4.2.4  CONVERGENCE BEHAVIOUR\n" + "=" * 76)
    marks = [df[f"gap_{m}"].mean() for m in (30, 60, 90, 120)] + [df.gap_final.mean()]
    print(
        "  mean-of-3 gap at 30 / 60 / 90 / 120 min and at the final cost: "
        + "  ".join(f"{v:.4f}" for v in marks)
    )
    quarters = [marks[i] - marks[i + 1] for i in range(4)]
    print("  gain per quarter (pp): " + "  ".join(f"{v:+.4f}" for v in quarters))
    ratios = [quarters[1] / quarters[0], quarters[2] / quarters[1]]
    rbar = float(np.mean(ratios))
    tail = quarters[2] * rbar / (1 - rbar)
    print(
        f"  quarter-on-quarter ratios {ratios[0]:.3f}, {ratios[1]:.3f} (mean {rbar:.3f}); "
        f"geometric tail beyond 120 min {tail:.4f} pp -> {marks[4] - tail:.4f} % (illustrative only)"
    )

    per_seed = [int((df[f"cost_seed{s}"] < df[f"mark90_seed{s}"]).sum()) for s in SEEDS]
    print(f"\n  runs improving after the 90-minute mark, by seed: {per_seed}  (total {sum(per_seed)} of 300)")
    print(
        "  runs improving after the 60-minute mark: "
        f"{sum(int((df[f'cost_seed{s}'] < df[f'mark60_seed{s}']).sum()) for s in SEEDS)} of 300"
    )
    print(
        "  runs logging an update past 7,200 s: "
        f"{sum(int((df[f'cost_seed{s}'] < df[f'mark120_seed{s}']).sum()) for s in SEEDS)} of 300"
    )
    both = np.all([(df[f"cost_seed{s}"] < df[f"mark90_seed{s}"]).to_numpy() for s in SEEDS], axis=0)
    none = ~np.any([(df[f"cost_seed{s}"] < df[f"mark90_seed{s}"]).to_numpy() for s in SEEDS], axis=0)
    print(
        f"  instances where all three seeds improved after 90 min: {int(both.sum())}; "
        f"where none did: {int(none.sum())}"
    )

    df["q4_pp"] = 100 * (df.mean_90 - df.mean_final) / df.bks_current
    print(
        f"\n  final-quarter gain, mean-of-3: mean {df.q4_pp.mean():.4f} pp, median {df.q4_pp.median():.4f}, "
        f"max {df.q4_pp.max():.4f} ({df.loc[df.q4_pp.idxmax(), 'instance']})"
    )
    rr, pr = stats.spearmanr(df.r, df.q4_pp)
    rn, pn = stats.spearmanr(df.n, df.q4_pp)
    print(f"  Spearman with r: {rr:+.3f} (p = {pr:.3g});  with n: {rn:+.3f} (p = {pn:.3f})")
    print(f"  long-route instances {df[df.long].q4_pp.mean():.4f} pp vs {df[~df.long].q4_pp.mean():.4f} pp for r < 50")
    print(f"  n > 3400 {df[df.n > 3400].q4_pp.mean():.4f} pp vs n < 3400 {df[df.n < 3400].q4_pp.mean():.4f} pp")

    print("\n  iterations completed")
    print(f"  Spearman(iterations, n) = {stats.spearmanr(df.n, df.iters_mean)[0]:+.3f}")
    print(
        f"  ten smallest instances {df.nsmallest(10, 'n').iters_mean.mean():.1f} iterations; "
        f"ten largest {df.nlargest(10, 'n').iters_mean.mean():.1f}"
    )
    print(f"  n < 3400 {df[df.n < 3400].iters_mean.mean():.1f}; n > 3400 {df[df.n > 3400].iters_mean.mean():.1f}")

    print("\n  Figure data, anytime trajectory (x = minutes, y = mean-of-3 gap %):")
    for lab, sub in [("all 100", df), ("r < 50", df[~df.long]), ("r >= 50", df[df.long])]:
        vals = [sub[f"gap_{m}"].mean() for m in (30, 60, 90, 120)] + [sub.gap_final.mean()]
        print(f"    {lab:8s} " + " ".join(f"({t},{v:.4f})" for t, v in zip([30, 60, 90, 120, 125], vals)))


def figure_data(df):
    print("\n" + "=" * 76 + "\n FIGURE DATA\n" + "=" * 76)
    print("-- Figure 4.2, third panel: standardized log qbar vs mean-of-3 gap --")
    fit = smf.ols("g ~ zq", data=df).fit()
    print(f"   simple fit {fit.params['zq']:+.4f} pp per SD, R2 = {fit.rsquared:.3f}")
    print("-- Figure 4.3 annotations --")
    for v, lab in [("zn", "n"), ("zr", "r")]:
        fit = smf.ols(f"seed_sd_pp ~ {v}", data=df).fit()
        print(f"   seed SD ~ {lab}: {fit.params[v]:+.4f} pp per SD, R2 = {fit.rsquared:.3f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = load()
    sec_421(d)
    sec_422(d)
    sec_424(d)
    figure_data(d)
