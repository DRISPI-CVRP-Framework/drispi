# BG-AILS ablation (Stage 2)

Campaign artifacts live under `artifacts/bg_ails_ablation/` (gitignored):

- `checkpoints/<instance>/seed<seed>.json`
- `performance/` and `measurement/` per cell

Committed figure inputs: `data/results/bg_ails_ablation/*.json`.

## Binaries

- Performance runs: `ext/ails2/build/AILSII.jar` (campaign pin `bf955a64`)
- Measurement runs: `ext/ails2/build/AILSII-touch.jar` from branch `instrument/touch-counters`

## Commands

Run from the repo root. Wave 1 = seeds 101–103, wave 2 = 104–106. Stop after
wave 1 if wall-clock overruns; that is still a complete 100×3 design. Do not
cut instances. Default concurrency is 5 checkpoint generators and 24 AILS-II
JVMs (`-Xmx4g`).

```text
python scripts/generate_bgails_checkpoints.py --wave 1
python scripts/run_bgails_ablation.py --campaign performance --wave 1
python scripts/run_bgails_ablation.py --campaign measurement --wave 1
python scripts/analyze_bgails_ablation.py
python scripts/generate_bgails_mechanism_plot.py
python scripts/generate_bgails_touchmap_plot.py
python scripts/generate_bgails_concentration_plot.py
```

Smoke (short AILS-II budget; checkpoint generation still uses FILO2):

```text
python scripts/generate_bgails_checkpoints.py --instance XL-n1281-k29 --seed 101 --jobs 1
python scripts/run_bgails_ablation.py --campaign performance --instance XL-n1281-k29 --seed 101 --time-limit 8 --jobs 1
python scripts/run_bgails_ablation.py --campaign measurement --instance XL-n1281-k29 --seed 101 --time-limit 8 --jobs 1
```

Completed cells are skipped unless `--force` is passed. Section 4.4 result
paragraphs are filled after `analyze_bgails_ablation.py` produces
`performance_stats.json`. The placeholder sentence stays until then.
