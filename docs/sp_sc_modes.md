# SC/SP modes: sync, async, and off

Set covering / set partitioning (SC/SP) plus post-SP standard AILS can run in three
modes. Mode and trigger are **orthogonal**: choose *when* to fire independently of
*how* it runs.

| Setting | Values | Default |
|---------|--------|---------|
| `sp_sc.mode` | `off` \| `sync` \| `async` | `sync` |
| `sp_sc.trigger` | `iteration` \| `wallclock` | `iteration` |
| `sp_sc.overlap_policy` | `skip` \| `queue_latest` | `skip` |
| `sp_sc.interval_minutes` | float | `20.0` |

Configure via a nested `sp_sc:` block in YAML (preferred) or flat keys
(`sp_sc_mode`, `sp_sc_trigger`, …). There is no CLI flag for mode yet — pass a
config file with `--config`.

```bash
drispi-pipeline path/to/instance.vrp --config configs/your_profile.yaml
# or
python -m drispi.pipeline.runner path/to/instance.vrp --config configs/your_profile.yaml
```

Profile YAMLs under `configs/` merge on top of sibling `default.yaml`.

---

## Sync (`mode: sync`) — default

SC/SP (+ standard AILS) runs **inline on the main process** after eviction in the
same iteration. The DRI loop blocks until the MIP and AILS finish.

**When to use:** baseline behaviour, debugging, or when you do not reserve a
dedicated SP core.

### Iteration trigger (default)

Fires when `warmup_iterations` / `sp_interval` / coverage gates say so (same
policy as before).

```yaml
sp_sc:
  mode: sync
  trigger: iteration
  warmup_iterations: 10
  sp_interval: 3
  min_coverage: 5
  sp_time_limit: 600.0
  mip_gap: 0.0005
```

Flat equivalent (also accepted):

```yaml
sp_sc_mode: sync
sp_sc_trigger: iteration
```

### Wall-clock trigger

Fires on a **start-anchored** schedule: after an enqueue/run starts, the next due
time is `now + interval_minutes`. Overlap policy still applies if a previous sync
run would somehow overlap (mainly relevant for async).

```yaml
sp_sc:
  mode: sync
  trigger: wallclock
  interval_minutes: 20.0
  overlap_policy: skip
  warmup_iterations: 10
  min_coverage: 5
  sp_time_limit: 600.0
```

### Cores (optional)

Without a `cores:` block, DRI parallelism uses `n_workers` and Gurobi/AILS use
default thread behaviour (AILS still gets SerialGC + `ActiveProcessorCount=1` +
`-Xmx4g`).

With a `cores:` block, DRI uses `cores.dri` workers and sync Gurobi is limited to
`cores.sp` threads. In sync mode the SP core(s) are **idle during DRI** and only
busy during the blocking SC/SP phase.

```yaml
n_workers: 8   # ignored for worker count when cores: is present

cores:
  total: 8
  dri: 7
  sp: 1
  # cpu_list: [0, 1, 2, 3, 4, 5, 6, 7]   # optional; or pass --cpus 0-7
```

---

## Async (`mode: async`)

A **long-lived SP worker process** owns Gurobi and post-SP standard AILS on the
reserved SP CPU set. The main thread:

1. Pickles a `RoutePool` snapshot to bytes (sync, on the main thread).
2. Enqueues the job when the trigger fires.
3. Continues the DRI loop without waiting.
4. **Drains** at most one ready result at the **start of an iteration** (before
   HAOS select), then adopts vs the **live** incumbent.

Adoption never rounds-trips a stale best: if the MIP returns `SolCount==0`, the
result reason is `mip_no_solution` (no fallback to a previous best).

**When to use:** keep DRI moving while SC/SP runs on a reserved core; A/B
comparisons of “reserve 1 core for async SP” vs “give that core to DRI” (both
arms typically use `cores: {total: 8, dri: 7, sp: 1}` with sync leaving the SP
core idle during DRI).

### Recommended async config

Async is most useful with an explicit `cores:` block so the worker can pin to
`cores.sp` CPUs and AILS gets `ActiveProcessorCount={cores.sp}`.

```yaml
cores:
  total: 8
  dri: 7
  sp: 1
  cpu_list: [0, 1, 2, 3, 4, 5, 6, 7]

sp_sc:
  mode: async
  trigger: wallclock          # or iteration
  interval_minutes: 20.0
  overlap_policy: skip        # or queue_latest
  warmup_iterations: 10
  sp_interval: 3              # used when trigger: iteration
  min_coverage: 5
  sp_time_limit: 600.0
  mip_gap: 0.0005
```

Run example:

```bash
drispi-pipeline data/some.vrp \
  --config configs/async_example.yaml \
  --cpus 0-7
```

`--cpus` supplies the CPU list when `cores.cpu_list` is omitted but `cores:` is
present.

### Overlap policy

Only matters when a new trigger fires while the worker is still busy:

| Policy | Behaviour |
|--------|-----------|
| `skip` | Drop the new trigger; keep the in-flight job. |
| `queue_latest` | Keep only the newest pending snapshot; replace any queued job. |

Wall-clock scheduling is **start-anchored** (`next_due = enqueue_time + interval`),
so overlap can occur when SC/SP (+ AILS) runs longer than `interval_minutes`.

### What to look for in logs / JSONL

- Console: `ASYNC` lines on apply (`adopted=…`, `drain_latency=…`).
- JSONL events: `spsc_apply` (per drain), `spsc_run_totals` (at finalize:
  adopted count, drain latency, `sp_core_busy_fraction`, batch_rounds stats).
- Phase logs still include `batch_rounds` for DRI subcluster waves.

---

## Off (`mode: off`)

No SC/SP invocations. Use this instead of any disable hack. A tripwire raises if
SP code is entered while mode is `off`.

```yaml
sp_sc:
  mode: off
```

---

## Behaviour cheat sheet

```text
trigger fires
    │
    ├─ mode=off     → never
    ├─ mode=sync    → run SC/SP+AILS on main thread; block DRI until done
    └─ mode=async   → pickle pool → enqueue → DRI continues
                           │
                           └─ next iteration start: poll → adopt vs live best
```

| | Sync | Async |
|--|------|-------|
| Who runs MIP / AILS | Main process | SP worker process |
| DRI blocked? | Yes, during SC/SP | No (except brief pickle) |
| Result apply | Same iteration, after SP | Next iteration start only |
| Needs `cores:`? | Optional | Strongly recommended |
| Default in `DRISPIConfig` | yes (`sync` + `iteration`) | — |

---

## Notes

- **AILS JVM flags** always include `-XX:+UseSerialGC`,
  `-XX:ActiveProcessorCount={cores.sp|1}`, `-Xmx4g` (heap chosen from XL
  diagnostics; not a claim of zero handicap vs unpinned G1).
- **Subcluster wall-timeout** is scaled by `ceil(k / dri_workers)` so multi-wave
  HAOS rolls get proportional time.
- Defaults when keys are omitted: see `drispi/pipeline/config.py`.
  `configs/default.yaml` may omit `sp_sc.mode`; missing keys fall back to
  dataclass defaults (`sync` / `iteration`).
