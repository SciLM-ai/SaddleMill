"""Measurement pass for the dataset-2/3 ("trajectory→saddle") build.

Samples saddles, locates their dimer/DM/initbasin debug trajectories, runs the
exact 6-node RMSD-uniform interpolation + dedup that the real build will use,
writes the assembled groups to a throwaway LMDB to measure bytes/frame, and
times every stage so we can extrapolate total build time and size for
dataset 2 (matched, 22 imgs/saddle) and dataset 3 (all, 18 imgs/saddle).

Run on a compute node:
  salloc -N1 -C cpu -q interactive -t 0:30:00 -A m1883
  python scripts/measure_dataset23.py --root <campaign> --n 800 -j 128
"""
import argparse, csv, glob, multiprocessing as mp, os, re, sys, tempfile, time, zipfile
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

RANK_RE = re.compile(r"_rank_(\d+)\.traj$")
LOG_PREFIX = "MinModeTranslate:"
_TARGETS = set()

# ---- shared helpers (mirrors make_comparison_trajs / consolidate) ----
def dimer_key(atoms):
    n = atoms.info
    while isinstance(n, dict):
        if "src_index" in n and "attempt_id" in n:
            return (int(n["src_index"]), int(n["attempt_id"]))
        n = n.get("orig_info")
    return None

def parse_log(text):
    """Return (sorted [(step,curv)], last_step_energy)."""
    per = {}
    last_e = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(LOG_PREFIX):
            continue
        p = line[len(LOG_PREFIX):].split()
        if len(p) < 6:
            continue
        try:
            step = int(p[0]); energy = float(p[2]); curv = float(p[5])
        except ValueError:
            continue
        if step not in per:
            per[step] = curv
        last_e = energy
    return sorted(per.items()), last_e

def first_neg(curv):
    if not curv or curv[-1][1] >= 0:
        return None
    f = curv[-1][0]
    for s, c in reversed(curv[:-1]):
        if c < 0:
            f = s
        else:
            break
    return f

def zbytes(zp, name):
    with zipfile.ZipFile(zp) as zf:
        return zf.read(name) if name in zf.namelist() else None

def ztraj(zp, name):
    d = zbytes(zp, name)
    if d is None:
        return None
    fd, t = tempfile.mkstemp(suffix=".traj"); os.close(fd)
    try:
        open(t, "wb").write(d)
        return read(t, index=":")
    finally:
        os.remove(t)

# ---- 6-node RMSD-uniform interpolation ----
def arc_resample(frames, n_nodes=6):
    """Return (positions list [n_nodes], arc_length, nframes). Endpoints exact,
    interior linearly interpolated in MIC-unwrapped space; result wrapped."""
    pos = [f.get_positions() for f in frames]
    cell = np.asarray(frames[0].cell.array, float)
    inv = np.linalg.inv(cell)
    # unwrap into a continuous path via MIC steps
    unw = [pos[0].copy()]
    for i in range(1, len(pos)):
        d = pos[i] - pos[i - 1]
        fr = d @ inv; fr -= np.round(fr)
        unw.append(unw[-1] + fr @ cell)
    unw = np.array(unw)
    seg = np.sqrt(((unw[1:] - unw[:-1]) ** 2).sum(2).mean(1)) if len(unw) > 1 else np.array([])
    cum = np.concatenate([[0.0], np.cumsum(seg)]) if len(seg) else np.array([0.0])
    total = float(cum[-1])
    out = []
    if total < 1e-9 or len(unw) == 1:
        out = [unw[0].copy() for _ in range(n_nodes)]
    else:
        for tg in np.linspace(0, total, n_nodes):
            j = min(max(int(np.searchsorted(cum, tg, "right") - 1), 0), len(cum) - 2)
            sl = cum[j + 1] - cum[j]
            w = 0.0 if sl < 1e-12 else (tg - cum[j]) / sl
            out.append((1 - w) * unw[j] + w * unw[j + 1])
    # wrap into cell
    wrapped = []
    for p in out:
        fr = p @ inv; fr -= np.floor(fr)
        wrapped.append(fr @ cell)
    return wrapped, total, len(frames)

def _ase(positions, numbers, cell, e=None, f=None):
    a = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    if e is not None or f is not None:
        a.calc = SinglePointCalculator(a, energy=e, forces=f)
    return a

# ---- locate sample saddles ----
def _scan_worker(path):
    m = RANK_RE.search(path); rank = int(m.group(1)) if m else -1
    hits = {}
    try:
        for at in read(path, index=":"):
            k = dimer_key(at)
            if k in _TARGETS and k not in hits:
                hits[k] = (int(at.info.get("src_index")), rank)
    except Exception:
        pass
    return hits

def scan(files, workers, label):
    found = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for i, h in enumerate(pool.imap_unordered(_scan_worker, files), 1):
            found.update(h)
            if i % 25 == 0 or i == len(files):
                print(f"  [{label}] {i}/{len(files)} shards, {len(found)} located", flush=True)
    return found

# ---- build one group (returns list of row-dicts + per-traj stats) ----
def build_group(args):
    src, att, group, dimer_loc, dm_loc, min_loc, root = args
    drank, atom = dimer_loc
    djob, dmr = dm_loc
    dz = f"{root}/Dimer_debug_zips/structure_rank_{drank}_data.zip"
    dmz = f"{root}/doubleopt/DoubleMinimization_debug_zips/structure_rank_{dmr}_data.zip"
    t0 = time.perf_counter()
    dm_m1 = ztraj(dmz, f"optimization_{djob}_-1.traj")
    dm_p1 = ztraj(dmz, f"optimization_{djob}_1.traj")
    dimer = ztraj(dz, f"dimer_{src}_{att}_{atom}.traj")
    logb = zbytes(dz, f"dimer_opt_{src}_{att}_{atom}.log")
    if None in (dm_m1, dm_p1, dimer, logb):
        return None
    curv, sad_e = parse_log(logb.decode("utf-8", "replace"))
    fneg = first_neg(curv)
    if fneg is None or fneg >= len(dimer):
        return None
    numbers = dimer[0].get_atomic_numbers()
    cell = dimer[0].cell.array
    rows, stats = [], []

    # core: saddle (energy only), reactant_min, product_min (E+F)
    sad = dimer[-1]
    rmin, pmin = dm_m1[-1], dm_p1[-1]
    rows.append(("saddle", 1, _ase(sad.get_positions(), numbers, cell, e=sad_e)))
    def ef(a):
        e = a.calc.results.get("energy") if a.calc else None
        f = a.calc.results.get("forces") if a.calc else None
        return e, f
    re_, rf_ = ef(rmin); pe_, pf_ = ef(pmin)
    rows.append(("reactant_min", 0, _ase(rmin.get_positions(), numbers, cell, re_, rf_)))
    rows.append(("product_min", 0, _ase(pmin.get_positions(), numbers, cell, pe_, pf_)))

    # DM -1: nodes 0..4 (start + 4 interior); node5=min already core
    p, L, nf = arc_resample(dm_m1); stats.append(("dm-1", nf, L))
    for q in p[:5]:
        rows.append(("reactant_traj", 0, _ase(q, numbers, cell)))
    # DM +1
    p, L, nf = arc_resample(dm_p1); stats.append(("dm+1", nf, L))
    for q in p[:5]:
        rows.append(("product_traj", 0, _ase(q, numbers, cell)))
    # dimer: frames[fneg:] = basin_entry..saddle; nodes 0..4 (drop saddle node5)
    dim_path = dimer[fneg:]
    p, L, nf = arc_resample(dim_path); stats.append(("dimer", nf, L))
    for q in p[:5]:
        rows.append(("dimer", 0, _ase(q, numbers, cell)))

    if group == "matched":  # dataset 2: + initbasin inner 4
        ijob, imr = min_loc
        iz = f"{root}/initbasinopt/Minimization_debug_zips/structure_rank_{imr}_data.zip"
        init = ztraj(iz, f"optimization_{ijob}.traj")
        if init is None:
            return None
        p, L, nf = arc_resample(init); stats.append(("initbasin", nf, L))
        for q in p[1:5]:  # drop both endpoints
            rows.append(("initbasin", 0, _ase(q, numbers, cell)))

    dt = time.perf_counter() - t0
    return {"src": src, "att": att, "group": group, "rows": rows,
            "nrows": len(rows), "dt": dt, "stats": stats,
            "saddle_e_ok": sad_e is not None,
            "rp_ef_ok": (rf_ is not None and pf_ is not None)}

def main():
    global _TARGETS
    ncpu = len(os.sched_getaffinity(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=300, help="sample saddles (half matched)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--dm-shards", type=int, default=50,
                    help="cap on DM output shards scanned (sample is taken from these)")
    ap.add_argument("--init-shards", type=int, default=250,
                    help="cap on initbasinopt output shards scanned")
    ap.add_argument("-j", "--workers", type=int, default=min(64, ncpu))
    a = ap.parse_args()
    root = a.root
    comp = f"{root}/basin_entry_vs_doublemin/comparison_rmsd.csv"

    import random
    rng = random.Random(a.seed)
    label = {}
    print(f"reading {comp} ...", flush=True)
    with open(comp) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row[11] != "1":
                continue
            try:
                mmd = float(row[9])
            except ValueError:
                continue
            label[(int(row[0]), int(row[1]))] = "matched" if mmd < a.threshold else "unmatched"
    _TARGETS = set(label)
    print(f"comparison saddles: {len(label):,}", flush=True)

    # Capped scan: only locate from a subset of shards (we just need a
    # representative sample, not every saddle). Full-scan time is known from
    # the comparison run (~8 min at 64 workers) and noted in the report.
    t = time.perf_counter()
    dm_files = sorted(glob.glob(f"{root}/doubleopt/DoubleMinimization_trajes/collected_opt_rank_*.traj"))[:a.dm_shards]
    min_files = sorted(glob.glob(f"{root}/initbasinopt/Minimization_trajes/collected_opt_rank_*.traj"))[:a.init_shards]
    dm_loc = scan(dm_files, a.workers, "dm")
    min_loc = scan(min_files, a.workers, "init")
    t_scan = time.perf_counter() - t
    print(f"capped scan ({len(dm_files)} DM + {len(min_files)} init shards) in "
          f"{t_scan:.0f}s: dm={len(dm_loc):,} init={len(min_loc):,}", flush=True)

    # sample from saddles actually found in the scanned shards
    nm = a.n // 2
    cand_m = [k for k in dm_loc if label.get(k) == "matched" and k in min_loc]
    cand_u = [k for k in dm_loc if label.get(k) == "unmatched"]
    sel = {k: "matched" for k in rng.sample(cand_m, min(nm, len(cand_m)))}
    sel.update({k: "unmatched" for k in rng.sample(cand_u, min(nm, len(cand_u)))})
    print(f"sample: {sum(v=='matched' for v in sel.values())} matched + "
          f"{sum(v=='unmatched' for v in sel.values())} unmatched "
          f"(from {len(cand_m)} matched / {len(cand_u)} unmatched candidates)", flush=True)

    # dimer rank+atom for the sample only
    sel_keys = set(sel)
    dimer_loc = {}
    for c in glob.glob(f"{root}/Dimer_status_csvs/status_rank_*.csv"):
        with open(c) as f:
            for line in f:
                p = line.split(",")
                if len(p) < 5:
                    continue
                try:
                    s, mpi, at_, atom = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                except ValueError:
                    continue
                if (s, at_) in sel_keys and (s, at_) not in dimer_loc:
                    dimer_loc[(s, at_)] = (mpi, atom)
        if len(dimer_loc) >= len(sel_keys):
            break
    print(f"dimer-located {len(dimer_loc)}/{len(sel)}", flush=True)

    # build groups (parallel)
    tasks = []
    for (s, at_), g in sel.items():
        if (s, at_) not in dimer_loc or (s, at_) not in dm_loc:
            continue
        if g == "matched" and (s, at_) not in min_loc:
            continue
        tasks.append((s, at_, g, dimer_loc[(s, at_)], dm_loc[(s, at_)],
                      min_loc.get((s, at_)), root))
    print(f"\nbuilding {len(tasks)} groups (parallel) ...", flush=True)
    t = time.perf_counter()
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = [x for x in pool.map(build_group, tasks) if x is not None]
    t_build = time.perf_counter() - t

    # write sample to throwaway LMDB to measure bytes/frame
    import fairchem.core.datasets  # noqa: register aselmdb backend
    from ase.db import connect
    lmdb_path = f"{root}/_measure_tmp.aselmdb"
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
    nframes = 0
    gid = 0
    with connect(lmdb_path, type="aselmdb") as db:
        for grp in res:
            gid += 1
            for ri, (role, is_sad, atoms) in enumerate(grp["rows"]):
                atoms.info["task_name"] = "omat"
                db.write(atoms, task_name="omat", group_id=gid, role=role,
                         is_saddle=is_sad,
                         data={"info": {"task_name": "omat", "charge": 0, "spin": 0,
                                        "group_id": gid, "role": role,
                                        "is_saddle": is_sad, "src_index": grp["src"],
                                        "attempt_id": grp["att"]}})
                nframes += 1
    lmdb_bytes = os.path.getsize(lmdb_path)
    os.remove(lmdb_path)

    # ---- report ----
    mt = [g for g in res if g["group"] == "matched"]
    um = [g for g in res if g["group"] == "unmatched"]
    per_dt = np.array([g["dt"] for g in res])
    bpf = lmdb_bytes / max(nframes, 1)
    all_stats = [(name, nf, L) for g in res for (name, nf, L) in g["stats"]]
    nfs = np.array([s[1] for s in all_stats]); Ls = np.array([s[2] for s in all_stats])

    N_D2, N_D3 = 992_422, 1_877_277
    def hr(b):  # human GB
        return f"{b/1e9:.1f} GB"
    print("\n================ MEASUREMENT ================")
    print(f"groups built: {len(res)} ({len(mt)} matched/22, {len(um)} unmatched/18)")
    print(f"row counts: matched={sorted(set(g['nrows'] for g in mt))} "
          f"unmatched={sorted(set(g['nrows'] for g in um))}  (expect 22 / 18)")
    print(f"saddle energy present: {sum(g['saddle_e_ok'] for g in res)}/{len(res)} ; "
          f"R/P forces present: {sum(g['rp_ef_ok'] for g in res)}/{len(res)}")
    print(f"\nLMDB size: {nframes:,} frames -> {hr(lmdb_bytes)}  ({bpf:.0f} bytes/frame)")
    print(f"  dataset 2 (~{N_D2:,}x22={N_D2*22:,} frames): ~{hr(N_D2*22*bpf)}")
    print(f"  dataset 3 (~{N_D3:,}x18={N_D3*18:,} frames): ~{hr(N_D3*18*bpf)}")
    print(f"\nper-saddle build time: median={np.median(per_dt)*1000:.0f}ms "
          f"mean={per_dt.mean()*1000:.0f}ms  (single-thread, repeated zip-open; "
          f"real build batches per-zip -> faster)")
    rate = len(res) / t_build * a.workers / a.workers  # observed parallel rate
    obs_rate = len(res) / t_build
    print(f"observed build throughput: {obs_rate:.1f} saddles/s on {a.workers} workers "
          f"({t_build:.0f}s for {len(res)})")
    print(f"  -> dataset 2 ({N_D2:,}): ~{N_D2/obs_rate/3600:.1f} h ; "
          f"dataset 3 ({N_D3:,}): ~{N_D3/obs_rate/3600:.1f} h  (per-saddle extraction, "
          f"repeated zip-open; real build batches per-zip -> faster)")
    print(f"capped location scan: {len(dm_files)} DM + {len(min_files)} init shards in "
          f"{t_scan:.0f}s (full scan of 384+2048 ~ 8 min one-time, from comparison run)")

    # IO-bound build estimate from total debug-zip volume (more reliable than
    # the repeated-open per-saddle timing above)
    def dirsize(d):
        tot = 0
        for r_, _, fs in os.walk(d):
            for fn in fs:
                try:
                    tot += os.path.getsize(os.path.join(r_, fn))
                except OSError:
                    pass
        return tot
    vols = {n: dirsize(f"{root}/{p}") for n, p in [
        ("Dimer_debug_zips", "Dimer_debug_zips"),
        ("DM_debug_zips", "doubleopt/DoubleMinimization_debug_zips"),
        ("init_debug_zips", "initbasinopt/Minimization_debug_zips")]}
    print(f"\ndebug-zip volumes to read for the build:")
    for n, v in vols.items():
        print(f"  {n}: {v/1e9:.0f} GB")
    print(f"  dataset 3 reads Dimer+DM ({(vols['Dimer_debug_zips']+vols['DM_debug_zips'])/1e9:.0f} GB); "
          f"dataset 2 also reads init (+{vols['init_debug_zips']/1e9:.0f} GB)")
    print(f"\ntrajectory lengths (n={len(all_stats)} trajs):")
    print(f"  frames/traj: median={np.median(nfs):.0f} p10={np.percentile(nfs,10):.0f} "
          f"p90={np.percentile(nfs,90):.0f} ; frac <6 frames = {100*np.mean(nfs<6):.1f}%")
    print(f"  RMSD arc length/traj (A): median={np.median(Ls):.3f} "
          f"p10={np.percentile(Ls,10):.3f} p90={np.percentile(Ls,90):.3f} ; "
          f"frac <0.5A = {100*np.mean(Ls<0.5):.1f}%")

if __name__ == "__main__":
    main()
