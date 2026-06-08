"""Build per-attempt "story" trajectories that stitch together, for a chosen
set of Dimer (src_index, attempt_id) cases, the four relaxation paths around
one saddle so they can be visualized together:

    section 1: DoubleMinimization relaxation, saddle -> reactant   (side -1)
    section 2: DoubleMinimization relaxation, saddle -> product    (side +1)
    section 3: the Dimer trajectory REVERSED, from the converged saddle back to
               the basin-entry frame (first step of the last negative-curvature
               stretch -- same frame consolidate_dimer_basin_entry.py picked)
    section 4: the initbasinopt Minimization relaxation, basin-entry -> minimum
               (its frame 0 is the basin-entry structure, so it intentionally
               REPEATS section 3's last frame -- a built-in check that
               initbasinopt started from the right structure)

Each output frame's info carries `section` (1-4), `section_name`, `src_index`,
`attempt_id`, `frame_in_section`. Energies/forces from the source optimizer
trajectories are preserved.

Alongside each stitched traj, the original Dimer optimizer log
`dimer_opt_{src}_{att}_{atom}.log` is copied **verbatim** (name unchanged; it
encodes src_index, attempt_id, atom_index).

Selection: reads the comparison output `comparison_rmsd.csv` and randomly
samples N matched (min_maxdev < threshold) and N unmatched (>= threshold)
cases, writing them into two directories.

ID mapping (resolved empirically -- the top-level src_index in each downstream
run is that run's own scan index, NOT the Dimer src):
  - Dimer  : Dimer_status_csvs row (src,mpi_rank,attempt,atom,status) gives
             rank+atom -> dimer_{src}_{att}_{atom}.traj / dimer_opt_*.log in
             Dimer_debug_zips/structure_rank_{rank}_data.zip
  - DoubleMin: scan DoubleMinimization_trajes for the frame whose Dimer key is
             the target; its top-level src_index is the debug job id ->
             optimization_{jobid}_{-1,1}.traj in
             DoubleMinimization_debug_zips/structure_rank_{outrank}_data.zip
  - initbasinopt: same, optimization_{jobid}.traj in
             Minimization_debug_zips/structure_rank_{outrank}_data.zip

Usage:
  python scripts/make_comparison_trajs.py --root <campaign_dir> [--n 10]
         [--threshold 0.3] [--seed 0] [-j N]
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
from ase.io import read, write


RANK_RE = re.compile(r"_rank_(\d+)\.traj$")
LOG_PREFIX = "MinModeTranslate:"
_TARGETS = set()        # set of (src, attempt) Dimer keys to locate


# ----- Dimer key / curvature helpers (mirror consolidate_dimer_basin_entry) -- #
def dimer_key(atoms):
    node = atoms.info
    while isinstance(node, dict):
        if "src_index" in node and "attempt_id" in node:
            return (int(node["src_index"]), int(node["attempt_id"]))
        node = node.get("orig_info")
    return None


def parse_curvatures(log_text):
    per_step = {}
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith(LOG_PREFIX):
            continue
        parts = line[len(LOG_PREFIX):].split()
        if len(parts) < 6:
            continue
        try:
            step = int(parts[0]); curv = float(parts[5])
        except ValueError:
            continue
        per_step.setdefault(step, curv)
    return sorted(per_step.items())


def first_step_of_last_neg_stretch(curvatures):
    if not curvatures or curvatures[-1][1] >= 0:
        return None
    first = curvatures[-1][0]
    for step, curv in reversed(curvatures[:-1]):
        if curv < 0:
            first = step
        else:
            break
    return first


# ----- zip extraction ------------------------------------------------------- #
def _zip_read_bytes(zip_path, name):
    with zipfile.ZipFile(zip_path) as zf:
        if name not in zf.namelist():
            return None
        return zf.read(name)


def _zip_read_traj(zip_path, name):
    data = _zip_read_bytes(zip_path, name)
    if data is None:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".traj")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        return read(tmp, index=":")
    finally:
        os.remove(tmp)


# ----- scan worker: locate targets in an output-traj shard ------------------ #
def _scan_worker(path):
    """Return {(src,att): (top_src_index, rank)} for target frames in `path`."""
    m = RANK_RE.search(path)
    rank = int(m.group(1)) if m else -1
    hits = {}
    try:
        frames = read(path, index=":")
    except Exception:
        return hits
    for at in frames:
        k = dimer_key(at)
        if k in _TARGETS and k not in hits:
            hits[k] = (int(at.info.get("src_index")), rank)
    return hits


def _scan(files, workers, label, need):
    """Scan output shards in parallel; stop once all `need` targets found."""
    found = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        it = pool.imap_unordered(_scan_worker, files)
        for done, hits in enumerate(it, 1):
            found.update(hits)
            if done % 50 == 0 or found:
                print(f"  [{label}] scanned {done}/{len(files)} shards, "
                      f"{len(found)}/{need} targets located", flush=True)
            if len(found) >= need:
                pool.terminate()
                break
    return found


# ----- per-target stitch ---------------------------------------------------- #
SECTION_NAMES = {
    1: "dm_saddle_to_reactant_side-1",
    2: "dm_saddle_to_product_side+1",
    3: "dimer_reverse_saddle_to_basinentry",
    4: "initbasinopt_basinentry_to_min",
}


def build_one(src, att, dimer_info, dm_info, min_info, root, out_dir):
    """Extract the four sections, stitch, copy the dimer log. Returns a status
    dict (includes the section-3/section-4 basin-entry consistency check)."""
    dimer_rank, atom = dimer_info
    dm_jobid, dm_rank = dm_info
    min_jobid, min_rank = min_info

    dimer_zip = os.path.join(root, "Dimer_debug_zips",
                             f"structure_rank_{dimer_rank}_data.zip")
    dm_zip = os.path.join(root, "doubleopt", "DoubleMinimization_debug_zips",
                          f"structure_rank_{dm_rank}_data.zip")
    min_zip = os.path.join(root, "initbasinopt", "Minimization_debug_zips",
                           f"structure_rank_{min_rank}_data.zip")

    dimer_traj_name = f"dimer_{src}_{att}_{atom}.traj"
    dimer_log_name = f"dimer_opt_{src}_{att}_{atom}.log"
    dm_m1_name = f"optimization_{dm_jobid}_-1.traj"
    dm_p1_name = f"optimization_{dm_jobid}_1.traj"
    min_name = f"optimization_{min_jobid}.traj"

    # section 1/2: DM per-side relaxations
    dm_m1 = _zip_read_traj(dm_zip, dm_m1_name)
    dm_p1 = _zip_read_traj(dm_zip, dm_p1_name)
    # section 3 source: dimer trajectory + log for the neg-curv cutoff
    dimer_frames = _zip_read_traj(dimer_zip, dimer_traj_name)
    log_bytes = _zip_read_bytes(dimer_zip, dimer_log_name)
    # section 4: initbasinopt relaxation
    init_frames = _zip_read_traj(min_zip, min_name)

    missing = [n for n, v in [(dm_m1_name, dm_m1), (dm_p1_name, dm_p1),
                              (dimer_traj_name, dimer_frames),
                              (dimer_log_name, log_bytes),
                              (min_name, init_frames)] if v is None]
    if missing:
        return {"src": src, "att": att, "ok": False, "error": f"missing {missing}"}

    first_neg = first_step_of_last_neg_stretch(parse_curvatures(
        log_bytes.decode("utf-8", "replace")))
    if first_neg is None or first_neg >= len(dimer_frames):
        return {"src": src, "att": att, "ok": False,
                "error": f"bad first_neg_step={first_neg} (ntraj={len(dimer_frames)})"}

    # section 3 = dimer frames [first_neg .. saddle] reversed -> saddle first
    sec3 = list(reversed(dimer_frames[first_neg:]))

    sections = [(1, dm_m1), (2, dm_p1), (3, sec3), (4, init_frames)]
    stitched = []
    for sidx, frames in sections:
        for j, at in enumerate(frames):
            at.info = {"section": sidx, "section_name": SECTION_NAMES[sidx],
                       "src_index": src, "attempt_id": att, "frame_in_section": j}
            stitched.append(at)

    # consistency check: sec3 last (basin-entry) == sec4 first (init input)
    be = sec3[-1]
    ini = init_frames[0]
    same_n = (len(be) == len(ini)
              and np.array_equal(be.get_atomic_numbers(), ini.get_atomic_numbers()))
    maxdev = (float(np.abs(be.get_positions() - ini.get_positions()).max())
              if same_n else float("nan"))
    basin_entry_ok = bool(same_n and maxdev < 1e-4)

    out_traj = os.path.join(out_dir, f"stitched_src{src}_att{att}.traj")
    write(out_traj, stitched)
    with open(os.path.join(out_dir, dimer_log_name), "wb") as f:
        f.write(log_bytes)

    return {"src": src, "att": att, "ok": True, "nframes": len(stitched),
            "nsec": [len(dm_m1), len(dm_p1), len(sec3), len(init_frames)],
            "first_neg_step": first_neg, "ntraj_dimer": len(dimer_frames),
            "basin_entry_ok": basin_entry_ok, "basin_entry_maxdev": maxdev,
            "traj": out_traj, "log": dimer_log_name}


# ----- main ----------------------------------------------------------------- #
def main():
    global _TARGETS
    ncpu = len(os.sched_getaffinity(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--comparison-csv", default=None)
    ap.add_argument("--n", type=int, default=10, help="cases per group")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="maxdev match cutoff used to split matched/unmatched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("-j", "--workers", type=int, default=ncpu)
    args = ap.parse_args()

    root = args.root
    comp = args.comparison_csv or os.path.join(
        root, "basin_entry_vs_doublemin", "comparison_rmsd.csv")
    out_root = args.out_dir or os.path.join(root, "comparison_examples")
    out_matched = os.path.join(out_root, "matched")
    out_unmatched = os.path.join(out_root, "unmatched")
    os.makedirs(out_matched, exist_ok=True)
    os.makedirs(out_unmatched, exist_ok=True)

    # ---- select targets ----
    import random
    rng = random.Random(args.seed)
    matched, unmatched = [], []
    print(f"reading {comp} ...", flush=True)
    with open(comp) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if row[11] != "1":           # numbers_ok
                continue
            try:
                mmd = float(row[9])       # min_maxdev
            except ValueError:
                continue
            key = (int(row[0]), int(row[1]), row[2])  # src, att, rxn
            (matched if mmd < args.threshold else unmatched).append(key)
    print(f"  pool: {len(matched):,} matched, {len(unmatched):,} unmatched")
    sel_matched = rng.sample(matched, min(args.n, len(matched)))
    sel_unmatched = rng.sample(unmatched, min(args.n, len(unmatched)))
    targets = {(s, a): ("matched", rxn) for s, a, rxn in sel_matched}
    targets.update({(s, a): ("unmatched", rxn) for s, a, rxn in sel_unmatched})
    _TARGETS = set(targets)
    print(f"selected {len(sel_matched)} matched + {len(sel_unmatched)} unmatched "
          f"= {len(_TARGETS)} targets (seed={args.seed})", flush=True)

    # ---- dimer rank+atom from Dimer CSVs (text) ----
    print("locating targets in Dimer_status_csvs ...", flush=True)
    dimer_loc = {}
    for csvf in glob.glob(os.path.join(root, "Dimer_status_csvs", "status_rank_*.csv")):
        with open(csvf) as f:
            for line in f:
                p = line.rstrip("\n").split(",")
                if len(p) < 5:
                    continue
                try:
                    s, mpi, a, atom = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                except ValueError:
                    continue
                if (s, a) in _TARGETS and (s, a) not in dimer_loc:
                    dimer_loc[(s, a)] = (mpi, atom)
        if len(dimer_loc) >= len(_TARGETS):
            break
    print(f"  dimer-located {len(dimer_loc)}/{len(_TARGETS)}", flush=True)

    # ---- DM + initbasinopt jobid/rank by scanning their output trajes ----
    dm_files = sorted(glob.glob(os.path.join(
        root, "doubleopt", "DoubleMinimization_trajes", "collected_opt_rank_*.traj")))
    min_files = sorted(glob.glob(os.path.join(
        root, "initbasinopt", "Minimization_trajes", "collected_opt_rank_*.traj")))
    print(f"scanning {len(min_files)} initbasinopt shards ...", flush=True)
    min_loc = _scan(min_files, args.workers, "init", len(_TARGETS))
    print(f"scanning {len(dm_files)} DoubleMinimization shards ...", flush=True)
    dm_loc = _scan(dm_files, args.workers, "dm", len(_TARGETS))

    # ---- build each ----
    print("\nbuilding stitched trajectories ...", flush=True)
    results = []
    for key, (group, rxn) in targets.items():
        out_dir = out_matched if group == "matched" else out_unmatched
        if key not in dimer_loc or key not in dm_loc or key not in min_loc:
            miss = [n for n, d in [("dimer", dimer_loc), ("dm", dm_loc),
                                   ("init", min_loc)] if key not in d]
            print(f"  SKIP src{key[0]} att{key[1]} ({group}): not located in {miss}")
            results.append({"src": key[0], "att": key[1], "group": group,
                            "ok": False, "error": f"not located in {miss}"})
            continue
        res = build_one(key[0], key[1], dimer_loc[key], dm_loc[key], min_loc[key],
                        root, out_dir)
        res["group"] = group
        res["reaction_type"] = rxn
        tag = "OK " if res["ok"] else "ERR"
        extra = (f"frames={res['nframes']} sec={res['nsec']} "
                 f"basin_entry_ok={res['basin_entry_ok']} "
                 f"(maxdev={res['basin_entry_maxdev']:.2e})"
                 if res["ok"] else res["error"])
        print(f"  [{tag}] {group:9s} src{key[0]} att{key[1]} {rxn}: {extra}", flush=True)
        results.append(res)

    ok = [r for r in results if r.get("ok")]
    be_ok = [r for r in ok if r.get("basin_entry_ok")]
    print(f"\ndone: {len(ok)}/{len(results)} built; "
          f"basin-entry check passed {len(be_ok)}/{len(ok)}")
    print(f"  matched   -> {out_matched}")
    print(f"  unmatched -> {out_unmatched}")
    if len(be_ok) != len(ok):
        print("  WARNING: some basin-entry checks FAILED (section3 end != "
              "section4 start) -- investigate mapping for those.")


if __name__ == "__main__":
    main()
