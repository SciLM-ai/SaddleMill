"""Compare each converged initbasinopt minimum against the two converged
DoubleMinimization endpoints for the same Dimer attempt.

Question answered
-----------------
For every converged Dimer saddle (keyed by ``(src_index, attempt_id)``) two
things were computed downstream:

- **initbasinopt** (`Minimization`): one downhill minimum from the dimer
  trajectory's negative-curvature *basin-entry* frame.
- **doubleopt** (`DoubleMinimization`): the two endpoints (``side=-1`` and
  ``side=+1``) obtained by displacing the converged TS along ±eigenmode and
  relaxing.

This script joins the two by ``(src_index, attempt_id)`` and asks: **does the
initbasinopt minimum coincide with either of the two DoubleMinimization
endpoints?** For each matched attempt it records the RMSD of the initbasinopt
minimum against each endpoint, the minimum of the two, and a binary match at a
configurable threshold. It then prints completion/agreement statistics
(overall and per reaction type) and plots histograms so a sensible RMSD
threshold can be chosen.

Why this works geometrically
----------------------------
The initbasinopt minimum and both DoubleMinimization endpoints all descend
from the *same supercell dimer TS* (same atoms, same order, fixed cell since
`relax_cell=False` in both runs). So atom i in one structure corresponds to
atom i in the other — no permutation/registration search is needed. RMSD is
computed on that i<->i correspondence with the minimum-image convention (to
absorb atoms that drifted across periodic boundaries) and with the rigid
global translation removed (a relaxed crystal in a fixed cell is free to sit
anywhere). Cell orientation is pinned by the fixed lattice, so no rotation
search is needed either.

Two dissimilarity measures are reported per comparison:
- ``rmsd``   : sqrt(mean_i |Δr_i|^2)            — whole-cell average.
- ``maxdev`` : max_i |Δr_i|                      — largest single-atom move.
For localized bulk reactions (a vacancy hop moves ~1 atom in a ~100-atom cell)
the RMSD is diluted by the many static atoms, so ``maxdev`` usually separates
"same basin" from "different basin" more sharply. Both are saved; the match
threshold is applied to ``min_rmsd`` by default (``--match-on maxdev`` switches
to the max-deviation metric).

Caveats: i<->i RMSD assumes atom identity/order is preserved (verified per
comparison via the atomic-number arrays — mismatches are flagged, not
silently scored); it does not treat two chemically-identical atoms swapping
sites as "the same" (that would need a permutation-invariant matcher such as
pymatgen ``StructureMatcher``); the round()-based MIC is exact for orthogonal
cells and a good approximation for the near-orthogonal supercells here.

Inputs (defaults relative to ``--root``, the campaign dir)
----------------------------------------------------------
- ``--initbasin-dir``  initbasinopt/Minimization_trajes/collected_opt_rank_*.traj
- ``--doublemin-dir``  doubleopt/DoubleMinimization_trajes/collected_opt_rank_*.traj

Outputs (into ``--out-dir``, default ``<root>/basin_entry_vs_doublemin/``)
-------------------------------------------------------------------------
- ``comparison_rmsd.csv``    one row per joined attempt (all metrics).
- ``matched_attempts.csv``   the subset with min metric < threshold:
                             ``src_index, attempt_id, reaction_type,
                             matched_side, min_rmsd, min_maxdev``.
- ``rmsd_histogram.png``     histograms of min_rmsd and min_maxdev.
- ``summary.txt``            text stats (overall + per reaction type).

Scaling / testing knobs
-----------------------
The full campaign is ~1.9M attempts. Reading is parallelized over rank files.
The initbasinopt index is built in the parent process and shared with the
DoubleMinimization workers by fork (copy-on-write), so the big structure set is
held once. For a quick functional test, cap the number of rank files scanned
with ``--initbasin-files`` and ``--doublemin-files`` (overlap is probabilistic
but nonzero for generous caps), or restrict to a deterministic key slice with
``--src-mod K [--src-rem R]`` (scans everything but keeps only
``src_index % K == R`` — guarantees overlap and bounds memory; also usable to
run the full set in K memory-bounded passes).
"""
import argparse
import csv
import gc
import glob
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter

import numpy as np

try:
    from ase.io import read
except Exception as e:  # pragma: no cover
    sys.exit(f"ASE import failed: {e}")


RANK_RE = re.compile(r"_rank_(\d+)\.traj$")

# Populated in the parent before the DoubleMinimization pool is forked; the
# fork lets workers read it copy-on-write without re-pickling ~GBs of structures.
_INDEX = {}
_ARGS = None


# --------------------------------------------------------------------------- #
# info helpers
# --------------------------------------------------------------------------- #
def info_get(atoms, key):
    """Return atoms.info[key], walking nested orig_info dicts if needed."""
    info = atoms.info
    if key in info:
        return info[key]
    oi = info.get("orig_info")
    while isinstance(oi, dict):
        if key in oi:
            return oi[key]
        oi = oi.get("orig_info")
    return None


def _key(atoms):
    """The Dimer identity ``(src_index, attempt_id)``.

    IMPORTANT: the top-level ``src_index`` of a DoubleMinimization /
    initbasinopt frame is *not* the Dimer source index — each downstream run
    overwrites the top-level ``src_index`` with its own input-scan position,
    and only ``attempt_id`` survives untouched in ``orig_info``. The original
    Dimer record is the shallowest ``info``/``orig_info`` dict that carries
    *both* ``src_index`` and ``attempt_id``; read both from that same dict so
    the two runs join on the actual Dimer crystal+attempt, not a coincidental
    collision of reassigned indices.
    """
    node = atoms.info
    while isinstance(node, dict):
        if "src_index" in node and "attempt_id" in node:
            return (int(node["src_index"]), int(node["attempt_id"]))
        node = node.get("orig_info")
    return None


def _is_converged(atoms):
    if atoms.info.get("status") == "converged":
        return True
    # fall back to the boolean if a top-level status string is absent
    return bool(atoms.info.get("converged", False)) and "status" not in atoms.info


def _src_kept(src):
    if _ARGS is None or _ARGS.src_mod is None:
        return True
    return (src % _ARGS.src_mod) == _ARGS.src_rem


# --------------------------------------------------------------------------- #
# RMSD
# --------------------------------------------------------------------------- #
def periodic_rmsd_maxdev(pos_a, pos_b, cell):
    """i<->i RMSD and max single-atom deviation under MIC, translation removed.

    Both structures must share `cell` (true here: fixed lattice). Returns
    (rmsd, maxdev) in Angstrom.
    """
    delta = np.asarray(pos_a, float) - np.asarray(pos_b, float)
    cell = np.asarray(cell, float)
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)                 # minimum image
    mic = frac @ cell
    mic -= mic.mean(axis=0)                # remove rigid translation
    d = np.sqrt((mic ** 2).sum(axis=1))
    return float(np.sqrt((d ** 2).mean())), float(d.max())


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
def _index_worker(path):
    """Return [(key, numbers, positions(f32), cell(f32), reaction_type), ...]
    for converged initbasinopt minima in one rank file."""
    out = []
    try:
        frames = read(path, index=":")
    except Exception as e:
        return ("error", path, f"{type(e).__name__}: {e}")
    for at in frames:
        if not _is_converged(at):
            continue
        key = _key(at)
        if key is None or not _src_kept(key[0]):
            continue
        out.append((
            key,
            np.asarray(at.get_atomic_numbers(), dtype=np.int16),
            np.asarray(at.get_positions(), dtype=np.float32),
            np.asarray(at.cell.array, dtype=np.float32),
            info_get(at, "reaction_type"),
        ))
    return ("ok", path, out)


def _dm_worker(path):
    """Read one DoubleMinimization rank file; return its both-converged minima
    pairs. **Does NOT touch the shared index** — the parent scores each pair.

    This is deliberate: having forked workers read the ~GB parent index causes
    CPython refcount-driven copy-on-write to duplicate its pages per worker
    (gc.freeze cannot prevent this), which OOM-kills workers at full scale. By
    keeping workers as pure readers of their own DM file and scoring in the
    single parent process, the index is never copied.

    Returns ("ok", path, candidates, stats), where each candidate is
    (src, attempt, reaction_type, numbers_i16, pos_side-1_f32, pos_side+1_f32,
    cell_f32). numbers/cell are shared by both sides (same TS), so sent once.
    """
    stats = Counter()
    cands = []
    try:
        frames = read(path, index=":")
    except Exception as e:
        return ("error", path, f"{type(e).__name__}: {e}")

    # group the two minima of each job (side -1 / +1); TS (side 0) ignored
    groups = {}
    for at in frames:
        side = at.info.get("side")
        if side not in (-1, 1):
            continue
        key = _key(at)
        if key is None or not _src_kept(key[0]):
            continue
        g = groups.setdefault(key, {"rxn": info_get(at, "reaction_type")})
        g[side] = (
            np.asarray(at.get_atomic_numbers(), dtype=np.int16),
            np.asarray(at.get_positions(), dtype=np.float32),
            np.asarray(at.cell.array, dtype=np.float32),
            _is_converged(at),
        )

    for key, g in groups.items():
        if -1 not in g or 1 not in g:
            stats["dm_incomplete"] += 1
            continue
        stats["dm_jobs"] += 1
        n1, p1, c1, conv1 = g[-1]
        n2, p2, c2, conv2 = g[1]
        if not (conv1 and conv2):
            stats["dm_side_not_converged"] += 1
            continue
        stats["dm_both_converged"] += 1
        cands.append((key[0], key[1], g["rxn"], n1, p1, p2, c1))

    return ("ok", path, cands, stats)


def _score_candidate(cand):
    """Score one both-converged DM pair against the parent-held _INDEX.

    Runs in the parent (so it can read the big index without copying it).
    Returns (row_or_None, stat_key).
    """
    src, attempt, rxn_dm, n_dm, p1, p2, cell_dm = cand
    ib = _INDEX.get((src, attempt))
    if ib is None:
        return None, "no_initbasin_match"
    ib_num, ib_pos, ib_cell, ib_rxn = ib
    rxn = rxn_dm if rxn_dm is not None else ib_rxn
    natoms = len(ib_num)
    if not (len(ib_num) == len(n_dm) and np.array_equal(ib_num, n_dm)):
        return ((src, attempt, rxn, natoms,
                 np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, 0),
                "numbers_mismatch")
    ib_cell = np.asarray(ib_cell, float)
    cell_ok = int(np.allclose(ib_cell, cell_dm, atol=1e-3))
    r1, m1 = periodic_rmsd_maxdev(ib_pos, p1, ib_cell)
    r2, m2 = periodic_rmsd_maxdev(ib_pos, p2, ib_cell)
    if r1 <= r2:
        matched_side, min_rmsd, min_md = -1, r1, m1
    else:
        matched_side, min_rmsd, min_md = 1, r2, m2
    return ((src, attempt, rxn, natoms,
             r1, r2, m1, m2, min_rmsd, min_md, matched_side, 1, cell_ok),
            "compared")


HEADER = ["src_index", "attempt_id", "reaction_type", "natoms",
          "rmsd_side-1", "rmsd_side+1", "maxdev_side-1", "maxdev_side+1",
          "min_rmsd", "min_maxdev", "matched_side", "numbers_ok", "cell_ok"]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _list_rank_files(d, limit):
    files = sorted(glob.glob(os.path.join(d, "collected_opt_rank_*.traj")),
                   key=lambda p: int(RANK_RE.search(p).group(1)))
    if limit is not None:
        files = files[:limit]
    return files


def main():
    global _INDEX, _ARGS
    ncpu = len(os.sched_getaffinity(0))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--initbasin-dir", default=None)
    ap.add_argument("--doublemin-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="match cutoff on the selected metric (Angstrom)")
    ap.add_argument("--match-on", choices=["rmsd", "maxdev"], default="rmsd",
                    help="metric the binary match/threshold uses (default rmsd)")
    ap.add_argument("--initbasin-files", type=int, default=None,
                    help="cap number of initbasinopt rank files (testing)")
    ap.add_argument("--doublemin-files", type=int, default=None,
                    help="cap number of DoubleMinimization rank files (testing)")
    ap.add_argument("--src-mod", type=int, default=None,
                    help="keep only src_index %% K == --src-rem (slice/memory bound)")
    ap.add_argument("--src-rem", type=int, default=0)
    ap.add_argument("-j", "--workers", type=int, default=ncpu)
    _ARGS = ap.parse_args()

    root = _ARGS.root
    ib_dir = _ARGS.initbasin_dir or os.path.join(root, "initbasinopt/Minimization_trajes")
    dm_dir = _ARGS.doublemin_dir or os.path.join(root, "doubleopt/DoubleMinimization_trajes")
    out_dir = _ARGS.out_dir or os.path.join(root, "basin_entry_vs_doublemin")
    os.makedirs(out_dir, exist_ok=True)

    ib_files = _list_rank_files(ib_dir, _ARGS.initbasin_files)
    dm_files = _list_rank_files(dm_dir, _ARGS.doublemin_files)
    if not ib_files:
        sys.exit(f"no initbasinopt rank files under {ib_dir}")
    if not dm_files:
        sys.exit(f"no DoubleMinimization rank files under {dm_dir}")
    workers = max(1, min(_ARGS.workers, max(len(ib_files), len(dm_files))))
    print(f"initbasinopt files: {len(ib_files)} | DoubleMinimization files: "
          f"{len(dm_files)} | workers: {workers}", flush=True)
    if _ARGS.src_mod:
        print(f"slice: src_index %% {_ARGS.src_mod} == {_ARGS.src_rem}", flush=True)

    ctx = mp.get_context("fork")
    t0 = time.time()
    def el():
        return time.time() - t0

    # ---- phase 1: build the initbasinopt index in the parent ----
    print("building initbasinopt index ...", flush=True)
    dup = 0
    with ctx.Pool(workers) as pool:
        for done, res in enumerate(pool.imap_unordered(_index_worker, ib_files), 1):
            if res[0] == "error":
                print(f"  WARN index read failed {res[1]}: {res[2]}", flush=True)
                continue
            for key, num, pos, cell, rxn in res[2]:
                if key in _INDEX:
                    dup += 1
                _INDEX[key] = (num, pos, cell, rxn)
            if done % 25 == 0 or done == len(ib_files):
                rate = done / max(el(), 1e-9)
                eta = (len(ib_files) - done) / max(rate, 1e-9)
                print(f"  [{el():5.0f}s] indexed {done}/{len(ib_files)} files "
                      f"({rate:.0f} f/s, ~{eta:.0f}s left), "
                      f"{len(_INDEX):,} minima", flush=True)
    print(f"  [{el():5.0f}s] index ready: {len(_INDEX):,} converged initbasinopt "
          f"minima{f' ({dup:,} dup keys, last kept)' if dup else ''}", flush=True)

    # Freeze GC so the long-lived ~GB index isn't repeatedly traversed by the
    # collector during the parent scoring loop below.
    gc.collect()
    gc.freeze()

    # ---- phase 2: workers READ DM files (no index access); parent SCORES ----
    # Workers return both-converged minima pairs; the parent looks each up in
    # _INDEX and computes RMSD/maxdev. Keeping the index out of the forked
    # workers is what prevents the copy-on-write OOM.
    print(f"  [{el():5.0f}s] joining DoubleMinimization ...", flush=True)
    t_dm = time.time()
    all_rows = []
    stats = Counter()
    with ctx.Pool(workers) as pool:
        for done, res in enumerate(pool.imap_unordered(_dm_worker, dm_files), 1):
            if res[0] == "error":
                print(f"  WARN dm read failed {res[1]}: {res[2]}", flush=True)
                continue
            _, _, cands, st = res
            stats.update(st)
            for cand in cands:
                row, sk = _score_candidate(cand)
                stats[sk] += 1
                if row is not None:
                    all_rows.append(row)
            if done % 10 == 0 or done == len(dm_files):
                rate = done / max(time.time() - t_dm, 1e-9)
                eta = (len(dm_files) - done) / max(rate, 1e-9)
                print(f"  [{el():5.0f}s] processed {done}/{len(dm_files)} files "
                      f"({rate:.1f} f/s, ~{eta:.0f}s left), "
                      f"{stats['compared']:,} comparisons", flush=True)
    stats["have_all_three"] = stats["compared"] + stats["numbers_mismatch"]

    # ---- write per-attempt CSV ----
    csv_path = os.path.join(out_dir, "comparison_rmsd.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(all_rows)
    print(f"wrote {csv_path} ({len(all_rows):,} rows)", flush=True)

    # ---- analyse ----
    metric_idx = 8 if _ARGS.match_on == "rmsd" else 9   # min_rmsd / min_maxdev
    thr = _ARGS.threshold
    valid = [r for r in all_rows if r[11] == 1 and np.isfinite(r[8])]
    min_rmsd = np.array([r[8] for r in valid], float)
    min_md = np.array([r[9] for r in valid], float)
    metric = np.array([r[metric_idx] for r in valid], float)
    matched_mask = metric < thr
    n_matched = int(matched_mask.sum())
    n_valid = len(valid)

    # matched attempts list
    matched_path = os.path.join(out_dir, "matched_attempts.csv")
    with open(matched_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_index", "attempt_id", "reaction_type",
                    "matched_side", "min_rmsd", "min_maxdev"])
        for r, ok in zip(valid, matched_mask):
            if ok:
                w.writerow([r[0], r[1], r[2], r[10], f"{r[8]:.4f}", f"{r[9]:.4f}"])
    print(f"wrote {matched_path} ({n_matched:,} matched at "
          f"{_ARGS.match_on} < {thr} A)", flush=True)

    # ---- summary text ----
    lines = []
    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    out("\n================ SUMMARY ================")
    out(f"DoubleMinimization jobs (both sides present)      : {stats['dm_jobs']:,}")
    out(f"  both sides converged                            : {stats['dm_both_converged']:,}")
    out(f"  + had a converged initbasinopt match            : {stats['have_all_three']:,}")
    out(f"  - atom-order/identity mismatch (excluded)       : {stats['numbers_mismatch']:,}")
    out(f"comparisons scored                                : {n_valid:,}")
    if n_valid:
        out(f"MATCHED ({_ARGS.match_on} < {thr} A)                       "
            f": {n_matched:,}  ({100.0*n_matched/n_valid:.1f}%)")
        out(f"min_rmsd   median={np.median(min_rmsd):.3f}  "
            f"p10={np.percentile(min_rmsd,10):.3f}  p90={np.percentile(min_rmsd,90):.3f}")
        out(f"min_maxdev median={np.median(min_md):.3f}  "
            f"p10={np.percentile(min_md,10):.3f}  p90={np.percentile(min_md,90):.3f}")
        side = np.array([r[10] for r in valid])
        out(f"matched-side balance (lower-RMSD endpoint): "
            f"side-1={int((side==-1).sum()):,}  side+1={int((side==1).sum()):,}")

    # per reaction type
    if n_valid:
        out("\nper reaction_type (matched / scored):")
        by = {}
        for r, ok in zip(valid, matched_mask):
            rt = str(r[2])
            c = by.setdefault(rt, [0, 0])
            c[1] += 1
            c[0] += int(ok)
        for rt in sorted(by, key=lambda k: -by[k][1]):
            m, t = by[rt]
            out(f"  {rt:24s} {m:>8,} / {t:>8,}  ({100.0*m/t:5.1f}%)")

    if stats.get("dm_side_not_converged"):
        out(f"\n(note: {stats['dm_side_not_converged']:,} DM jobs had a "
            f"non-converged side; {stats['no_initbasin_match']:,} had no "
            f"converged initbasinopt match)")

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- histograms ----
    if n_valid:
        _plot(min_rmsd, min_md, thr, _ARGS.match_on,
              os.path.join(out_dir, "rmsd_histogram.png"))
        print(f"wrote {os.path.join(out_dir, 'rmsd_histogram.png')}", flush=True)
    else:
        print("no valid comparisons — nothing to plot (try larger file caps "
              "or a smaller --src-mod).", flush=True)


def _plot(min_rmsd, min_md, thr, match_on, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, name in ((axes[0], min_rmsd, "min_rmsd"),
                           (axes[1], min_md, "min_maxdev")):
        hi = np.percentile(data, 99) if len(data) else 1.0
        hi = max(hi, thr * 1.5, 1e-3)
        clipped = np.clip(data, 0, hi)
        ax.hist(clipped, bins=120, color="steelblue", edgecolor="none")
        if match_on == name.split("_")[1] or (match_on == "rmsd" and name == "min_rmsd") \
                or (match_on == "maxdev" and name == "min_maxdev"):
            ax.axvline(thr, ls="--", color="k", label=f"threshold={thr}")
            ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel(f"{name} (A, clipped at p99={hi:.2f})")
        ax.set_ylabel("count")
        ax.set_title(f"{name}: initbasinopt vs nearer DoubleMin endpoint\n"
                     f"n={len(data):,}, median={np.median(data):.3f} A")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
