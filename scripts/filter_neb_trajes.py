"""Filter NEB_trajes/ -> NEB_trajes_filtered/.

A band is one (src_index, subband_idx) segment within a rank file; a frame
without 'subband_idx' counts as sub-band 0. Each sub-band is judged on its
own, against its own status and climbing image. A band is dropped if any of:

  1. it is not converged: status does not start with "converged"  (drops
     "not_converged", "error", "not_converged_StopRun", etc; keeps
     "converged" and "converged_CI"). A frame with no 'status' in its info
     falls back to the band_converged / band_converged_CI flags nebopt
     writes; "converged*" is equivalent to either flag being True.
  2. any frame in the band has no energy/forces  (on mp20bat this caught
     the 2 truncated bands rank 24 src=88937 and rank 59 src=89231 that we
     couldn't recover)
  3. no frame carries info['barrier'], so there is no climbing image (CI)
  4. an intermediate minimum is present -- see
     _band_has_intermediate_minimum. Walking image by image from each
     endpoint toward the CI, with dE the energy change of each step:
       - a single step with dE < -big_drop_thresh (0.05 eV) flags the band;
       - two consecutive drops (dE < 0 on two adjacent steps of the same
         walk) flag it;
       - a small drop (-big_drop_thresh <= dE < 0) is tolerated as long as
         neither adjacent step of the same walk is also a drop
         (endpoint-relaxation jitter).
     The band is dropped if either the left or the right walk flags.
     Energies are deduplicated by image_idx first.

Frames in surviving bands are written in the original order.

Usage:
    python filter_neb_trajes.py                                # defaults
    python filter_neb_trajes.py --in-dir NEB_trajes \\
                                --out-dir NEB_trajes_filtered \\
                                --workers 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import OrderedDict

from ase.io import read
from ase.io.trajectory import Trajectory


def _band_has_intermediate_minimum(
    energies_by_imgidx: dict[int, float],
    ci_imgidx: int,
    big_drop_thresh: float = 0.05,
) -> bool:
    """Return True if a real intermediate minimum is detected.

    Walks from each endpoint inward toward the CI (climbing image, located
    at `ci_imgidx`). Along each walk we track whether the previous step was
    a (small) drop:

        delta = E[next] - E[current]
        - delta < -big_drop_thresh   : big drop -> FLAG (real descent)
        - delta < 0                  : small drop
            - if previous step was also a drop -> FLAG (two in a row)
            - else mark "saw_drop" and continue
        - delta >= 0                 : reset "saw_drop", continue

    Endpoint not-fully-relaxed jitter is tolerated as long as it's within
    `big_drop_thresh` (0.05 eV) -- a single such drop is allowed, but two
    consecutive drops anywhere on the walk to CI signal a real basin.

    `energies_by_imgidx` maps image_idx -> energy for the band (already
    deduplicated). `ci_imgidx` is the image_idx of the CI (the frame that
    carries info['barrier']).
    """
    n = len(energies_by_imgidx)
    if n < 3:
        return False
    items = sorted(energies_by_imgidx.items())  # by image_idx
    indices = [k for k, _ in items]
    energies = [e for _, e in items]
    if ci_imgidx not in indices:
        # CI image got deduped away or never present -- can't run the walk.
        return False
    ci_pos = indices.index(ci_imgidx)

    # Walk from left endpoint (pos 0) toward CI.
    saw_drop = False
    for i in range(0, ci_pos):
        delta = energies[i + 1] - energies[i]
        if delta < -big_drop_thresh:
            return True
        if delta < 0:
            if saw_drop:
                return True
            saw_drop = True
        else:
            saw_drop = False

    # Walk from right endpoint (pos n-1) toward CI.
    saw_drop = False
    for i in range(n - 1, ci_pos, -1):
        delta = energies[i - 1] - energies[i]
        if delta < -big_drop_thresh:
            return True
        if delta < 0:
            if saw_drop:
                return True
            saw_drop = True
        else:
            saw_drop = False

    return False


def _band_has_energy(frames) -> bool:
    """Every frame must carry an energy (and forces — they always come together
    here)."""
    for a in frames:
        if a.calc is None or "energy" not in a.calc.results:
            return False
    return True


def _frame_converged(a) -> bool:
    """True if the frame's status starts with 'converged'. Without a 'status'
    key, fall back to nebopt's flags: status "converged" is written exactly
    when band_converged is True and "converged_CI" exactly when only
    band_converged_CI is, so either flag being True is the same test."""
    st = a.info.get("status")
    if st is not None:
        return str(st).startswith("converged")
    return bool(a.info.get("band_converged") or a.info.get("band_converged_CI"))


def _all_converged(frames) -> bool:
    """All frames of the band must be converged (all frames in a sub-band
    share status) — anything not_converged → drop."""
    return all(_frame_converged(a) for a in frames)


def _band_key(a) -> tuple[int, int]:
    """(src_index, subband_idx) of a frame; frames without 'subband_idx'
    belong to sub-band 0."""
    return a.info["src_index"], a.info.get("subband_idx", 0)


def filter_rank(rank, in_dir, out_dir, big_drop_thresh=0.05):
    in_path = os.path.join(in_dir, f"collected_ts_rank_{rank}.traj")
    out_path = os.path.join(out_dir, f"collected_ts_rank_{rank}.traj")
    stats = {
        "rank": rank,
        "n_bands_in": 0,
        "n_bands_out": 0,
        "drop_status": 0,
        "drop_no_ef": 0,
        "drop_imin": 0,
        "drop_no_ci_tag": 0,
        "n_frames_in": 0,
        "n_frames_out": 0,
        "error": None,
    }
    if not os.path.exists(in_path):
        stats["error"] = "input missing"
        return stats

    frames = read(in_path, index=":")
    stats["n_frames_in"] = len(frames)

    # Group by (src_index, subband_idx), preserving first-seen order.
    band_order: list[tuple[int, int]] = []
    band_frames: dict[tuple[int, int], list] = OrderedDict()
    for a in frames:
        key = _band_key(a)
        if key not in band_frames:
            band_order.append(key)
            band_frames[key] = []
        band_frames[key].append(a)

    stats["n_bands_in"] = len(band_order)

    keep_keys: set[tuple[int, int]] = set()
    for key in band_order:
        chunk = band_frames[key]

        if not _all_converged(chunk):
            stats["drop_status"] += 1
            continue
        if not _band_has_energy(chunk):
            stats["drop_no_ef"] += 1
            continue

        # Imin via walk from each endpoint to CI. Dedupe by image_idx so a
        # repeated image doesn't create false neighbors. CI is the image
        # whose info has 'barrier' (matches extract_neb_triplets.py).
        e_by_idx: dict[int, float] = {}
        ci_imgidx: int | None = None
        for a in chunk:
            ii = a.info["image_idx"]
            if ii not in e_by_idx:
                e_by_idx[ii] = a.get_potential_energy()
            if "barrier" in a.info and ci_imgidx is None:
                ci_imgidx = ii
        if ci_imgidx is None:
            # No CI tag in any frame -- can't run the walk; drop band.
            stats["drop_no_ci_tag"] += 1
            continue
        if _band_has_intermediate_minimum(
            e_by_idx, ci_imgidx, big_drop_thresh=big_drop_thresh
        ):
            stats["drop_imin"] += 1
            continue

        keep_keys.add(key)

    # Write output preserving the original frame order across kept bands.
    os.makedirs(out_dir, exist_ok=True)
    n_out = 0
    with Trajectory(out_path, "w") as tr:
        for a in frames:
            if _band_key(a) in keep_keys:
                tr.write(a)
                n_out += 1
    stats["n_frames_out"] = n_out
    stats["n_bands_out"] = len(keep_keys)
    return stats


def _worker(args):
    rank, in_dir, out_dir, big_drop_thresh = args
    try:
        return filter_rank(rank, in_dir, out_dir, big_drop_thresh)
    except Exception as e:
        return {
            "rank": rank, "n_bands_in": 0, "n_bands_out": 0,
            "drop_status": 0, "drop_no_ef": 0, "drop_imin": 0,
            "drop_no_ci_tag": 0,
            "n_frames_in": 0, "n_frames_out": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def _discover_ranks(in_dir):
    out = []
    for fn in os.listdir(in_dir):
        if fn.startswith("collected_ts_rank_") and fn.endswith(".traj"):
            try:
                out.append(int(fn[len("collected_ts_rank_"):-len(".traj")]))
            except ValueError:
                pass
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="NEB_trajes")
    p.add_argument("--out-dir", default="NEB_trajes_filtered")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--big-drop-thresh", type=float, default=0.05,
                   help="energy drop (eV) on the way to CI that flags a "
                        "real intermediate basin in a single step. Two "
                        "consecutive smaller drops also flag. Default 0.05.")
    args = p.parse_args()

    ranks = _discover_ranks(args.in_dir)
    if not ranks:
        print(f"no ranks found in {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Filtering {len(ranks)} ranks  ({args.in_dir} -> {args.out_dir})  with {args.workers} worker(s)")

    t0 = time.time()
    jobs = [(r, args.in_dir, args.out_dir, args.big_drop_thresh) for r in ranks]
    if args.workers <= 1:
        results = [_worker(j) for j in jobs]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    dt = time.time() - t0

    n_bands_in = sum(r["n_bands_in"] for r in results)
    n_bands_out = sum(r["n_bands_out"] for r in results)
    drop_status = sum(r["drop_status"] for r in results)
    drop_no_ef = sum(r["drop_no_ef"] for r in results)
    drop_imin = sum(r["drop_imin"] for r in results)
    n_frames_in = sum(r["n_frames_in"] for r in results)
    n_frames_out = sum(r["n_frames_out"] for r in results)
    errors = [r for r in results if r["error"]]

    print(f"\nDone in {dt:.1f}s")
    print(f"Bands  : in={n_bands_in}, out={n_bands_out} ({100*n_bands_out/max(n_bands_in,1):.2f}%)")
    print(f"  dropped (status not converged*) : {drop_status}")
    print(f"  dropped (no E/F)                : {drop_no_ef}")
    print(f"  dropped (intermediate minimum)  : {drop_imin}")
    drop_no_ci = sum(r["drop_no_ci_tag"] for r in results)
    if drop_no_ci:
        print(f"  dropped (no 'barrier' tag)       : {drop_no_ci}")
    print(f"Frames : in={n_frames_in}, out={n_frames_out}")
    if errors:
        print(f"\nErrors in {len(errors)} ranks:")
        for r in errors[:5]:
            print(f"  rank {r['rank']}: {r['error'].splitlines()[0]}")


if __name__ == "__main__":
    main()
