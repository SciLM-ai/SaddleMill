#!/usr/bin/env python
"""Backfill ``atoms.info['status']`` (and optionally ``task_name``) onto
output traj frames.

Run from a directory containing one or more of
``{Dimer,NEB,Minimization,DoubleMinimization}_status_csvs/`` and
``{Dimer,NEB,Minimization,DoubleMinimization}_trajes/``. Rewrites each
output traj in place; the original is moved to ``*.bak`` next to it.

Re-running is idempotent: files that already have a ``.bak`` sibling are
skipped, and so are files whose first frame already carries the targeted
keys (``status`` and, when applicable, ``task_name``). Presence is
checked once on the first frame — if it's there, the rest of the file is
left untouched.

If ``config.ini`` is present in the working directory and contains
``[FAIRChemCalculator] task_name = ...``, that value is also written to
``atoms.info['task_name']`` on every frame that doesn't already have it.

Usage:
    python backfill_status.py             # patch every method present
    python backfill_status.py Dimer NEB   # subset
"""
import configparser
import csv
import glob
import os
import shutil
import sys
from ase.io import Trajectory

# method -> (frame info key for sub-unit, csv column index of sub-unit id).
# Minimization has no sub-unit: match by src_index alone.
# DoubleMinimization sub-unit is `side` (-1, 0, +1). The TS frame (side=0) has
# no CSV row — it's always 'converged' — so we synthesize that entry below.
METHOD_KEYS = {
    "Dimer":              ("attempt_id", 2),  # csv: src, rank, attempt_id, selected_idx, status
    "NEB":                ("subband_idx", 2),  # csv: src, rank, sub_band_id, status
    "Minimization":       (None, None),        # csv: src, rank, status
    "DoubleMinimization": ("side", 2),         # csv: src, rank, side_id, parent_ts_idx, status
}


def read_task_name(config_path="config.ini"):
    """Return ``[FAIRChemCalculator] task_name`` from config.ini, or None."""
    if not os.path.exists(config_path):
        return None
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cp.read(config_path)
    if not cp.has_option("FAIRChemCalculator", "task_name"):
        return None
    return cp["FAIRChemCalculator"]["task_name"].strip()


def load_status_map(method):
    """Return {(src_index, subunit_id_or_None): status} from all rank CSVs."""
    _, subunit_col = METHOD_KEYS[method]
    out = {}
    for path in sorted(glob.glob(f"{method}_status_csvs/status_rank_*.csv")):
        with open(path) as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                src = int(row[0])
                sub = int(row[subunit_col]) if subunit_col is not None else None
                out[(src, sub)] = row[-1].strip()
    if method == "DoubleMinimization":
        # TS frame (side=0) is always 'converged' and has no CSV row.
        for (src, _) in list(out.keys()):
            out.setdefault((src, 0), "converged")
    return out


def patch_method(method):
    if not (os.path.isdir(f"{method}_status_csvs") and os.path.isdir(f"{method}_trajes")):
        return
    info_key, _ = METHOD_KEYS[method]
    status_map = load_status_map(method)
    task_name = read_task_name()
    print(f"[{method}] loaded {len(status_map)} status entries")
    if task_name is not None:
        print(f"[{method}] task_name from config.ini: {task_name!r}")

    for traj_path in sorted(glob.glob(f"{method}_trajes/collected_*.traj")):
        bak = traj_path + ".bak"
        if os.path.exists(bak):
            print(f"  skip {traj_path} (.bak already exists)")
            continue

        with Trajectory(traj_path, "r") as src:
            if len(src) == 0:
                print(f"  skip {traj_path} (empty)")
                continue
            first_info = src[0].info
            need_status = "status" not in first_info
            need_task = task_name is not None and "task_name" not in first_info
            if not need_status and not need_task:
                if task_name is not None:
                    print(f"  skip {traj_path} (status and task_name already present)")
                else:
                    print(f"  skip {traj_path} (status already present)")
                continue
            frames = [src[i] for i in range(len(src))]

        shutil.copy2(traj_path, bak)
        n_status = 0
        n_task = 0
        for img in frames:
            if need_status:
                src_idx = img.info.get("src_index")
                sub = None if info_key is None else img.info.get(info_key)
                if src_idx is not None and (info_key is None or sub is not None):
                    key = (src_idx, sub)
                    if key in status_map:
                        img.info["status"] = status_map[key]
                        n_status += 1
            if need_task:
                img.info["task_name"] = task_name
                n_task += 1

        os.remove(traj_path)
        with Trajectory(traj_path, "w") as out:
            for img in frames:
                out.write(img)

        msg_parts = []
        if need_status:
            msg_parts.append(f"status {n_status}/{len(frames)}")
        if need_task:
            msg_parts.append(f"task_name {n_task}/{len(frames)}")
        print(f"  {traj_path}: " + ", ".join(msg_parts))


if __name__ == "__main__":
    methods = sys.argv[1:] or list(METHOD_KEYS.keys())
    for m in methods:
        if m not in METHOD_KEYS:
            print(f"unknown method: {m}", file=sys.stderr)
            continue
        patch_method(m)
