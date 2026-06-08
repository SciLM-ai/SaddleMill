"""Build SaddleFlow "trajectory->saddle" datasets 2 and 3 from a SaddleMill
campaign (Dimer + DoubleMinimization + initbasinopt + basin-entry comparison).

Dataset 3 (all both-converged saddles):       18 rows/saddle
Dataset 2 (matched saddles, maxdev<0.3):       22 rows/saddle (= 18 + 4 initbasin)

Per-saddle rows, in order:
  0 saddle        is_saddle=1  energy-only (from dimer_opt log)   + eigenmode
  1 reactant_min               real E+F  (DM side -1 final)
  2 product_min                real E+F  (DM side +1 final)
  3-7   reactant_traj  DM side -1: start + 4 interior              (no E/F)
  8-12  product_traj   DM side +1: start + 4 interior              (no E/F)
  13-17 dimer          basin-entry + 4 interior (saddle dropped)   (no E/F)
  18-21 initbasin (D2) 4 interior (both endpoints dropped)         (no E/F)

Every trajectory is resampled to exactly 6 nodes, uniformly spaced in cumulative
minimum-image RMSD arc length, by *interpolation* between frames (endpoints
exact, interior interpolated). Stored positions are wrapped into the cell.

Each row's LMDB key_value_pairs: task_name=omat, group_id (=src*16+att, identical
in D2/D3), is_saddle, role, section, src_index, attempt_id, charge, spin, split.
data['info'] mirrors those (+ eigenmode on the saddle row). Energy/forces are set
on the 3 core rows only.

Two phases (run inside one salloc; see launch comment at bottom):
  --phase locmap : scan DM/init collected_opt output for (jobid,rank)+eigenmode,
                   grep Dimer CSVs for (rank,atom), assign group_id/split, sort by
                   zip-rank locality, persist location_map.csv + eigenmode_flat.npy.
  --phase build  : each SLURM task (one per node) takes a contiguous block, splits
                   it across worker processes (contiguous -> zip locality), and
                   each worker extracts/interpolates/writes its own D2+D3 LMDB
                   shard. Resumable via per-shard .done markers.
"""
import argparse, csv, glob, multiprocessing as mp, os, re, sys, tempfile, time, zipfile
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read

RANK_RE = re.compile(r"_rank_(\d+)\.traj$")
LOG_PREFIX = "MinModeTranslate:"
TASK = "omat"
_TARGETS = set()
_WANT_EIG = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def dimer_key(atoms):
    return dimer_key_from_info(atoms.info)


def dimer_key_from_info(info):
    """Dimer (src_index, attempt_id) = shallowest info/orig_info dict holding both."""
    n = info
    while isinstance(n, dict):
        if "src_index" in n and "attempt_id" in n:
            return (int(n["src_index"]), int(n["attempt_id"]))
        n = n.get("orig_info")
    return None


def parse_log(text):
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
        per.setdefault(step, curv)
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


def wrap(pos, cell, inv):
    fr = pos @ inv
    fr -= np.floor(fr)
    return fr @ cell


def arc_resample(frames, n_nodes=6):
    """6 positions uniform in MIC-RMSD arc length; endpoints exact, interior
    interpolated in unwrapped space; returns wrapped positions + arc length."""
    pos = [f.get_positions() for f in frames]
    cell = np.asarray(frames[0].cell.array, float)
    inv = np.linalg.inv(cell)
    unw = [pos[0].copy()]
    for i in range(1, len(pos)):
        d = pos[i] - pos[i - 1]
        fr = d @ inv; fr -= np.round(fr)
        unw.append(unw[-1] + fr @ cell)
    unw = np.array(unw)
    if len(unw) > 1:
        seg = np.sqrt(((unw[1:] - unw[:-1]) ** 2).sum(2).mean(1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
    else:
        cum = np.array([0.0])
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
    return [wrap(p, cell, inv) for p in out], total


class ZipCache:
    """Small LRU cache of open ZipFile handles + namelist sets."""
    def __init__(self, maxsize=8):
        self.maxsize = maxsize
        self.d = OrderedDict()

    def _get(self, path):
        z = self.d.get(path)
        if z is not None:
            self.d.move_to_end(path)
            return z
        try:
            zf = zipfile.ZipFile(path)
        except Exception:
            self.d[path] = (None, set())
            return self.d[path]
        names = set(zf.namelist())
        self.d[path] = (zf, names)
        if len(self.d) > self.maxsize:
            old, (oz, _) = self.d.popitem(last=False)
            try:
                if oz is not None:
                    oz.close()
            except Exception:
                pass
        return self.d[path]

    def read(self, path, name):
        zf, names = self._get(path)
        if zf is None or name not in names:
            return None
        return zf.read(name)

    def traj(self, path, name):
        data = self.read(path, name)
        if data is None:
            return None
        fd, t = tempfile.mkstemp(suffix=".traj"); os.close(fd)
        try:
            with open(t, "wb") as f:
                f.write(data)
            return read(t, index=":")
        finally:
            os.remove(t)


def group_id(src, att):
    return src * 16 + att


def split_of(gid):
    r = (gid * 2654435761) % 100
    return "test" if r < 5 else ("val" if r < 10 else "train")


# --------------------------------------------------------------------------- #
# phase: locmap
# --------------------------------------------------------------------------- #
def _scan_dm(path):
    """{(src,att): (top_src, rank, eigenmode_or_None)} from a DM output shard
    (eigenmode taken from the side==0 frame)."""
    m = RANK_RE.search(path); rank = int(m.group(1)) if m else -1
    hits = {}
    try:
        frames = read(path, index=":")
    except Exception:
        return hits
    for at in frames:
        k = dimer_key(at)
        if k not in _TARGETS:
            continue
        if k not in hits:
            hits[k] = [int(at.info.get("src_index")), rank, None]
        if _WANT_EIG and at.info.get("side") == 0 and hits[k][2] is None:
            eig = at.info.get("eigenmode")
            if eig is not None:
                hits[k][2] = np.asarray(eig, np.float32)
    return {k: tuple(v) for k, v in hits.items()}


def _scan_min(path):
    m = RANK_RE.search(path); rank = int(m.group(1)) if m else -1
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


def _scan(files, fn, workers, label):
    found = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for i, h in enumerate(pool.imap_unordered(fn, files), 1):
            found.update(h)
            if i % 50 == 0 or i == len(files):
                print(f"  [{label}] {i}/{len(files)} shards, {len(found):,} located", flush=True)
    return found


def phase_locmap(a):
    global _TARGETS, _WANT_EIG
    root = a.root
    _WANT_EIG = not a.no_eigenmode
    print(f"reading comparison {a.comparison_csv} ...", flush=True)
    label = {}
    with open(a.comparison_csv) as f:
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
    print(f"saddles: {len(label):,}", flush=True)

    dm_files = sorted(glob.glob(f"{root}/doubleopt/DoubleMinimization_trajes/collected_opt_rank_*.traj"))
    min_files = sorted(glob.glob(f"{root}/initbasinopt/Minimization_trajes/collected_opt_rank_*.traj"))
    if a.cap_shards:
        dm_files = dm_files[:a.cap_shards]
        min_files = min_files[:a.cap_shards * 5]
    t = time.perf_counter()
    dm_loc = _scan(dm_files, _scan_dm, a.workers, "dm")
    min_loc = _scan(min_files, _scan_min, a.workers, "init")
    print(f"scan done in {time.perf_counter()-t:.0f}s: dm={len(dm_loc):,} init={len(min_loc):,}", flush=True)

    print("grepping Dimer CSVs ...", flush=True)
    want = set(dm_loc)
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
                if (s, at_) in want and (s, at_) not in dimer_loc:
                    dimer_loc[(s, at_)] = (mpi, atom)
        if len(dimer_loc) >= len(want):
            break

    # assemble records (need DM + dimer; matched also needs init)
    records = []
    eigs = []
    eig_off = 0
    skipped = {"no_dimer": 0, "no_init_matched": 0}
    for k, (dm_job, dm_rank, eig) in dm_loc.items():
        if k not in dimer_loc:
            skipped["no_dimer"] += 1
            continue
        lab = label[k]
        if lab == "matched" and k not in min_loc:
            skipped["no_init_matched"] += 1
            continue
        d_rank, atom = dimer_loc[k]
        i_job, i_rank = min_loc.get(k, (-1, -1))
        gid = group_id(*k)
        if eig is not None and eig.size:
            nat = eig.shape[0]
            eigs.append(eig.reshape(-1))
            eo, en = eig_off, nat        # eo = float offset, en = #atoms
            eig_off += 3 * nat           # flat array is in floats (3 per atom)
        else:
            eo, en = -1, 0
        records.append((k[0], k[1], lab, d_rank, atom, dm_job, dm_rank,
                        i_job, i_rank, gid, split_of(gid), eo, en))
    # sort by zip locality
    records.sort(key=lambda r: (r[6], r[3], r[8]))  # dm_rank, dimer_rank, init_rank

    os.makedirs(a.out_dir, exist_ok=True)
    map_path = os.path.join(a.out_dir, "location_map.csv")
    with open(map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_index", "attempt_id", "label", "dimer_rank", "atom",
                    "dm_job", "dm_rank", "init_job", "init_rank", "group_id",
                    "split", "eig_off", "eig_nat"])
        w.writerows(records)
    if eigs:
        np.save(os.path.join(a.out_dir, "eigenmode_flat.npy"),
                np.concatenate(eigs).astype(np.float32))
    # split manifest
    with open(os.path.join(a.out_dir, "split_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "src_index", "attempt_id", "label", "split"])
        for rec in records:
            w.writerow([rec[9], rec[0], rec[1], rec[2], rec[10]])

    nmat = sum(1 for r in records if r[2] == "matched")
    print(f"\nlocation_map: {len(records):,} saddles ({nmat:,} matched / "
          f"{len(records)-nmat:,} unmatched)", flush=True)
    print(f"  skipped: {skipped}", flush=True)
    print(f"  eigenmodes: {len(eigs):,} stored", flush=True)
    sc = {"train": 0, "val": 0, "test": 0}
    for r in records:
        sc[r[10]] += 1
    print(f"  split: {sc}", flush=True)
    print(f"  -> {map_path}", flush=True)


# --------------------------------------------------------------------------- #
# phase: build
# --------------------------------------------------------------------------- #
def _ef(atoms):
    if atoms.calc is None:
        return None, None
    return atoms.calc.results.get("energy"), atoms.calc.results.get("forces")


def _row(positions, numbers, cell, constraints, gid, src, att, label, split,
         role, section, is_saddle, energy=None, forces=None, eigenmode=None):
    a = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    if constraints:
        a.set_constraint(constraints)
    if energy is not None or forces is not None:
        a.calc = SinglePointCalculator(a, energy=energy, forces=forces)
    info = {"task_name": TASK, "charge": 0, "spin": 0, "group_id": gid,
            "src_index": src, "attempt_id": att, "label": label, "split": split,
            "role": role, "section": section, "is_saddle": is_saddle}
    if eigenmode is not None:
        info["eigenmode"] = np.asarray(eigenmode, np.float32)
    a.info.update(info)
    kvp = {"task_name": TASK, "group_id": gid, "is_saddle": is_saddle,
           "role": role, "section": section, "src_index": src,
           "attempt_id": att, "split": split, "label": label}
    return a, kvp, info


def assemble(rec, zc, eig_flat, root):
    (src, att, label, d_rank, atom, dm_job, dm_rank,
     i_job, i_rank, gid, split, eo, en) = rec
    dz = f"{root}/Dimer_debug_zips/structure_rank_{d_rank}_data.zip"
    dmz = f"{root}/doubleopt/DoubleMinimization_debug_zips/structure_rank_{dm_rank}_data.zip"
    dm_m1 = zc.traj(dmz, f"optimization_{dm_job}_-1.traj")
    dm_p1 = zc.traj(dmz, f"optimization_{dm_job}_1.traj")
    dimer = zc.traj(dz, f"dimer_{src}_{att}_{atom}.traj")
    logb = zc.read(dz, f"dimer_opt_{src}_{att}_{atom}.log")
    if None in (dm_m1, dm_p1, dimer, logb):
        return None, None, "missing_debug"
    curv, sad_e = parse_log(logb.decode("utf-8", "replace"))
    fneg = first_neg(curv)
    if fneg is None or fneg >= len(dimer):
        return None, None, "bad_fneg"

    numbers = dimer[0].get_atomic_numbers()
    cell = np.asarray(dimer[0].cell.array, float)
    inv = np.linalg.inv(cell)
    cons = dimer[0].constraints
    eig = (np.asarray(eig_flat[eo:eo + 3 * en]).reshape(en, 3)
           if (eig_flat is not None and en > 0 and len(numbers) == en) else None)

    def mk(role, sec, pos, is_sad=0, e=None, f=None, em=None):
        return _row(wrap(np.asarray(pos, float), cell, inv), numbers, cell, cons,
                    gid, src, att, label, split, role, sec, is_sad, e, f, em)

    rmin, pmin = dm_m1[-1], dm_p1[-1]
    re_, rf_ = _ef(rmin); pe_, pf_ = _ef(pmin)
    rows = [
        mk("saddle", 0, dimer[-1].get_positions(), is_sad=1, e=sad_e, em=eig),
        mk("reactant_min", 0, rmin.get_positions(), e=re_, f=rf_),
        mk("product_min", 0, pmin.get_positions(), e=pe_, f=pf_),
    ]
    for q in arc_resample(dm_m1)[0][:5]:
        rows.append(mk("reactant_traj", 1, q))
    for q in arc_resample(dm_p1)[0][:5]:
        rows.append(mk("product_traj", 2, q))
    for q in arc_resample(dimer[fneg:])[0][:5]:
        rows.append(mk("dimer", 3, q))
    rows_d3 = rows  # 18

    rows_d2 = None
    if label == "matched":
        iz = f"{root}/initbasinopt/Minimization_debug_zips/structure_rank_{i_rank}_data.zip"
        init = zc.traj(iz, f"optimization_{i_job}.traj")
        if init is not None:
            rows_d2 = rows_d3 + [mk("initbasin", 4, q) for q in arc_resample(init)[0][1:5]]
    return rows_d3, rows_d2, None


def _write_rows(db, rows):
    for atoms, kvp, info in rows:
        db.write(atoms, **kvp, data={"info": info})


def build_worker(args):
    chunk, wid, out_dir, root, eig_path = args
    d3_path = os.path.join(out_dir, "dataset3", f"shard_{wid:05d}.aselmdb")
    d2_path = os.path.join(out_dir, "dataset2", f"shard_{wid:05d}.aselmdb")
    done = os.path.join(out_dir, ".done", f"shard_{wid:05d}")
    if os.path.exists(done):
        return wid, 0, 0, 0, "resumed"
    for p in (d3_path, d2_path):
        if os.path.exists(p):
            os.remove(p)
    eig_flat = np.load(eig_path, mmap_mode="r") if os.path.exists(eig_path) else None
    zc = ZipCache(8)
    n3 = n2 = errs = 0
    err_kinds = {}
    db3 = connect(d3_path, type="aselmdb")
    db2 = connect(d2_path, type="aselmdb")
    try:
        for rec in chunk:
            try:
                r3, r2, err = assemble(rec, zc, eig_flat, root)
            except Exception as e:
                err = f"exc:{type(e).__name__}"
                r3 = r2 = None
            if err:
                errs += 1
                err_kinds[err] = err_kinds.get(err, 0) + 1
                continue
            _write_rows(db3, r3); n3 += 1
            if r2 is not None:
                _write_rows(db2, r2); n2 += 1
    finally:
        db3.close(); db2.close()
    os.makedirs(os.path.dirname(done), exist_ok=True)
    open(done, "w").write(f"{n3} {n2} {errs} {err_kinds}\n")
    return wid, n3, n2, errs, str(err_kinds)


def phase_build(a):
    root = a.root
    node_rank = int(os.environ.get("SLURM_PROCID", "0"))
    num_nodes = int(os.environ.get("SLURM_NTASKS", str(a.num_nodes)))
    out_dir = a.out_dir
    for sub in ("dataset2", "dataset3", ".done"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    eig_path = os.path.join(out_dir, "eigenmode_flat.npy")

    records = []
    with open(os.path.join(out_dir, "location_map.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            records.append((int(row[0]), int(row[1]), row[2], int(row[3]), int(row[4]),
                            int(row[5]), int(row[6]), int(row[7]), int(row[8]),
                            int(row[9]), row[10], int(row[11]), int(row[12])))
    if a.max_saddles:
        records = records[:a.max_saddles]
    n = len(records)
    per = (n + num_nodes - 1) // num_nodes
    block = records[node_rank * per:(node_rank + 1) * per]
    W = a.workers
    csize = (len(block) + W - 1) // W
    tasks = []
    for w in range(W):
        chunk = block[w * csize:(w + 1) * csize]
        if not chunk:
            continue
        wid = node_rank * W + w
        tasks.append((chunk, wid, out_dir, root, eig_path))
    print(f"[node {node_rank}/{num_nodes}] {len(block):,} saddles -> {len(tasks)} workers",
          flush=True)
    import fairchem.core.datasets  # noqa: register aselmdb backend once (parent, pre-fork)
    t = time.perf_counter()
    tot3 = tot2 = tote = 0
    with mp.get_context("fork").Pool(len(tasks)) as pool:
        for wid, n3, n2, errs, ek in pool.imap_unordered(build_worker, tasks):
            tot3 += n3; tot2 += n2; tote += errs
    dt = time.perf_counter() - t
    print(f"[node {node_rank}] done in {dt/60:.1f} min: d3 groups={tot3:,} "
          f"d2 groups={tot2:,} errors={tote:,} ({tot3/max(dt,1):.1f} saddles/s)",
          flush=True)


def phase_verify(a):
    import fairchem.core.datasets  # noqa
    from ase.db import connect
    from collections import defaultdict
    ORDER = ["saddle", "reactant_min", "product_min", "reactant_traj",
             "product_traj", "dimer", "initbasin"]
    for ds, expect in [("dataset3", 18), ("dataset2", 22)]:
        shards = sorted(glob.glob(os.path.join(a.out_dir, ds, "*.aselmdb")))[:a.verify_shards]
        if not shards:
            print(f"\n{ds}: no shards"); continue
        groups = defaultdict(list)
        for sh in shards:
            with connect(sh) as db:
                for row in db.select():
                    groups[row.get("group_id")].append(row)
        sizes = {}
        for rws in groups.values():
            sizes[len(rws)] = sizes.get(len(rws), 0) + 1
        print(f"\n{ds}: {len(groups):,} groups in {len(shards)} shards; "
              f"row-count histogram={sizes} (expect all {expect})")
        if not groups:
            print("  (no groups — nothing to inspect)")
            continue
        gid = next(iter(groups))
        rws = sorted(groups[gid], key=lambda r: r.id)
        print(f"  sample group_id={gid}:")
        roles = []
        for r in rws:
            info = r.data["info"]
            roles.append(info["role"])
            e = getattr(r, "energy", None)
            try:
                f = r.toatoms().get_forces(); fmax = float(np.abs(f).max())
            except Exception:
                fmax = None
            if info["role"] in ("saddle", "reactant_min", "product_min"):
                print(f"    {info['role']:13s} is_saddle={info['is_saddle']} "
                      f"sec={info['section']} split={info['split']} "
                      f"energy={e} max|F|={fmax} eig={'eigenmode' in info}")
        a0 = rws[0].toatoms()
        fr = a0.get_scaled_positions(wrap=False)
        wrapped = bool(fr.min() >= -1e-6 and fr.max() < 1 + 1e-6)
        order_ok = roles[:3] == ["saddle", "reactant_min", "product_min"]
        size_ok = all(s == expect for s in sizes)
        print(f"  CHECKS: size_ok={size_ok}  order_ok={order_ok}  "
              f"row0_wrapped={wrapped}  saddle_is_flagged={rws[0].get('is_saddle')==1}")


def _d1split_worker(shard):
    out = []
    with connect(shard) as db:
        for row in db.select("side=-1"):  # reactant frame = one triplet
            if row.get("side") != -1:
                continue
            info = row.data.get("info", {}) if row.data else {}
            k = dimer_key_from_info(info)
            ms = row.get("ms_id")
            if k is not None and ms is not None:
                out.append((int(ms), k[0], k[1]))
    return out


def phase_d1split(a):
    """Emit a dataset-1 split manifest consistent with datasets 2/3: read the
    dataset-1 triplet LMDB, recover each triplet's Dimer (src,attempt) from
    data['info'].orig_info, and apply the same split_of(group_id)."""
    import fairchem.core.datasets  # noqa: register aselmdb backend (parent, pre-fork)
    d1_dir = a.d1_dir or f"{a.root}/doubleopt/aselmdb_no-EF"
    out = a.d1_manifest or os.path.join(a.out_dir, "dataset1_split_manifest.csv")
    shards = sorted(glob.glob(os.path.join(d1_dir, "*.aselmdb")))
    if not shards:
        sys.exit(f"no dataset-1 shards under {d1_dir}")
    print(f"reading {len(shards)} dataset-1 shards from {d1_dir} ...", flush=True)
    rows = []
    with mp.get_context("fork").Pool(min(a.workers, len(shards))) as pool:
        for i, part in enumerate(pool.imap_unordered(_d1split_worker, shards), 1):
            rows.extend(part)
            print(f"  {i}/{len(shards)} shards, {len(rows):,} triplets", flush=True)
    rows.sort()  # by ms_id -> positional triplet order
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sc = {"train": 0, "val": 0, "test": 0}
    dups = set()
    ndup = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["triplet_index", "ms_id_R", "src_index", "attempt_id", "group_id", "split"])
        for ti, (ms, src, att) in enumerate(rows):
            gid = group_id(src, att)
            sp = split_of(gid)
            sc[sp] = sc.get(sp, 0) + 1
            if gid in dups:
                ndup += 1
            dups.add(gid)
            w.writerow([ti, ms, src, att, gid, sp])
    print(f"\nwrote {out}", flush=True)
    print(f"  {len(rows):,} triplets, {len(dups):,} unique group_ids "
          f"({ndup:,} triplets share a group_id — expected 0 for 1 attempt/saddle)",
          flush=True)
    print(f"  split: {sc}", flush=True)


def main():
    ncpu = len(os.sched_getaffinity(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["locmap", "build", "verify", "d1split"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--comparison-csv", default=None)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--num-nodes", type=int, default=4)
    ap.add_argument("--no-eigenmode", action="store_true")
    ap.add_argument("--cap-shards", type=int, default=0, help="locmap: cap DM shards (testing)")
    ap.add_argument("--max-saddles", type=int, default=0, help="build: cap saddles (testing)")
    ap.add_argument("--verify-shards", type=int, default=8,
                    help="verify: number of shards to sample per dataset")
    ap.add_argument("--d1-dir", default=None,
                    help="d1split: dataset-1 LMDB dir (default <root>/doubleopt/aselmdb_no-EF)")
    ap.add_argument("--d1-manifest", default=None,
                    help="d1split: output CSV (default <out-dir>/dataset1_split_manifest.csv)")
    ap.add_argument("-j", "--workers", type=int, default=min(64, ncpu))
    a = ap.parse_args()
    if a.comparison_csv is None:
        a.comparison_csv = f"{a.root}/basin_entry_vs_doublemin/comparison_rmsd.csv"
    if a.phase == "locmap":
        phase_locmap(a)
    elif a.phase == "build":
        phase_build(a)
    elif a.phase == "verify":
        phase_verify(a)
    else:
        phase_d1split(a)


if __name__ == "__main__":
    main()

# Launch (4 nodes, 4 h):
#   salloc -N 4 -C cpu -q interactive -t 4:00:00 -A m1883 bash -c '
#     set -e
#     srun -N1 -n1 python scripts/make_traj_datasets.py --phase locmap \
#       --root <ROOT> --out-dir <OUT> -j 128
#     srun -N4 --ntasks-per-node=1 python scripts/make_traj_datasets.py --phase build \
#       --root <ROOT> --out-dir <OUT> --num-nodes 4 -j 64
#   '
