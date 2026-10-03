import csv
import os, re, glob, copy, pathlib, zipfile
from ase.io import Trajectory

from saddlemill.config_defaults import DEFAULTS, RENAMED_KEYS
from saddlemill.config_factories import (
    load_calculator_factory,
    load_method_factory,
    load_optimizer_factory,
)
from saddlemill.config_parsing import merge_config_file, migrate_renamed_keys, parse_value
from saddlemill.config_validation import normalize_dimer_method_config, validate_method_config

VALID_RUN_CATEGORIES = frozenset({"converged", "not_converged", "errored", "remaining"})
_RUN_CATEGORY_ALIASES = {"not_started": "remaining", "error": "errored"}

class ConfigManager:
    # 1. Define your Safe Defaults here
    DEFAULTS = DEFAULTS

    def __init__(self, config_file="config.ini"):
        self._config = copy.deepcopy(self.DEFAULTS)
        self.user_config_file = config_file
        
        # Load user config if it exists
        if os.path.exists(self.user_config_file):
            self._load_from_file()
        else:
            print(f"Warning: {self.user_config_file} not found. Using default parameters.")

    def _load_from_file(self):
        """Load the INI file through the compatibility parsing helper."""
        merge_config_file(self._config, self.user_config_file, self.DEFAULTS)

    # Renamed keys remain visible on the facade for compatibility.
    _RENAMED_KEYS = list(RENAMED_KEYS)

    def _migrate_renamed_keys(self):
        """Silently migrate old config key names to new names."""
        migrate_renamed_keys(self._config)

    def _parse_value(self, val):
        """Interpret one config value using the historical coercion rules."""
        return parse_value(val)

    def __getitem__(self, key):
        """Allow dict-like access: config['Main']"""
        return self._config.get(key, {})

    def __contains__(self, key):
        return key in self._config

    def __iter__(self):
        return iter(self._config)


    def get(self, key, fallback=None):
        """
        Standard dict-like get. 
        Usage: config.get("Main", {}) 
        """
        return self._config.get(key, fallback)

    def get_value(self, section, key, fallback=None):
        """
        Specific helper to get a value deep inside a section.
        Usage: config.get_value("Main", "fmax", 0.05)
        """
        return self._config.get(section, {}).get(key, fallback)

    @property
    def as_dict(self):
        """Return the raw dictionary."""
        return self._config

    def __str__(self):
        """Enables pretty printing via print(config)"""
        import json
        # default=str handles objects that aren't natively JSON serializable
        return json.dumps(self._config, indent=4, default=str)

# --- Helper function to mimic your old parse_inputfile ---
def load_config(path="config.ini"):
    return ConfigManager(path)


def load_calculator(config_dict):
    return load_calculator_factory(config_dict["Main"]["Calculator"])




def load_method(config_dict):
    method_name = validate_method_config(
        config_dict, normalize_run_jobs=_normalize_run_jobs
    )
    return load_method_factory(method_name)


def _load_optimizer(optimizer_name):
    return load_optimizer_factory(optimizer_name)


def load_optimizer(config_dict):
    Optimizer = _load_optimizer(config_dict["Main"]["Optimizer"])
    if config_dict["Main"]["method"] == "NEB":
        if config_dict["ourNEB"]["endpoint_relax_Optimizer"] is None:
            return Optimizer, Optimizer
        endpoint_relax_Optimizer = _load_optimizer(
            config_dict["ourNEB"]["endpoint_relax_Optimizer"]
        )
        return endpoint_relax_Optimizer, Optimizer
    return Optimizer


def get_trajes_and_indices(config_dict):

    main_cfg = config_dict.get("Main", {})
    dir_path = os.path.expandvars(os.path.expanduser(main_cfg.get("dir_path", ".")))
    input_format = main_cfg.get("input_format", "traj")
    method_name = main_cfg.get("method")

    if input_format == "lmdb":
        # LMDB inputs are only supported for SinglePoint (enforced in load_method).
        # fairchem.core.datasets must be imported so ase.db recognizes the aselmdb backend.
        import fairchem.core.datasets  # noqa: F401
        from ase.db import connect

        fpj = config_dict["ourSinglePoint"].get("frames_per_job", 1)
        input_pattern = os.path.join(dir_path, "**", "*.aselmdb")
        all_lmdb_files = sorted(glob.glob(input_pattern, recursive=True))

        trajes_and_idxs = []
        for lmdb_path in all_lmdb_files:
            db = connect(lmdb_path, type='aselmdb', readonly=True)
            n_rows = db.count()
            # ASE LMDB ids are 1-indexed and dense. The last chunk may be smaller
            # than fpj — user is responsible for choosing fpj that keeps the
            # leftover meaningful for their use case (e.g. multiple of 3 for triplets).
            for start in range(1, n_rows + 1, fpj):
                end = min(start + fpj, n_rows + 1)
                trajes_and_idxs.append([lmdb_path, start, end])
        return trajes_and_idxs

    input_pattern = os.path.join(dir_path, "**", "*.traj")
    all_traj_files = sorted(glob.glob(input_pattern, recursive=True))

    if config_dict["ourNEB"]["images_location_in_input_traj"] in (":", -1):
        traj_lens = []
        for traj_name in all_traj_files:
            with Trajectory(traj_name, 'r') as traj:
                traj_lens.append(len(traj))

    if method_name == "NEB":
        if config_dict["ourNEB"]["only_endpoints_in_input_traj"]:
            nimages = 2
        else:
            nimages = config_dict["ourNEB"]["num_frames"]
    elif method_name == "SinglePoint":
        nimages = config_dict["ourSinglePoint"].get("frames_per_job", 1)
    else:
        nimages = 1

    trajes_and_idxs = []
    for i, traj_name in enumerate(all_traj_files):
        if config_dict["ourNEB"]["images_location_in_input_traj"] == 0:
            trajes_and_idxs.append([traj_name, 0, nimages])
        elif config_dict["ourNEB"]["images_location_in_input_traj"] == -1:
            trajes_and_idxs.append([traj_name, traj_lens[i]-nimages, traj_lens[i]])
        elif config_dict["ourNEB"]["images_location_in_input_traj"] == ":":
            traj_len = traj_lens[i]
            if method_name == "SinglePoint":
                # SP: allow a smaller final batch.
                for start in range(0, traj_len, nimages):
                    trajes_and_idxs.append([traj_name, start, min(start + nimages, traj_len)])
            else:
                if traj_len%nimages != 0: raise ValueError(f"Can't divide a traj file with {traj_len} atoms objects into batches of {nimages} atoms objects")
                for j in range(traj_len//nimages):
                    trajes_and_idxs.append([traj_name, j*nimages, (j+1)*nimages])

    return trajes_and_idxs


def create_results_directories(config_dict):
    method_name = config_dict["Main"]["method"]
    dirs = [f"{method_name}_status_csvs"]
    if method_name == "SinglePoint":
        # SP output dir matches input_format. Debug zips only for VASP — DFT leaves
        # real artifacts worth keeping (FAIRChem SP produces none, so no zip dir).
        input_format = config_dict["Main"].get("input_format", "traj")
        out_subdir = "lmdbs" if input_format == "lmdb" else "trajes"
        dirs.append(f"{method_name}_{out_subdir}")
        if config_dict["Main"]["Calculator"] in ("Vasp", "VaspInteractive"):
            dirs.append(f"{method_name}_debug_zips")
    else:
        dirs.extend([f"{method_name}_trajes", f"{method_name}_debug_zips"])
        if method_name == "DoubleMinimization":
            dirs.append(f"{method_name}_timing_csvs")
        if method_name in {"Minimization", "DoubleMinimization"} and str(
            config_dict["Main"].get("Optimizer", "")
        ).lower() in {"firelbfgs", "fire_lbfgs", "warmfirelbfgs"}:
            dirs.append(f"{method_name}_optimizer_csvs")
    for d in dirs:
        pathlib.Path(d).mkdir(exist_ok=False)


def read_status_csv_rows(method_name, directory="."):
    """Read all status CSV rows. Returns list of lists of strings (one per row)."""
    csv_dir = os.path.join(directory, f"{method_name}_status_csvs")
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(csv_dir, "status_rank_*.csv"))):
        with open(csv_path) as fh:
            for row in csv.reader(fh):
                if row:
                    rows.append(row)
    return rows


def _normalize_run_jobs(run_jobs_value):
    """Convert parsed run_jobs config value into a set of job categories."""
    if isinstance(run_jobs_value, str):
        if run_jobs_value == "all":
            return set(VALID_RUN_CATEGORIES)
        cats = {run_jobs_value}
    elif isinstance(run_jobs_value, list):
        cats = {str(c) for c in run_jobs_value}
    else:
        raise ValueError(f"Invalid run_jobs value: {run_jobs_value!r}")
    cats = {_RUN_CATEGORY_ALIASES.get(c, c) for c in cats}
    invalid = cats - VALID_RUN_CATEGORIES
    if invalid:
        raise ValueError(
            f"Invalid run_jobs categories: {invalid}. "
            f"Valid: {sorted(VALID_RUN_CATEGORIES)} or 'all'")
    return cats


def _categorize_status(status):
    """Categorize a single status string into a run_jobs category."""
    if status.startswith("converged"):
        return "converged"
    if status.startswith("error"):
        return "errored"
    if status.startswith("not_converged"):
        return "not_converged"
    return "errored"


def _categorize_statuses(statuses, method_name=None):
    """Return the set of categories for a job based on its status lines.

    For NEB (band-level): 'converged' only if ALL sub-bands are converged/converged_CI.
    For other methods: ANY matching status adds its category (original behavior).
    """
    cats_per_line = [_categorize_status(s) for s in statuses]
    if method_name == "NEB":
        # Band-level: converged only if ALL sub-bands converged
        result = set()
        if all(c == "converged" for c in cats_per_line):
            result.add("converged")
        if any(c == "not_converged" for c in cats_per_line):
            result.add("not_converged")
        if any(c == "errored" for c in cats_per_line):
            result.add("errored")
        return result
    return set(cats_per_line)

def _expected_dimer_entries(config_dict):
    """Total attempt slots produced by the configured Dimer generator contract.

    ``initial_guess`` is exclusive and always produces exactly one attempt.
    Otherwise a scalar count applies to every reaction type, while a parsed
    list must provide one position-aligned count per type, matching
    ``dimertools.structure_edit._resolve_attempts_per_type``.  ``None`` is
    returned only when reaction types are not configured.
    """
    rt = config_dict["ourDimer"].get("reaction_types")
    if not rt:
        return None
    types = rt.split() if isinstance(rt, str) else list(rt)
    if "initial_guess" in types:
        return 1

    raw = config_dict["ourDimer"].get("num_attempts_per_type", 1)
    if isinstance(raw, list):
        counts = [int(x) for x in raw]
        if len(counts) != len(types):
            raise ValueError(
                f"[ourDimer] num_attempts_per_type has {len(counts)} values but "
                f"reaction_types has {len(types)} entries. Give one count per "
                "type (aligned by position), or a single integer for all."
            )
    else:
        n = int(raw) if raw is not None else 1
        counts = [n] * len(types)
    return sum(counts)

def _get_subunit_config(method_name):
    """Return (CSV column index, trajectory info key) for method sub-units."""
    if method_name == "Dimer":
        return 2, "attempt_id"
    if method_name == "NEB":
        return 2, "subband_idx"
    if method_name == "DoubleMinimization":
        return 2, "side"
    return None, None


def _subunit_namespace(method_name):
    """Stable namespace used to keep method/sub-unit identities explicit."""
    return {
        "Dimer": "attempt_id",
        "NEB": "subband_idx",
        "DoubleMinimization": "side",
    }.get(method_name, "job")


def _expected_subunit_ids(config_dict):
    """Return the exact current sub-unit identity set when it is knowable."""
    method_name = config_dict["Main"]["method"]
    if method_name == "Dimer":
        count = _expected_dimer_entries(config_dict)
        return None if count is None else frozenset(range(count))
    if method_name == "DoubleMinimization":
        # The TS frame (side=0) is an output/continuation endpoint, not a status
        # sub-unit. The two independently resumable work identities are +/-1.
        return frozenset({-1, 1})
    return None


def _resume_identity(method_name, job_id, subunit_id=None):
    """Return a method-qualified resume identity tuple."""
    return (method_name, int(job_id), _subunit_namespace(method_name), subunit_id)


def _status_record(method_name, row, path, line_num):
    """Validate one status row and return its explicit resume identity."""
    subunit_col, _ = _get_subunit_config(method_name)
    minimum_columns = 3 if subunit_col is None else subunit_col + 2
    if len(row) < minimum_columns:
        raise ValueError(
            f"Unreadable/truncated {method_name} status row at {path}:{line_num}: "
            f"expected at least {minimum_columns} columns, got {len(row)}"
        )
    try:
        job_id = int(row[0])
        int(row[1])  # rank; validate but do not make it part of scientific identity
        subunit_id = int(row[subunit_col]) if subunit_col is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {method_name} status identity at {path}:{line_num}: {row!r}"
        ) from exc
    status = row[-1].strip()
    if not status:
        raise ValueError(f"Empty {method_name} status at {path}:{line_num}")
    return {
        "identity": _resume_identity(method_name, job_id, subunit_id),
        "job_id": job_id,
        "subunit_id": subunit_id,
        "status": status,
        "row": row,
        "path": path,
        "line_num": line_num,
    }


def _read_resume_status_records(method_name):
    """Strictly read active status shards; malformed CSVs fail closed."""
    status_dir = f"{method_name}_status_csvs"
    csv_files = sorted(glob.glob(os.path.join(status_dir, "status_rank_*.csv")))
    records = []
    for path in csv_files:
        try:
            with open(path, newline="") as handle:
                reader = csv.reader(handle, strict=True)
                for row in reader:
                    if not row:
                        continue
                    records.append(_status_record(method_name, row, path, reader.line_num))
        except csv.Error as exc:
            raise ValueError(f"Unreadable/truncated status CSV {path}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Unable to read status CSV {path}: {exc}") from exc
    return csv_files, records


def _effective_status_records(records):
    """Resolve duplicate status identities deterministically with last-row wins.

    Status shards are append-oriented and executor retries can legitimately leave
    repeated identities.  The last row in deterministic (path, line) order is
    authoritative for selection, while all physical duplicate rows are removed
    together if that identity is later archived for redo.
    """
    import warnings

    effective = {}
    for record in records:
        identity = record["identity"]
        previous = effective.get(identity)
        if previous is not None:
            warnings.warn(
                "Duplicate resume status identity "
                f"{identity!r}: {previous['path']}:{previous['line_num']} "
                f"({previous['status']!r}) -> {record['path']}:{record['line_num']} "
                f"({record['status']!r}); last row wins deterministically.",
                RuntimeWarning,
                stacklevel=3,
            )
        effective[identity] = record
    return effective


def _compute_redo_info(job_ids, config_dict, categories, effective):
    """Return exact per-job work identities selected by the resume policy."""
    method_name = config_dict["Main"]["method"]
    job_ids = [int(jid) for jid in job_ids]
    expected = _expected_subunit_ids(config_dict)
    by_job = {}
    for record in effective.values():
        by_job.setdefault(record["job_id"], []).append(record)

    redo_info = {}
    if method_name == "NEB":
        for jid in job_ids:
            records = by_job.get(jid, [])
            if not records:
                if "remaining" in categories:
                    # None is an internal full-band sentinel; NEB itself does not
                    # consume entries_to_run and continuation extraction retains
                    # every frame/end point for the job.
                    redo_info[jid] = {None}
                continue
            cats = _categorize_statuses([r["status"] for r in records], method_name)
            if cats & categories:
                redo_info[jid] = {r["subunit_id"] for r in records}
        return redo_info

    if expected is not None:
        for jid in job_ids:
            records = [r for r in by_job.get(jid, []) if r["subunit_id"] in expected]
            seen = {r["subunit_id"] for r in records}
            selected = {
                r["subunit_id"]
                for r in records
                if _categorize_status(r["status"]) in categories
            }
            if "remaining" in categories:
                selected.update(expected - seen)
            if selected:
                redo_info[jid] = selected
        return redo_info

    subunit_col, _ = _get_subunit_config(method_name)
    if subunit_col is not None:
        # Defensive legacy path for a sub-unit method whose expected set cannot
        # be determined from the current config (normally only invalid Dimer
        # configs). Do not invent an identity range.
        for jid in job_ids:
            records = by_job.get(jid, [])
            selected = {
                r["subunit_id"]
                for r in records
                if _categorize_status(r["status"]) in categories
            }
            if selected:
                redo_info[jid] = selected
        return redo_info

    # Single-unit methods: one explicit job identity.  A missing status can still
    # have an orphan trajectory from an interrupted append; returning {None}
    # lets continuation extraction recover that output before cleanup.
    for jid in job_ids:
        records = by_job.get(jid, [])
        if records:
            if _categorize_status(records[-1]["status"]) in categories:
                redo_info[jid] = {None}
        elif "remaining" in categories:
            redo_info[jid] = {None}
    return redo_info


def _warn_unexpected_subunits(config_dict, effective):
    import warnings

    expected = _expected_subunit_ids(config_dict)
    if expected is None:
        return
    method_name = config_dict["Main"]["method"]
    unexpected = sorted({
        (record["job_id"], record["subunit_id"])
        for record in effective.values()
        if record["subunit_id"] not in expected
    })
    if unexpected:
        warnings.warn(
            f"Preserving unexpected {method_name} resume identities outside the "
            f"current expected {_subunit_namespace(method_name)} set: {unexpected!r}. "
            "They do not count toward completion and are not selected for redo.",
            RuntimeWarning,
            stacklevel=3,
        )


def get_remaining_trajes(trajes_and_idxs, config_dict):
    categories_to_run = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
    method_name = config_dict["Main"]["method"]
    _, records = _read_resume_status_records(method_name)
    effective = _effective_status_records(records)
    _warn_unexpected_subunits(config_dict, effective)

    redo_info = _compute_redo_info(
        range(len(trajes_and_idxs)), config_dict, categories_to_run, effective
    )
    remaining = [
        [idx, item]
        for idx, item in enumerate(trajes_and_idxs)
        if idx in redo_info
    ]
    if not remaining:
        return [], []
    job_ids, selected = zip(*remaining)
    return list(job_ids), list(selected)


def build_redo_info(job_ids, config_dict):
    """Return exact method-qualified sub-unit identities selected for redo.

    The public return shape remains ``{job_id: set(subunit_ids)}``:
      - Dimer: exact current attempt IDs;
      - DoubleMinimization: exact sides -1/+1;
      - NEB: all known sub-bands (or ``{None}`` for an orphan/no-status full band);
      - single-unit methods: ``{None}``.
    """
    categories_to_run = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
    method_name = config_dict["Main"]["method"]
    _, records = _read_resume_status_records(method_name)
    effective = _effective_status_records(records)
    _warn_unexpected_subunits(config_dict, effective)
    return _compute_redo_info(job_ids, config_dict, categories_to_run, effective)


def _resume_archive_boundary(name):
    """No-op hook used by Task-07 interruption/fault-injection tests."""


def _fsync_parent(path):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_zip_member(zf, info):
    import hashlib

    digest = hashlib.sha256()
    with zf.open(info, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_archive_path(directory, prefix="previous_"):
    os.makedirs(directory, exist_ok=True)
    index = 0
    while os.path.exists(os.path.join(directory, f"{prefix}{index}.zip")):
        index += 1
    return os.path.join(directory, f"{prefix}{index}.zip"), index


def _write_validated_backup_zip(archive_path, files, *, arcnames=None, extra_entries=None):
    """Create, validate, fsync, and atomically publish a full backup ZIP."""
    import uuid

    files = list(files)
    arcnames = dict(arcnames or {})
    extra_entries = dict(extra_entries or {})
    expected = {}
    for path in files:
        expected[arcnames.get(path, os.path.basename(path))] = _sha256_file(path)

    temp = os.path.join(
        os.path.dirname(archive_path),
        f".{os.path.basename(archive_path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, arcnames.get(path, os.path.basename(path)))
            for name, payload in extra_entries.items():
                zf.writestr(name, payload)
        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())

        with zipfile.ZipFile(temp, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"Backup ZIP validation failed for member {bad!r}")
            infos = {info.filename: info for info in zf.infolist()}
            for name, sha in expected.items():
                if name not in infos:
                    raise ValueError(f"Backup ZIP missing expected member {name!r}")
                if _sha256_zip_member(zf, infos[name]) != sha:
                    raise ValueError(f"Backup ZIP hash mismatch for member {name!r}")
        _resume_archive_boundary(f"backup_validated:{archive_path}")
        os.replace(temp, archive_path)
        _fsync_parent(archive_path)
        _resume_archive_boundary(f"backup_published:{archive_path}")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return archive_path


def _atomic_write_csv_rows(path, rows):
    """Rewrite one CSV via validated same-directory replacement."""
    import uuid

    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with open(temp, "w", newline="") as handle:
            csv.writer(handle).writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        with open(temp, newline="") as handle:
            parsed = [row for row in csv.reader(handle, strict=True) if row]
        if parsed != rows:
            raise ValueError(f"CSV rewrite validation mismatch for {path}")
        _resume_archive_boundary(f"csv_rewrite_validated:{path}")
        os.replace(temp, path)
        _fsync_parent(path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _hessian_managed_paths(job_ids):
    """Return active Task-06 files bounded to the selected Hessian job IDs."""
    paths = set()
    for jid in sorted(set(int(x) for x in job_ids)):
        exact = [
            f"Hessian_artifacts/commits/hessian_{jid}.json",
            f"Hessian_artifacts/pending/hessian_{jid}.json",
            f"Hessian_summary_csvs/hessian_{jid}.csv",
            f"Hessian_hessians/hessian_{jid}.npz",
        ]
        patterns = [
            f"Hessian_artifacts/events/hessian_{jid}.*.traj",
            f"Hessian_artifacts/events/.hessian_{jid}.*.stage.*.traj",
            f"Hessian_summary_csvs/.hessian_{jid}.csv.stage.*",
            f"Hessian_hessians/.hessian_{jid}.npz.stage.*",
        ]
        paths.update(path for path in exact if os.path.exists(path))
        for pattern in patterns:
            paths.update(glob.glob(pattern))
    return sorted(paths)


def _legacy_hessian_indexes_with_jobs(job_ids):
    """Validate Task-06 legacy snapshots and return those needing filtering."""
    selected = set(int(x) for x in job_ids)
    status_paths, traj_paths = [], []
    for path in sorted(glob.glob("Hessian_artifacts/legacy_indexes/status_rank_*.csv")):
        has_selected = False
        try:
            with open(path, newline="") as handle:
                reader = csv.reader(handle, strict=True)
                for row in reader:
                    if not row:
                        continue
                    if len(row) < 3:
                        raise ValueError(f"truncated legacy Hessian status row: {row!r}")
                    if int(row[0]) in selected:
                        has_selected = True
        except Exception as exc:
            raise ValueError(f"Unreadable Task-06 legacy status snapshot {path}: {exc}") from exc
        if has_selected:
            status_paths.append(path)

    for path in sorted(glob.glob("Hessian_artifacts/legacy_indexes/collected_hessian_rank_*.traj")):
        has_selected = False
        try:
            with Trajectory(path, "r") as traj:
                for idx in range(len(traj)):
                    info = getattr(traj[idx], "info", {}) or {}
                    if "src_index" not in info:
                        raise ValueError(f"frame {idx} lacks top-level src_index")
                    if int(info["src_index"]) in selected:
                        has_selected = True
        except Exception as exc:
            raise ValueError(f"Unreadable Task-06 legacy trajectory snapshot {path}: {exc}") from exc
        if has_selected:
            traj_paths.append(path)
    return status_paths, traj_paths


def _archive_and_invalidate_hessian_artifacts(job_ids):
    """Archive/invalidate Task-06 completion state for an explicit forced redo.

    Task-06 commit identity remains authoritative; this helper does not invent a
    competing completion marker.  It only retires active commit/pending/event/fixed
    outputs after a validated backup, and removes the same jobs from legacy index
    snapshots so Task-06 index reconstruction cannot resurrect pre-redo evidence.
    """
    import json

    job_ids = sorted(set(int(x) for x in job_ids))
    if not job_ids:
        return None
    managed = _hessian_managed_paths(job_ids)
    legacy_status, legacy_traj = _legacy_hessian_indexes_with_jobs(job_ids)
    files_to_backup = sorted(set(managed + legacy_status + legacy_traj))
    if not files_to_backup:
        return None

    archive_path, archive_index = _next_archive_path("Hessian_artifacts", "redo_previous_")
    arcnames = {path: path.replace(os.sep, "/") for path in files_to_backup}
    manifest = {
        "schema": "task07_hessian_forced_redo_archive_v1",
        "jobs": job_ids,
        "checkpoint_contract": "Task-06 active commit/pending/event state retired for forced redo",
        "files": [
            {"path": arcnames[path], "sha256": _sha256_file(path)}
            for path in files_to_backup
        ],
    }
    _write_validated_backup_zip(
        archive_path,
        files_to_backup,
        arcnames=arcnames,
        extra_entries={
            "TASK07_REDO_MANIFEST.json": json.dumps(
                manifest, sort_keys=True, indent=2
            ).encode() + b"\n"
        },
    )

    selected = set(job_ids)
    for path in legacy_status:
        with open(path, newline="") as handle:
            rows = [row for row in csv.reader(handle, strict=True) if row]
        kept = [row for row in rows if int(row[0]) not in selected]
        if kept:
            _atomic_write_csv_rows(path, kept)
        else:
            _resume_archive_boundary(f"legacy_status_remove:{path}")
            os.unlink(path)
            _fsync_parent(path)

    for path in legacy_traj:
        _atomic_filter_trajectory(
            path,
            lambda atoms: int(atoms.info["src_index"]) not in selected,
            validation_label="Task-06 legacy Hessian index",
        )

    invalidated_root = os.path.join(
        "Hessian_artifacts", "redo_invalidated", f"redo_{archive_index}"
    )
    for path in managed:
        if not os.path.exists(path):
            continue
        destination = os.path.join(invalidated_root, path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _resume_archive_boundary(f"hessian_invalidate:{path}")
        os.replace(path, destination)
        _fsync_parent(path)
        _fsync_parent(destination)
    return archive_path


def archive_and_clean_csvs(config_dict, job_ids, categories_to_clean):
    """Archive status shards and clean exactly the identities selected for redo.

    Duplicate status identities use deterministic last-row selection, but every
    physical row for a selected identity is removed together. Missing expected
    identities are returned in ``cleaned`` so orphan output frames can be archived
    before rerun. Unexpected Dimer/DoubleMin identities are preserved.
    """
    if not job_ids:
        return {}
    method_name = config_dict["Main"]["method"]
    categories_to_clean = set(categories_to_clean)
    csv_files, records = _read_resume_status_records(method_name)
    effective = _effective_status_records(records)
    _warn_unexpected_subunits(config_dict, effective)
    cleaned = _compute_redo_info(job_ids, config_dict, categories_to_clean, effective)
    if not cleaned:
        return {}

    # Resolve the raw rows to remove. NEB redo is always full-band; other
    # sub-unit methods remove all duplicate rows for only the selected identity.
    target_identities = set()
    if method_name == "NEB":
        selected_jobs = set(cleaned)
        target_identities = {
            record["identity"] for record in records if record["job_id"] in selected_jobs
        }
    else:
        for jid, subunits in cleaned.items():
            for subunit in subunits:
                target_identities.add(_resume_identity(method_name, jid, subunit))

    rows_to_remove = [r for r in records if r["identity"] in target_identities]
    if not rows_to_remove:
        return cleaned

    # A status-backed Hessian selection is an explicit forced redo. Retire the
    # Task-06 reusable commit state *before* mutating the status index so a fault
    # cannot make a still-valid commit silently win the requested redo.
    if method_name == "Hessian":
        forced_jobs = sorted({r["job_id"] for r in rows_to_remove})
        _archive_and_invalidate_hessian_artifacts(forced_jobs)

    status_dir = f"{method_name}_status_csvs"
    archive_path, _ = _next_archive_path(status_dir)
    _write_validated_backup_zip(archive_path, csv_files)

    target = set(target_identities)
    by_path = {path: [] for path in csv_files}
    for record in records:
        by_path[record["path"]].append(record)
    for path in csv_files:
        file_records = by_path.get(path, [])
        kept = [record["row"] for record in file_records if record["identity"] not in target]
        original_count = len(file_records)
        if len(kept) == original_count:
            continue
        if kept:
            _atomic_write_csv_rows(path, kept)
        else:
            _resume_archive_boundary(f"csv_remove:{path}")
            os.unlink(path)
            _fsync_parent(path)
    return cleaned


def _get_debug_filename_patterns(method_name):
    """Return compiled regex patterns that capture (job_id, subunit_id) from debug filenames.

    Each pattern should have group(1)=job_id. Group(2), if present, is the subunit_id.
    """
    if method_name == "NEB":
        return [
            re.compile(r'^(?:ERROR_)?neb_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?neb_refine_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?(?:reactant|product)_relaxation_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?diffusion_barrier_(\d+)(?:_sub(\d+))?\.'),
            re.compile(r'^(?:ERROR_)?dimer_ci_(?:control_)?(\d+)(?:_sub(\d+))?_img\d+\.'),
            re.compile(r'^(?:ERROR_)?imin_relax_(\d+)(?:_sub(\d+))?_img\d+\.'),
            re.compile(r'^VASP_(\d+)(?:_sub(\d+))?_'),
        ]
    elif method_name == "Dimer":
        return [
            re.compile(r'^(?:ERROR_)?dimer_(?:control_|opt_)?(\d+)_(\d+)_'),
            # VASP debug entries: per-attempt dir → (job, attempt) per-subunit cleanup.
            re.compile(r'^(?:ERROR_)?VASP_(\d+)_(\d+)/'),
        ]
    elif method_name == "DoubleMinimization":
        return [
            re.compile(r'^(?:ERROR_)?optimization_(\d+)_(-?\d+)'),
            re.compile(r'^(?:ERROR_)?dimer_refine_(\d+)'),
            # VASP debug entries: per-side dirs (VASP_{job}_-1/0/1) — capture only job_id
            # so the DM remove-all-for-job branch fires (DM always re-runs all 3 sides).
            re.compile(r'^(?:ERROR_)?VASP_(\d+)_-?\d+/'),
        ]
    elif method_name == "Minimization":
        return [
            re.compile(r'^(?:ERROR_)?optimization_(\d+)'),
            re.compile(r'^(?:ERROR_)?VASP_(\d+)/'),
        ]
    elif method_name == "SinglePoint":
        # SP+VASP debug entries are only the per-job VASP dir (no log/traj temps).
        # No subunit captured → the generic remove-whole-job branch in
        # _should_remove_debug fires (SP's cleaned dict is {job_id: {None}}).
        return [re.compile(r'^(?:ERROR_)?VASP_(\d+)/')]
    return []


def _extract_debug_ids(filename, patterns):
    """Extract (job_id, subunit_id) from a debug filename.

    Returns (int, int) or (int, None) or (None, None).
    For Dimer: subunit_id is the attempt_id.
    For NEB: subunit_id is the subband_idx (from _sub{N} suffix), or None for full-band files.
    For DoubleMinimization: subunit_id is the file_idx (0→side=-1, 1→side=1).
    """
    for pat in patterns:
        m = pat.match(filename)
        if m:
            job_id = int(m.group(1))
            subunit_id = int(m.group(2)) if m.lastindex >= 2 and m.group(2) is not None else None
            return job_id, subunit_id
    return None, None


def _should_remove_debug(filename, patterns, cleaned, method_name):
    """Check if a debug file should be removed based on cleaned entries."""
    job_id, subunit_id = _extract_debug_ids(filename, patterns)
    if job_id is None or job_id not in cleaned:
        return False
    if method_name == "Minimization":
        return True  # No subunit, remove all for job
    if method_name == "DoubleMinimization":
        if subunit_id is not None:
            return subunit_id in cleaned[job_id]
        return True  # Can't determine side, remove to be safe
    # Dimer and NEB: subunit_id directly matches
    if subunit_id is not None:
        return subunit_id in cleaned[job_id]
    # No subunit in filename (e.g., full-band NEB file) — remove if job matches
    return True


def _nested_info_value(info, key, aliases=()):
    """Read a metadata key through legacy orig_info nesting without rewriting it."""
    current = info if isinstance(info, dict) else {}
    seen = set()
    keys = (key,) + tuple(aliases)
    for _ in range(32):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        for candidate in keys:
            if candidate in current:
                return current[candidate]
        child = current.get("orig_info")
        if not isinstance(child, dict):
            break
        current = child
    return None


def _should_remove_frame(img, cleaned, info_key, remove_all_sides=False):
    """Check whether an active output frame belongs to an exact redo identity."""
    info = getattr(img, "info", {}) or {}
    if "src_index" not in info:
        raise ValueError(
            "Output frame lacks top-level src_index; refusing to substitute an "
            "upstream orig_info src_index after stage renumbering."
        )
    try:
        jid = int(info["src_index"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid output-frame src_index {info.get('src_index')!r}") from exc
    if jid not in cleaned:
        return False
    if info_key is None or remove_all_sides or None in cleaned[jid]:
        return True
    aliases = ("sub_band_id",) if info_key == "subband_idx" else ()
    subunit = _nested_info_value(info, info_key, aliases)
    if subunit is None:
        raise ValueError(
            f"Output frame for selected job {jid} lacks {_subunit_namespace_from_info_key(info_key)}"
        )
    try:
        subunit = int(subunit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid output-frame {info_key} {subunit!r} for job {jid}") from exc
    return subunit in cleaned[jid]


def _subunit_namespace_from_info_key(info_key):
    return info_key or "job"


def _atomic_filter_trajectory(path, keep_predicate, *, validation_label="output trajectory"):
    """Filter a trajectory without unlinking the live file before replacement."""
    import uuid

    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}.traj",
    )
    kept_count = 0
    removed_count = 0
    try:
        with Trajectory(path, "r") as source:
            with Trajectory(temp, "w") as writer:
                for idx in range(len(source)):
                    atoms = source[idx]
                    if keep_predicate(atoms):
                        writer.write(atoms)
                        kept_count += 1
                    else:
                        removed_count += 1
        if removed_count == 0:
            return kept_count, removed_count
        if kept_count == 0:
            _resume_archive_boundary(f"trajectory_remove:{path}")
            os.unlink(path)
            _fsync_parent(path)
            return kept_count, removed_count

        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())
        try:
            with Trajectory(temp, "r") as check:
                if len(check) != kept_count:
                    raise ValueError(
                        f"{validation_label} rewrite count mismatch for {path}: "
                        f"expected {kept_count}, got {len(check)}"
                    )
                for idx in range(len(check)):
                    if not keep_predicate(check[idx]):
                        raise ValueError(
                            f"{validation_label} rewrite retained a removed identity "
                            f"at {path} frame {idx}"
                        )
        except Exception as exc:
            raise ValueError(f"Unable to validate rewritten {validation_label} {path}: {exc}") from exc
        _resume_archive_boundary(f"trajectory_rewrite_validated:{path}")
        os.replace(temp, path)
        _fsync_parent(path)
        return kept_count, removed_count
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_filter_zip(path, keep_predicate):
    """Filter one debug ZIP using exact ZipInfo entries and atomic replacement."""
    import uuid

    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with zipfile.ZipFile(path, "r") as source:
            bad = source.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure in member {bad!r}")
            infos = source.infolist()
            kept = [info for info in infos if keep_predicate(info.filename)]
            if len(kept) == len(infos):
                return len(kept), 0
            if not kept:
                _resume_archive_boundary(f"debug_zip_remove:{path}")
                os.unlink(path)
                _fsync_parent(path)
                return 0, len(infos)
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as target:
                for info in kept:
                    # Passing ZipInfo reads the exact physical member even when
                    # duplicate filenames exist in the source archive.
                    target.writestr(info, source.read(info))

        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())
        with zipfile.ZipFile(temp, "r") as check:
            bad = check.testzip()
            if bad is not None:
                raise ValueError(f"rewritten ZIP CRC failure in member {bad!r}")
            actual = [
                (info.filename, info.CRC, info.file_size)
                for info in check.infolist()
            ]
        expected = [(info.filename, info.CRC, info.file_size) for info in kept]
        if actual != expected:
            raise ValueError(f"Debug ZIP rewrite validation mismatch for {path}")
        _resume_archive_boundary(f"debug_zip_rewrite_validated:{path}")
        os.replace(temp, path)
        _fsync_parent(path)
        return len(kept), len(infos) - len(kept)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Unreadable/truncated debug ZIP {path}: {exc}") from exc
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def archive_and_clean_outputs(config_dict, cleaned):
    """Archive and atomically clean outputs for the exact selected identities.

    Every live source is validated before mutation. A full, hash-checked backup
    is atomically published before any trajectory/debug rewrite. Unreadable
    sources fail closed instead of being silently omitted from continuation data.
    """
    if not cleaned:
        return

    method_name = config_dict["Main"]["method"]
    _, info_key = _get_subunit_config(method_name)
    remove_all_sides = method_name == "DoubleMinimization"

    # ---- Output trajectories ----
    traj_dir = f"{method_name}_trajes"
    traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.traj")))
    stale_trajs = []
    for traj_path in traj_files:
        has_stale = False
        try:
            with Trajectory(traj_path, "r") as traj:
                for idx in range(len(traj)):
                    if _should_remove_frame(
                        traj[idx], cleaned, info_key, remove_all_sides
                    ):
                        has_stale = True
        except Exception as exc:
            raise ValueError(f"Unreadable output trajectory {traj_path}: {exc}") from exc
        if has_stale:
            stale_trajs.append(traj_path)

    if stale_trajs:
        archive_path, _ = _next_archive_path(traj_dir)
        _write_validated_backup_zip(archive_path, traj_files)
        for traj_path in stale_trajs:
            _atomic_filter_trajectory(
                traj_path,
                lambda atoms: not _should_remove_frame(
                    atoms, cleaned, info_key, remove_all_sides
                ),
            )

    # ---- Debug ZIPs ----
    zip_dir = f"{method_name}_debug_zips"
    zip_files = sorted(
        path for path in glob.glob(os.path.join(zip_dir, "*.zip"))
        if not os.path.basename(path).startswith("previous_")
    )
    patterns = _get_debug_filename_patterns(method_name)
    stale_zips = []
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise ValueError(f"CRC failure in member {bad!r}")
                if any(
                    _should_remove_debug(info.filename, patterns, cleaned, method_name)
                    for info in zf.infolist()
                ):
                    stale_zips.append(zip_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Unreadable/truncated debug ZIP {zip_path}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Unable to read debug ZIP {zip_path}: {exc}") from exc

    if stale_zips:
        archive_path, _ = _next_archive_path(zip_dir)
        _write_validated_backup_zip(archive_path, zip_files)
        for zip_path in stale_zips:
            _atomic_filter_zip(
                zip_path,
                lambda name: not _should_remove_debug(
                    name, patterns, cleaned, method_name
                ),
            )


def get_flux_resources(config_dict):
    from flux import Flux, resource

    handle = Flux()
    rset = resource.list.resource_list(handle).get().all
    all_ncores = rset.ncores
    all_ngpus = rset.ngpus
    nnodes = rset.nnodes
    print(f"Number of nodes: {nnodes}, total number of CPU cores: {all_ncores}, total number of GPUs: {all_ngpus}")

    jobs_per_gpu = config_dict["Main"]["jobs_per_gpu"]
    jobs_per_node = config_dict["Main"]["jobs_per_node"]

    if config_dict["Main"]["device"] == 'cuda':
        max_workers = all_ngpus * jobs_per_gpu
        gpus_per_core = 1 if jobs_per_gpu == 1 else 0
        cores = 1
        threads_per_core = all_ncores // max_workers # - 1
    elif config_dict["Main"]["device"] == 'cpu':
        max_workers = nnodes * jobs_per_node
        gpus_per_core = 0
        # Always cores=1 per worker so executorlib spawns single-rank Python
        # workers (no internal mpi4py dependency). The user's vasp_command
        # (or FAIRChem CPU calc) handles its own threading / MPI ranks.
        # threads_per_core is partitioned across all workers so the total
        # resource request fits inside the node's physical core budget.
        cores = 1
        threads_per_core = max(1, all_ncores // max_workers)
    else:
        raise ValueError("Only devices cuda and cpu available. Please set one of the two in Main section of config.ini")
    return max_workers, cores, gpus_per_core, threads_per_core

# SADDLEMILL_CANONICAL_HISTORY_REFACTOR_20260831_V4

# SADDLEMILL_PERSISTENT_ROTATION_LBFGS_20260901_V2
