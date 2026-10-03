import numpy as np
import json, os, glob, shutil, tempfile, zipfile, fnmatch
from ase.neighborlist import neighbor_list, natural_cutoffs
from ase.io import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from saddlemill.config import VALID_RUN_CATEGORIES, _RUN_CATEGORY_ALIASES, _get_subunit_config


#==============================================================================
### FLUX LOG BACKUP

def backup_flux_logs(worker_id):
    """Append current flux log files into backup files before worker restart.

    Flux overwrites flux_{id}.out/.err on each new job submission, so we
    append their contents to persistent backup files before sys.exit(1).
    """
    for ext in (".out", ".err"):
        src = f"flux_{worker_id}{ext}"
        dst = f"flux_{worker_id}{ext}.bak"
        if os.path.exists(src):
            with open(src, 'r') as f_in, open(dst, 'a') as f_out:
                f_out.write(f_in.read())


#==============================================================================
### ATOMS LOADING

def load_and_sanitize(traj, i, j):
    """Load images from trajectory and stash original .info into orig_info.

    This prevents per-atom array data (e.g. forces, stress) in .info from
    causing size mismatches when atoms are later added or removed (e.g. vacancy
    mechanism in Dimer). Applied uniformly across all methods for consistency.
    """
    if j != i + 1:
        images = list(traj[i:j])
        for img in images:
            img.info = {"orig_info": dict(img.info)}
    else:
        images = traj[i]
        images.info = {"orig_info": dict(images.info)}
    return images


def passes_input_filter(images, config_dict):
    """Return True if a sanitized input's status matches ``input_statuses``.

    Patterns support ``fnmatch`` wildcards (e.g. ``converged*`` matches
    ``converged``, ``converged_CI``, ``converged_after_extension``, etc.).
    The special value ``"all"`` (the default) bypasses the filter entirely.
    """
    raw = config_dict["Main"]["input_statuses"]
    if raw in ("all", None):
        return True

    main_atoms = images[0] if isinstance(images, list) else images
    orig = main_atoms.info.get('orig_info', {})
    status = orig.get('status')

    patterns = [raw] if isinstance(raw, str) else list(raw)
    return any(fnmatch.fnmatchcase(status or '', p) for p in patterns)


def get_task_name(config_dict):
    """Return [FAIRChemCalculator] task_name if FAIRChem is the calculator, else None."""
    if config_dict["Main"]["Calculator"] == "FAIRChemCalculator":
        return config_dict["FAIRChemCalculator"].get("task_name")
    return None


#==============================================================================
### VASP HELPERS

def vasp_incar_kwargs(config_dict, atoms=None):
    """Return the INCAR/k-point/setup kwargs for a VASP calculator.

    Starts from an optional ``[ourVasp] input_generator`` (built-in name, dotted
    ``module:func``, or ``file.py:func``) evaluated on ``atoms``, then layers the
    explicit ``[Vasp]`` section keys on top so the user's ``[Vasp]`` always wins.
    With no generator (or no ``atoms``), this is just the ``[Vasp]`` section.
    ``[Vasp]`` is a pure pass-through to ASE's Vasp calculator; SaddleMill's own
    knobs live in ``[ourVasp]`` and are never forwarded to the calculator.
    """
    vasp_section = dict(config_dict.get("Vasp", {}))
    gen_spec = config_dict.get("ourVasp", {}).get("input_generator")
    if gen_spec and atoms is not None:
        from saddlemill.vasp_io import load_input_generator
        gen_kwargs = load_input_generator(gen_spec)(atoms)
        return {**gen_kwargs, **vasp_section}  # [Vasp] keys override generator
    return vasp_section


def _with_extra_io(calc_cls, writers, parsers):
    """Subclass *calc_cls* to run extra-input writers and extra-output parsers.

    Writers ``(calc, atoms, directory) -> None`` run after ASE writes its inputs
    (directory exists, ``calc.sort`` set) and before VASP runs — via ``write_input``.
    Parsers ``(calc, atoms, directory) -> dict`` run after VASP finishes (directory
    populated, ``calc.resort`` set) — via ``read_results`` — and their merged dict is
    stashed on ``calc.sm_extra_outputs`` for the method to stamp onto output frames.
    """
    class _CalcWithExtraIO(calc_cls):
        def write_input(self, atoms, *args, **kwargs):
            super().write_input(atoms, *args, **kwargs)
            directory = kwargs.get("directory", getattr(self, "directory", "."))
            for writer in writers:
                writer(self, atoms, directory)

        def read_results(self):
            super().read_results()
            info = {}
            directory = getattr(self, "directory", ".")
            for parser in parsers:
                info.update(parser(self, self.atoms, directory) or {})
            self.sm_extra_outputs = info

    _CalcWithExtraIO.__name__ = f"{calc_cls.__name__}WithExtraIO"
    return _CalcWithExtraIO


def resolve_vasp_calc_class(config_dict, calc):
    """Return *calc*, wrapped for ``[ourVasp] extra_input_files`` / ``extra_outputs`` (if set).

    No-op for FAIRChem or when neither key is set. Shared by ``resolve_vasp_calc``
    and ``nebopt._build_neb_vasp_calc`` so the hooks are identical across all methods.
    Each value is one spec or a space-separated list (built-in name, ``module:func``,
    or ``file.py:func``). Output parsers leave their merged dict on
    ``calc.sm_extra_outputs``; the method decides whether to stamp it onto frames.
    """
    if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):
        return calc
    our_vasp = config_dict.get("ourVasp", {})
    in_spec = our_vasp.get("extra_input_files")
    out_spec = our_vasp.get("extra_outputs")
    if not in_spec and not out_spec:
        return calc
    from saddlemill.vasp_io import (load_extra_input_writer,
                                                  load_extra_output_parser)
    _aslist = lambda s: [s] if isinstance(s, str) else list(s)
    writers = [load_extra_input_writer(s) for s in _aslist(in_spec)] if in_spec else []
    parsers = [load_extra_output_parser(s) for s in _aslist(out_spec)] if out_spec else []
    return _with_extra_io(calc, writers, parsers)


def resolve_vasp_calc(config_dict, calc, i, subunit_id, section, atoms=None):
    """Return an instantiated calculator for this (job, subunit).

    For FAIRChem, returns the shared instance unchanged. For VASP/VaspInteractive,
    builds a fresh calculator pointing at ``VASP_{i}[_{subunit_id}]/`` with the
    section's ``vasp_command`` / ``vasp_ncore`` and the INCAR kwargs from
    ``vasp_incar_kwargs`` (``[Vasp]`` plus an optional per-structure
    ``[ourVasp] input_generator``). The class is first wrapped by
    ``resolve_vasp_calc_class`` so ``[ourVasp] extra_input_files`` (e.g. a VTST
    MODECAR) are written too.
    ``subunit_id=None`` produces ``VASP_{i}/`` (Minimization, SinglePoint). Pass
    ``atoms`` to enable ``input_generator`` and the extra-file writers.
    """
    if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):
        return calc
    suffix = f"_{subunit_id}" if subunit_id is not None else ""
    kwargs = {"directory": f"VASP_{i}{suffix}",
              "command": config_dict[section]["vasp_command"],
              **vasp_incar_kwargs(config_dict, atoms)}
    ncore = config_dict[section].get("vasp_ncore")
    if ncore is not None:
        kwargs["ncore"] = int(ncore)
    return resolve_vasp_calc_class(config_dict, calc)(**kwargs)


def remove_vasp_heavies(dir_path):
    """Delete WAVECAR / CHG / CHGCAR from *dir_path* if they exist."""
    for name in ("WAVECAR", "CHG", "CHGCAR"):
        p = os.path.join(dir_path, name)
        if os.path.exists(p):
            os.remove(p)


def archive_and_clear_temp_files(temp_files, zip_name, prefix="", enabled=True):
    """Zip existing temp files/directories into *zip_name* and remove them.

    Mirrors the per-method temp-file cleanup that previously lived inline in
    each method. Walks directories (e.g. VASP working dirs) so every file inside
    is archived under its relative path. Set ``enabled=False`` to skip zipping
    and just remove the entries.
    """
    existing = [f for f in temp_files if os.path.exists(f)]
    if not existing:
        return
    if enabled:
        with zipfile.ZipFile(zip_name, 'a', zipfile.ZIP_DEFLATED) as zf:
            for f_name in existing:
                if os.path.isdir(f_name):
                    for root, _dirs, files in os.walk(f_name):
                        for file in files:
                            filepath = os.path.join(root, file)
                            zf.write(filepath, arcname=f"{prefix}{filepath}")
                else:
                    zf.write(f_name, arcname=f"{prefix}{f_name}")
    for f_name in existing:
        if os.path.isdir(f_name):
            shutil.rmtree(f_name)
        else:
            os.remove(f_name)


def finalize_if_vasp_interactive(config_dict, calc_instance):
    """Call ``.finalize()`` on a VaspInteractive instance; no-op otherwise.

    The matching guard is on the active calculator class, not on the instance
    type — that keeps the call site readable next to other VASP-only branches.
    """
    if config_dict["Main"]["Calculator"] == "VaspInteractive":
        try:
            calc_instance.finalize()
        except Exception:
            pass


def vasp_final_scf_converged(directory):
    """Return True iff the LAST electronic (SCF) loop in OUTCAR reached EDIFF.

    VASP 6 labels each SCF exit: ``aborting loop because EDIFF is reached`` when an
    ionic step's electronic loop converges, and ``aborting loop because EDIFF was
    not reached (unconverged)`` (a NELM miss) when it does not. We keep the verdict
    of the LAST such marker, so an intermediate step that blew NELM but later
    recovered does not fail the job — only the final structure's SCF must be sound.
    Returns True when OUTCAR is missing/unreadable or has no marker (can't tell ->
    don't block; a genuinely broken run errors out elsewhere on parsing).
    """
    outcar = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(outcar):
        return True
    result = True
    try:
        with open(outcar) as f:
            for line in f:
                if "aborting loop" in line:
                    result = "because EDIFF is reached" in line
    except OSError:
        return True
    return result


#==============================================================================
### FILE IO

def save_ordered_traj_names(trajes_and_idxs):
    with open('traj_files_ordered.json', 'w') as f:
        json.dump(trajes_and_idxs, f)


def read_ordered_traj_names():
    with open('traj_files_ordered.json', 'r') as f:
        trajes_and_idxs = json.load(f)
    return trajes_and_idxs


def clean_up_files(config_dict):
    """Clean only documented per-run scratch names from the working directory.

    Flux stdout/stderr are diagnostics, so current numeric worker logs are moved
    into ``resume_cleanup_diagnostics/`` instead of being destroyed; existing
    ``*.bak`` files are left untouched. VASP cleanup is method-specific and only
    removes numeric per-job scratch directories, never an arbitrary ``VASP_*``.
    """
    import re

    method_name = config_dict["Main"]["method"]
    file_patterns = {
        "NEB": [
            r"neb_\d+\.(?:log|traj)",
            r"reactant_relaxation_\d+\.(?:log|traj)",
            r"product_relaxation_\d+\.(?:log|traj)",
            r"diffusion_barrier_\d+\.png",
            r"imin_relax_\d+_img\d+\.(?:log|traj)",
            r"dimer_ci(?:_control)?_\d+_img\d+\.(?:log|traj)",
            r"neb_refine_\d+\.(?:log|traj)",
        ],
        "Dimer": [
            r"dimer_control_\d+(?:_\d+(?:_-?\d+)?)?\.log",
            r"dimer_opt_\d+(?:_\d+(?:_-?\d+)?)?\.log",
            r"dimer_sella_opt_\d+_\d+_-?\d+\.log",
            r"dimer_mode_\d+_\d+_-?\d+\.log",
            r"dimer_\d+(?:_\d+(?:_-?\d+)?)?\.traj",
            r"dimer_sella_\d+_\d+_-?\d+\.traj",
        ],
        "Minimization": [r"optimization_\d+\.(?:log|traj)"],
        "DoubleMinimization": [
            r"optimization_\d+(?:_(?:-1|1))?\.(?:log|traj)",
            r"dimer_refine_\d+\.log",
        ],
        "SinglePoint": [],
        "Hessian": [],
    }
    compiled_files = [re.compile(rf"^(?:{pattern})$") for pattern in file_patterns.get(method_name, [])]

    for name in list(os.listdir(".")):
        path = os.path.join(".", name)
        if os.path.isfile(path) and any(pattern.fullmatch(name) for pattern in compiled_files):
            os.remove(path)

    if config_dict["Main"].get("Calculator") in ("Vasp", "VaspInteractive"):
        vasp_pattern = {
            "NEB": r"^VASP_\d+_\d+$",
            "Dimer": r"^VASP_\d+_\d+$",
            "DoubleMinimization": r"^VASP_\d+_(?:-1|0|1)$",
            "Minimization": r"^VASP_\d+$",
            "SinglePoint": r"^VASP_\d+$",
        }.get(method_name)
        if vasp_pattern:
            matcher = re.compile(vasp_pattern)
            for name in list(os.listdir(".")):
                path = os.path.join(".", name)
                if matcher.fullmatch(name) and os.path.isdir(path):
                    shutil.rmtree(path)

    # Preserve useful worker diagnostics while still freeing the canonical names
    # that executorlib may reuse on the next launch. Deliberately do not touch
    # flux_*.out.bak / flux_*.err.bak.
    flux_matcher = re.compile(r"^flux_\d+\.(?:out|err)$")
    flux_logs = [name for name in os.listdir(".") if flux_matcher.fullmatch(name)]
    if flux_logs:
        diag_dir = "resume_cleanup_diagnostics"
        os.makedirs(diag_dir, exist_ok=True)
        for name in sorted(flux_logs):
            src = os.path.join(".", name)
            index = 0
            while True:
                dst = os.path.join(diag_dir, f"{name}.previous_{index}")
                if not os.path.exists(dst):
                    break
                index += 1
            os.replace(src, dst)


#==============================================================================
### PREVIOUS RESULT EXTRACTION (for continue-from-result on resume)


class ResumeIntegrityError(ValueError):
    """Active resume evidence is unreadable or has conflicting identities."""


def _info_chain(info):
    current = info if isinstance(info, dict) else {}
    seen = set()
    for _ in range(32):
        marker = id(current)
        if marker in seen:
            return
        seen.add(marker)
        yield current
        child = current.get("orig_info")
        if not isinstance(child, dict):
            return
        current = child


def _legacy_info_value(info, key, aliases=()):
    keys = (key,) + tuple(aliases)
    for level in _info_chain(info):
        for candidate in keys:
            if candidate in level:
                return level[candidate]
    return None


def _frame_job_id(atoms):
    """Return the current stage job ID; never substitute an upstream lineage ID."""
    info = getattr(atoms, "info", {}) or {}
    if "src_index" not in info:
        raise ResumeIntegrityError(
            "Output frame lacks top-level src_index. Refusing to use an orig_info "
            "src_index because that may be an upstream source after stage renumbering."
        )
    try:
        return int(info["src_index"])
    except (TypeError, ValueError) as exc:
        raise ResumeIntegrityError(f"Invalid output-frame src_index {info.get('src_index')!r}") from exc


def _frame_subunit_id(method_name, atoms):
    info = getattr(atoms, "info", {}) or {}
    if method_name == "Dimer":
        key, aliases = "attempt_id", ()
    elif method_name == "DoubleMinimization":
        key, aliases = "side", ()
    elif method_name == "NEB":
        key, aliases = "subband_idx", ("sub_band_id",)
    else:
        return None
    value = _legacy_info_value(info, key, aliases)
    if value is None:
        raise ResumeIntegrityError(
            f"{method_name} output frame for job {_frame_job_id(atoms)} lacks {key} metadata"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResumeIntegrityError(f"Invalid {method_name} {key} metadata {value!r}") from exc


def _frame_image_idx(atoms):
    value = _legacy_info_value(getattr(atoms, "info", {}) or {}, "image_idx")
    if value is None:
        raise ResumeIntegrityError("NEB output frame lacks image_idx metadata")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResumeIntegrityError(f"Invalid NEB image_idx metadata {value!r}") from exc


def _frame_resume_identity(method_name, atoms):
    jid = _frame_job_id(atoms)
    if method_name == "NEB":
        return (method_name, jid, "subband_idx", _frame_subunit_id(method_name, atoms),
                "image_idx", _frame_image_idx(atoms))
    if method_name in ("Dimer", "DoubleMinimization"):
        namespace = "attempt_id" if method_name == "Dimer" else "side"
        return (method_name, jid, namespace, _frame_subunit_id(method_name, atoms))
    return (method_name, jid, "job", None)


def _value_equivalent(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            a = np.asarray(left)
            b = np.asarray(right)
            if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
                return False
            try:
                return bool(np.array_equal(a, b, equal_nan=True))
            except TypeError:
                return bool(np.array_equal(a, b))
        except Exception:
            return False
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_value_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _value_equivalent(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return True
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


def _constraint_signature(atoms):
    constraints = getattr(atoms, "constraints", []) or []
    signatures = []
    for constraint in constraints:
        if hasattr(constraint, "todict"):
            value = constraint.todict()
        elif hasattr(constraint, "__dict__"):
            value = dict(constraint.__dict__)
        else:
            value = repr(constraint)
        signatures.append((
            f"{type(constraint).__module__}.{type(constraint).__name__}", value
        ))
    return signatures


def _frame_equivalent(left, right):
    """Exact duplicate check; positions alone are intentionally insufficient."""
    for attr in ("numbers", "positions", "cell", "pbc"):
        if not _value_equivalent(getattr(left, attr, None), getattr(right, attr, None)):
            return False
    if not _value_equivalent(getattr(left, "arrays", {}) or {}, getattr(right, "arrays", {}) or {}):
        return False
    if not _value_equivalent(getattr(left, "info", {}) or {}, getattr(right, "info", {}) or {}):
        return False
    if not _value_equivalent(_constraint_signature(left), _constraint_signature(right)):
        return False
    left_calc = getattr(left, "calc", None)
    right_calc = getattr(right, "calc", None)
    left_results = getattr(left_calc, "results", {}) if left_calc is not None else {}
    right_results = getattr(right_calc, "results", {}) if right_calc is not None else {}
    return _value_equivalent(left_results or {}, right_results or {})


def _build_output_traj_index(method_name, wanted_job_ids=None, wanted_subunits=None):
    """Build a selected-job continuation index with explicit frame identities.

    Active output trajectories are fail-closed: unreadable/truncated files raise
    ``ResumeIntegrityError``. Exact duplicate frames for the same identity collapse
    deterministically to the last physical occurrence; conflicting duplicates raise
    instead of silently selecting the first. Scientifically distinct attempt/sub-band
    identities are never merged merely because positions match.
    """
    import warnings

    if method_name == "SinglePoint":
        return {}
    wanted_jobs = None if wanted_job_ids is None else {int(x) for x in wanted_job_ids}
    wanted_subunits = wanted_subunits or {}
    selected = {}
    origins = {}
    traj_dir = f"{method_name}_trajes"
    for traj_path in sorted(glob.glob(os.path.join(traj_dir, "*.traj"))):
        try:
            with Trajectory(traj_path, "r") as traj:
                for frame_idx in range(len(traj)):
                    img = traj[frame_idx]
                    info = getattr(img, "info", {}) or {}
                    if "src_index" not in info:
                        warnings.warn(
                            f"Skipping output frame without top-level src_index: "
                            f"{traj_path} frame {frame_idx}. Upstream orig_info "
                            "src_index is preserved as lineage and is not reused as "
                            "the current-stage resume identity.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    jid = _frame_job_id(img)
                    if wanted_jobs is not None and jid not in wanted_jobs:
                        continue
                    identity = _frame_resume_identity(method_name, img)
                    if method_name == "Dimer" and jid in wanted_subunits:
                        requested = wanted_subunits[jid]
                        if None not in requested and _frame_subunit_id(method_name, img) not in requested:
                            continue
                    per_job = selected.setdefault(jid, {})
                    previous = per_job.get(identity)
                    if previous is not None and not _frame_equivalent(previous, img):
                        old_path, old_idx = origins[(jid, identity)]
                        raise ResumeIntegrityError(
                            "Conflicting duplicate continuation result for identity "
                            f"{identity!r}: {old_path} frame {old_idx} vs "
                            f"{traj_path} frame {frame_idx}"
                        )
                    # Deterministic last physical occurrence wins only when the
                    # complete stored result is equivalent.
                    per_job[identity] = img
                    origins[(jid, identity)] = (traj_path, frame_idx)
        except ResumeIntegrityError:
            raise
        except Exception as exc:
            raise ResumeIntegrityError(
                f"Unreadable/truncated output trajectory {traj_path}: {exc}"
            ) from exc
    return {jid: list(frames.values()) for jid, frames in selected.items()}


def _sanitize_with_continuation(atoms):
    """Return a continuation copy with lineage and cached energy/forces intact."""
    cached = {}
    calc = getattr(atoms, "calc", None)
    results = getattr(calc, "results", {}) if calc is not None else {}
    for key in ("energy", "forces"):
        if key in results:
            value = results[key]
            cached[key] = value.copy() if hasattr(value, "copy") else value

    result = atoms.copy()
    result.info = {"orig_info": dict(getattr(result, "info", {}) or {})}
    if cached:
        result.calc = SinglePointCalculator(result, **cached)
    return result


def extract_previous_results(job_ids, config_dict, redo_info):
    """Extract unambiguous previous outputs needed for continuation.

    The current stage ``src_index`` is used only to locate the resume job. The
    full frame metadata, including upstream ``parent_ts_index`` / Hessian parent
    lineage and any nested legacy ``orig_info``, is preserved under the returned
    continuation's ``orig_info``.
    """
    method_name = config_dict["Main"]["method"]
    if method_name == "SinglePoint" or not job_ids:
        return {}

    requested_jobs = [int(jid) for jid in job_ids if jid in redo_info]
    if not requested_jobs:
        return {}
    wanted_subunits = redo_info if method_name == "Dimer" else None
    output_traj_index = _build_output_traj_index(
        method_name,
        wanted_job_ids=requested_jobs,
        wanted_subunits=wanted_subunits,
    )
    results = {}

    for job_id in requested_jobs:
        frames = output_traj_index.get(job_id, [])
        if not frames:
            continue

        if method_name in ("Minimization", "Hessian"):
            if len(frames) != 1:
                raise ResumeIntegrityError(
                    f"Expected one {method_name} continuation frame for job {job_id}, "
                    f"found {len(frames)}"
                )
            results[job_id] = _sanitize_with_continuation(frames[0])
            continue

        grouped = {}
        for frame in frames:
            subunit_id = _frame_subunit_id(method_name, frame)
            grouped.setdefault(subunit_id, []).append(frame)

        if method_name == "NEB":
            for subunit_id, atoms_list in grouped.items():
                atoms_list.sort(key=_frame_image_idx)
                grouped[subunit_id] = [
                    _sanitize_with_continuation(atoms) for atoms in atoms_list
                ]
            results[job_id] = grouped
            continue

        # Dimer and DoubleMin identities are one frame per attempt/side after
        # conflict-aware indexing. DoubleMin deliberately retains all available
        # -1/0/+1 frames so a one-sided redo can keep the completed endpoint.
        flattened = {}
        for subunit_id, atoms_list in grouped.items():
            if len(atoms_list) != 1:
                raise ResumeIntegrityError(
                    f"Expected one {method_name} result for job {job_id} "
                    f"{_get_subunit_config(method_name)[1]}={subunit_id}, "
                    f"found {len(atoms_list)}"
                )
            flattened[subunit_id] = _sanitize_with_continuation(atoms_list[0])
        results[job_id] = flattened

    return results


def get_bond_set(atoms, cutoffs, tag_filter=None):
    """
    Returns a python set of bonds tuple(atom_index_A, atom_index_B).
    
    Args:
        atoms: The ASE atoms object
        cutoffs: Dictionary or list of cutoff radii
        tag_filter: (Optional) Only include bonds where BOTH atoms have this tag.
    """
    # 'i' and 'j' are indices of bonded atoms
    i_list, j_list = neighbor_list('ij', atoms, cutoffs)
    
    bonds = set()
    tags = atoms.get_tags()
    
    for k in range(len(i_list)):
        a, b = i_list[k], j_list[k]
        
        # We only want each bond once (0-1 is same as 1-0)
        # So we sort them: tuple((min, max))
        bond = tuple(sorted((a, b)))
        
        # If a filter is applied (e.g., tag==2), check tags
        if tag_filter is not None:
            if tags[a] == tag_filter and tags[b] == tag_filter:
                bonds.add(bond)
        else:
            bonds.add(bond)
            
    return bonds


def check_reaction(atoms_initial, atoms_final, neighbor_fudge=1.25):
    """
    Compares connectivity of two structures.
    """
    # 1. Get bonds for both
    assert np.array_equal(atoms_initial.numbers, atoms_final.numbers), \
            "Error: Atomic numbers do not match between initial and final states."
    cutoffs = natural_cutoffs(atoms_initial, mult=neighbor_fudge)
    bonds_ini = get_bond_set(atoms_initial, cutoffs)
    bonds_fin = get_bond_set(atoms_final, cutoffs)
    
    # 2. Compare sets
    # Bonds present in Initial but NOT in Final = BROKEN
    broken = bonds_ini - bonds_fin
    
    # Bonds present in Final but NOT in Initial = FORMED
    formed = bonds_fin - bonds_ini
    
    reaction_occurred = len(broken) > 0 or len(formed) > 0
    
    return {
        "occurred": reaction_occurred,
        "broken_bonds": broken,
        "formed_bonds": formed,
        "n_broken": len(broken),
        "n_formed": len(formed)
    }

def check_adsorbate_reaction(atoms_initial, atoms_final, neighbor_fudge=1.25, target_tag=2):
    """
    Checks for reactions ONLY within atoms having specific tag (e.g. tag=2).
    """
    # 1. Get filtered bonds
    assert np.array_equal(atoms_initial.numbers, atoms_final.numbers), \
            "Error: Atomic numbers do not match between initial and final states."
    cutoffs = natural_cutoffs(atoms_initial, mult=neighbor_fudge)
    bonds_ini = get_bond_set(atoms_initial, cutoffs, tag_filter=target_tag)
    bonds_fin = get_bond_set(atoms_final, cutoffs, tag_filter=target_tag)
    
    # 2. Calculate differences
    broken = bonds_ini - bonds_fin
    formed = bonds_fin - bonds_ini
    
    return {
        "occurred": len(broken) > 0 or len(formed) > 0,
        "broken_bonds": broken,
        "formed_bonds": formed,
        "n_broken": len(broken),
        "n_formed": len(formed)
    }

#==============================================================================
