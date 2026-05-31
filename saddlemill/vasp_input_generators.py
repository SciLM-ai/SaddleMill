"""User-pluggable VASP input generation.

A *generator* maps an ASE ``Atoms`` object to a dict of ASE-``Vasp`` keyword
arguments (lowercased INCAR tags plus ``kpts``/``gamma``/``setups``/``magmom``).
SaddleMill merges this dict UNDER the ``[Vasp]`` section — explicit ``[Vasp]``
keys always win — and hands the result to the calculator. Because ASE then
writes every input file itself, atom sorting, the resort that maps forces back,
POTCAR selection, and the ``VaspInteractive`` interactive flags are all handled
correctly. The generator only decides *what settings to use*, never writes files.

Selection lives in ``[ourVasp] input_generator`` and may be:
  - a built-in name: ``omat24_static``, ``omat24_relax``, ``cheap_omat``, ``oc20``
  - ``package.module:func`` — import an installed module and use ``func``
  - ``/abs/or/rel/file.py:func`` — load a local ``.py`` file and use ``func``

A custom generator is any callable ``generator(atoms) -> dict`` of ASE-Vasp
kwargs, so users can plug in their own input recipes without touching SaddleMill.

**Ionic-driver tags are stripped.** ``IBRION``/``NSW``/``POTIM``/``EDIFFG`` are
removed from generator output because SaddleMill drives geometry through ASE
optimizers (each VASP call is a single-point force evaluation), and
``VaspInteractive`` forbids overriding ``IBRION``/``POTIM``. Set ``ISIF`` in
``[Vasp]`` if you need stress for cell relaxation. Built-in input sets such as
OMat24/OC20 are authored for *standalone* VASP relaxations, so their relaxation
tags are intentionally dropped here.

----

This module also provides **extra-input-file writers** (``[ourVasp]
extra_input_files``): callables ``writer(calc, atoms, directory) -> None`` that
drop additional files into the VASP working directory *after* ASE has written
INCAR/POSCAR/etc. (so ``calc.sort`` and the directory exist) and *before* VASP
runs. The motivating case is ``modecar`` — a VTST MODECAR built from
``atoms.info['eigenmode']``, reordered to POSCAR order via ``calc.sort``. Same
selection grammar as ``input_generator`` (built-in name, ``module:func``,
``file.py:func``), and a space-separated list runs several writers in order.
Unlike ``input_generator`` (which only computes settings), these write files,
so the hook lives in a ``write_input`` subclass — see ``tools._with_extra_input_files``.
"""
import importlib
import importlib.util
import os
import warnings

# INCAR tags selecting VASP's *internal* ionic driver. SaddleMill controls
# geometry via ASE optimizers, so these must never come from a generator.
_DRIVER_KEYS = {"ibrion", "nsw", "potim", "ediffg"}


def _to_native(v):
    """Convert numpy scalars / arrays to plain Python for clean INCAR writing."""
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_to_native(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_to_native(x) for x in v]
    return v


def _pmg_set_to_ase_kwargs(input_set):
    """Translate a pymatgen ``VaspInputSet`` into ASE ``Vasp`` kwargs.

    The input set must be built with ``sort_structure=False`` so that any
    per-site ``MAGMOM`` stays aligned to the ASE atom order — ASE's
    ``set_magmom`` then re-sorts it into POSCAR (symbol) order itself.

    DFT+U is special-cased. pymatgen emits ``LDAUU``/``LDAUL``/``LDAUJ`` as
    *positional* lists aligned to its own POSCAR species order
    (``poscar.site_symbols``). ASE re-sorts the atoms and would write those
    lists verbatim, landing U on the wrong element. We instead emit ASE's
    element-keyed ``ldau_luj`` dict, which ASE re-orders to its own POSCAR — so
    U follows the species. (ASE rejects ``ldau_luj`` alongside the raw lists,
    so they are dropped.) ``MAGMOM`` needs no such treatment: it is per-atom and
    ASE's ``set_magmom`` already re-sorts it.
    """
    kwargs = {}
    incar = input_set.incar
    for k, v in incar.items():
        key = k.lower()
        if key in _DRIVER_KEYS:
            continue
        kwargs[key] = _to_native(v)

    # DFT+U: positional lists -> element-keyed ldau_luj (see docstring).
    if "LDAUU" in incar:
        luj = {}
        for sym, ll, uu, jj in zip(input_set.poscar.site_symbols,
                                   incar["LDAUL"], incar["LDAUU"], incar["LDAUJ"]):
            luj.setdefault(sym, {"L": int(ll), "U": float(uu), "J": float(jj)})
        for raw in ("ldauu", "ldaul", "ldauj"):
            kwargs.pop(raw, None)
        kwargs["ldau_luj"] = luj

    # KPOINTS -> explicit k-mesh + gamma flag.
    kp = getattr(input_set, "kpoints", None)
    if kp is not None and getattr(kp, "kpts", None):
        kwargs["kpts"] = [int(x) for x in kp.kpts[0]]
        kwargs["gamma"] = str(kp.style).lower().startswith("gamma")

    # POTCAR symbols -> per-element ASE `setups` suffix ('Fe_pv' -> {'Fe': '_pv'}).
    setups = {}
    for sym in getattr(input_set, "potcar_symbols", []):
        el, _, suf = sym.partition("_")
        if suf:
            setups[el] = "_" + suf
    if setups:
        kwargs["setups"] = setups
    return kwargs


def _omat24(atoms, set_cls_name, **set_kwargs):
    from pymatgen.io.ase import AseAtomsAdaptor
    from fairchem.data.omat.vasp import sets as omat_sets
    set_cls = getattr(omat_sets, set_cls_name)
    struct = AseAtomsAdaptor.get_structure(atoms)
    iset = set_cls(struct, sort_structure=False, **set_kwargs)
    return _pmg_set_to_ase_kwargs(iset)


def omat24_static(atoms):
    """OMat24 single-point (static) accuracy settings."""
    return _omat24(atoms, "OMat24StaticSet")


def omat24_relax(atoms):
    """OMat24 relaxation accuracy settings (ionic-driver tags stripped)."""
    return _omat24(atoms, "OMat24RelaxSet")


def cheap_omat(atoms):
    """OMat24 recipe tuned DOWN for a cheap first-pass saddle search.

    Two changes vs :func:`omat24_static`: the k-point reciprocal density drops
    from 64 to 16, and the POTCARs are lightened to ASE's ``minimal`` base
    (plain potentials for transition metals, mandatory semicore only for
    alkali/alkaline-earth) with soft ``_s`` O/C/N — so ENCUT can fall to ~300.
    Reconverge the resulting saddle with :func:`omat24_static`. Electronic and
    accuracy knobs (ENCUT, EDIFF, PREC, ISMEAR, SIGMA, ALGO) are intentionally
    left to the ``[Vasp]`` section.

    Note: there is no soft ``_s`` POTCAR for F (and a few other hard anions),
    so fluorine-bearing systems still need a higher ENCUT than 300.
    """
    kwargs = _omat24(atoms, "OMat24StaticSet",
                     user_kpoints_settings={"reciprocal_density": 16})
    kwargs["setups"] = {"base": "minimal", "O": "_s", "C": "_s", "N": "_s"}
    return kwargs


def oc20(atoms):
    """OC20 slab/adslab VASP settings (RPBE, ENCUT 350, surface k-points)."""
    from fairchem.data.oc.utils.vasp_flags import VASP_FLAGS
    from fairchem.data.oc.utils.vasp import calculate_surface_k_points
    kwargs = {k: v for k, v in VASP_FLAGS.items() if k not in _DRIVER_KEYS}
    if "kpts" not in kwargs:
        kwargs["kpts"] = tuple(calculate_surface_k_points(atoms))
    kwargs.setdefault("setups", "minimal")
    return kwargs


_BUILTINS = {
    "omat24_static": omat24_static,
    "omat24_relax": omat24_relax,
    "cheap_omat": cheap_omat,
    "oc20": oc20,
}


def _import_callable(spec):
    """Resolve ``package.module:func`` or ``/path/to/file.py:func`` to a callable."""
    target, func_name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(target)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input file not found: {path}")
        mod_name = "_sm_vasp_dyn_" + str(abs(hash(path)))
        mod_spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise AttributeError(f"{func_name!r} not found in {target!r}.")


def load_input_generator(spec):
    """Resolve an ``input_generator`` config value to a callable ``atoms -> dict``.

    ``spec`` is a built-in name, ``package.module:func``, or
    ``/path/to/file.py:func``. Raises a clear error if it cannot be resolved.
    """
    if callable(spec):
        return spec
    if spec in _BUILTINS:
        return _BUILTINS[spec]
    if ":" not in spec:
        raise ValueError(
            f"Unknown input_generator {spec!r}. Use a built-in "
            f"({', '.join(sorted(_BUILTINS))}), 'package.module:func', "
            f"or '/path/to/file.py:func'."
        )
    return _import_callable(spec)


#==============================================================================
### EXTRA INPUT FILES (written after ASE writes its inputs, e.g. VTST MODECAR)

def write_modecar(calc, atoms, directory):
    """Write a VTST ``MODECAR`` (initial dimer mode) from ``atoms.info['eigenmode']``.

    The eigenmode (in atoms order, with the usual ``orig_info`` fallback) is
    reshaped to ``(natoms, 3)``, reordered to POSCAR order via ``calc.sort``,
    normalized, and written one ``nx ny nz`` line per atom. No-op (with a
    warning) when no eigenmode is present, so a batch never hard-fails on it.
    """
    import numpy as np
    eig = atoms.info.get("eigenmode")
    if eig is None:
        eig = atoms.info.get("orig_info", {}).get("eigenmode")
    if eig is None:
        warnings.warn(
            "extra_input_files=modecar but atoms.info has no 'eigenmode'; "
            "skipping MODECAR (VTST will use its default initial mode).")
        return
    eig = np.asarray(eig, dtype=float).reshape(len(atoms), 3)
    sort = getattr(calc, "sort", None)
    if sort is not None:
        eig = eig[sort]                       # atoms order -> POSCAR (symbol) order
    norm = np.linalg.norm(eig)
    if norm > 0:
        eig = eig / norm
    with open(os.path.join(directory, "MODECAR"), "w") as f:
        for vx, vy, vz in eig:
            f.write(f"{vx:.16f} {vy:.16f} {vz:.16f}\n")


_EXTRA_FILE_BUILTINS = {
    "modecar": write_modecar,
}


def load_extra_input_writer(spec):
    """Resolve one ``extra_input_files`` value to a callable ``(calc, atoms, dir) -> None``.

    Same grammar as :func:`load_input_generator`: a built-in name (``modecar``),
    ``package.module:func``, or ``/path/to/file.py:func``.
    """
    if callable(spec):
        return spec
    if spec in _EXTRA_FILE_BUILTINS:
        return _EXTRA_FILE_BUILTINS[spec]
    if ":" not in spec:
        raise ValueError(
            f"Unknown extra_input_files writer {spec!r}. Use a built-in "
            f"({', '.join(sorted(_EXTRA_FILE_BUILTINS))}), 'package.module:func', "
            f"or '/path/to/file.py:func'."
        )
    return _import_callable(spec)


#==============================================================================
### EXTRA OUTPUTS (parsed from the VASP dir after the run, merged into .info)

def read_vtst_dimer(calc, atoms, directory):
    """Parse a finished VTST dimer run: ``eigenmode`` (NEWMODECAR) + ``curvature`` (DIMCAR).

    Returns a dict of ``.info`` keys to merge onto the output frame. The mode in
    ``NEWMODECAR`` is in POSCAR (symbol) order; it's mapped back to atoms order via
    ``calc.resort`` — the inverse of what :func:`write_modecar` does on the way in.
    Missing files are skipped, so this is safe to set even on non-dimer runs.
    """
    import numpy as np
    info = {}

    newmodecar = os.path.join(directory, "NEWMODECAR")
    if os.path.isfile(newmodecar):
        try:
            mode = np.loadtxt(newmodecar, dtype=float).reshape(len(atoms), 3)
            resort = getattr(calc, "resort", None)
            if resort is not None:
                mode = mode[resort]              # POSCAR order -> atoms order
            info["eigenmode"] = mode
        except (ValueError, OSError):
            pass

    # DIMCAR columns: Step Force Torque Energy Curvature Angle -> curvature is col 4.
    dimcar = os.path.join(directory, "DIMCAR")
    if os.path.isfile(dimcar):
        last = None
        try:
            with open(dimcar) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            float(parts[1])      # numeric row (skips the header)
                            last = parts
                        except ValueError:
                            continue
            if last is not None:
                info["curvature"] = float(last[4])
        except OSError:
            pass

    return info


_EXTRA_OUTPUT_BUILTINS = {
    "vtst_dimer": read_vtst_dimer,
}


def load_extra_output_parser(spec):
    """Resolve one ``extra_outputs`` value to a callable ``(calc, atoms, dir) -> dict``.

    Same grammar as :func:`load_input_generator`: a built-in name (``vtst_dimer``),
    ``package.module:func``, or ``/path/to/file.py:func``. The returned dict is
    merged into the output frame's ``.info``.
    """
    if callable(spec):
        return spec
    if spec in _EXTRA_OUTPUT_BUILTINS:
        return _EXTRA_OUTPUT_BUILTINS[spec]
    if ":" not in spec:
        raise ValueError(
            f"Unknown extra_outputs parser {spec!r}. Use a built-in "
            f"({', '.join(sorted(_EXTRA_OUTPUT_BUILTINS))}), 'package.module:func', "
            f"or '/path/to/file.py:func'."
        )
    return _import_callable(spec)
