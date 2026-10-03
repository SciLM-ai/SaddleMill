# Branch Context: `ndn454_branch`

This branch is Nathan Nguyen's experimental/benchmark SaddleMill fork. It is intentionally **not** upstream `main`.

Important rules for coding agents and CLI assistants working in this branch:

- Do not merge, rebase, force-push, or commit changes to `main` unless explicitly requested by the repository owner.
- Treat `ndn454_branch` as a runnable experimental branch containing substantial changes from the original SaddleMill source.
- The imported source baseline for this publication is the validated immutable snapshot `0.1.1rc3-new7-real-force-fmax-diagnostics-v1` with runtime-tree SHA-256 `dd76677f3d6cfb696a28a3a7ba59a8123659d4fd57ee87e98331e2103c76447c`.
- Preserve scientific-method definitions when making implementation or performance changes. Do not silently alter optimizer settings, convergence thresholds, calculator/task definitions, structure-equivalence rules, or initialization semantics.
- Historical benchmark Sella configs used with this source are stored under `benchmark_configs/sella_cohort0_20260929/` for reference. Do not treat those configs as global defaults.
- When asked to change this source, prefer complete tested changes on this branch and keep upstream/main separation explicit.

---

# SaddleMill - High-Throughput Transition State Search Manual

## Purpose

This file is the working manual for the SaddleMill source tree. It is intended to save future developers and coding agents from repeatedly reverse-engineering the repository before making a change.

It combines two kinds of information:

1. **Operational/runtime knowledge** carried forward from the earlier `CLAUDE.md`: what SaddleMill does, how it is launched, calculator and HPC assumptions, configuration ownership, resume semantics, outputs, worker behavior, and important deployment caveats.
2. **A current source-code map** of the refactored codebase: which module owns each behavior, how the Dimer/minimum-mode subsystems fit together, and which tests protect each implementation family.

The code has changed substantially since the original manual. Where the old manual described an implementation that no longer exists, this document preserves the useful concept but describes the current implementation instead. For example, Sella is now an engine under the Dimer attempt runner rather than a separate top-level `Sella` method, and exact Hessian work now has a dedicated `Hessian` method rather than being primarily a SinglePoint option.

This manual assumes a normal Git repository containing one active SaddleMill source tree. It does **not** require an external archive of immutable source versions. Repository history belongs in Git; this file describes the currently checked-out code.

---

# Overview

SaddleMill is a Python library for high-throughput transition-state search, geometry optimization, Hessian evaluation, and related atomistic workflows. It supports neural-network potentials through FAIRChem/UMA and DFT through ASE VASP or the maintained `VaspInteractive` package.

The runtime can execute serially or distribute work through executorlib + Flux. The earlier production manual records deployments on several GPU node layouts, including:

- **4 x NVIDIA A100 GPUs per node**;
- **3 x NVIDIA A100 GPUs per node**;
- **NVIDIA GH200 systems**.

Those are deployment profiles, not scientific assumptions. GPU count, partition name, scheduler syntax, worker count, and node shape must remain outside the scientific method and should not be baked into optimizer logic.

On Lonestar6 deployments, the production A100 partition is commonly `gpu-a100`, while `gpu-a100-dev` is useful for short validation or diagnostic work when its limits fit the job. These names are cluster-specific operational information; a different site should replace them rather than changing the scientific configuration.

The main package is `saddlemill/`. Most algorithm-development complexity lives in `saddlemill/dimertools/`, which contains the minimum-mode, rotation, translation, quasi-Newton, Hessian-model, RFO/RAS, mode-scheduling, and diagnostic machinery used by the Dimer search framework.

## Dependencies and calculator assumptions

The earlier package baseline required:

- `ase >= 3.26.0`;
- `fairchem-core >= 2.19.0`.

The target repository's `pyproject.toml` or environment specification is authoritative if those pins change. The important compatibility facts to preserve are:

- `fairchem-core` registers the ASE LMDB backend used for `.aselmdb` files.
- Code that opens ASE LMDB files must import `fairchem.core.datasets` before `ase.db.connect(...)` so the backend is registered.
- FAIRChem/UMA is the normal GPU calculator path.
- `Vasp` and `VaspInteractive` are instantiated per job unit rather than shared globally like a neural-network calculator.

### VaspInteractive

Use the maintained `tiangroup-uofa` `vasp_interactive` package (`from vasp_interactive import VaspInteractive`). The old upstream `ulissigroup` implementation is frozen and is not the supported runtime. Do not substitute ASE's bundled `ase.calculators.vasp.interactive.VaspInteractive` for the maintained package.

Installation details should live in the repository's setup/README documentation rather than being duplicated as hard-coded commands here.

---

# Supported Runtime Methods

The current top-level `method` values are:

| Method | Main module | Purpose |
|---|---|---|
| `NEB` | `saddlemill/nebopt.py` | Nudged Elastic Band search, including optional switched DNEB behavior and band-management features. |
| `Dimer` | `saddlemill/dimeropt.py` | First-order saddle search and the main experimental minimum-mode framework. |
| `Minimization` | `saddlemill/geomopt.py` | Ordinary geometry minimization. |
| `DoubleMinimization` | `saddlemill/geomopt.py` | Displace from a saddle along a mode in both directions, relax both endpoints, and characterize the reaction. |
| `SinglePoint` | `saddlemill/geomopt.py` | Energy/force evaluation, including batched FAIRChem execution and LMDB I/O. |
| `Hessian` | `saddlemill/hessian_job.py` | Standalone exact physical-Hessian evaluation/certification workflow. |

### Sella is now a Dimer engine

The old manual documented `method = Sella` and a standalone `sellaopt.py`. That is no longer the architecture. In the current source, Sella is selected as an alternate first-order saddle engine inside the Dimer orchestration layer. Attempt generation, attempt identity, resume handling, output bookkeeping, and much of the surrounding lifecycle therefore remain shared while the optimizer engine changes.

The current implementation lives primarily in:

- `saddlemill/dimeropt.py`;
- `saddlemill/sella_engine.py`;
- `saddlemill/sella_ablation.py`;
- `saddlemill/sella_diagnostics.py`.

This is an important architectural fact: do not reintroduce a parallel top-level Sella workflow unless there is a compelling reason to duplicate Dimer orchestration.

---

# Runtime Architecture

At the highest level:

```text
config.ini
    |
    v
config_defaults.py + config_parsing.py + config_validation.py + config_factories.py
    |
    v
config.py
    |
    v
__main__.py
    |\
    | +--> input discovery / resume / selective redo / continuation extraction
    |
    +--> init_function.py
    |       |
    |       +--> calculator setup
    |       +--> GPU/MPS worker setup
    |
    +--> selected top-level method
            |
            +--> nebopt.nebopt
            +--> dimeropt.dimeropt
            +--> geomopt.geomopt
            +--> geomopt.doublegeomopt
            +--> geomopt.singlepoint
            +--> hessian_job.hessian_job
```

The Dimer path is deeper because it is also the experimental optimizer framework:

```text
dimeropt.py
    |
    +--> dimertools/structure_edit.py     initial attempt generation
    +--> dimer_lifecycle.py              per-attempt timing/state/lifecycle
    +--> dimertools/dimer_factory.py     component assembly
    |       |
    |       +--> minimum-mode solver
    |       +--> rotation solver/history/model
    |       +--> translation optimizer
    |       +--> quasi-Newton / Hessian model
    |       +--> mode scheduler/predictor
    |
    +--> sella_engine.py                 alternate Sella execution engine
    +--> diagnostics_io.py / attempt_metrics.py
```

Auxiliary user-facing entry points retained from the earlier manual include:

```text
python -m saddlemill.status [run-directory]
python -m saddlemill.analyze_neb [run-directory]
```

Use these rather than writing one-off filesystem scanners when they already answer the question.

---

# Configuration Model

## Config-section ownership rule

This rule from the original manual remains important:

> A section named for an external class or library is a pass-through section. A SaddleMill-owned option belongs in `[Main]` or an `our*` section.

Examples of pass-through/external sections include sections such as:

- `[Vasp]`;
- `[FAIRChemCalculator]`;
- `[DimerControl]`;
- `[BaseNEB]`;
- ordinary ASE optimizer sections such as `[MDMin]`, `[LBFGS]`, and `[FIRE]` when present in the active configuration.

Do **not** put SaddleMill-only keys into these sections. For example, an unknown key in `[Vasp]` can be forwarded into ASE VASP handling and end up as an unintended INCAR parameter.

SaddleMill-owned behavior belongs in sections such as:

- `[Main]`;
- `[ourNEB]`;
- `[ourDimer]`;
- `[ourSella]`;
- `[ourMinimization]`;
- `[ourDoubleMinimization]`;
- `[ourSinglePoint]`;
- `[ourHessian]`;
- `[ourVasp]`;
- `[ourLBFGS]`;
- the advanced `ourDimer*`, `ourMode*`, `ourPhysicalHessian`, `ourPartitionedLBFGS`, `ourRFO`, `ourIsopotential`, and related experimental sections.

The authoritative current defaults are in `saddlemill/config_defaults.py`; parsing/migration lives in `config_parsing.py`; legality and cross-field constraints live in `config_validation.py`.

## Core `[Main]` behavior

Current defaults include the following important execution semantics:

| Key | Current default/meaning |
|---|---|
| `executorlib` | `True`; distributed executor path when enabled. |
| `method` | required; one of the supported top-level methods above. |
| `dir_path` | input directory, default `.`. |
| `Optimizer` | ordinary optimizer selector; default `MDMin`. |
| `fmax` | default `0.05`. |
| `steps` | default `1000`. |
| `Calculator` | default `FAIRChemCalculator`. |
| `jobs_per_gpu` | default `1`; GPU-sharing is opt-in. |
| `run_jobs` | default `remaining`. |
| `input_statuses` | default `all`. |
| `continue_from_result` | default `True`. |
| `zip` | default `True` for supported debug artifacts. |
| `max_consecutive_errors` | worker self-health threshold; default `5`. |
| `restart_limit` | executor worker restart limit; default `3`. |
| `input_format` | `traj` by default; `lmdb` is currently SinglePoint-only. |
| `attempt_chunk_size` | Dimer launcher chunking; `0` disables splitting. |
| `rng_scheme` | legacy stream by default; keyed deterministic schemes are opt-in. |

Do not duplicate these defaults in scientific modules. If a default changes, `config_defaults.py` should remain the single source of truth.

## Important advanced configuration families

The current Dimer framework has many independent selector families. Before adding a new key, determine whether it belongs to an existing family:

- `[ourDimer]`: top-level Dimer engine, initialization, rotation, translation, and search-policy selectors;
- `[ourDimerLBFGS]`: custom rotational/translation L-BFGS settings and reconstruction options;
- `[ourDimerHistory]`: canonical raw-force history and reconstructed secant behavior;
- `[ourDimerHybrid]`: FIRE/L-BFGS hybrid translator;
- `[ourDimerCG]`, `[ourDimerBroyden]`: rotation/model families;
- `[ourPartitionedLBFGS]`: P/Q partitioned translator;
- `[ourPhysicalHessian]`: physical-Hessian model updates;
- `[ourMinMode]`: iterative minimum-mode solver selection and convergence;
- `[ourModeSchedule]`: mode-solve skip/refresh policy;
- `[ourModePredictor]`: predicted-mode proposal mechanisms;
- `[ourModeDiagnostics]`: diagnostic-only mode-quality/reference behavior;
- `[ourRFO]`: RFO/P-RFO/QN-MMF and restricted-step controls;
- `[ourIsopotential]`: isopotential estimators/diagnostics;
- `[ourQNShadow]`: shadow/dense QN diagnostics;
- `[ourSella]`: Sella engine controls and ablations;
- `[ourHessian]`: standalone Hessian performance/storage policy.

Many of these selectors are scientific-method choices. Do not silently turn a diagnostic or performance refactor into a changed optimizer definition.

---

# Input, Resume, and Redo Semantics

## `run_jobs`

`run_jobs` selects categories of previously observed work. The canonical categories are:

| Category | Meaning |
|---|---|
| `converged` | Stored status begins with `converged`. |
| `not_converged` | Stored status begins with `not_converged`. |
| `errored` | Stored status begins with `error`, or is otherwise treated as an execution error. |
| `remaining` | No completed status record exists for the logical work unit. |
| `all` | All four categories. |

Historical aliases include `not_started -> remaining` and `error -> errored`.

For ordinary methods, a job can be selected if any of its status rows matches a requested category. NEB is special: convergence is band-level, so all sub-bands must be converged for the band to be categorized as converged.

Selective redo is not a disposable rerun. The orchestration layer archives/reconciles prior status/output artifacts and keeps unselected completed work active.

## `continue_from_result`

`continue_from_result=True` means selected redo work should reuse the previous result structure/state when the method supports it. `False` means selected work should restart from the original input or regenerate its fresh attempt state.

The granularity differs by method:

- Dimer: per attempt;
- Sella engine: per Dimer attempt;
- NEB: full band;
- DoubleMinimization: per side;
- Minimization: per job;
- Hessian: logical Hessian job/artifact identity;
- SinglePoint: normally no iterative continuation, though resume bookkeeping still applies.

Continuation is distinct from an externally supplied `initial_guess`: an initial guess is scientific input; continuation is recovery/re-execution of prior SaddleMill work.

## `input_statuses`

`input_statuses` filters source frames before they become work. Patterns use status strings stored on prior SaddleMill outputs, with wildcard matching supported by the runtime helpers.

`all` means no filtering. An explicit filter is deliberate: a frame with no stored status does not implicitly match a narrow status filter.

This is different from `run_jobs`:

- `input_statuses` decides **which source frames enter this run**;
- `run_jobs` decides **which logical work from this run's own prior state should execute/re-execute**.

Do not conflate the two.

## LMDB

`input_format=lmdb` is currently supported only for `method=SinglePoint`. The ASE LMDB backend must be registered by importing `fairchem.core.datasets` before connecting.

Current LMDB resume support is intentionally conservative: the validated path supports `run_jobs=remaining`; do not assume the trajectory-output selective-cleaning machinery applies identically to LMDB outputs.

---


# Attempt Generation Vocabulary

The original manual documented the reaction/initialization vocabulary because it is part of the user-facing scientific interface. That information should remain easy to find.

`dimertools/structure_edit.py` currently dispatches **bulk** attempts with:

| Reaction type | Purpose |
|---|---|
| `vacancy` | Vacancy-centered hop / local rearrangement family. |
| `hop_reuse` | Move an existing atom toward an interstitial-like site. |
| `hop_insert` | Insert a new atom into an interstitial-like site. |
| `kickout_reuse` | Existing-atom kickout mechanism. |
| `displace_kickout_reuse` | Displacement/kickout variant using an existing atom. |
| `kickout_insert` | Inserted-atom kickout mechanism. |
| `ring` | Cooperative ring/exchange motion. |
| `all_atoms` | Broad all-atom displacement initialization. |
| `random_bubble` | Local randomized bubble-style initialization. |
| `initial_guess` | Use the supplied geometry as the attempt geometry. |

The **OC/surface** dispatch currently includes:

| Reaction type | Purpose |
|---|---|
| `all_movable` | Displace all non-fixed/movable atoms. |
| `adsorbate_atom` | Localized perturbation of one adsorbate atom. |
| `adsorbate_atom_neighbors` | Adsorbate-centered perturbation including neighbors. |
| `adsorbate` | Adsorbate-wide perturbation. |
| `diffusion` | Translational adsorbate diffusion guess. |
| `rotation` | Rigid/collective adsorbate rotation guess. |
| `adsorbate_surface` | Joint adsorbate/surface perturbation. |
| `surface` | Surface-reconstruction perturbation. |
| `custom` | Let `[DimerControl]` fully define the displacement. |
| `random_bubble` | Local randomized bubble-style initialization. |
| `initial_guess` | Use the supplied geometry as the attempt geometry. |

The exact displacement amplitudes, concentration/bubble rules, atom-selection policies, Gaussian tails, supercell handling, and keyed RNG behavior live in `structure_edit.py` and the relevant config sections. Do not recreate the reaction-type logic in a campaign wrapper.

`initial_guess` deserves special treatment: it represents an externally prepared saddle guess and should not be conflated with `continue_from_result`, which is SaddleMill resume behavior.

---

# Output and Metadata Conventions

Every scientific output must remain attributable to its source identity and run semantics. The exact files differ by method, but the established conventions include:

- status CSV shards;
- collected output trajectories or LMDBs;
- optional debug archives;
- method-specific sidecars such as Hessian summaries/artifacts;
- top-level `Atoms.info` metadata for scientific/result identity.

Common metadata concepts include:

- `src_index` or equivalent source identity;
- `status`;
- `task_name` for calculator/model provenance where applicable;
- `orig_info` lineage;
- method-specific attempt/side/sub-band identities;
- convergence and cost counters;
- mode/curvature or Hessian metadata where applicable.

### `.info` lineage rule

The earlier manual's `orig_info` rule remains important: source metadata is preserved rather than repeatedly overwritten. Current code may nest lineage through continuations, while each method writes its new result metadata at the top level.

When reading inherited information, inspect the current top level first where appropriate and then the preserved source lineage according to the helper used by that subsystem. Do not casually flatten or rewrite provenance.

### Mode naming

Approximate search modes and exact Hessian eigenmodes are different kinds of evidence. Do not rename one as the other merely because both are vectors. The current Hessian and Dimer/Sella code deliberately preserve that distinction.

### Physical cost accounting

An optimizer step count is not automatically an exact PES evaluation count. Dimer, Sella, Hessian, and diagnostic code may make extra force/HVP/eigensolver calls. Preserve the distinction between:

- optimizer iterations;
- physical energy/force evaluations;
- HVP/mode-solver work;
- Hessian work;
- wall time.

---

# Calculator and VASP Behavior

## FAIRChem / UMA

FAIRChem calculators are normally instantiated once per worker and reused across work units. Batched SinglePoint execution can evaluate multiple frames per calculator call when configured.

Model caching should use `FAIRCHEM_CACHE_DIR` or the deployment's equivalent cache location rather than repeatedly downloading model data into transient job directories.

## VASP / VaspInteractive

VASP-family calculators are per-job-unit objects because each calculation needs an isolated working directory and process/filesystem lifecycle.

The `[Vasp]` section is an ASE/VASP pass-through. SaddleMill orchestration belongs in `[ourVasp]` and method-owned `our*` sections.

`[ourVasp]` supports three extension points:

- `input_generator`: calculate VASP settings from the structure;
- `extra_input_files`: write additional files such as a MODECAR into the VASP directory;
- `extra_outputs`: parse extra VASP/VTST outputs back into SaddleMill metadata.

The current implementation is in `saddlemill/vasp_io.py` and the shared calculator lifecycle helpers in `saddlemill/tools.py`.

Important retained rules from the earlier manual:

- explicit `[Vasp]` user settings override generator-produced settings;
- external calculator sections must not contain SaddleMill-only keys;
- `VaspInteractive.finalize()` must be called when required so persistent VASP subprocesses do not outlive a job;
- heavy VASP files can be removed/archived according to the run's debug policy rather than accumulating indefinitely;
- VASP SinglePoint does not support batched `frames_per_job > 1`;
- method-specific VASP launcher commands/ncore settings belong in the relevant `our*` section.

The historical manual also documented built-in VASP recipe/input hooks such as OMat/OC presets and VTST modecar/dimer parsing. Those capabilities are owned by `vasp_io.py`; inspect that module before duplicating recipe logic elsewhere.


## Built-in VASP input and VTST hooks

The built-in `[ourVasp] input_generator` names currently include:

- `omat24_static`;
- `omat24_relax`;
- `cheap_omat`;
- `oc20`;
- `cheap_oc20`;
- `oc22`;
- `cheap_oc22`.

Custom generators use either `package.module:func` or `file.py:func` syntax and return ASE-Vasp keyword arguments from an `Atoms` object. The generator computes settings only; ASE still owns input-file writing, atom sorting, POTCAR construction, and force resorting.

Generator-produced ionic-driver tags are stripped where SaddleMill itself drives geometry. This prevents a reusable VASP recipe from silently taking over ionic motion through `IBRION`/`NSW`/`POTIM`/`EDIFFG` when the ASE/SaddleMill optimizer is supposed to own the geometry.

Built-in extra-file/output hooks include:

- `modecar`: write a VTST MODECAR from the available mode, reordered into VASP/POSCAR atom order;
- `vtst_dimer`: read VTST dimer outputs such as `NEWMODECAR`/`DIMCAR` back into SaddleMill metadata.

The old manual contained detailed electronic-structure recipes for the OMat/OC presets. Those recipes remain implementation knowledge in `vasp_io.py`; do not replace them with a newer pymatgen default merely because the external library has changed. Presets intended to reproduce a published dataset/protocol should remain reproducible until deliberately changed.


---

# NEB Operational Model

The original manual contained useful behavior that remains conceptually important:

1. optional endpoint relaxation occurs before band optimization;
2. interpolation may be supplied or generated;
3. `OCPNEB` owns the band force logic;
4. intermediate-minimum detection is an explicit event, not a hidden periodic background process;
5. detected minima can become segment boundaries;
6. climbing images are selected per segment when segmented behavior is active;
7. frozen images cache their effective NEB state and avoid unnecessary reevaluation where supported;
8. optional post-NEB Dimer refinement and/or additional band optimization can run without discarding the existing band state;
9. VASP execution cannot assume every FAIRChem-specific batching/freezing/refinement optimization is available.

`nebopt.py` owns workflow/lifecycle decisions. `catsunami/ocpneb.py` owns NEB force construction, image state, DNEB switching, and effective per-image force bookkeeping. Keep those responsibilities separate.

---

# Dimer / Sella Operational Model

Dimer attempt generation is shared infrastructure. A structure can produce multiple reaction-type/initialization attempts, each with a stable attempt identity and its own execution/error handling.

The current Dimer framework is much broader than the old ASE-only dimer implementation. It can compose different:

- engines (including Sella);
- minimum-mode finders;
- rotational optimizers;
- translation optimizers;
- force-history policies;
- quasi-Newton reconstructions;
- physical-Hessian models;
- skip/refresh schedulers;
- diagnostic/shadow analyses.

Do not infer that two configurations are equivalent merely because both are called "LBFGS" or "Dimer". The exact selected engine, history source, projector, pair admission policy, H0 scaling, rotation geometry, and translation algorithm can materially change the method.

Sella shares the outer Dimer attempt lifecycle but has its own native optimizer semantics. Its diagnostics and ablations should remain instance-local and should not silently patch global Sella behavior.

---

# Hessian Operational Model

Exact Hessian work is now a dedicated top-level `Hessian` method.

The standalone Hessian path:

- computes physical Hessian/eigen information using the validated calculator route;
- treats fixed-atom restriction and chunk size as performance/storage concerns rather than redefining the scientific Hessian;
- emits status, summary, optional stored Hessian, trajectory, and transactional artifact records;
- uses atomic/transactional artifact handling so interrupted jobs do not masquerade as completed Hessians;
- uses the configured negative-eigenvalue tolerance for order classification;
- is currently validated on CUDA;
- enforces `jobs_per_gpu=1` for the production Hessian method rather than running Hessians under MPS.

Older documentation that described exact-Hessian work as a SinglePoint option should be read as historical context. The dedicated Hessian workflow is the current implementation to extend.

---

# Execution Modes and HPC Deployment

## Serial mode

With executorlib disabled, SaddleMill runs one work item at a time in the local process. This is useful for debugging and small validation runs.

## Distributed mode

With executorlib enabled, work is distributed through executorlib/Flux. The typical model is one long-lived worker process per GPU, with the calculator initialized once and jobs processed sequentially by that worker. `jobs_per_gpu > 1` intentionally changes this to GPU sharing.

The historical launch pattern was conceptually:

```text
Slurm allocation
    -> one Flux broker/process group per allocated node
    -> SaddleMill worker initialization
    -> one or more worker slots per GPU
    -> many logical SaddleMill jobs over the life of the allocation
```

Do not copy one site's literal `srun`/`sbatch` flags into another cluster. In particular, a historical 4-A100 launcher that passed `--gpus-per-node=4` described that site/layout; it is not a scientific requirement and may be invalid on another system.

### Known GPU deployment profiles

Operational knowledge retained from the old manual:

- 4-A100-per-node HPC deployments have been used;
- 3-A100-per-node deployments have been used;
- GH200 deployments have been used;
- the code must not assume a fixed GPU count from any one of those systems.

For Lonestar6-specific workflows, `gpu-a100` is the production A100 partition and `gpu-a100-dev` is commonly useful for short diagnostics. Keep partition/resource selection in launch/orchestration code, never in scientific method definitions.

## GPU sharing and MPS

When `jobs_per_gpu > 1`, multiple workers may share a GPU through CUDA MPS. `init_function.py` and `worker_resources.py` own worker/GPU slot assignment and MPS environment setup.

Important invariants:

- do not silently fall back to GPU 0 when an expected MPS control socket is missing;
- do not start/use MPS when there is only one job per GPU;
- GPU assignment must remain stable across workers on a node;
- method-specific restrictions override generic sharing capability (for example, the standalone Hessian method currently requires one job per GPU).

## CUDA/process failure handling

A CUDA device-side assert can poison the process' CUDA context. The worker-health design from the old manual remains relevant:

1. methods track consecutive all-error work;
2. successful work resets the counter;
3. after the configured threshold, the worker exits instead of continuing with a poisoned context;
4. executorlib may restart the worker up to `restart_limit`;
5. a restarted worker reruns initialization and obtains a fresh process/CUDA context.

Do not convert deterministic scientific/configuration failures into endless automatic retries.

## Environment notes

- Set `FAIRCHEM_CACHE_DIR` appropriately for model caching.
- CUDA libraries must be discoverable through the deployment environment.
- Flux, Slurm, MPS, VASP, and module/environment setup are deployment concerns; keep them outside the scientific kernels.
- Site-specific launch wrappers may impose additional rules. Follow the target cluster's current wrapper/submission conventions rather than an old command copied from documentation.

---

# Testing and Validation

The earlier manual separated CPU, GPU, and Flux tests. Keep that distinction.

Typical pytest patterns are:

```bash
# CPU-only / non-scheduler tests when markers are available
pytest -m "not gpu and not flux" -v

# GPU-marked tests on a suitable CUDA node
pytest -m gpu -v

# Full suite when the complete runtime environment is available
pytest -v --timeout=600
```

The exact collected test count is not a stable contract and should not be hard-coded into this manual. The current focused regression files are mapped later in this document.

When adding a feature or fixing a bug:

- add a regression test at the narrowest layer that captures the behavior;
- add an integration test when the bug involved composition between layers;
- keep scientific-method changes explicit in the test names/configuration;
- avoid using a diagnostic replay as proof that the live production path changed correctly unless the production path itself is also covered.

---

# Source Layout

A normal single-source repository should look conceptually like:

```text
.
├── AGENTS.md
├── README.md / pyproject.toml / environment files   # when present in the host repository
├── saddlemill/
│   ├── __main__.py
│   ├── config*.py
│   ├── dimeropt.py
│   ├── geomopt.py
│   ├── nebopt.py
│   ├── hessian_job.py
│   ├── sella_*.py
│   ├── dimertools/
│   ├── catsunami/
│   └── ...
└── tests/
```

The remainder of this manual is the **deep source-code addendum**: it explains where the implementation lives so an agent can jump directly to the correct layer instead of rediscovering the tree.

---

# Configuration System

Configuration is deliberately split across several modules. Do not assume `config.py` contains all defaults or all validation.

## `saddlemill/config_defaults.py`

Data-only definition of canonical configuration sections and default values. Importing it should not import heavy calculators, CUDA libraries, optimizers, or method implementations.

This is the first place to look when asking:

- Does a config key exist?
- Which section owns it?
- What is the default when it is omitted?
- Is a feature default-off or active by default?

The configuration schema contains general execution sections, method-specific `our*` sections, external-class pass-through sections, and advanced Dimer/minimum-mode sections. Do not copy numeric defaults into another module unless there is a strong reason; keep the authoritative default here.

## `saddlemill/config_parsing.py`

Small parsing layer for INI values and backwards-compatible key migration. It provides the historical string-to-Python coercion behavior, renamed-key migration, and file merge into a default-filled config mapping.

Edit this when changing **syntax or migration**, not when changing method semantics.

## `saddlemill/config_validation.py`

Method-specific normalization and fail-closed validation. This is where incompatible selector combinations, invalid values, unsupported engine combinations, and method-specific requirements are rejected.

For Dimer work, this module is critical because many advanced components are independently selectable. If you add a new selector or composition, update validation here rather than relying on a later attribute error.

## `saddlemill/config_factories.py`

Lazy dispatch layer that turns validated names into runtime callables/classes without importing every heavy dependency up front.

Key responsibilities:

- calculator factory selection;
- top-level method selection;
- ordinary optimizer selection;
- construction of resolved Dimer run metadata for provenance.

Use this file when a new top-level method, calculator family, or ordinary optimizer must become selectable.

## `saddlemill/config.py`

Public configuration and resume/orchestration API. `ConfigManager` combines defaults, parsing, migration, and validation-facing access. The remainder of the module owns much of the filesystem-level execution state surrounding a run.

Major responsibilities include:

- loading configuration;
- loading method/calculator/optimizer factories through the split config modules;
- discovering trajectory/LMDB work units;
- creating result directories;
- reading status CSVs;
- classifying `run_jobs` categories;
- computing exact method-qualified resume identities;
- determining which subunits must be redone;
- archiving existing CSV/output/debug artifacts before selective redo;
- atomically filtering active outputs so kept work remains in place;
- invalidating/reconciling Hessian artifacts when a forced redo selects them;
- computing executor resources.

If the problem is “resume chose the wrong work,” “redo deleted the wrong frame,” “status rows conflict,” or “existing results were archived incorrectly,” start here rather than in the scientific optimizer.

---

# Program Entry and Worker Setup

## `saddlemill/__main__.py`

Primary `python -m saddlemill` entry point.

It:

1. loads `config.ini`;
2. resolves the top-level method;
3. discovers and freezes input ordering;
4. detects fresh versus resumable execution;
5. computes selected redo identities;
6. extracts prior continuation structures before active outputs are archived/filtered;
7. creates either a distributed executor or serial execution path;
8. loads input frames/rows;
9. applies input-status filtering;
10. optionally chunks Dimer attempts into smaller launcher tasks;
11. dispatches method calls;
12. distinguishes launcher/task failure from scientific non-convergence.

This file should remain orchestration-heavy and science-light. A new scientific method should normally be implemented elsewhere and added to factory dispatch.

## `saddlemill/init_function.py`

Per-worker initialization used by executorlib and serial mode. It loads config, establishes worker-local GPU/MPS visibility when GPU sharing is active, validates that the expected MPS control socket exists instead of silently collapsing workers onto a GPU, instantiates non-VASP calculators, selects the ordinary optimizer, and returns worker state including the consecutive-error counter.

This is process/resource setup, not the place for Dimer algorithm logic.

## `saddlemill/worker_resources.py`

Race-safe node-local GPU-slot allocation helpers for shared-GPU workers. It builds an explicit allocation/host scope, stores small local state atomically, validates the state, resolves the local GPU inventory, and assigns stable physical GPU slots.

Use this file for worker-to-GPU mapping bugs, allocation identity, or MPS resource discovery. Do not duplicate this logic in `init_function.py` or submission wrappers.

---

# Top-Level Scientific Methods

## `saddlemill/nebopt.py`

NEB workflow driver. Owns the high-level band lifecycle rather than the underlying NEB force formula.

Responsibilities include:

- endpoint extraction/relaxation;
- interpolation;
- constructing `OCPNEB`;
- optional one-shot intermediate-minimum detection;
- relaxing/freezing detected intermediate minima;
- optional band expansion by inserting images;
- optional Dimer refinement of climbing images;
- continuing refinement while preserving band state;
- status/output metadata and debug artifacts.

If the issue is “when is an event triggered?” or “how is a band expanded/resumed?”, start here. If the issue is the actual modified NEB force on an image, use `catsunami/ocpneb.py`.

## `saddlemill/catsunami/ocpneb.py`

Core NEB object. `OCPNEB` extends ASE’s NEB behavior with SaddleMill/CatSunami batching, frozen images, segment-aware climbing images, per-image effective force tracking, and optional switched DNEB mechanics.

`swDNEB` is the switched doubly-nudged force method. `_find_segment_ci()` chooses a segment climbing image.

This file owns **band force mechanics**. It should not own campaign orchestration or resume logic.

## `saddlemill/catsunami/autoframe.py`

Large CatSunami-derived reaction-frame generation module for NEB inputs. Contains `AutoFrame` plus dissociation, transfer, and desorption subclasses and the geometry/unwrapping/interpolation helpers they use.

It handles adsorbate reordering, symmetric-site choices, atom mappings, periodic unwrapping, edge-list preservation, and construction/correction of initial/final NEB frames.

This is mainly **NEB input chemistry/geometry generation**, separate from the optimizer itself.

## `saddlemill/catsunami/reaction.py`

Small reaction data model used by CatSunami frame generation. `Reaction` stores reaction-side structures/mappings/edge information and includes desorption mapping support.

## `saddlemill/geomopt.py`

Owns three top-level methods:

- `geomopt()` for ordinary minimization;
- `doublegeomopt()` for two-sided minimum relaxation from a transition-state structure;
- `singlepoint()` for energy/force evaluation and optional batched output handling.

It also owns minimizer-side diagnostic recording and helper logic for continuation endpoints, timing CSVs, and ordinary optimizer kwargs.

DoubleMinimization uses a supplied or computed unstable mode, creates the two displaced sides, relaxes them, and applies reaction checks. Hessian-specific pre-processing is delegated to `doublemin_hessian.py` / `hessian_job.py` rather than being implemented inline here.

## `saddlemill/doublemin_hessian.py`

Bridge between standalone Hessian results and DoubleMinimization.

It reads and validates stored Hessian metadata, determines the free Cartesian subspace, checks saddle order according to the standalone-Hessian convention, extracts the lowest mode, compares an input mode/curvature against Hessian results, and retains a legacy inline Hessian helper.

Use this file when the question is specifically “how does DoubleMin consume or gate on a Hessian?”

## `saddlemill/hessian_job.py`

Standalone analytical Hessian method implementation.

It:

- builds or reuses a Hessian-capable FairChem calculator;
- computes the active Cartesian Hessian in chunks;
- adapts chunk size after CUDA OOM when configured;
- handles fixed-coordinate projection;
- analyzes the order-defining subspace;
- returns eigenvalue/order information and optional full Hessian;
- delegates restart-safe publication to `hessian_artifacts.py`.

This module owns Hessian **calculation and scientific analysis**, not artifact transaction semantics.

## `saddlemill/hessian_artifacts.py`

Restart-safe artifact store for standalone Hessian jobs. It treats result publication as a transaction rather than “write a few files and hope.”

Responsibilities include:

- stable task identity;
- per-job locking;
- staged NPZ/summary/event writes;
- checksums and validation;
- recovery of interrupted publication;
- reuse of already committed results;
- quarantine/preservation of inconsistent fixed outputs;
- rebuilding rank indexes.

If Hessian computation is scientifically correct but files are missing, duplicated, stale, or half-published, debug here.

## `saddlemill/analyze_neb.py`

Standalone post-run NEB diagnostic/report generator. Parses optimizer and Dimer logs, reconstructs trajectory evolution, extracts files from debug archives, and produces overview/refinement/energy/fmax plots plus a detailed text report.

It is analysis-only and should not be imported into production NEB force mechanics.

## `saddlemill/status.py`

Read-only command-line status summary. Reads config, frozen input ordering, and status shards; computes expected entries where possible; and prints method-specific summaries such as Dimer reaction-type counts or NEB sub-band convergence.

Use it for human monitoring, not as the authoritative source of resume selection logic. Resume logic lives in `config.py`.

## `saddlemill/nebtools/create_endpoints_for_MP_batteries.py`

Standalone NEB input-preparation utility for battery/material endpoints. It is not part of the main `python -m saddlemill` runtime path.

---

# Dimer / Minimum-Mode Orchestration

## `saddlemill/dimeropt.py`

Main per-structure Dimer attempt runner and the central integration point for minimum-mode search experiments.

It is responsible for the attempt lifecycle rather than implementing every optimizer itself. Its jobs include:

- mapping configured attempt IDs to reaction-generation types;
- preparing fresh versus continuation attempts;
- setting up the selected Dimer or Sella engine;
- applying deterministic RNG provenance;
- running the optimizer;
- enforcing per-attempt error isolation;
- recording status/reaction/timing/diagnostic outputs;
- materializing final mode/curvature/metrics metadata;
- archiving per-attempt scratch;
- maintaining the structure-level consecutive-error health counter.

Two recorder classes live here:

- `ModeDiagnosticRecorder`: mode-quality history;
- `OptimizerDiagnosticRecorder`: accepted-translation diagnostics and final summary.

When a Dimer result has the wrong scientific step, the bug is usually in `dimertools/` or `sella_engine.py`. When an attempt was generated, resumed, recorded, or archived incorrectly, the bug is usually here or in `config.py`.

## `saddlemill/dimer_lifecycle.py`

Small state containers and timing helpers extracted from the large Dimer runner. `DimerRunContext` holds per-structure state; `DimerAttemptContext` holds mutable state for one attempt.

Helpers apply generated metadata, continuations, historical scratch filenames, and timing boundaries. This file exists to keep lifecycle bookkeeping separate from optimizer science.

## `saddlemill/rng_keyed.py`

Deterministic random-number architecture for order-independent attempts. It builds stable structure/attempt keys, named Python/NumPy substreams, provenance, and compatibility scopes for external code that still uses global RNG state.

Use this module whenever adding stochastic attempt generation or a stochastic fallback that must remain reproducible independent of execution order.

## `saddlemill/attempt_metrics.py`

Engine-independent terminal metrics for saddle-search attempts. Builds stable science-config/source/initial-geometry identifiers and records real/projection force criteria and audited physical PES-call counters with explicit availability semantics.

Its purpose is to make results from different engines comparable without pretending that incompatible historical counters mean the same thing.

## `saddlemill/diagnostics_io.py`

Low-overhead buffered JSONL/CSV appenders with simple I/O accounting. Used by passive instrumentation so diagnostic logging does not turn every optimizer step into a shared-filesystem write.

`append_jsonl_durable()` is the explicit durable append path; the buffered classes are for batched recording.

---

# Sella Integration

## `saddlemill/sella_engine.py`

Adapter that makes Sella behave as an alternate engine inside SaddleMill’s existing Dimer attempt runner.

It owns:

- supported Sella environment/version checks;
- construction of Sella runs from SaddleMill attempts;
- mapping input Cartesian modes into Sella coordinates;
- optional FairChem direct-Hessian callback setup/cache;
- extracting the lowest Sella model mode/spectrum;
- Sella-specific convergence classification;
- force-call accounting;
- translation of SaddleMill config into Sella constructor options.

If the issue is normal Sella setup, seeding, convergence, or Hessian injection, start here.

## `saddlemill/sella_ablation.py`

Controlled, attempt-scoped modifications of the pinned Sella API for benchmark ablations. It resolves supported component compositions, installs temporary instance/class hooks, captures exact restricted-step or eigensolver failure state when requested, and restores the original Sella state afterward.

This module is intentionally fail-closed: an ablation is allowed only when the expected Sella API is present. It should not silently patch an unknown Sella version.

## `saddlemill/sella_diagnostics.py`

Passive Sella diagnostics. `SellaPassiveQNRecorder` observes Sella QN state without changing its step. `SellaLinalgFailureRecorder` captures bounded numerical context for rare linear-algebra failures and then leaves failure semantics unchanged.

Use this for observation/debugging, not to implement a new Sella algorithm.

---

# Ordinary Minimizer / ASE L-BFGS Layer

## `saddlemill/fire_lbfgs.py`

Warm-start hybrid optimizer for ordinary Minimization/DoubleMinimization: ASE FIRE first, then ASE L-BFGS under configurable hysteresis. Includes optional L-BFGS pair safeguards and passive diagnostics.

This is **ordinary geometry optimization**, not the Dimer-specific custom L-BFGS reconstruction stack.

---

# `dimertools/`: Foundation and Shared Evidence

The `dimertools/` directory is the main research-method subsystem. It contains legacy Dimer adapters, current minimum-mode solvers, alternative rotation/translation optimizers, shared force history, QN reconstruction, RFO/RAS methods, mode scheduling, and diagnostics.

Do not treat every file here as an independent optimizer. Several modules are kernels, adapters, compatibility facades, or diagnostics around the same underlying data.

## `saddlemill/dimertools/foundation_types.py`

Typed contracts shared across advanced minimum-mode code. Defines enums and immutable dataclasses for:

- work/evaluation purpose;
- operator origin and residual semantics;
- root selection/reference status;
- active coordinate spaces;
- force accounting;
- admission policies/ledgers;
- common result metadata;
- resolved-run metadata;
- mode-predictor and mode-scheduler interfaces.

Also provides stable exact-geometry fingerprinting and immutable/JSON-safe helpers.

This file is the vocabulary layer. If two subsystems need to exchange HVP/mode/scheduler metadata, define the contract here rather than passing an undocumented dict.

## `saddlemill/dimertools/force_history.py`

Canonical projection-independent raw force history (“ForceBank”). Stores immutable `ForceObservation`s grouped into accepted translation-center `TranslationState`s, plus explicit finite-difference `DerivativeStencil`s and pair candidates.

It owns:

- observation identity and provenance;
- center/probe recording;
- bounded state retention;
- pair-candidate construction/admission bookkeeping;
- stencil relationships;
- serialization/checkpointing and optional raw NPZ dumps.

This file stores **evidence**, not a projected optimizer model. Consumers should project/reconstruct from it rather than mutating the raw history into one optimizer’s coordinate system.

## `saddlemill/dimertools/force_evaluator.py`

Observation boundary used by minimum-mode code to record forces that were already requested by the active algorithm. `PhysicalForceEvaluator` is explicitly **not a calculator wrapper** and should not add PES evaluations.

Its source-context stack labels observations as Dimer rotation, Lanczos, Davidson, reference-Hessian probes, etc., while preserving raw physical force data.

## `saddlemill/dimertools/force_stencil_history.py`

Views raw force observations as derivative stencils. Reconstructs same-center physical HVPs, Dimer torques, and force-bank rotational secants from already recorded center/probe endpoints.

Use this when a consumer needs derivative information from existing physical probes without taking another force call.

## `saddlemill/dimertools/hvp_interfaces.py`

Typed HVP and low-spectrum interface layer. Defines request/result contracts, memory budgets for explicit matrices, finite-difference and explicit-matrix backends, residual/Ritz result types, and helpers for physical eigen-residual calculations.

This is the boundary that distinguishes:

- physical/reference HVP information;
- approximate-model HVP information;
- certification eligibility;
- explicit dense-matrix resource limits.

New eigensolvers or diagnostics should consume this typed interface rather than inventing another HVP tuple format.

## `saddlemill/dimertools/force_policy.py`

Pure policies for decomposing and modifying minimum-mode translation forces, especially mode-parallel damping. Contains the resolved damping config, decomposition result types, Shang–Liu lambda rule, and sequence-level selection logic.

It should remain calculator-free and state-light: this is policy math, not force acquisition.

## `saddlemill/dimertools/runtime_state.py`

Serialization/restoration helpers for per-attempt advanced runtime state: canonical force evaluator/history plus rotational histories and related Stage-A state.

Use this for resume/checkpoint compatibility rather than giving each optimizer a private incompatible checkpoint format.

---

# Dimer State, Rotation, and Legacy Adapters

## `saddlemill/dimertools/minmode_atoms.py`

State-ownership adapters around minimum-mode atoms. `CanonicalHistoryRecordingMixin` records projection-independent center/probe observations while preserving the parent numerical path. Concrete standard and Kappa classes combine that recording with the appropriate minimum-mode implementation.

This is where raw history recording is attached to a live minimum-mode object.

## `saddlemill/dimertools/legacy_dimer_adapter.py`

Large compatibility layer for the historical Dimer implementation. It contains:

- bowl/convex-region breakout confinement;
- a small dependency-free inverse L-BFGS model;
- historical rotational L-BFGS mixin/search;
- passive entry-torque Dimer search;
- `ConfigurableRotationMinModeAtoms`, which selects Dimer or iterative minimum-mode solvers while preserving ASE-compatible behavior.

This module contains intentionally preserved legacy numerical paths. Refactors here require strong regression/equivalence tests because old benchmark behavior may depend on details that look redundant.

## `saddlemill/dimertools/ase_lbfgs_adapter.py`

ASE-specific L-BFGS compatibility and Dimer translation layer. It includes:

- optional curvature-guarded ASE L-BFGS;
- real-force convergence override for non-Sella Dimer translation;
- stock ASE Dimer translation plus diagnostics;
- an adapter around native ASE L-BFGS history/counters;
- direct ASE-LBFGS Dimer translation;
- FIRE→ASE-LBFGS hybrid Dimer translation.

This is the file to inspect when a path claims to be using **ASE L-BFGS** rather than SaddleMill’s custom reconstructed L-BFGS.

## `saddlemill/dimertools/lbfgs_dimer.py`

Compatibility facade. Historical code imported many names from this file when one monolithic implementation contained L-BFGS, Dimer rotation, diagnostics, and hybrid logic.

The real implementations now live mainly in `ase_lbfgs_adapter.py` and `legacy_dimer_adapter.py`; this file re-exports names and preserves historical module metadata/import compatibility.

Do not add new implementation here unless compatibility specifically requires it.

## `saddlemill/dimertools/dimer_rotation.py`

Generalized rotational L-BFGS search built on Dimer trial/Fourier mechanics. Supports alternative rotation geometry/transport/step policies, external force-bank pairs, and diagnostic reporting while retaining a separate legacy rotation path elsewhere.

`advanced_rotation_requested()` is the gate for when this generalized search is needed.

## `saddlemill/dimertools/dimer_entry_torque.py`

Tiny passive mixin/helper that captures the first already-computed Dimer rotational force (“entry torque”) without requesting extra PES work. Used by mode-schedule entry-gating/diagnostics.

## `saddlemill/dimertools/kappa_dimer.py`

Kappa Dimer variant with separate Phase-A/Phase-B rotation behavior. Provides isolated controls, a Phase-B search constrained to the isopotential hyperplane, an optional L-BFGS rotation variant, and `KappaMinModeAtoms` which blends/recovers translation behavior.

## `saddlemill/dimertools/sphere_manifold.py`

Calculator-free geometry for treating an eigenmode as an **unoriented unit-sphere axis**. Implements sign alignment, tangent projection, great-circle distance, exponential/retraction steps, logarithm map, and parallel transport.

Rotation methods should reuse these operations rather than open-code slightly different tangent/sphere geometry.

---

# Rotational Optimization Models

## `saddlemill/dimertools/riemannian_lbfgs.py`

Limited-memory BFGS for rotation on the unoriented unit sphere. Defines rotational secants, sequential history, transport policies, and `RotationLBFGSModel` for applying raw-pair L-BFGS after appropriate tangent-space transport.

This is the modern geometric rotational-LBFGS model. It is distinct from the simpler historical inverse-Hessian implementation in `legacy_dimer_adapter.py`.

## `saddlemill/dimertools/rotation_history.py`

Alternative state-window rotational L-BFGS history keyed to canonical center states. Retains admissible rotational secants over a bounded state window and applies the corresponding two-loop model.

This is used when rotation history semantics are tied to ForceBank/canonical-state retention rather than just the sequential accepted optimizer path.

## `saddlemill/dimertools/cg_rotation.py`

Calculator-free nonlinear conjugate-gradient direction state for Dimer rotation. Implements PR+ recurrence on the unoriented sphere, reset/recovery policies, and serialization.

It supplies directions; Dimer trial/Fourier mechanics remain elsewhere.

## `saddlemill/dimertools/broyden_rotation.py`

Sphere/tangent adapter that wraps the generic Broyden kernels for rotational use. Handles identity resets, tangent application, and state serialization.

## `saddlemill/dimertools/wave_b_rotation.py`

Thin shared-runtime adapters that plug verified PR+ CG or Broyden direction models into the existing generalized Dimer rotational mechanics. It deliberately disables L-BFGS pair admission when using those alternative direction kernels so only the selected rotation model drives the direction.

---

# Translation Optimizers and Quasi-Newton Reconstruction

## `saddlemill/dimertools/translation_optimizers.py`

ASE-facing translation optimizer classes for the advanced Dimer path:

- `CanonicalHistoryLBFGSMinModeTranslate`: rebuilds custom L-BFGS from raw canonical force history under the current mode/projector;
- `PartitionedLBFGSMinModeTranslate`: adapter for P/Q-partitioned or Q-only+Dimer-axial translation;
- `RFOPRFOMinModeTranslate`: adapter for native RFO/P-RFO/QN-MMF translation.

This file connects pure reconstruction/step kernels to the live ASE optimizer lifecycle, logging, step caps, trial behavior, and diagnostics.

## `saddlemill/dimertools/quasi_newton.py`

Projection/reconstruction consumer for raw ForceBank observations. `ProjectionSnapshot` freezes the current effective force-field definition; `reconstruct_lbfgs()` projects raw observations consistently and invokes canonical reconstruction to produce a custom L-BFGS direction.

Use this file when the scientific question is “how do raw physical observations become the current projected translation QN model?”

## `saddlemill/dimertools/qn_reconstruction_core.py`

Pure canonical quasi-Newton reconstruction stages. It separates:

1. projection of referenced observations;
2. raw secant construction;
3. pair safeguard/admission;
4. memory capping;
5. L-BFGS two-loop application;
6. optional dense reconstructed model;
7. final direction selection.

This is one of the most important shared numerical kernels. New custom L-BFGS consumers should reuse it instead of copying another two-loop implementation.

## `saddlemill/dimertools/qn_reconstruction_diagnostics.py`

Optional diagnostic payload construction around canonical reconstruction. Builds deep diagnostics and exact lossless replay state from an already constructed core result without changing pair admission or production arithmetic.

## `saddlemill/dimertools/canonical_diagnostics.py`

Shared additive diagnostic schema for canonical-history and translation-regularization paths. It defines common regularization fields, summarizes any history-enabled minimum-mode object into stable CSV-style fields, and optionally writes bounded raw-history dumps when explicitly requested.

Use this module to keep diagnostic column names/semantics consistent across translation implementations; it should not own optimizer state transitions or take calculator calls.

## `saddlemill/dimertools/qn_deep_diagnostics.py`

Low-level QN diagnostics plus low-memory shifted-LBFGS solvers. Includes pair/block diagnostics, rigid-translation projections, ordinary/reference two-loop helpers, and compact solves of shifted direct-BFGS systems for regularized/trust-region variants.

Some functions are retained specifically as regression/reference implementations. Do not assume every implementation here is a production path.

## `saddlemill/dimertools/lbfgs_state_dump.py`

Lossless diagnostic serialization and offline replay for translation L-BFGS. Can trace the two-loop recursion, write compressed state dumps, load them, and replay a state independently.

Use this when debugging “why did this exact L-BFGS step happen?” without rerunning the PES calculation.

## `saddlemill/dimertools/dense_bfgs.py`

Dense BFGS reconstruction and numerical helper library. Contains safe norms/cosines/caps, sequential dense reconstruction, multisecant BFGS reconstruction, spectral/condition/secant diagnostics, and scalar-shift trust-region solves.

Dense reconstructions are valuable for diagnostics/reference/compact-model validation but are not automatically appropriate for large production systems.

## `saddlemill/dimertools/dense_replay_diagnostics.py`

Bounded diagnostic-only dense replay of an exact frozen L-BFGS window. Enforces explicit resource ceilings and reconstructs the dense direct-BFGS model only when within limits.

## `saddlemill/dimertools/qn_shadow_diagnostics.py`

Common schema and builder for passive “shadow QN” diagnostics. Consumes an exact frozen replay payload and compares/reports what a dense reconstruction would do without changing the real optimizer trajectory.

## `saddlemill/dimertools/wave_b_shadow.py`

Small runtime glue for the shadow-QN diagnostic. Resolves options, ensures an explicit production active mask is used, returns structured unavailable results when the production path lacks a replay window, and invokes the passive shadow calculation.

---

# Partitioned L-BFGS and Physical-Hessian Models

## `saddlemill/dimertools/partitioned_lbfgs.py`

Pure physical-space partitioned L-BFGS translation core. Builds P/Q projectors from the current mode, reconstructs branch-specific raw pairs from ForceBank, applies branch models, supports a Q-only plus direct Dimer-axial variant, and optionally regularizes the combined step through a shared shifted solve.

The class is intentionally separated from the ASE-facing wrapper in `translation_optimizers.py`.

## `saddlemill/dimertools/physical_hessian.py`

Calculator-free approximate physical-Hessian model in typed active coordinates. Supports dense and compact representations, ordinary BFGS/TS-BFGS-style updates, same-center HVP probe blocks, translation secants, low-spectrum/inertia queries, compact QN-MMF capability, window rebuilds, and state serialization.

This model is built from already available physical observations/HVPs. Whether it is authoritative for a production decision is decided by consumers/schedulers, not by the model itself.

## `saddlemill/dimertools/mode_predictor.py`

Mode predictors that use already available physical information rather than launching a full fresh mode solve. Includes:

- translation-secant predictor;
- lowest-eigenvector predictor from the physical-Hessian model;
- projected Olsen/JD correction predictor.

Outputs include explicit identity/safety/provenance so the scheduler can fail closed when a prediction is stale or incompatible.

---

# Iterative Minimum-Mode Solvers

## `saddlemill/dimertools/minmode_solvers.py`

Iterative eigensolver library for minimum-mode searches. Contains stabilized generic Lanczos, historical SoftSaddle Lanczos, generic Davidson, SoftSaddle Davidson/hybrid variants, projected Olsen/Jacobi-Davidson correction, and a Sella-2.5-compatible Olsen/JD path through the typed HVP interface.

It also includes solver audit metadata, work accounting, retained-action summaries, and an optional persistent physical-Hessian BFGS helper used by solver variants.

If a failure concerns Ritz convergence, residual stopping, retained HVP blocks, or solver preconditioning, start here.

## `saddlemill/dimertools/mode_diagnostics.py`

Standardized mode-quality diagnostics across different solvers/operators. Converts available HVP, torque, solver-audit, or physical-Hessian information into comparable residual/gap/root-validity records; evaluates refresh criteria; and can run an explicitly paid reference measurement.

This module is designed to avoid treating an approximate/model residual as physical truth without provenance.

## `saddlemill/dimertools/hindsight.py`

Passive post hoc mode/trajectory diagnostics. Computes geometry-to-final comparisons under an explicit alignment convention and builds records comparing modes along a path to later/final reference modes.

It should not influence the trajectory being analyzed.

---

# Mode Scheduling and Skip/Prediction Runtime

## `saddlemill/dimertools/mode_schedule.py`

Pure bounded mode-solve scheduler and serializable state machine. Decides whether a real mode solve is required or a bounded skip may be used based only on supplied signals such as displacement, force trends, curvature/model state, residuals, angle estimates, and entry-gate evidence.

It does not itself compute a Hessian, force, or prediction. The caller supplies already available evidence. This separation is important for force-call accounting.

## `saddlemill/dimertools/wave_b_runtime.py`

Large shared runtime/controller that wires together the advanced Stage-B/C style components: canonical force history, physical-Hessian updates, mode scheduling, mode predictors, isopotential estimates, mode diagnostics, paid references, hindsight, rotational histories, and resume state.

`WaveBRuntimeController` is the attempt-scoped orchestrator for those features. It observes already-computed centers/probes, updates the selected physical model, performs pre/post mode-solve bookkeeping, evaluates schedule decisions, handles fail-closed prediction safety, and exports metadata/state.

This file is **integration glue plus state**, not the correct place to reimplement the numerical kernels it calls.

## `saddlemill/dimertools/isopotential.py`

Cheap directional isopotential-curvature estimator inspired by K-dimer ideas. `prepare_probe()` defines a single physical probe without evaluating the calculator; `finish_estimate()` turns the returned probe force into a directional estimate with explicit failure semantics.

It is deliberately separated into prepare/finish so force acquisition remains visible to the central observation/accounting machinery.

---

# RFO / P-RFO / Restricted-Step Family

## `saddlemill/dimertools/rfo_step.py`

Pure native rational-function step kernels. Implements augmented RFO root solves, root homing, unpartitioned order-0/order-1 branches, partitioned P/Q RFO, and fixed-radius restricted partitioned RFO.

This is algebra on matrices/vectors. It does not own force acquisition or physical-Hessian construction.

## `saddlemill/dimertools/ras.py`

Generic restricted-atomic-step and adaptive trust implementation used by SaddleMill-owned RFO/P-RFO/QN-MMF paths. Contains fixed-alpha step equations, the max-per-atom constraint, bisection solve, and adaptive trust controller.

## `saddlemill/dimertools/sella_fixed_ras.py`

Backwards-compatible Sella-parity fixed-RAS API. Provides NumPy transcriptions of Sella-style RFO, P-RFO, and QN/MMF fixed-alpha/restricted-step behavior for comparisons and compatibility.

New generic SaddleMill RAS work should normally go through `ras.py`; use this file when exact Sella-parity semantics are the point.

## `saddlemill/dimertools/rfo_translation.py`

Adapter from the sealed physical-Hessian model + same-center raw gradient into a production translation proposal. It validates matrix/gradient identity, chooses the requested partition, invokes RFO/P-RFO/QN-MMF kernels, handles trust/norm control, and records detailed provenance/failure diagnostics.

The attempt-scoped state/trust wrapper for this path is created in `dimer_factory.py`.

---

# Broyden Family

## `saddlemill/dimertools/broyden.py`

Dependency-light root-finding kernels:

- generic good-Broyden inverse Jacobian model;
- Johnson modified Broyden history-space model.

Provides bounded history, update diagnostics, proposal objects, conditioning/rank guards, and state serialization.

## `saddlemill/dimertools/broyden_translation.py`

Cartesian translation adapter around the generic Broyden kernels. Adds restart/history identity handling and returns a translation-step result suitable for the Dimer runtime.

## `saddlemill/dimertools/wave_b_translation.py`

Thin live Dimer translation class that wires Broyden translation into the existing minimum-mode optimizer lifecycle and defines the history identity used for resume/reset decisions.

---

# Dimer Component Assembly

## `saddlemill/dimertools/dimer_factory.py`

Central validated component factory for minimum-mode searches. This is the best single file for answering “which implementation is actually selected by this config?”

It:

- normalizes and validates history options;
- extracts partitioned-LBFGS settings;
- selects the `MinModeAtoms` class;
- selects the translation optimizer class;
- builds Dimer controls and selected minimum-mode implementation;
- attaches keyed RNG and canonical history recording;
- restores checkpointed histories/runtime state;
- constructs the translation optimizer;
- creates the physical-model RFO translation runtime;
- constructs the shared mode-diagnostics runtime.

Do not duplicate selector branching in `dimeropt.py`; keep component selection centralized here and configuration legality in `config_validation.py`.

---

# Attempt Generation

## `saddlemill/dimertools/structure_edit.py`

Large attempt-generation library used by Dimer/Sella searches. Owns the geometry-generation side of initialization rather than the optimizer.

It includes:

- bulk and adsorbate/surface attempt families;
- supercell preparation;
- periodic interstitial-site discovery;
- Gaussian/concentrated displacement scheduling;
- vacancy/hop/kickout/ring mechanisms;
- adsorbate atom, diffusion, rigid rotation, surface, all-movable, random-bubble, custom, and initial-guess mechanisms;
- deterministic per-type attempt-count mapping;
- exact placeholder slots for ungenerated attempts;
- attempt-keyed RNG integration.

If an attempt ID maps to the wrong initial geometry or reaction type, debug here before touching the optimizer.

---

# VASP and Calculator-Specific I/O

## `saddlemill/vasp_io.py`

Pluggable VASP input/output layer. Provides built-in input-generator recipes plus dynamic `module:func` / file callable loading, extra input-file writers, and extra output parsers.

It translates generator settings to ASE VASP kwargs, handles special mapping issues such as element-associated settings, writes optional files such as a mode file, and parses optional outputs such as VTST Dimer mode/curvature.

The external `[Vasp]` section should remain an external-calculator pass-through; SaddleMill orchestration belongs in SaddleMill-owned config sections and this module.

## `saddlemill/tools.py`

Cross-method utility module. Major responsibilities:

- input sanitization and `orig_info` handling;
- input-status filtering;
- VASP calculator construction and extra-I/O wrapping;
- VASP scratch finalization/archive cleanup;
- persistent input-order files;
- method-aware temporary-file cleanup;
- continuation/result extraction with exact frame identities;
- duplicate/equivalence checks for resume safety;
- connectivity/reaction checks.

Because many methods import this module, keep helpers genuinely cross-cutting. Method-specific optimizer math does not belong here.

---

# Important Data and Identity Rules

## `.info` lineage

Fresh inputs are sanitized so prior metadata is preserved under `orig_info`; continuations can therefore produce nested `orig_info` chains. Readers that need upstream data generally search top-level first and then historical levels.

Do not casually flatten or rewrite the chain: it contains provenance and resume identity used across stages.

## Exact identity versus geometric equivalence

Several modules compute exact hashes/fingerprints for reproducibility or resume identity. Those hashes mean “exact serialized/coordinate identity,” not scientific geometric equivalence.

Do not use an exact hash as a substitute for a scientific structure-matching definition, and do not use a topology check as a substitute for geometry when geometry is required.

## Raw physical force history versus projected forces

`CanonicalForceHistory` stores projection-independent physical observations. Projection into minimum-mode translation/rotation coordinates happens downstream.

Preserve that separation. If raw history is overwritten with reflected/projected forces, other consumers can no longer reconstruct alternative models correctly.

## Physical PES calls versus optimizer bookkeeping

The code distinguishes real calculator work from optimizer iterations and from model/diagnostic operations. When adding instrumentation, keep counters explicit and do not relabel a step counter as a physical PES-call counter.

---

# File-by-File Test Map

The `tests/` directory is a focused regression set for the integrated source tree. The tests are organized around the advanced minimum-mode, quasi-Newton, diagnostics, and convergence behaviors that are easiest to regress during development.

## `tests/test_external_mode_prfo_ras.py`

Checks generic P-RFO/RAS behavior when an external unstable mode defines the P partition, verifies that QN/MMF does not accept that same external-partition shortcut, and ensures the shared RAS solve adds no PES work.

## `tests/test_lanczos_residual_convergence_manager.py`

Manager-level acceptance tests for corrected generic Lanczos residual convergence. Guards against false convergence from stale curvature/eigenvalue state while preserving historical SoftSaddle stopping semantics.

## `tests/test_lanczos_residual_convergence_worker.py`

Focused numerical regressions for the same Lanczos change: exact residual definition and reporting from retained actions without an extra HVP.

## `tests/test_manager_seven_arm_integration.py`

Cross-component integration tests for the integrated multi-arm minimum-mode/QN feature set. Compares dense versus compact model actions/spectra/steps, Q-only+Dimer-axial behavior, legacy partitioned control behavior, valid selector compositions, and default-path inertness.

## `tests/test_mode_scheduler_convergence_recheck.py`

Guards the non-Sella Dimer convergence contract: real atomic `fmax` is authoritative for the new convergence path, projected-force behavior is kept distinct, configured threshold semantics are tested, and final mode refreshes may diagnose but cannot veto an already satisfied real-force criterion.

## `tests/test_partitioned_lbfgs_shifted_regularization.py`

Regression matrix for shifted/regularized L-BFGS in partitioned and unpartitioned custom translation. Tests common config surface, identical behavior when regularization is inactive, shared-radius solves, pair/history nonmutation, final step caps, serialization, and composition with mode-skip/predictor features.

## `tests/test_rc3_native_sella_policy.py`

Small policy regression for native Sella control selectors, especially separation of trust-policy changes from QN-specific `newton_safe` behavior.

## `tests/test_sella_native_control_hooks.py`

Unit tests around instance-scoped Sella hooks using fakes. Verifies temporary native restricted-step modifications, fail-closed method matching, incompatible composition rejection, supported-version gating, and trust-isolation behavior.

## `tests/test_sella_native_controls_installed_sella.py`

Checks assumptions against the actually installed Sella API: native QN/RFO flags and P-RFO restricted-step class behavior.

## `tests/test_worker1_attempt_metrics.py`

Ensures the terminal metric layer reports audited physical PES counters rather than silently substituting a legacy counter.

## `tests/test_worker1_diagnostics_io.py`

Tests buffered CSV/JSONL diagnostics, additive header migration, byte-equivalent serialization, flush behavior, and the noninterference of passive instrumentation.

## `tests/test_worker1_sella_passive_io.py`

Confirms passive Sella recording does not change optimizer/PES state.

## `tests/test_worker2_qspace_dimer_axial_lbfgs.py`

Deep regressions for the Q-space L-BFGS + direct Dimer axial translator. Covers current-projector reconstruction, isolation of Q history from P-gradient changes, axial curvature/sign rules, safeguards, pair caps, disallowing derived nonphysical translation secants, resume equivalence, legacy partitioned behavior, explicit failure on unusable curvature, and composition with the scheduler/predictor.

## `tests/test_worker3_skip_policies.py`

Large regression suite for bounded mode-skip policies and predictors. Covers entry-torque gating, bounded holds, physical-model-loss refresh, fail-closed identity/safety checks, ForceBank rotational prediction, serialization, near-degenerate model handling, and preservation of legacy behavior when new selectors are disabled.

## `tests/test_worker_1_windowed_tsbfgs.py`

Tests windowed physical TS-BFGS representations. Validates dense/compact equivalence, QN/MMF multi-negative-mode semantics, bounded-pair eviction, no hidden dense materialization on compact paths, no added PES calls, resume equivalence, config legality, and identity changes when the pair window is scientifically changed.

## `tests/test_worker_f_physical_hessian_family.py`

Tests the physical-Hessian update family and same-center probe-block semantics: historical BFGS admission, fail-closed block selection, use of already paid physical Dimer stencils, matched TS-BFGS probe reuse, and representative composition config snippets.

---

# Where to Start for Common Tasks

| Question / change | Start here | Usually also inspect |
|---|---|---|
| What top-level method runs? | `config_factories.py` | `config_validation.py`, `__main__.py` |
| What is the default for a key? | `config_defaults.py` | `config_validation.py` |
| Why is a config rejected? | `config_validation.py` | `dimer_factory.py`, engine-specific module |
| Wrong resume/redo selection | `config.py` | `tools.py`, `__main__.py` |
| Worker/GPU/MPS mapping | `init_function.py` | `worker_resources.py` |
| Dimer attempt generated incorrectly | `dimertools/structure_edit.py` | `rng_keyed.py`, `dimer_lifecycle.py` |
| Dimer attempt orchestration/status wrong | `dimeropt.py` | `dimer_lifecycle.py`, `config.py` |
| Which Dimer components does a config select? | `dimertools/dimer_factory.py` | `config_validation.py` |
| Stock/ASE Dimer translation behavior | `dimertools/ase_lbfgs_adapter.py` | `legacy_dimer_adapter.py` |
| Custom ForceBank L-BFGS translation | `dimertools/translation_optimizers.py` | `quasi_newton.py`, `qn_reconstruction_core.py` |
| Partitioned/Q-only L-BFGS | `dimertools/partitioned_lbfgs.py` | `translation_optimizers.py` |
| Rotational L-BFGS | `dimertools/dimer_rotation.py` | `riemannian_lbfgs.py`, `rotation_history.py` |
| Iterative minimum-mode eigensolver | `dimertools/minmode_solvers.py` | `hvp_interfaces.py` |
| Mode is being skipped/reused unexpectedly | `dimertools/mode_schedule.py` | `wave_b_runtime.py`, `mode_predictor.py` |
| Physical-Hessian model/update | `dimertools/physical_hessian.py` | `force_history.py`, `wave_b_runtime.py` |
| RFO/P-RFO/QN-MMF step | `dimertools/rfo_translation.py` | `rfo_step.py`, `ras.py` |
| Sella normal run | `sella_engine.py` | `dimeropt.py` |
| Sella ablation/native hook | `sella_ablation.py` | `sella_diagnostics.py` |
| Ordinary Minimization/DoubleMin | `geomopt.py` | `fire_lbfgs.py`, `doublemin_hessian.py` |
| Standalone Hessian numerical result | `hessian_job.py` | `doublemin_hessian.py` |
| Standalone Hessian publication/recovery | `hessian_artifacts.py` | `config.py` |
| NEB workflow event/order | `nebopt.py` | `catsunami/ocpneb.py` |
| NEB force mechanics/freezing | `catsunami/ocpneb.py` | `nebopt.py` |
| VASP input/output integration | `vasp_io.py` | `tools.py` |
| Exact physical work accounting | `attempt_metrics.py` | engine-specific counters, `foundation_types.py` |
| Passive diagnostic I/O performance | `diagnostics_io.py` | recorder using it |

---

