"""Method-specific SaddleMill configuration normalization and validation.

This module validates configuration only.  It must not instantiate a calculator,
create a CUDA context, or import method implementations merely to choose dispatch.
Optional implementation modules are imported only inside the branches that need
their existing validation helpers.
"""

from __future__ import annotations

_VASP_METHOD_SECTION = {
    "NEB": ("ourNEB", ["vasp_command_endpoints", "vasp_command_intermediates"]),
    "Dimer": ("ourDimer", ["vasp_command"]),
    "Minimization": ("ourMinimization", ["vasp_command"]),
    "DoubleMinimization": ("ourDoubleMinimization", ["vasp_command"]),
    "SinglePoint": ("ourSinglePoint", ["vasp_command"]),
}

def normalize_dimer_method_config(dimer_cfg):
    """Return the effective saddle/MMF selectors without mutating config.

    Existing configs keep their exact meanings:
      engine=ase   -> MMF + dimer + standard
      engine=kappa -> MMF + dimer + kappa
      engine=sella -> Sella

    New composable selectors are enabled only by engine=mmf.  This avoids
    silently changing legacy configs that happen to acquire new default keys.
    """
    engine = str(dimer_cfg.get("engine", "ase")).strip().lower()
    if engine == "ase":
        return {
            "engine_label": "ase",
            "saddle_engine": "mmf",
            "min_mode_finder": "dimer",
            "mmf_variant": "standard",
            "mode_reuse": "none",
            "convex_escape": "standard",
        }
    if engine == "kappa":
        return {
            "engine_label": "kappa",
            "saddle_engine": "mmf",
            "min_mode_finder": "dimer",
            "mmf_variant": "kappa",
            "mode_reuse": "none",
            "convex_escape": "standard",
        }
    if engine == "sella":
        return {
            "engine_label": "sella",
            "saddle_engine": "sella",
            "min_mode_finder": None,
            "mmf_variant": None,
            "mode_reuse": "none",
            "convex_escape": "standard",
        }
    if engine == "mmf":
        return {
            "engine_label": "mmf",
            "saddle_engine": "mmf",
            "min_mode_finder": str(
                dimer_cfg.get("min_mode_finder", "dimer")
            ).strip().lower(),
            "mmf_variant": str(
                dimer_cfg.get("mmf_variant", "standard")
            ).strip().lower(),
            "mode_reuse": str(
                dimer_cfg.get("mode_reuse", "none")
            ).strip().lower(),
            "convex_escape": str(
                dimer_cfg.get("convex_escape", "standard")
            ).strip().lower(),
        }
    raise ValueError(
        "[ourDimer] engine must be ase, kappa, sella, or mmf; "
        f"got {engine!r}."
    )


def validate_method_config(config_dict, *, normalize_run_jobs):
    method_name = config_dict["Main"]["method"]
    if method_name is None:
        raise ValueError("Configuration error: 'Main' -> 'method' is not set. Please specify a method (e.g., 'minimization') in config.ini")

    input_format = config_dict["Main"].get("input_format", "traj")
    if input_format not in ("traj", "lmdb"):
        raise ValueError(f"Unknown input_format={input_format!r}; expected 'traj' or 'lmdb'.")
    if input_format == "lmdb" and method_name != "SinglePoint":
        raise ValueError(
            f"input_format='lmdb' is only supported for method='SinglePoint'; got method={method_name!r}."
        )

    rng_scheme = str(config_dict["Main"].get("rng_scheme", "legacy_stream")).strip().lower()
    if rng_scheme not in {"legacy_stream", "attempt_keyed_v1"}:
        raise ValueError(
            "[Main] rng_scheme must be legacy_stream or attempt_keyed_v1; "
            f"got {rng_scheme!r}."
        )
    if rng_scheme == "attempt_keyed_v1":
        if method_name != "Dimer":
            raise ValueError(
                "[Main] rng_scheme=attempt_keyed_v1 is currently implemented only for "
                "method=Dimer (including ASE/Kappa/MMF/Sella Dimer engines)."
            )
        seed_group = str(config_dict["Main"].get("rng_seed_group", "")).strip()
        if not seed_group:
            raise ValueError("[Main] rng_seed_group must be non-empty for attempt_keyed_v1")
        # rng_structure_identity may be supplied per input frame, so it is
        # intentionally validated at runtime rather than required globally here.

    calc_name = config_dict["Main"]["Calculator"]
    optimizer_name = str(config_dict["Main"].get("Optimizer", "")).lower()
    if optimizer_name in {"firelbfgs", "fire_lbfgs", "warmfirelbfgs"} and method_name not in {
        "Minimization", "DoubleMinimization"
    }:
        raise NotImplementedError(
            "Optimizer=FIRELBFGS is currently supported only for "
            "Minimization and DoubleMinimization. Dimer translation uses "
            "[ourDimer] translation_optimizer=fire_lbfgs instead."
        )

    # SaddleMill L-BFGS dimer validation
    if method_name == "Dimer":
        dimer_cfg = config_dict["ourDimer"]

        # Power-law concentrated initialization: size-intensive random peak
        # amplitude separated from the Gaussian-derived shape.  Legacy
        # concentrate_std/concentrate_max_disp semantics are not silently
        # migrated because changing normalization changes the scientific method.
        conc_prob = float(dimer_cfg.get("concentrate_prob", 0.0))
        conc_power = float(dimer_cfg.get("concentrate_power", 1.5))
        conc_peak_median = float(
            dimer_cfg.get("concentrate_peak_median", 0.5)
        )
        conc_peak_log_std = float(
            dimer_cfg.get("concentrate_peak_log_std", 0.25)
        )
        conc_envelope = float(dimer_cfg.get("concentrate_envelope", 0.0))
        if not 0.0 <= conc_prob <= 1.0:
            raise ValueError(
                "[ourDimer] concentrate_prob must be in [0, 1]"
            )
        if conc_power <= 0.0:
            raise ValueError(
                "[ourDimer] concentrate_power must be > 0"
            )
        if conc_peak_median <= 0.0:
            raise ValueError(
                "[ourDimer] concentrate_peak_median must be > 0 A"
            )
        if conc_peak_log_std < 0.0:
            raise ValueError(
                "[ourDimer] concentrate_peak_log_std must be >= 0"
            )
        if conc_envelope < 0.0:
            raise ValueError(
                "[ourDimer] concentrate_envelope must be >= 0 A"
            )
        if conc_prob > 0.0:
            legacy = []
            for key in ("concentrate_std", "concentrate_max_disp"):
                value = dimer_cfg.get(key, None)
                if value not in (None, "", "None", "none"):
                    legacy.append(f"{key}={value!r}")
            if legacy:
                raise ValueError(
                    "Legacy concentration normalization is no longer accepted "
                    "when concentrate_prob > 0: " + ", ".join(legacy) +
                    ". Use concentrate_peak_median and "
                    "concentrate_peak_log_std instead."
                )

        method_cfg = normalize_dimer_method_config(dimer_cfg)
        engine = method_cfg["engine_label"]
        min_mode_finder = method_cfg["min_mode_finder"]
        mmf_variant = method_cfg["mmf_variant"]
        mode_reuse = method_cfg["mode_reuse"]
        convex_escape = method_cfg["convex_escape"]
        rotation_optimizer = str(
            dimer_cfg.get("rotation_optimizer", "ase")
        ).lower()
        translation_optimizer = str(
            dimer_cfg.get("translation_optimizer", "ase")
        ).lower()

        if rotation_optimizer not in {"ase", "lbfgs", "cg", "generic_good_broyden", "johnson_modified_broyden"}:
            raise ValueError(
                "[ourDimer] rotation_optimizer must be ase, lbfgs, cg, generic_good_broyden, or johnson_modified_broyden; "
                f"got {rotation_optimizer!r}."
            )
        if translation_optimizer not in {"ase", "lbfgs", "hybrid", "fire_lbfgs", "canonical_lbfgs", "partitioned_lbfgs", "q_lbfgs_dimer_axial", "rfo", "prfo", "qn_mmf", "generic_good_broyden", "johnson_modified_broyden"}:
            raise ValueError(
                "[ourDimer] translation_optimizer must be ase, lbfgs, canonical_lbfgs, partitioned_lbfgs, q_lbfgs_dimer_axial, rfo, prfo, qn_mmf, fire_lbfgs, generic_good_broyden, or johnson_modified_broyden; "
                f"got {translation_optimizer!r}."
            )
        from saddlemill.dimertools.dimer_factory import history_options_from_config
        _canonical_history_options = history_options_from_config(
            config_dict, translation_optimizer=translation_optimizer
        )
        from saddlemill.dimertools.wave_b_runtime import validate_wave_b_config
        _wave_b_options = validate_wave_b_config(
            config_dict,
            rotation_optimizer=rotation_optimizer,
            translation_optimizer=translation_optimizer,
            min_mode_finder=min_mode_finder,
            saddle_engine=str(method_cfg.get("saddle_engine", "mmf")),
        )
        if translation_optimizer in {"partitioned_lbfgs", "q_lbfgs_dimer_axial"}:
            from saddlemill.dimertools.dimer_factory import partitioned_options_from_config
            from saddlemill.dimertools.partitioned_lbfgs import PartitionedLBFGS
            PartitionedLBFGS(**partitioned_options_from_config(
                config_dict, translation_optimizer=translation_optimizer
            ))
            if mmf_variant == "kappa":
                raise ValueError(
                    f"{translation_optimizer} is not defined for mmf_variant=kappa"
                )
        if translation_optimizer in {"rfo", "prfo", "qn_mmf"}:
            from saddlemill.dimertools.dimer_factory import create_rfo_translation_runtime
            create_rfo_translation_runtime(config_dict)
            if mmf_variant == "kappa":
                raise ValueError("native RFO/P-RFO/QN-MMF translation is not defined for mmf_variant=kappa")
        if (
            method_cfg.get("saddle_engine") == "sella"
            and bool(_canonical_history_options.get("enabled", False))
        ):
            raise ValueError(
                "[ourDimerHistory] is an MMF/Dimer feature and cannot be enabled "
                "with the Sella saddle engine."
            )

        if method_cfg["saddle_engine"] == "sella" and (
            rotation_optimizer != "ase"
            or translation_optimizer != "ase"
            or convex_escape != "standard"
        ):
            raise ValueError(
                "[ourDimer] engine=sella cannot be combined with the MMF "
                "rotation_optimizer, translation_optimizer, or convex_escape "
                "selectors. Leave them at their defaults and configure Sella "
                "in [ourSella]."
            )

        if method_cfg["saddle_engine"] == "mmf":
            allowed_min_mode_finders = {
                "dimer",
                "lanczos",
                "davidson",
                "softsaddle_lanczos",
                "softsaddle_davidson",
                "softsaddle_davidson_textbook",
                "olsen_jd",
            }
            if min_mode_finder not in allowed_min_mode_finders:
                raise ValueError(
                    "[ourDimer] min_mode_finder must be one of "
                    f"{sorted(allowed_min_mode_finders)}; got "
                    f"{min_mode_finder!r}."
                )
            if mmf_variant not in {"standard", "kappa"}:
                raise ValueError(
                    "[ourDimer] mmf_variant must be standard or kappa; "
                    f"got {mmf_variant!r}."
                )
            if mmf_variant == "kappa" and min_mode_finder != "dimer":
                raise ValueError(
                    "[ourDimer] mmf_variant=kappa currently requires "
                    "min_mode_finder=dimer. Lanczos/Davidson Kappa has not "
                    "been scientifically defined in SaddleMill yet."
                )
            if min_mode_finder != "dimer" and rotation_optimizer != "ase":
                raise ValueError(
                    "[ourDimer] rotation_optimizer applies only to the dimer "
                    "mode finder; use rotation_optimizer=ase for "
                    "Lanczos/Davidson."
                )
            if mode_reuse not in {"none", "sm50"}:
                raise ValueError(
                    "[ourDimer] mode_reuse must be none or sm50; "
                    f"got {mode_reuse!r}."
                )
            if mode_reuse == "sm50" and min_mode_finder == "dimer":
                raise ValueError(
                    "[ourDimer] mode_reuse=sm50 is currently supported only "
                    "for Lanczos/Davidson."
                )

            if convex_escape not in {"standard", "bowl_breakout"}:
                raise ValueError(
                    "[ourDimer] convex_escape must be standard or "
                    f"bowl_breakout; got {convex_escape!r}."
                )
            if int(dimer_cfg.get("bowl_active_atoms", 20)) < 1:
                raise ValueError(
                    "[ourDimer] bowl_active_atoms must be >= 1"
                )

            minmode_cfg = config_dict.get("ourMinMode", {}) or {}
            if int(minmode_cfg.get("max_iterations", 8)) < 1:
                raise ValueError("[ourMinMode] max_iterations must be >= 1")
            sella_maxiter = minmode_cfg.get("maxiter", None)
            if sella_maxiter not in (None, "", "none", "None") and int(sella_maxiter) < 1:
                raise ValueError("[ourMinMode] maxiter must be >= 1 or None")
            if (
                min_mode_finder != "olsen_jd"
                and sella_maxiter not in (None, "", "none", "None")
            ):
                raise ValueError(
                    "[ourMinMode] maxiter is Sella-compatible and applies only to "
                    "min_mode_finder=olsen_jd; use max_iterations for the other "
                    "minimum-mode finders"
                )
            if float(minmode_cfg.get("eigenvalue_tolerance", 0.01)) < 0.0:
                raise ValueError(
                    "[ourMinMode] eigenvalue_tolerance must be >= 0"
                )
            for key in (
                "finite_difference",
                "breakdown_tolerance",
                "davidson_update_tolerance",
                "davidson_preconditioner_floor",
                "reference_hessian_finite_difference",
            ):
                default = {
                    "finite_difference": 1.0e-4,
                    "breakdown_tolerance": 1.0e-12,
                    "davidson_update_tolerance": 1.0e-12,
                    "davidson_preconditioner_floor": 1.0e-8,
                    "reference_hessian_finite_difference": 1.0e-4,
                }[key]
                if float(minmode_cfg.get(key, default)) <= 0.0:
                    raise ValueError(f"[ourMinMode] {key} must be > 0")
            if float(
                minmode_cfg.get("softsaddle_preconditioner_floor", 1.0e-12)
            ) < 0.0:
                raise ValueError(
                    "[ourMinMode] softsaddle_preconditioner_floor must be >= 0"
                )
            if float(
                minmode_cfg.get("softsaddle_switch_threshold", 12.0)
            ) < 0.0:
                raise ValueError(
                    "[ourMinMode] softsaddle_switch_threshold must be >= 0"
                )
            hessian_source = str(
                minmode_cfg.get("davidson_initial_hessian_source", "identity")
            ).strip().lower()
            if hessian_source not in {"identity", "reference_fd"}:
                raise ValueError(
                    "[ourMinMode] davidson_initial_hessian_source must be "
                    "identity or reference_fd"
                )
            hvp_origin = str(minmode_cfg.get("hvp_origin", "physical_fd")).strip().lower()
            if hvp_origin == "effective_force_jacobian":
                raise ValueError("[ourMinMode] hvp_origin=effective_force_jacobian is not a physical HVP backend")
            if hvp_origin not in {"physical_fd", "explicit_physical_matrix", "approximate_physical_model"}:
                raise ValueError("[ourMinMode] hvp_origin must be physical_fd, explicit_physical_matrix, or approximate_physical_model")
            root_policy = str(minmode_cfg.get("root_policy", "lowest")).strip().lower()
            if root_policy not in {"lowest", "homed_overlap"}:
                raise ValueError("[ourMinMode] root_policy must be lowest or homed_overlap")
            if min_mode_finder == "olsen_jd" and root_policy != "lowest":
                raise ValueError(
                    "[ourMinMode] root_policy must be lowest for Sella-parity "
                    "min_mode_finder=olsen_jd"
                )
            if float(minmode_cfg.get("residual_tolerance", 1.0e-3)) < 0.0:
                raise ValueError("[ourMinMode] residual_tolerance must be >= 0")
            restart_dimension = minmode_cfg.get("olsen_restart_dimension", None)
            if restart_dimension not in (None, "", "none", "None") and int(restart_dimension) < 2:
                raise ValueError("[ourMinMode] olsen_restart_dimension must be >= 2 or blank")
            if min_mode_finder == "olsen_jd" and restart_dimension not in (None, "", "none", "None"):
                raise ValueError(
                    "[ourMinMode] olsen_restart_dimension is incompatible with the "
                    "Sella-parity olsen_jd path; use maxiter instead"
                )
            if float(minmode_cfg.get("olsen_condition_threshold", 1.0e12)) <= 0.0:
                raise ValueError("[ourMinMode] olsen_condition_threshold must be > 0")
            if float(minmode_cfg.get("olsen_pseudoinverse_rcond", 1.0e-12)) < 0.0:
                raise ValueError("[ourMinMode] olsen_pseudoinverse_rcond must be >= 0")
            if min_mode_finder == "olsen_jd":
                if bool(minmode_cfg.get("residual_stop", False)):
                    raise ValueError(
                        "[ourMinMode] residual_stop is obsolete for Sella-parity "
                        "olsen_jd; Sella gamma=0.1 residual stopping is always used"
                    )
                if float(minmode_cfg.get("eigenvalue_tolerance", 0.01)) != 0.01:
                    raise ValueError(
                        "[ourMinMode] eigenvalue_tolerance is not a convergence "
                        "control for Sella-parity olsen_jd; leave it at the default "
                        "used by the other mode finders"
                    )
                if int(minmode_cfg.get("max_iterations", 8)) != 8:
                    raise ValueError(
                        "[ourMinMode] max_iterations is not an Olsen/JD subspace cap; "
                        "use [ourMinMode] maxiter for Sella-parity olsen_jd"
                    )
                if float(minmode_cfg.get("residual_tolerance", 1.0e-3)) != 1.0e-3:
                    raise ValueError(
                        "[ourMinMode] residual_tolerance is superseded by Sella's "
                        "fixed gamma=0.1 criterion for olsen_jd"
                    )
                if float(minmode_cfg.get("olsen_condition_threshold", 1.0e12)) != 1.0e12:
                    raise ValueError(
                        "[ourMinMode] olsen_condition_threshold belongs to the old "
                        "custom correction solver and is not used by Sella JD0"
                    )
                if float(minmode_cfg.get("olsen_pseudoinverse_rcond", 1.0e-12)) != 1.0e-12:
                    raise ValueError(
                        "[ourMinMode] olsen_pseudoinverse_rcond belongs to the old "
                        "custom correction solver and is not used by Sella JD0"
                    )
            initial_hessian = float(
                minmode_cfg.get("davidson_initial_hessian", 1.0)
            )
            if initial_hessian == 0.0:
                raise ValueError(
                    "[ourMinMode] davidson_initial_hessian must be nonzero"
                )
            sm50_factor = float(minmode_cfg.get("sm50_factor", 50.0))
            if not 0.0 <= sm50_factor <= 100.0:
                raise ValueError("[ourMinMode] sm50_factor must be in [0, 100]")
            if float(
                minmode_cfg.get("sm50_line_search_tolerance", 0.01)
            ) < 0.0:
                raise ValueError(
                    "[ourMinMode] sm50_line_search_tolerance must be >= 0"
                )

        if engine == "sella":
            from saddlemill.sella_engine import (
                normalize_hessian_engine,
                validate_direct_hessian_config,
                validate_sella_environment,
            )
            validate_sella_environment()
            sella_cfg = config_dict.get("ourSella", {}) or {}
            normalize_hessian_engine(sella_cfg.get("hessian_engine", "sella"))
            validate_direct_hessian_config(config_dict)
            method = str(sella_cfg.get("method", "prfo")).strip()
            if not method:
                raise ValueError("[ourSella] method must be a non-empty string")
            for key in (
                "internal", "eig", "threepoint", "allow_fragments",
                "require_first_order_model", "check_desorption",
                "check_delocalization",
            ):
                if not isinstance(sella_cfg.get(key), bool):
                    raise ValueError(f"[ourSella] {key} must be True or False")
            for key in ("project_translations", "project_rotations"):
                value = sella_cfg.get(key, None)
                if value not in (None, "", "None", "none") and not isinstance(value, bool):
                    raise ValueError(
                        f"[ourSella] {key} must be True, False, or blank"
                    )
            for key in ("delta0", "eta", "gamma", "constraints_tol"):
                if float(sella_cfg.get(key, 0.0)) <= 0.0:
                    raise ValueError(f"[ourSella] {key} must be > 0")
            for key in ("sigma_inc", "sigma_dec", "rho_inc", "rho_dec"):
                value = sella_cfg.get(key, None)
                if value not in (None, "", "None", "none") and float(value) <= 0.0:
                    raise ValueError(f"[ourSella] {key} must be > 0 or None")
            if int(sella_cfg.get("nsteps_per_diag", 3)) < 1:
                raise ValueError("[ourSella] nsteps_per_diag must be >= 1")
            maxiter = sella_cfg.get("maxiter", None)
            if maxiter not in (None, "", "None", "none") and int(maxiter) < 1:
                raise ValueError("[ourSella] maxiter must be >= 1 or None")
            diag_every_n = sella_cfg.get("diag_every_n", None)
            if diag_every_n not in (None, "", "None", "none") and int(diag_every_n) < 1:
                raise ValueError("[ourSella] diag_every_n must be >= 1 or None")
            if float(sella_cfg.get("negative_eigenvalue_tolerance", 1.0e-6)) < 0.0:
                raise ValueError("[ourSella] negative_eigenvalue_tolerance must be >= 0")
            if int(sella_cfg.get("check_interval", 5)) < 1:
                raise ValueError("[ourSella] check_interval must be >= 1")
            ablation = sella_cfg.get("ablation", None)
            if ablation not in (None, "", "none", "None"):
                from saddlemill.sella_ablation import (
                    LEARNING_CENTER_ONLY, PARTITION_NATIVE_RITZ, QN_NEWTON_SAFE_FALSE,
                    resolve_sella_ablation, SellaAblationError,
                )
                spec = resolve_sella_ablation(ablation)
                if spec.qn_newton_safe is False:
                    if method.lower() != "qn":
                        raise SellaAblationError("non_qn_baseline", f"{QN_NEWTON_SAFE_FALSE} requires [ourSella] method=qn")
                elif not spec.is_normal and method.lower() != "prfo":
                    raise SellaAblationError("non_prfo_baseline", "PRFO Sella ablations require [ourSella] method=prfo")
                if not bool(sella_cfg.get("eig", True)):
                    raise SellaAblationError("eig_required", "Sella ablations require [ourSella] eig=True")
                if bool(sella_cfg.get("internal", False)):
                    raise SellaAblationError("internal_coordinates_unsupported", "Sella ablations require [ourSella] internal=False")
                if float(sella_cfg.get("delta0", 0.1)) != 0.1:
                    raise SellaAblationError("delta0_mismatch", "Sella-ablation requires unchanged Sella delta0=0.1")
                if normalize_hessian_engine(sella_cfg.get("hessian_engine", "sella")) == "fairchem_direct" and spec.learning == LEARNING_CENTER_ONLY:
                    raise SellaAblationError(
                        "probe_learning_not_applicable_direct_hessian",
                        "center_only_learning is undefined with hessian_engine=fairchem_direct",
                    )
                if normalize_hessian_engine(sella_cfg.get("hessian_engine", "sella")) == "fairchem_direct" and spec.partition == PARTITION_NATIVE_RITZ:
                    raise SellaAblationError(
                        "ritz_partition_requires_native_diag",
                        "ritzmode_prfo_partition requires hessian_engine=sella/native PES.diag",
                    )

        lbfgs_cfg = config_dict.get("ourDimerLBFGS", {}) or {}
        for key in ("rotation_memory", "translation_memory"):
            if int(lbfgs_cfg.get(key, 10)) < 1:
                raise ValueError(f"[ourDimerLBFGS] {key} must be >= 1")
        for key in (
            "rotation_initial_hessian",
            "translation_initial_hessian",
            "translation_damping",
        ):
            if float(lbfgs_cfg.get(key, 1.0)) <= 0.0:
                raise ValueError(f"[ourDimerLBFGS] {key} must be > 0")
        if float(lbfgs_cfg.get("curvature_epsilon", 1.0e-12)) < 0.0:
            raise ValueError(
                "[ourDimerLBFGS] curvature_epsilon must be >= 0"
            )
        rotation_geometry = str(
            lbfgs_cfg.get("rotation_geometry", "projected")
        ).strip().lower()
        if rotation_geometry not in {"projected", "riemannian"}:
            raise ValueError(
                "[ourDimerLBFGS] rotation_geometry must be projected or riemannian"
            )
        rotation_transport_policy = str(
            lbfgs_cfg.get("rotation_transport_policy", "auto")
        ).strip().lower()
        valid_transport = {
            "auto", "projected", "riemannian", "projection", "transport",
            "legacy_projection", "double_projection", "sequential_projection",
            "direct_transport", "double_transport", "sequential_transport",
        }
        if rotation_transport_policy not in valid_transport:
            raise ValueError("[ourDimerLBFGS] invalid rotation_transport_policy")
        rotation_step_method = str(
            lbfgs_cfg.get("rotation_step_method", "fourier")
        ).strip().lower()
        if rotation_step_method not in {"fourier", "direct"}:
            raise ValueError(
                "[ourDimerLBFGS] rotation_step_method must be fourier or direct"
            )
        for key in ("rotation_reconstruction_model", "translation_reconstruction_model"):
            value = str(lbfgs_cfg.get(key, "lbfgs")).strip().lower()
            if value not in {"lbfgs", "reconstructed_bfgs"}:
                raise ValueError(f"[ourDimerLBFGS] {key} must be lbfgs or reconstructed_bfgs")
        for key in ("rotation_bfgs_update", "translation_bfgs_update"):
            value = str(lbfgs_cfg.get(key, "sequential")).strip().lower()
            if value not in {"sequential", "multisecant"}:
                raise ValueError(f"[ourDimerLBFGS] {key} must be sequential or multisecant")
        for key in ("rotation_first_angle_degrees", "rotation_max_angle_degrees"):
            value = float(lbfgs_cfg.get(key, 45.0))
            if not 0.0 < value <= 90.0:
                raise ValueError(f"[ourDimerLBFGS] {key} must be in (0, 90]")
        if float(lbfgs_cfg.get("rotation_first_angle_degrees", 45.0)) > float(
            lbfgs_cfg.get("rotation_max_angle_degrees", 45.0)
        ):
            raise ValueError(
                "[ourDimerLBFGS] rotation_first_angle_degrees cannot exceed "
                "rotation_max_angle_degrees"
            )
        valid_rotation_guards = {
            "auto", "legacy", "legacy_skip", "legacy_rotation_guard",
            "off", "skip", "damp", "powell", "reset",
        }
        rotation_guard = str(
            lbfgs_cfg.get("rotation_curvature_guard", "legacy_skip")
        ).strip().lower()
        if rotation_guard not in valid_rotation_guards:
            raise ValueError(
                "[ourDimerLBFGS] invalid rotation_curvature_guard"
            )
        valid_translation_guards = {
            "legacy", "legacy_damp", "legacy_translation_guard",
            "off", "skip", "damp", "powell", "reset", "cautious",
        }
        translation_guard = str(
            lbfgs_cfg.get("translation_curvature_guard", "skip")
        ).strip().lower()
        if translation_guard not in valid_translation_guards:
            raise ValueError(
                "[ourDimerLBFGS] invalid translation_curvature_guard"
            )
        for key, default in (
            ("rotation_curvature_floor", 1.0e-3),
            ("translation_curvature_floor", 1.0e-3),
        ):
            if float(lbfgs_cfg.get(key, default)) <= 0.0:
                raise ValueError(f"[ourDimerLBFGS] {key} must be > 0")
        for key, default in (
            ("rotation_powell_eta", 0.2),
            ("translation_powell_eta", 0.2),
        ):
            value = float(lbfgs_cfg.get(key, default))
            if not 0.0 < value < 1.0:
                raise ValueError(f"[ourDimerLBFGS] {key} must be in (0, 1)")
        translation_regularization = str(
            lbfgs_cfg.get("translation_regularization", "off")
        ).strip().lower()
        if translation_regularization not in {"off", "shifted_trust_region", "shifted_lbfgs_fixed", "shifted_lbfgs_trust"}:
            raise ValueError("[ourDimerLBFGS] invalid translation_regularization")
        if float(lbfgs_cfg.get("translation_regularization_mu", 1.0)) < 0.0:
            raise ValueError("[ourDimerLBFGS] translation_regularization_mu must be >= 0")
        if float(lbfgs_cfg.get("translation_regularization_radius", 0.1)) <= 0.0:
            raise ValueError("[ourDimerLBFGS] translation_regularization_radius must be > 0")
        if float(lbfgs_cfg.get("translation_regularization_tolerance", 1.0e-8)) <= 0.0:
            raise ValueError(
                "[ourDimerLBFGS] translation_regularization_tolerance must be > 0"
            )

        # Exact-replay diagnostics are output-only and disabled by default.
        _dump_steps = str(
            lbfgs_cfg.get("translation_state_dump_steps", "all")
        ).strip().lower()
        if _dump_steps not in {"", "all", "*", "none", "off"}:
            try:
                _parsed_dump_steps = [
                    int(x) for x in _dump_steps.replace(",", " ").split()
                ]
            except ValueError as exc:
                raise ValueError(
                    "[ourDimerLBFGS] translation_state_dump_steps must be "
                    "all/none or integer steps"
                ) from exc
            if any(x < 0 for x in _parsed_dump_steps):
                raise ValueError(
                    "[ourDimerLBFGS] translation_state_dump_steps cannot "
                    "contain negative steps"
                )

        hybrid_cfg = config_dict.get("ourDimerHybrid", {}) or {}
        if float(hybrid_cfg.get("exit_fmax", 0.50)) < float(
            hybrid_cfg.get("enter_fmax", 0.30)
        ):
            raise ValueError(
                "[ourDimerHybrid] exit_fmax must be >= enter_fmax"
            )
        if float(hybrid_cfg.get("exit_curvature", 0.0)) < float(
            hybrid_cfg.get("enter_curvature", -0.05)
        ):
            raise ValueError(
                "[ourDimerHybrid] exit_curvature must be >= "
                "enter_curvature"
            )
        if int(hybrid_cfg.get("enter_stable_steps", 3)) < 1 or int(
            hybrid_cfg.get("exit_stable_steps", 2)
        ) < 1:
            raise ValueError(
                "[ourDimerHybrid] stable-step counts must be >= 1"
            )
        if int(hybrid_cfg.get("minimum_history_pairs", 3)) < 0:
            raise ValueError(
                "[ourDimerHybrid] minimum_history_pairs must be >= 0"
            )
        for key in ("fire_dt", "fire_dtmax", "fire_finc", "fire_fdec", "fire_astart", "fire_fa"):
            if float(hybrid_cfg.get(key, 0.1)) <= 0.0:
                raise ValueError(f"[ourDimerHybrid] {key} must be > 0")
        if int(hybrid_cfg.get("fire_Nmin", 5)) < 0:
            raise ValueError("[ourDimerHybrid] fire_Nmin must be >= 0")
        if translation_optimizer in {"hybrid", "fire_lbfgs"} and not bool(
            hybrid_cfg.get("enabled", False)
        ):
            print(
                "Note: translation_optimizer=fire_lbfgs but "
                "[ourDimerHybrid] enabled=False; translation remains in "
                "the FIRE state."
            )
    # SADDLEMILL_DOUBLEMIN_FAIRCHEM_HESSIAN_V2
    if method_name == "DoubleMinimization":
        dm_cfg = config_dict.get("ourDoubleMinimization", {}) or {}
        displacement = float(dm_cfg.get('displacement', 0.25))
        if displacement <= 0.0:
            raise ValueError('[ourDoubleMinimization] displacement must be > 0 A')
        pre_dimer = dm_cfg.get("pre_dimer_refine", False)
        pre_hessian = dm_cfg.get("pre_hessian_eigenmode", False)
        store_hessian = dm_cfg.get("pre_hessian_store_full", True)
        require_first_order = dm_cfg.get("pre_hessian_require_first_order", True)
        for key, value in (("pre_dimer_refine", pre_dimer),
                           ("pre_hessian_eigenmode", pre_hessian),
                           ("pre_hessian_store_full", store_hessian),
                           ("pre_hessian_require_first_order", require_first_order)):
            if not isinstance(value, bool):
                raise ValueError(f"[ourDoubleMinimization] {key} must be True or False")
        if pre_dimer and pre_hessian:
            raise ValueError(
                "[ourDoubleMinimization] pre_dimer_refine and "
                "pre_hessian_eigenmode are mutually exclusive."
            )
        htol = float(dm_cfg.get("hessian_negative_eigenvalue_tolerance", 1.0e-6))
        if htol < 0.0:
            raise ValueError(
                "[ourDoubleMinimization] "
                "hessian_negative_eigenvalue_tolerance must be >= 0"
            )
        if pre_hessian:
            from saddlemill.sella_engine import validate_fairchem_hessian_config
            validate_fairchem_hessian_config(config_dict)

    if method_name in {"Minimization", "DoubleMinimization"} and str(
        config_dict["Main"].get("Optimizer", "")
    ).lower() == "lbfgs":
        cfg = config_dict.get("ourLBFGS", {}) or {}
        guard = str(cfg.get("curvature_guard", "off")).strip().lower()
        allowed_guards = {"off", "skip", "damp", "powell", "reset"}
        if guard not in allowed_guards:
            raise ValueError(
                "[ourLBFGS] curvature_guard must be one of "
                f"{sorted(allowed_guards)}"
            )
        floor = float(cfg.get("curvature_floor", 1.0e-3))
        if guard in {"skip", "damp", "reset"} and floor <= 0.0:
            raise ValueError(
                "[ourLBFGS] curvature_floor must be > 0 for "
                "skip/damp/reset"
            )
        eta = float(cfg.get("powell_eta", 0.2))
        if guard == "powell" and not (0.0 < eta < 1.0):
            raise ValueError("[ourLBFGS] powell_eta must satisfy 0 < eta < 1")

    if method_name in {"Minimization", "DoubleMinimization"} and str(
        config_dict["Main"].get("Optimizer", "")
    ).lower() in {"firelbfgs", "fire_lbfgs", "warmfirelbfgs"}:
        cfg = config_dict.get("FIRELBFGS", {}) or {}
        if int(cfg.get("lbfgs_memory", 10)) < 1:
            raise ValueError("[FIRELBFGS] lbfgs_memory must be >= 1")
        for key in ("maxstep", "fire_dt", "fire_dtmax", "fire_finc", "fire_fdec",
                    "fire_astart", "fire_fa", "lbfgs_initial_hessian",
                    "lbfgs_damping", "enter_fmax", "exit_fmax"):
            if float(cfg.get(key, 1.0)) <= 0.0:
                raise ValueError(f"[FIRELBFGS] {key} must be > 0")
        if float(cfg.get("exit_fmax", 0.35)) < float(cfg.get("enter_fmax", 0.20)):
            raise ValueError("[FIRELBFGS] exit_fmax must be >= enter_fmax")
        if int(cfg.get("enter_stable_steps", 3)) < 1 or int(
            cfg.get("exit_stable_steps", 2)
        ) < 1:
            raise ValueError("[FIRELBFGS] stable-step counts must be >= 1")
        if int(cfg.get("minimum_history_pairs", 3)) < 0:
            raise ValueError("[FIRELBFGS] minimum_history_pairs must be >= 0")
        if float(cfg.get("lbfgs_curvature_epsilon", 1.0e-12)) < 0.0:
            raise ValueError("[FIRELBFGS] lbfgs_curvature_epsilon must be >= 0")
        guard = str(cfg.get("lbfgs_curvature_guard", "off")).strip().lower()
        allowed_guards = {"off", "skip", "damp", "powell", "reset"}
        if guard not in allowed_guards:
            raise ValueError(
                "[FIRELBFGS] lbfgs_curvature_guard must be one of "
                f"{sorted(allowed_guards)}"
            )
        floor = float(cfg.get("lbfgs_curvature_floor", 1.0e-3))
        if guard in {"skip", "damp", "reset"} and floor <= 0.0:
            raise ValueError(
                "[FIRELBFGS] lbfgs_curvature_floor must be > 0 for "
                "skip/damp/reset"
            )
        eta = float(cfg.get("lbfgs_powell_eta", 0.2))
        if guard == "powell" and not (0.0 < eta < 1.0):
            raise ValueError(
                "[FIRELBFGS] lbfgs_powell_eta must satisfy 0 < eta < 1"
            )
        if not isinstance(cfg.get("lbfgs_use_line_search", False), bool):
            raise ValueError("[FIRELBFGS] lbfgs_use_line_search must be True or False")

    # SADDLEMILL_STANDALONE_HESSIAN_JOB
    if method_name == "Hessian":
        if calc_name != "FAIRChemCalculator":
            raise NotImplementedError(
                "method='Hessian' currently requires Calculator=FAIRChemCalculator."
            )
        hcfg = config_dict.get("ourHessian", {}) or {}
        if not isinstance(hcfg.get("restrict_fixed_atoms", True), bool):
            raise ValueError("[ourHessian] restrict_fixed_atoms must be True or False")
        if not isinstance(hcfg.get("oom_retry", True), bool):
            raise ValueError("[ourHessian] oom_retry must be True or False")
        if not isinstance(hcfg.get("store_hessian", True), bool):
            raise ValueError("[ourHessian] store_hessian must be True or False")
        chunk = hcfg.get("chunk_size", 32)
        if not (isinstance(chunk, str) and chunk.strip().lower() == "auto"):
            if int(chunk) < 1:
                raise ValueError("[ourHessian] chunk_size must be 'auto' or >=1")
        if int(hcfg.get("min_chunk_size", 1)) < 1:
            raise ValueError("[ourHessian] min_chunk_size must be >=1")
        if int(hcfg.get("max_oom_retries", 5)) < 0:
            raise ValueError("[ourHessian] max_oom_retries must be >=0")
        if float(hcfg.get("negative_eigenvalue_tolerance", 1.0e-6)) < 0.0:
            raise ValueError("[ourHessian] negative_eigenvalue_tolerance must be >=0")
        ccfg = config_dict.get("FAIRChemCalculator", {}) or {}
        if int(ccfg.get("workers", 1)) != 1:
            raise ValueError("method='Hessian' requires [FAIRChemCalculator] workers=1")
        device = str(ccfg.get("device", config_dict["Main"].get("device", "cuda"))).lower()
        if not device.startswith("cuda"):
            raise ValueError("method='Hessian' is currently validated only on CUDA")
        if int(config_dict["Main"].get("jobs_per_gpu", 1)) != 1:
            raise ValueError(
                "method='Hessian' production policy requires [Main] jobs_per_gpu=1 "
                "(no MPS). Use separate MPS-enabled methods for saddle search/DoubleMin."
            )

    if method_name == "SinglePoint":
        if calc_name not in ("FAIRChemCalculator", "Vasp", "VaspInteractive"):
            raise NotImplementedError(
                f"method='SinglePoint' supports FAIRChemCalculator, Vasp, and "
                f"VaspInteractive; got Calculator={calc_name!r}."
            )
        if calc_name in ("Vasp", "VaspInteractive"):
            fpj = config_dict["ourSinglePoint"].get("frames_per_job", 1)
            if fpj != 1:
                raise NotImplementedError(
                    f"SinglePoint with Calculator={calc_name!r} requires "
                    f"frames_per_job=1 (no batched DFT forward pass); got frames_per_job={fpj}."
                )
        # v1: LMDB output cleaning is not implemented; restrict resume categories.
        if input_format == "lmdb":
            cats = normalize_run_jobs(config_dict["Main"]["run_jobs"])
            if cats != {"remaining"}:
                raise NotImplementedError(
                    "v1: SinglePoint with input_format='lmdb' supports only "
                    "run_jobs='remaining' (the default). To re-process specific "
                    "categories, delete SinglePoint_lmdbs/ and "
                    "SinglePoint_status_csvs/ first."
                )

    if calc_name in ("Vasp", "VaspInteractive"):
        section, required_keys = _VASP_METHOD_SECTION[method_name]
        missing = [k for k in required_keys if not config_dict[section].get(k)]
        if missing:
            raise ValueError(
                f"Calculator={calc_name!r} requires [{section}] {', '.join(missing)} "
                f"to be set. Add them to config.ini (the launcher command for VASP, "
                f"e.g. 'mpirun -n 64 vasp_std')."
            )
        # Fail fast on unresolvable [ourVasp] input_generator / extra_input_files
        # (built-in name, 'module:func', or 'file.py:func'). Resolved, not called.
        gen_spec = config_dict.get("ourVasp", {}).get("input_generator")
        if gen_spec:
            from saddlemill.vasp_io import load_input_generator
            load_input_generator(gen_spec)
        extra_spec = config_dict.get("ourVasp", {}).get("extra_input_files")
        if extra_spec:
            from saddlemill.vasp_io import load_extra_input_writer
            for s in ([extra_spec] if isinstance(extra_spec, str) else extra_spec):
                load_extra_input_writer(s)
        out_spec = config_dict.get("ourVasp", {}).get("extra_outputs")
        if out_spec:
            from saddlemill.vasp_io import load_extra_output_parser
            for s in ([out_spec] if isinstance(out_spec, str) else out_spec):
                load_extra_output_parser(s)
    elif any(config_dict.get("ourVasp", {}).get(k) for k in
             ("input_generator", "extra_input_files", "extra_outputs")):
        print(f"Warning: [ourVasp] settings are set but Calculator={calc_name!r} "
              f"is not VASP; they will be ignored.")
    return method_name


__all__ = ["normalize_dimer_method_config", "validate_method_config"]
