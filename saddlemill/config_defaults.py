"""Canonical SaddleMill configuration defaults.

This module is data-only: importing it must not import calculators, optimizers,
CUDA libraries, or method implementations.  ``saddlemill.config`` re-exports
these defaults through ``ConfigManager.DEFAULTS`` for compatibility.
"""

DEFAULTS = {
    "Main": {
        "executorlib": True,
        "method": None,  # requires user input
        "dir_path": ".",
        "Optimizer": "MDMin",
        "fmax": 0.05,
        "steps": 1000,
        "Calculator": "FAIRChemCalculator",
        "device": 'cuda',
        "jobs_per_node": 1,  # this only used if device = 'cpu', otherwise jobs_per_gpu is used
        "jobs_per_gpu": 1,
        "run_jobs": "remaining",
        "input_statuses": "all",
        "continue_from_result": True,
        "zip": True,
        "max_consecutive_errors": 5,
        "restart_limit": 3,
        # Random-initialization policy. Legacy behavior keeps the historical
        # process-global stream seeded from src_index/seed_offset. Order-independent
        # campaigns should explicitly select attempt_keyed_v1 and provide a
        # stable rng_structure_identity on each input frame.
        "rng_scheme": "legacy_stream",  # legacy_stream | attempt_keyed_v1
        "rng_base_seed": 0,
        "rng_seed_group": "default",
        "rng_structure_identity": None,  # single-structure fallback; per-frame metadata is preferred
        "seed_offset": 0, # legacy_stream only: historical seed offset
        "input_format": "traj",  # traj (default) | lmdb. lmdb is only supported for method=SinglePoint.
        "attempt_chunk_size": 0, # 0 = off (one job per structure). >0 = split a structure's redo attempts into jobs of this many, spread across workers.
        },
    "ourMinimization": {
        "relax_cell": False,
        "vasp_command": None,
        "vasp_ncore": None,
    },
    "ourLBFGS": {
        # SaddleMill-owned safeguards for ordinary Minimization and each
        # DoubleMinimization side when [Main] Optimizer = LBFGS.  [LBFGS]
        # remains a pure ASE pass-through section.
        "curvature_guard": "off",  # off | skip | damp | powell | reset
        "curvature_floor": 1.0e-3,  # eV/A^2 directional secant floor for skip/damp/reset
        "powell_eta": 0.2,  # classical Powell damped-BFGS eta
    },
    "ourSinglePoint": {
        "frames_per_job": 1,  # 1 (default) | 3. With 3, each executorlib job processes a triplet (e.g. DM min1/TS/min2) in a single batched FAIRChem forward pass. VASP requires 1.
        "vasp_command": None,
        "vasp_ncore": None,
    },
    "ourDoubleMinimization": {
        "relax_cell": False,
        "displacement": 0.25,  # A; scientific DoubleMin initial mode amplitude
        "pre_dimer_refine": False,
        "pre_hessian_eigenmode": False,  # legacy inline Hessian recomputation only
        "pre_hessian_store_full": True,
        "pre_hessian_require_first_order": True,
        "hessian_negative_eigenvalue_tolerance": 1.0e-6,
        "vasp_command": None,
        "vasp_ncore": None,
    },
    # SADDLEMILL_STANDALONE_HESSIAN_JOB
    "ourHessian": {
        # Performance knobs only; scientific classification matches the
        # existing DoubleMin pre-Hessian default tolerance.
        "restrict_fixed_atoms": True,
        "chunk_size": 32,
        "oom_retry": True,
        "max_oom_retries": 5,
        "min_chunk_size": 1,
        "store_hessian": True,
        "negative_eigenvalue_tolerance": 1.0e-6,
    },
    "FIRELBFGS": {
        # Used when [Main] Optimizer = FIRELBFGS.  Applies to both
        # Minimization and each side of DoubleMinimization.
        "maxstep": 0.2,
        "fire_dt": 0.1,
        "fire_dtmax": 1.0,
        "fire_Nmin": 5,
        "fire_finc": 1.1,
        "fire_fdec": 0.5,
        "fire_astart": 0.1,
        "fire_fa": 0.99,
        "lbfgs_memory": 10,
        "lbfgs_initial_hessian": 70.0,
        "lbfgs_dynamic_h0": False,
        "lbfgs_curvature_epsilon": 1.0e-12,
        "lbfgs_damping": 1.0,
        "lbfgs_curvature_guard": "off",  # off | skip | damp | powell | reset
        "lbfgs_curvature_floor": 1.0e-3,  # eV/A^2 directional secant floor for skip/damp/reset
        "lbfgs_powell_eta": 0.2,  # classical Powell damped-BFGS eta
        "lbfgs_use_line_search": False,
        "enter_fmax": 0.20,
        "exit_fmax": 0.35,
        "enter_stable_steps": 3,
        "exit_stable_steps": 2,
        "minimum_history_pairs": 3,
        "warm_start_history": True,
        "reset_history_on_exit": True,
    },
    "ourNEB": {
        "only_endpoints_in_input_traj": False,
        "images_location_in_input_traj": ":",  # can also be 0 or -1, meaning begining or end of file. This defines where are the initial endpoints or the band in the input traj
        "relax_endpoints": True,
        "endpoint_relax_Optimizer": None,
        "endpoint_relax_fmax": 0.01,
        "endpoint_relax_steps": 500,
        "interpolate_method": "ase_linear",
        "num_frames": 10,
        "max_num_frames": None,
        "batch_size": 4,
        "DNEB": False,
        "intermediate_minima_check_step": 0,  # 0 = disabled; >0 = one-shot imin detection at this optimizer step
        "intermediate_minima_min_depth": 0.05,
        "add_images_step": 0,  # 0 = disabled; >0 = one-shot image addition at this optimizer step
        "dimer_refine_ci": False,
        "dimer_refine_steps": 300,
        "refine_band_steps": 0,
        "vasp_command_endpoints": None,
        "vasp_ncore_endpoints": None,
        "vasp_command_intermediates": None,
        "vasp_ncore_intermediates": None,
    },
    "ourPhysicalHessian": {

        # physical-Hessian model. Entire feature is off by default.

        "update": "off",                 # off | bfgs | ts_bfgs
        # online_dense preserves the existing carried matrix. The two ForceBank
        # windowed selectors rebuild TS-BFGS from B0 using only retained raw pairs.
        "representation": "online_dense", # online_dense | forcebank_window_dense | forcebank_window_compact
        "probe_updates": "off",          # off | same_center_physical_hvp
        # sequential admits each complete same-center physical HVP stencil
        # independently. same_center_block suppresses incremental admission and
        # commits one rank-revealed block after a completed Dimer mode solve.
        # solver_retained_block commits the final retained iterative-solver HVP
        # block after its validated solve/audit.
        "probe_batching": "sequential",  # sequential | same_center_block | solver_retained_block
        "initial_hessian": 1.0,
        "max_matrix_bytes": 536870912,
        "denominator_tolerance": 1.0e-12,
        "rank_relative_tolerance": 1.0e-10,
        "rank_absolute_tolerance": 1.0e-12,
        "dependence_noise_tolerance": 1.0e-6,
        "block_symmetry_noise_tolerance": 1.0e-3,
        "secant_residual_tolerance": 1.0e-8,
        "spectral_zero_tolerance": 1.0e-12,
    },
    "ourDimerBroyden": {
        # quasi-Newton/rotation kernels are scientific selectors only when chosen as an optimizer.
        "generic_initial_inverse_scale": -1.0,
        "generic_history_cap": 20,
        "generic_denominator_tolerance": 1.0e-12,
        "generic_vector_tolerance": 1.0e-14,
        "generic_rank_tolerance": 1.0e-12,
        "generic_max_condition": 1.0e12,
        "johnson_initial_step_scale": 1.0,
        "johnson_history_cap": 20,
        "johnson_regularization_w0": 0.01,
        "johnson_default_weight": 1.0,
        "johnson_vector_tolerance": 1.0e-14,
        "johnson_rank_tolerance": 1.0e-12,
        "johnson_max_condition": 1.0e12,
    },
    "ourDimerCG": {
        "formula": "pr_plus",
        # No scientific reset-policy default was supplied by rotation-CG. Require an
        # explicit value only when rotation_optimizer=cg is selected.
        "reset_policy": None,             # every_translation | bad_direction | never
        "denominator_tolerance": 1.0e-24,
        "tangent_tolerance": 1.0e-14,
        "descent_tolerance": 0.0,
    },
    "ourModeSchedule": {
        # Global selector remains disabled. New scheduling behavior is opt-in;
        # legacy enabled behavior remains bounded with configurable K (default 3).
        "enabled": False,
        "max_skips": 3,
        "skip_entry_gate": "none",          # none | dimer_initial_torque
        "refresh_policy": "bounded",        # bounded | physical_model_loss
        "parallel_force_increase_trigger": True,
        "physical_model_negative_mode_trigger": False,
        "displacement_trigger": False,
        "stale_data_trigger": True,
        "real_residual_trigger": False,
        "model_residual_trigger": False,
        "angle_trigger": False,
        "max_cumulative_displacement": None,
        "max_curvature_age": None,
        "real_residual_threshold": None,
        "model_residual_threshold": None,
        "angle_threshold_radians": None,
        "required_negative_modes": 1,
        "negative_curvature_tolerance": 0.0,
        "displacement_tolerance": 1.0e-12,
        "parallel_metric_absolute_tolerance": 1.0e-12,
        "parallel_metric_relative_tolerance": 1.0e-8,
        "predictor_hook": "none",
        "parallel_force_damping": "none", # none | fixed | shang_liu
        "fixed_lambda": 1.0,
    },
    "ourModeDiagnostics": {
        # mode diagnostics are strictly opt-in.  Legacy scheduling remains
        # authoritative unless an explicit criterion is selected.
        "enabled": False,
        "mode_refresh_criterion": "legacy",
        "mode_residual_threshold": None,
        "mode_angular_error_threshold_degrees": None,
        "comparison_tolerance": 0.0,
        "residual_epsilon": 1.0e-12,
        "gap_tolerance": 1.0e-6,
        "model_gap_solver_tolerance": 1.0e-4,
        "require_root_validity": True,
        "paid_reference": False,
        "hindsight": False,
    },
    "ourModePredictor": {
        # mode predictor is inert unless [ourModeSchedule] predictor_hook selects it.
        # Direct translation-secant SD/live-history predictors require an explicit scale.
        # translation_secant_forcebank_lbfgs uses the reconstructed inverse rotational
        # stiffness itself as the step scale (effective scale = 1) and records the
        # uncapped requested angle explicitly.
        "angular_step_scale": None,
        "max_angle_degrees": 5.0,
        "alignment_tolerance": 1.0e-8,
        "displacement_tolerance_A": 1.0e-12,
        "tangent_tolerance": 1.0e-14,
        # New force-bank reconstructed rotational L-BFGS predictor. Dynamic H0
        # makes the uncapped direction norm an empirically secant-scaled angular
        # proposal rather than a legacy fixed-H0 torque multiplier.
        "forcebank_lbfgs_dynamic_h0": True,
        "physical_root_policy": "lowest",   # lowest | overlap
        "physical_low_spectrum_count": 4,
        "physical_degeneracy_tolerance": 1.0e-8,
        "physical_vector_tolerance": 1.0e-14,
    },
    "ourPartitionedLBFGS": {
        # partitioned-LBFGS is a fully opt-in physical-space partitioned translator.
        "projector_policy": "reconstruct",
        "projector_overlap_tolerance": 0.95,
        "sign_reference_overlap_tolerance": 1.0e-10,
        "p_initial_hessian": 70.0,
        "q_initial_hessian": 70.0,
        "p_dynamic_h0": False,
        "q_dynamic_h0": False,
        "p_safeguard": "skip",
        "q_safeguard": "skip",
        "p_curvature_floor": 1.0e-3,
        "q_curvature_floor": 1.0e-3,
        "curvature_epsilon": 1.0e-12,
        "powell_eta": 0.2,
        "p_memory": 10,
        "q_memory": 10,
        "step_damping": 1.0,
    },
    "ourRFO": {
        # Generic RFO/P-RFO/QN-MMF translation; inert unless explicitly selected.
        # Selecting the stepper/partition/trust policy is a scientific-method choice.
        "partition": None,
        "step_control": None,             # unrestricted | norm_cap | restricted_prfo | ras | legacy aliases
        "target_order": 1,
        "physical_root_policy": "lowest",
        "negative_mode_policy": "allow",
        "trust_policy": "fixed",         # fixed | adaptive (adaptive only with step_control=ras)
        "trust_radius": None,
        "restricted_step_root_solver": "bisection",
        "ras_tolerance": None,            # None -> exact algorithm-specific Sella 2.5.0 parity tolerance
        "ras_max_iterations": 1000,
        # Sella 2.5.0 saddle trust defaults, used only for trust_policy=adaptive.
        "trust_sigma_inc": 1.15,
        "trust_sigma_dec": 0.65,
        "trust_rho_inc": 1.035,
        "trust_rho_dec": 5.0,
        "trust_delta_min": 1.0e-4,
        "norm_cap": None,
        "negative_tolerance": 1.0e-12,
        "physical_degeneracy_tolerance": 1.0e-10,
        "external_coupling_tolerance": 1.0e-6,
        "restricted_tolerance": 1.0e-10,
        "restricted_max_iterations": 100,
        "restricted_alpha_min": 1.0e-12,
        # Backward-compatible RC1 aliases only. New production configs use generic RAS.
        "sella_ras_tolerance": None,
        "sella_ras_max_iterations": 1000,
        "sella_qn_curvature_regularization": 1.0e-12,
        "denominator_tolerance": 1.0e-12,
        "root_cluster_tolerance": 1.0e-10,
        "matrix_symmetry_tolerance": 1.0e-12,
    },
    "ourIsopotential": {
        "estimator": "off",               # off | directional
        "regime_policy": "source_faithful_guarded",
        "displacement": None,              # required explicitly when enabled
        "release_f": 0.0,
        "purpose": "diagnostic",          # diagnostic | algorithm
        "promote_observation_to_algorithm_consumers": False,
        "force_tolerance": 1.0e-14,
        "direction_tolerance": 1.0e-14,
        "mode_tolerance": 1.0e-14,
    },
    "ourQNShadow": {
        "dense_diagnostic": False,
        "max_dimension": 512,
        "max_matrix_bytes": 134217728,
        "max_work_units": 200000000,
        "store_dense_matrix": False,
        "near_zero_absolute": 1.0e-12,
        "near_zero_relative": 1.0e-10,
    },
    "ourDimer": {
        "dataset_type": None,
        "reaction_types": None, # Bulk: vacancy hop_reuse hop_insert kickout_reuse displace_kickout_reuse
                               #       kickout_insert ring initial_guess all_atoms random_bubble
                               # OC: all_movable adsorbate_atom adsorbate_atom_neighbors adsorbate diffusion
                               #     rotation adsorbate_surface surface custom initial_guess random_bubble
        "num_attempts_per_type": 1,
        # Deterministic schedule: N normal attempts, then one Gaussian replacement.
        # 0 disables scheduled Gaussian replacements.
        "gaussian_normal_attempts": 0,
        # Deprecated integer-only compatibility alias. Decimal probabilities fail loudly.
        "gaussian_swap_prob": None,
        # Ranked mechanisms never synthesize Gaussian attempts after candidate exhaustion.
        "reuse_exhaustion": "stop",
        "ring_sizes": "3 4",
        "ring_mode": "arc",
        "ring_frac": 0.2,
        "ring_neighbor_mult": 1.20,
        "ring_neighbor_cutoff": None,
        "ring_max_cycles": 20000,
        "supercell": True,
        "delocalization_threshold": 0.8,
        "extension_check_fmax": 0.4,
        "extension_check_curvature": -0.2,
        # Legacy engine values remain fully supported: ase | kappa | sella.
        # New configurations should use engine=mmf and select the pieces
        # independently below.  [ourDimer] keeps its historical name for
        # compatibility even though it now configures the general saddle search.
        "engine": "ase",            # legacy ase|kappa|sella, or new mmf
        # dimer | lanczos | davidson | softsaddle_lanczos |
        # softsaddle_davidson | softsaddle_davidson_textbook | olsen_jd
        "min_mode_finder": "dimer",
        "mmf_variant": "standard",  # standard | kappa (kappa currently dimer-only)
        "mode_reuse": "none",       # none | sm50 (Lanczos/Davidson only)
        # Positive-curvature translation policy. Bowl breakout confines
        # each accepted MMF translation to the atoms carrying the largest
        # physical force norms; all atoms are released once curvature is non-positive.
        "convex_escape": "standard",  # standard | bowl_breakout
        "bowl_active_atoms": 20,       # Pedersen/Luisier N_confine
        "rotation_optimizer": "ase",    # ase | lbfgs | cg | generic_good_broyden | johnson_modified_broyden
        "translation_optimizer": "ase", # ase | lbfgs | fire_lbfgs | canonical_lbfgs | partitioned_lbfgs | q_lbfgs_dimer_axial | rfo | prfo | qn_mmf | generic_good_broyden | johnson_modified_broyden
        "kappa_beta": 2.0,          # only used when mmf_variant = kappa
        "kappa_recover_fmax": 0.3,  # only used when engine = kappa
        "vasp_command": None,
        "vasp_ncore": None,
        # Ranked-candidate offset. Effective default is 0. None permits the
        # sm_offset/SM_OFFSET compatibility fallback in structure_edit.py.
        "bulk_reuse_offset": None,
        "sm_offset": None,
        "concentrate_prob": 0.0,     # fraction of Gaussian attempts replaced by concentration; 0 = off
        "concentrate_power": 1.5,    # >1 increasingly localizes relative atomic magnitudes; 1 preserves raw Gaussian ratios
        "concentrate_peak_median": 0.5,   # A; median of the independent largest-atom displacement distribution
        "concentrate_peak_log_std": 0.25, # dimensionless log-space sigma; central 90% ~= 0.331-0.754 A at median 0.5 A
        "concentrate_envelope": 0.0, # >0 Gaussian spatial envelope width (A) around the kick center
        # Legacy normalization keys are recognized only to fail loudly when
        # concentration is enabled. Do not use in new configs.
        "concentrate_std": None,
        "concentrate_max_disp": None,
    },
    "ourMinMode": {
        # Iterative minimum-mode solver settings.  Defaults mirror the
        # public SoftSaddle benchmark where directly transferable.
        "max_iterations": 8,
        # Sella-compatible Rayleigh-Ritz/JD0 subspace cap used only by
        # min_mode_finder=olsen_jd. None preserves Sella 2.5.0's normal
        # dimension-based limit. This is intentionally separate from
        # max_iterations, which keeps its existing meaning for other finders.
        "maxiter": None,
        # Historical SoftSaddle semantics differ by solver: Lanczos uses
        # relative lowest-Ritz change; Davidson uses absolute eV/A^2 change.
        "eigenvalue_tolerance": 0.01,
        "finite_difference": 1.0e-4,   # A; one-sided HVP displacement
        "breakdown_tolerance": 1.0e-12,
        # Generic Davidson defaults to B0 = initial_hessian * I. It may
        # instead use a full finite-difference Hessian at the attempt's
        # pre-displacement reference geometry. SoftSaddle-Davidson forces
        # reference-Hessian initialization regardless of this selector.
        "davidson_initial_hessian": 1.0,
        "davidson_initial_hessian_source": "identity", # identity | reference_fd
        "davidson_update_tolerance": 1.0e-12,
        "davidson_preconditioner_floor": 1.0e-8,
        # Historical libdavidson hybrid: trust B when
        # ||H q0 - B q0|| * ||v0|| <= 12, otherwise use its embedded
        # Lanczos branch. The historical source has no denominator guard;
        # this tiny floor is a numerical safety guard only.
        "softsaddle_switch_threshold": 12.0,
        "softsaddle_preconditioner_floor": 1.0e-12,
        # Reference Hessian used to seed SoftSaddle Davidson. False
        # matches historical 2N reference/probe force-call accounting.
        # True reuses one reference force and gives the same deterministic
        # one-sided FD matrix in N+1 calls.
        "reference_hessian_finite_difference": 1.0e-4,
        "reference_hessian_reuse_center_force": False,
        # Historical SoftSaddle sm50 rule:
        # tol = ls_tol + (maximum_translation-ls_tol)*factor/100.
        "sm50_factor": 50.0,
        "sm50_line_search_tolerance": 0.01,
        # Historical SoftSaddle refreshes the final mode after convergence
        # but does not re-test convergence with that refreshed mode.
        "sm50_final_refresh_recheck": False,
        # projected-Olsen/JD typed HVP provenance and Olsen/JD options. Legacy finders keep
        # their existing numerical stopping defaults when these remain default.
        "hvp_origin": "physical_fd", # physical_fd | explicit_physical_matrix | approximate_physical_model
        "root_policy": "lowest",     # lowest | homed_overlap
        "residual_stop": False,
        "residual_tolerance": 1.0e-3,
        "olsen_restart_dimension": None,
        "olsen_condition_threshold": 1.0e12,
        "olsen_pseudoinverse_rcond": 1.0e-12,
    },
    "ourDimerLBFGS": {
        # Inert unless an L-BFGS optimizer is explicitly selected.
        "rotation_memory": 10,
        "translation_memory": 10,
        "rotation_initial_hessian": 1.0,
        # Passed to ASE LBFGS as alpha. The initial inverse Hessian is 1/alpha.
        "translation_initial_hessian": 70.0,
        "rotation_dynamic_h0": False,
        "translation_dynamic_h0": False,  # compatibility only; ASE LBFGS keeps H0=1/alpha fixed
        "translation_damping": 1.0,
        "curvature_epsilon": 1.0e-12,
        # Rotation choices are orthogonal: geometry controls secant transport,
        # step_method controls Fourier versus direct displacement, and the
        # history section controls local versus force-bank reuse.
        "rotation_geometry": "projected",       # projected | riemannian
        "rotation_transport_policy": "auto",   # auto | legacy_projection | double_projection | sequential_projection | direct_transport | double_transport | sequential_transport
        "rotation_step_method": "fourier",      # fourier | direct
        "rotation_first_angle_degrees": 45.0,
        "rotation_max_angle_degrees": 45.0,
        # Default-off shadow diagnostic reconstructs dense sequential BFGS
        # from the exact L-BFGS pairs without changing the trajectory.
        "dense_bfgs_diagnostic": False,
        "translation_deep_qn_diagnostics": False,
        # Diagnostic-only lossless translation L-BFGS state dump.
        "translation_state_dump": False,
        # One-based accepted translation-step numbers, or "all" / "none".
        "translation_state_dump_steps": "all",
        "translation_state_dump_two_loop_trace": True,
        # Empty => <method>_lbfgs_state_dumps in the attempt working directory.
        "translation_state_dump_directory": "",

        # Driving quasi-Newton representation for generalized/canonical consumers.
        "rotation_reconstruction_model": "lbfgs", # lbfgs | reconstructed_bfgs
        "translation_reconstruction_model": "lbfgs", # lbfgs | reconstructed_bfgs
        "rotation_bfgs_update": "sequential", # sequential | multisecant
        "translation_bfgs_update": "sequential", # sequential | multisecant
        # Historical rotation behavior: admit only sufficiently positive s.y pairs.
        "rotation_curvature_guard": "legacy_skip", # legacy_skip | off | skip | damp | shifted_secant(alias) | powell | reset
        "rotation_curvature_floor": 1.0e-3,
        "rotation_powell_eta": 0.2,
        # Default to pair skipping; Powell remains an explicit experiment.
        "translation_curvature_guard": "skip", # legacy_damp | off | skip | damp | shifted_secant(alias) | powell | reset
        "translation_curvature_floor": 1.0e-3,
        "translation_powell_eta": 0.2,
        "translation_cautious_epsilon": 1.0e-6,
        "translation_cautious_alpha": 1.0,
        "translation_regularization": "off",   # off | shifted_lbfgs_fixed | shifted_lbfgs_trust | shifted_trust_region(alias)
        "translation_regularization_mu": 1.0,
        "translation_regularization_radius": 0.1,
        "translation_regularization_tolerance": 1.0e-8,
        # Opt-in canonical-force-bank globalization. Existing behavior remains off.
        "translation_trial_step": "off",       # off | secant_force_root
        "reset_translation_on_regime_change": True,
    },
    "ourDimerHistory": {
        # Generic raw PES observation history.  False leaves every legacy
        # numerical path unchanged; canonical_lbfgs enables it automatically.
        "enabled": False,
        # Reconstruct rotational L-BFGS secants from raw mode-tagged
        # Dimer stencils across canonical center states.
        "rotation_reuse": False,
        # Compatibility default: a historical rotation_reuse=True config
        # continues to use the previously installed derived-pair cache.
        # Select force_bank explicitly for canonical raw-force reuse.
        "rotation_history_source": "legacy_derived", # legacy_derived | force_bank
        "rotation_max_pairs": 0,                   # 0 = all admissible pairs in retained states
        "rotation_pair_sources": "consecutive_physical", # consecutive_physical | fourier_trial | accepted_rotation | direct_accepted
        # accepted_rotation can optionally consume ASE Fourier-extrapolated
        # accepted-point torques. Default preserves physical-only force-bank behavior.
        "rotation_accepted_force_source": "physical_only", # physical_only | allow_extrapolated
        "rotation_trial_pairs_future_only": False,
        # Optional independent cosine admission for force-bank rotation pairs.
        "rotation_cosine_threshold": "off",
        "memory_states": 20,
        "max_probes_per_state": 0,
        "center_tolerance": 1.0e-12,
        "sample_tolerance": 1.0e-12,
        "record_sources": "center rotation lanczos davidson reference_hessian translation_trial",
        # center_center isolates current-state reprojection.  Add
        # center_rotation to benchmark reuse of already-paid-for Dimer probes.
        "pair_sources": "center_center",
        "pair_order": "acquisition",
        "projection_policy": "current_snapshot",
        "kappa_projection_policy": "fixed_current_center_coefficients",
        # Default to pair skipping; Powell remains available explicitly.
        "pair_safeguard": "skip",
        "curvature_floor": 1.0e-3,
        "curvature_epsilon": 1.0e-12,
        "powell_eta": 0.2,
        # Optional independent cosine admission for canonical translation pairs.
        "cosine_threshold": "off",
        # Li-Fukushima cautious update is opt-in; baseline skip remains unchanged.
        "cautious_epsilon": 1.0e-6,
        "cautious_alpha": 1.0,
        # Scientific variant: remove all-atom rigid translations from force/s/y/final step.
        "rigid_translation_projection": False,
        # Analysis-only sidecar; no extra calculator calls.
        "deep_qn_diagnostics": False,
        "diagnostic_shift_mus": "0.01 0.1 1 10",
        "dynamic_h0": False,
        "max_pairs": 0,  # 0 = all admissible canonical pairs
        # Diagnostic only by default. The comparison is still computed and
        # written to canonical_projection_validation_* fields. Fatal
        # behavior requires the explicit new value ``fatal``.
        "projection_validation": "off",
        "projection_validation_tolerance": 1.0e-8,
        "reuse_ablation": True,
        "reconstruction_timing": True,
        "raw_dump": False,
        "raw_dump_directory": "Dimer_history_npzs",
    },
    "ourDimerHybrid": {
        # Dormant unless translation_optimizer=fire_lbfgs (or hybrid alias)
        # and enabled=True. FIRE steps can populate the live L-BFGS history.
        "enabled": False,
        "enter_fmax": 0.30,
        "exit_fmax": 0.50,
        "enter_curvature": -0.05,
        "exit_curvature": 0.00,
        "enter_stable_steps": 3,
        "exit_stable_steps": 2,
        "minimum_history_pairs": 3,
        "warm_start_history": True,
        "reset_history_on_exit": True,
        "fire_dt": 0.10,
        "fire_dtmax": 1.0,
        "fire_Nmin": 5,
        "fire_finc": 1.1,
        "fire_fdec": 0.5,
        "fire_astart": 0.1,
        "fire_fa": 0.99,
    },
    "ourSella": {
        # Passive diagnostics only; does not alter Sella trajectory.
        "passive_qn_diagnostics": False,
        # Exception-only W5-009 capture. Records bounded least-squares state and
        # re-raises the original LinAlgError; no retry/fallback is performed.
        "linalg_failure_diagnostics": False,
        # Sella-ablation/W5: None preserves the native Sella path exactly; any explicit
        # selector installs one attempt-local, instance-scoped session.
        # Ritz-partition ablation uses ritzmode_prfo_partition to change only PRFO's unstable
        # partition direction while retaining native PES.diag cadence/calls.
        # qn_newton_safe_false is QN-only and changes only the live native Sella
        # QuasiNewton stepper flag consumed by RestrictedAtomicStep.
        "ablation": None,
        # Used only when [ourDimer] engine = sella. Sella is a distinct
        # first-order saddle engine; Dimer rotation/translation selectors
        # remain at their defaults and are not applied to Sella.
        # sella preserves Sella's stock approximate-Hessian path;
        # fairchem_direct supplies a full UMA Cartesian Hessian through
        # Sella's hessian_function callback using a second cached predictor.
        "hessian_engine": "sella",  # sella | fairchem_direct
        "internal": False,
        "eig": True,
        "method": "prfo",
        "delta0": 0.1,
        "eta": 1.0e-4,
        "gamma": 0.1,
        "threepoint": False,
        "constraints_tol": 1.0e-5,
        "nsteps_per_diag": 3,
        # Maximum Rayleigh-Ritz subspace size per native Sella diagonalization.
        # None preserves Sella 2.5.0 stock behavior (full free-space allowed).
        "maxiter": None,
        "diag_every_n": None,
        "restricted_step": None,
        "sigma_inc": None,
        "sigma_dec": None,
        "rho_inc": None,
        "rho_dec": None,
        "allow_fragments": False,
        "project_translations": None,
        "project_rotations": None,
        "require_first_order_model": True,
        "negative_eigenvalue_tolerance": 1.0e-6,
        "check_desorption": True,
        "check_delocalization": False,
        "check_interval": 5,
    },
    # SaddleMill-side VASP-input orchestration (the [Vasp] section itself is a
    # pure pass-through to ASE's Vasp calculator and never holds our keys).
    "ourVasp": {
        "input_generator": None,    # built-in (omat24_static|omat24_relax|cheap_omat|oc20|cheap_oc20|oc22|cheap_oc22) | module:func | file.py:func
        "extra_input_files": None,  # built-in (modecar) | module:func | file.py:func | space-separated list
        "extra_outputs": None,      # built-in (vtst_dimer) | module:func | file.py:func | space-separated list
    },
}

RENAMED_KEYS = (
    ("ourNEB", "intermediate_minima_check_interval", "intermediate_minima_check_step"),
    ("ourNEB", "add_images_check_interval", "add_images_step"),
)

__all__ = ["DEFAULTS", "RENAMED_KEYS"]
