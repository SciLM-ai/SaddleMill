"""Warm-started ASE FIRE to ASE L-BFGS optimizer.

Both algorithms are delegated to ASE.  SaddleMill only owns the hysteretic
switch controller, accepted-state snapshot buffer, and diagnostics.  A warm
FIRE -> L-BFGS switch rebuilds ASE's native L-BFGS state in generalized
coordinates, so ordinary Atoms and filters such as FrechetCellFilter use the
same path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer

from saddlemill.dimertools.lbfgs_dimer import (
    _ASELBFGSState,
    _CurvatureGuardedLBFGS,
    _ase_lbfgs_diagnostic_metrics,
    _cosine_alignment,
    _lbfgs_step_metrics,
)
from saddlemill.dimertools.ase_lbfgs_adapter import _ase_lbfgs_get


@dataclass
class HybridDecision:
    state: str
    switch_event: str = ""
    history_pairs_at_switch: int = 0


class ForceThresholdController:
    """Two-state FIRE/L-BFGS controller with fmax hysteresis."""

    def __init__(
        self,
        enter_fmax=0.20,
        exit_fmax=0.35,
        enter_stable_steps=3,
        exit_stable_steps=2,
        minimum_history_pairs=3,
        warm_start_history=True,
    ):
        self.enter_fmax = float(enter_fmax)
        self.exit_fmax = float(exit_fmax)
        self.enter_stable_steps = int(enter_stable_steps)
        self.exit_stable_steps = int(exit_stable_steps)
        self.minimum_history_pairs = int(minimum_history_pairs)
        self.warm_start_history = bool(warm_start_history)
        if self.enter_fmax <= 0.0 or self.exit_fmax <= 0.0:
            raise ValueError("FIRELBFGS force thresholds must be > 0")
        if self.exit_fmax < self.enter_fmax:
            raise ValueError("FIRELBFGS exit_fmax must be >= enter_fmax")
        if self.enter_stable_steps < 1 or self.exit_stable_steps < 1:
            raise ValueError("FIRELBFGS stable-step counts must be >= 1")
        if self.minimum_history_pairs < 0:
            raise ValueError("FIRELBFGS minimum_history_pairs must be >= 0")
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0

    def force_fire(self, event):
        self.state = "fire"
        self._enter_count = 0
        self._exit_count = 0
        return HybridDecision("fire", str(event), 0)

    def update(self, fmax, history_pairs):
        history_pairs = int(history_pairs)
        if self.state == "fire":
            history_ready = (
                not self.warm_start_history
                or history_pairs >= self.minimum_history_pairs
            )
            enter = float(fmax) <= self.enter_fmax and history_ready
            self._enter_count = self._enter_count + 1 if enter else 0
            self._exit_count = 0
            if self._enter_count >= self.enter_stable_steps:
                self.state = "lbfgs"
                self._enter_count = 0
                return HybridDecision(
                    "lbfgs", "fire_to_lbfgs", history_pairs
                )
        else:
            exit_now = float(fmax) >= self.exit_fmax
            self._exit_count = self._exit_count + 1 if exit_now else 0
            self._enter_count = 0
            if self._exit_count >= self.exit_stable_steps:
                self.state = "fire"
                self._exit_count = 0
                return HybridDecision(
                    "fire", "lbfgs_to_fire_threshold", history_pairs
                )
        return HybridDecision(self.state, "", 0)


class DiagnosticLBFGS(_CurvatureGuardedLBFGS):
    """ASE L-BFGS plus optional secant safeguards and diagnostics.

    With ``curvature_guard='off'`` (the SaddleMill default), the numerical
    history update and step are stock ASE L-BFGS.  ``skip``, shifted-secant
    ``damp``, classical ``powell`` damping, and ``reset`` change only how a
    secant pair is handled before ASE's native two-loop recursion is called.
    """

    def __init__(
        self,
        *args,
        curvature_guard="off",
        curvature_floor=1.0e-3,
        powell_eta=0.2,
        **kwargs,
    ):
        qn_shadow_options = kwargs.pop("qn_shadow_options", None)
        from saddlemill.dimertools.wave_b_shadow import normalize_qn_shadow_options
        self.qn_shadow_options = normalize_qn_shadow_options(qn_shadow_options)
        self.last_qn_shadow_row = None
        self.last_qn_shadow_matrix = None
        self.qn_shadow_cumulative_ns = 0
        alpha = kwargs.get("alpha", 70.0)
        memory = kwargs.get("memory", 100)
        damping = kwargs.get("damping", 1.0)
        self.sm_alpha = 70.0 if alpha is None else float(alpha)
        self.sm_memory = int(memory)
        self.sm_damping = float(damping)
        super().__init__(
            *args,
            curvature_guard=curvature_guard,
            curvature_floor=curvature_floor,
            powell_eta=powell_eta,
            **kwargs,
        )
        self.last_step_diagnostics = None
        self._diagnostic_serial = 0
        self._sm_raw_direction = None

    def determine_step(self, dr):
        # ASE rescales this array in place. Preserve the original proposal for
        # diagnostics, then delegate the numerical operation unchanged.
        self._sm_raw_direction = np.asarray(dr, dtype=float).reshape(-1).copy()
        return super().determine_step(dr)

    def line_search(self, *args, **kwargs):
        # The line-search path bypasses determine_step(). Capture the native
        # quasi-Newton direction before ASE's line-search machinery uses it.
        self._sm_raw_direction = np.asarray(self.p, dtype=float).reshape(-1).copy()
        return super().line_search(*args, **kwargs)

    def step(self, forces=None):
        if forces is None:
            forces = -self.optimizable.get_gradient()
        force = np.asarray(forces, dtype=float).reshape(-1)
        before = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1).copy()
        force_calls_before = int(getattr(self, "force_calls", 0))
        function_calls_before = int(getattr(self, "function_calls", 0))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Please do not pass forces to step\(\)\..*",
                category=UserWarning,
                module=r"ase\.optimize\.optimize",
            )
            super().step(forces=force)
        after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        displacement = after - before
        raw_source = self._sm_raw_direction
        if raw_source is None:
            raw_source = self.p
        raw_direction = np.asarray(raw_source, dtype=float).reshape(-1)
        step_metrics = _lbfgs_step_metrics(
            self.optimizable, raw_direction, displacement, self.maxstep, self.damping
        )
        metrics = _ase_lbfgs_diagnostic_metrics(self)
        guard = self.sm_guard_metrics()
        metrics.update({
            "alpha": self.sm_alpha,
            "initial_inverse_hessian_scale": 1.0 / self.sm_alpha,
            "memory": self.sm_memory,
            "damping": self.sm_damping,
        })
        if self.use_line_search:
            # Line-search LBFGS does not use determine_step()*damping. Keep the
            # proposal/accepted displacement diagnostics but do not mislabel a
            # line-search scale as native maxstep clipping.
            step_metrics["step_clipped"] = ""
            step_metrics["maxstep_rescaled"] = ""
            step_metrics["clip_scale"] = ""
            step_metrics["damping"] = ""
        line_search_force_calls = int(getattr(self, "force_calls", 0)) - force_calls_before
        line_search_function_calls = (
            int(getattr(self, "function_calls", 0)) - function_calls_before
        )
        self._diagnostic_serial += 1
        self.last_step_diagnostics = {
            "diagnostic_serial": self._diagnostic_serial,
            "optimizer_step": int(self.nsteps) + 1,
            "active_optimizer": "lbfgs",
            "switch_event": "",
            "fmax": float(self.optimizable.gradient_norm(force)),
            "step_norm": float(step_metrics["actual_step_norm"]),
            "step_clipped": (
                "" if step_metrics["step_clipped"] == ""
                else int(step_metrics["step_clipped"])
            ),
            "direction_alignment": _cosine_alignment(raw_direction, force),
            "raw_step_norm": step_metrics["raw_step_norm"],
            "raw_step_max": step_metrics["raw_step_max"],
            "actual_step_norm": step_metrics["actual_step_norm"],
            "actual_step_max": step_metrics["actual_step_max"],
            "maxstep": step_metrics["maxstep"],
            "maxstep_rescaled": step_metrics["maxstep_rescaled"],
            "clip_scale": step_metrics["clip_scale"],
            "damping": step_metrics["damping"],
            "applied_scale": step_metrics["applied_scale"],
            "warm_start_history": "",
            "history_pairs_at_switch": "",
            "lbfgs_history_size": int(metrics["history_size"]),
            "lbfgs_pairs_accepted_total": int(guard.get("pairs_accepted", 0)),
            "lbfgs_pairs_rejected_total": int(guard.get("pairs_skipped", 0))
            + int(guard.get("pairs_damped", 0)),
            "lbfgs_pairs_skipped_total": int(guard.get("pairs_skipped", 0)),
            "lbfgs_pairs_damped_total": int(guard.get("pairs_damped", 0)),
            "lbfgs_pairs_powell_damped_total": int(guard.get("pairs_powell_damped", 0)),
            "lbfgs_worst_raw_s_dot_y": guard.get("worst_sy", ""),
            "lbfgs_history_resets": int(guard.get("curvature_resets", 0)),
            "lbfgs_last_reset_reason": "",
            "lbfgs_curvature_guard": guard.get("guard_mode", "off"),
            "lbfgs_curvature_floor": guard.get("curvature_floor", ""),
            "lbfgs_powell_eta": guard.get("powell_eta", ""),
            "lbfgs_latest_guard_action": guard.get("last_guard_action", ""),
            "lbfgs_latest_powell_theta": guard.get("last_powell_theta", ""),
            "lbfgs_latest_powell_s_dot_Bs": guard.get("last_powell_s_dot_Bs", ""),
            "lbfgs_alpha": metrics.get("alpha", ""),
            "lbfgs_initial_inverse_hessian_scale": metrics.get(
                "initial_inverse_hessian_scale", ""
            ),
            "lbfgs_memory": metrics.get("memory", ""),
            "lbfgs_use_line_search": int(bool(self.use_line_search)),
            "lbfgs_force_calls": int(getattr(self, "force_calls", 0)),
            "lbfgs_function_calls": int(getattr(self, "function_calls", 0)),
            "lbfgs_step_force_calls": line_search_force_calls,
            "lbfgs_step_function_calls": line_search_function_calls,
            "lbfgs_alpha_k": (
                getattr(self, "alpha_k", "") if self.use_line_search else ""
            ),
            "lbfgs_latest_s_norm": metrics.get("latest_s_norm", ""),
            "lbfgs_latest_y_norm": metrics.get("latest_y_norm", ""),
            "lbfgs_latest_s_dot_y": metrics.get("latest_s_dot_y", ""),
            "lbfgs_latest_secant_curvature": metrics.get(
                "latest_secant_curvature", ""
            ),
            "lbfgs_latest_secant_cosine": metrics.get("latest_secant_cosine", ""),
            "lbfgs_latest_force_change_norm": metrics.get(
                "latest_force_change_norm", ""
            ),
            "lbfgs_latest_pair_damped": int(guard.get("last_pair_damped", 0)),
            "lbfgs_latest_raw_s_dot_y": guard.get("last_raw_pair", {}).get(
                "s_dot_y", ""
            ),
            "lbfgs_latest_raw_secant_curvature": guard.get(
                "last_raw_pair", {}
            ).get("secant_curvature", ""),
            "lbfgs_latest_stored_s_dot_y": guard.get(
                "last_stored_pair", {}
            ).get("s_dot_y", ""),
            "lbfgs_latest_stored_secant_curvature": guard.get(
                "last_stored_pair", {}
            ).get("secant_curvature", ""),
            "fire_dt": "",
        }
        self._run_qn_shadow(force, raw_direction)
        if self.last_qn_shadow_row is not None:
            self.last_step_diagnostics["t08_qn_shadow"] = dict(self.last_qn_shadow_row)

    def _run_qn_shadow(self, force, raw_direction):
        self.last_qn_shadow_row = None
        self.last_qn_shadow_matrix = None
        if not bool(self.qn_shadow_options.get("enabled", False)):
            return
        s_hist = [np.asarray(x, dtype=float).reshape(-1) for x in list(_ase_lbfgs_get(self, "s", []) or [])]
        y_hist = [np.asarray(x, dtype=float).reshape(-1) for x in list(_ase_lbfgs_get(self, "y", []) or [])]
        ndim = int(np.asarray(force).size)
        S = np.stack(s_hist, axis=0) if s_hist else np.empty((0, ndim), dtype=np.float64)
        Y = np.stack(y_hist, axis=0) if y_hist else np.empty((0, ndim), dtype=np.float64)
        rho_hist = list(_ase_lbfgs_get(self, "rho", []) or [])
        if len(rho_hist) == S.shape[0] and all(np.isfinite(float(r)) and float(r) != 0.0 for r in rho_hist):
            sy = np.asarray([1.0 / float(r) for r in rho_hist], dtype=np.float64)
            rho = np.asarray([float(r) for r in rho_hist], dtype=np.float64)
        else:
            sy = np.asarray([float(np.dot(S[i], Y[i])) for i in range(S.shape[0])], dtype=np.float64)
            rho = np.divide(1.0, sy, out=np.full_like(sy, np.nan), where=np.isfinite(sy) & (sy != 0.0))
        # ASE L-BFGS uses the fixed inverse scale H0=I/alpha.  This is the
        # exact production H0; do not rerun two-loop recursion to rediscover it.
        h0_scale = 1.0 / float(self.sm_alpha)
        payload = {
            "current_two_loop_vector": np.asarray(force, dtype=np.float64).reshape(-1),
            "s_history": S, "y_history": Y, "sy_history": sy, "rho_history": rho,
            "raw_lbfgs_direction": np.asarray(raw_direction, dtype=np.float64).reshape(-1),
            "h0_inverse_scale": np.asarray(h0_scale, dtype=np.float64),
            "initial_hessian": np.asarray(self.sm_alpha, dtype=np.float64),
            "dynamic_h0": np.asarray(False, dtype=np.bool_),
            "replay_active_dof_mask": np.ones(ndim, dtype=np.bool_),
            "history_order": np.asarray("oldest_to_newest"),
            "input_vector_role": np.asarray("optimizer_force"),
            "gradient_sign_convention": np.asarray("g=-F"),
            "two_loop_sign_convention": np.asarray("raw_lbfgs_direction=H*F=-H*g"),
            "two_loop_arithmetic": np.asarray("rho_multiply"),
            "pair_safeguard": np.asarray(str(self.curvature_guard)),
            "max_pairs": np.asarray(int(self.sm_memory), dtype=np.int64),
        }
        from saddlemill.dimertools.wave_b_shadow import run_shadow
        result, elapsed = run_shadow(
            payload, options=self.qn_shadow_options,
            consumer=str(self.qn_shadow_options.get("consumer") or "minimization"),
            residual_kind=str(self.qn_shadow_options.get("residual_kind") or "physical_force"),
            force_interpretation=str(self.qn_shadow_options.get("force_interpretation") or "raw_physical_force"),
            model_type=str(self.qn_shadow_options.get("model_type") or "ordinary_bfgs_hessian"),
            owner=None,
        )
        if result is not None:
            self.last_qn_shadow_row = dict(result.row)
            self.last_qn_shadow_matrix = result.matrix
            self.qn_shadow_cumulative_ns += int(elapsed)

    def hybrid_summary(self):
        metrics = _ase_lbfgs_diagnostic_metrics(self)
        guard = self.sm_guard_metrics()
        metrics.update({
            "alpha": self.sm_alpha,
            "initial_inverse_hessian_scale": 1.0 / self.sm_alpha,
            "memory": self.sm_memory,
            "damping": self.sm_damping,
        })
        return {
            "final_active_optimizer": "lbfgs",
            "switch_count": 0,
            "lbfgs_history_size": int(metrics["history_size"]),
            "lbfgs_pairs_accepted_total": int(guard.get("pairs_accepted", 0)),
            "lbfgs_pairs_rejected_total": int(guard.get("pairs_skipped", 0))
            + int(guard.get("pairs_damped", 0)),
            "lbfgs_pairs_skipped_total": int(guard.get("pairs_skipped", 0)),
            "lbfgs_pairs_damped_total": int(guard.get("pairs_damped", 0)),
            "lbfgs_pairs_powell_damped_total": int(guard.get("pairs_powell_damped", 0)),
            "lbfgs_history_resets": int(guard.get("curvature_resets", 0)),
            "lbfgs_last_reset_reason": "",
            "lbfgs_curvature_guard": guard.get("guard_mode", "off"),
            "lbfgs_curvature_floor": guard.get("curvature_floor", ""),
            "lbfgs_powell_eta": guard.get("powell_eta", ""),
            "lbfgs_latest_guard_action": guard.get("last_guard_action", ""),
            "lbfgs_latest_powell_theta": guard.get("last_powell_theta", ""),
            "lbfgs_latest_powell_s_dot_Bs": guard.get("last_powell_s_dot_Bs", ""),
            "lbfgs_alpha": metrics.get("alpha", ""),
            "lbfgs_initial_inverse_hessian_scale": metrics.get(
                "initial_inverse_hessian_scale", ""
            ),
            "lbfgs_memory": metrics.get("memory", ""),
            "lbfgs_use_line_search": int(bool(self.use_line_search)),
            "lbfgs_force_calls": int(getattr(self, "force_calls", 0)),
            "lbfgs_function_calls": int(getattr(self, "function_calls", 0)),
            "maxstep": float(self.maxstep),
            "damping": metrics.get("damping", ""),
        }


class FIRELBFGS(Optimizer):
    """ASE FIRE warm-up followed by optionally safeguarded ASE L-BFGS."""

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=None,
        fire_dt=0.1,
        fire_dtmax=1.0,
        fire_Nmin=5,
        fire_finc=1.1,
        fire_fdec=0.5,
        fire_astart=0.1,
        fire_fa=0.99,
        lbfgs_memory=10,
        lbfgs_initial_hessian=70.0,
        lbfgs_dynamic_h0=False,
        lbfgs_curvature_epsilon=1.0e-12,
        lbfgs_damping=1.0,
        lbfgs_curvature_guard="off",
        lbfgs_curvature_floor=1.0e-3,
        lbfgs_powell_eta=0.2,
        lbfgs_use_line_search=False,
        enter_fmax=0.20,
        exit_fmax=0.35,
        enter_stable_steps=3,
        exit_stable_steps=2,
        minimum_history_pairs=3,
        warm_start_history=True,
        reset_history_on_exit=True,
        **kwargs,
    ):
        if restart is not None:
            raise NotImplementedError(
                "FIRELBFGS restart files are not implemented; use "
                "SaddleMill continuation from the saved structure instead."
            )
        if float(lbfgs_damping) <= 0.0:
            raise ValueError("FIRELBFGS lbfgs_damping must be > 0")
        # Retained for config compatibility. ASE's LBFGS uses a fixed H0=1/alpha
        # and stores all update pairs; these custom-recursion options are inert.
        self.lbfgs_dynamic_h0 = bool(lbfgs_dynamic_h0)
        self.lbfgs_curvature_epsilon = float(lbfgs_curvature_epsilon)

        Optimizer.__init__(
            self,
            atoms,
            restart=None,
            logfile=logfile,
            trajectory=trajectory,
            **kwargs,
        )
        self.maxstep = (
            float(maxstep) if maxstep is not None else float(self.defaults["maxstep"])
        )
        self.fire_options = {
            "dt": float(fire_dt),
            "dtmax": float(fire_dtmax),
            "Nmin": int(fire_Nmin),
            "finc": float(fire_finc),
            "fdec": float(fire_fdec),
            "astart": float(fire_astart),
            "fa": float(fire_fa),
        }
        self.lbfgs_memory = int(lbfgs_memory)
        self.lbfgs_alpha = float(lbfgs_initial_hessian)
        self.lbfgs_damping = float(lbfgs_damping)
        self.lbfgs_curvature_guard = str(lbfgs_curvature_guard).strip().lower()
        self.lbfgs_curvature_floor = float(lbfgs_curvature_floor)
        self.lbfgs_powell_eta = float(lbfgs_powell_eta)
        self.lbfgs_use_line_search = bool(lbfgs_use_line_search)
        if self.lbfgs_use_line_search and self.lbfgs_damping != 1.0:
            warnings.warn(
                "FIRELBFGS lbfgs_damping is ignored while "
                "lbfgs_use_line_search=True because ASE uses alpha_k*p "
                "directly on the line-search path.",
                UserWarning,
            )
        self.warm_start_history = bool(warm_start_history)
        self.reset_history_on_exit = bool(reset_history_on_exit)
        self.controller = ForceThresholdController(
            enter_fmax=enter_fmax,
            exit_fmax=exit_fmax,
            enter_stable_steps=enter_stable_steps,
            exit_stable_steps=exit_stable_steps,
            minimum_history_pairs=minimum_history_pairs,
            warm_start_history=warm_start_history,
        )
        self.fire_optimizer = self._new_fire_optimizer()
        self.lbfgs_state = _ASELBFGSState(
            atoms,
            maxstep=self.maxstep,
            memory=self.lbfgs_memory,
            damping=self.lbfgs_damping,
            alpha=self.lbfgs_alpha,
            curvature_guard=self.lbfgs_curvature_guard,
            curvature_floor=self.lbfgs_curvature_floor,
            powell_eta=self.lbfgs_powell_eta,
            use_line_search=self.lbfgs_use_line_search,
        )
        self.snapshots = deque(maxlen=self.lbfgs_memory + 1)
        self.last_step_diagnostics = None
        self._diagnostic_serial = 0
        self.switch_count = 0

    def initialize(self):
        # Optimizer.__init__ calls this before the internal optimizers exist.
        pass

    def read(self):
        raise NotImplementedError("FIRELBFGS restart files are not implemented")

    def _new_fire_optimizer(self):
        return FIRE(
            self.atoms,
            restart=None,
            logfile=None,
            trajectory=None,
            maxstep=self.maxstep,
            downhill_check=False,
            **self.fire_options,
        )

    def _append_snapshot(self, position, force):
        item = (
            np.asarray(position, dtype=float).reshape(-1).copy(),
            np.asarray(force, dtype=float).reshape(-1).copy(),
        )
        if self.snapshots and np.array_equal(self.snapshots[-1][0], item[0]):
            self.snapshots[-1] = item
        else:
            self.snapshots.append(item)

    def step(self, forces=None):
        force = -self._get_gradient(forces)
        force = np.asarray(force, dtype=float).reshape(-1)
        position = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
        fmax = float(self.optimizable.gradient_norm(force))
        self._append_snapshot(position, force)
        available_pairs = max(0, len(self.snapshots) - 1)

        decision = self.controller.update(fmax, available_pairs)
        if decision.switch_event:
            self.switch_count += 1

        if decision.switch_event == "fire_to_lbfgs":
            if self.warm_start_history:
                self.lbfgs_state.rebuild(
                    self.snapshots, reason="warm_start_fire_to_lbfgs"
                )
            else:
                self.lbfgs_state.reset("cold_start_fire_to_lbfgs")
        elif decision.switch_event == "lbfgs_to_fire_threshold":
            self.fire_optimizer = self._new_fire_optimizer()
            if self.reset_history_on_exit:
                self.lbfgs_state.reset("lbfgs_to_fire_threshold")
                self.snapshots.clear()
                self._append_snapshot(position, force)

        active = decision.state
        alignment = ""
        before = position.copy()
        if active == "lbfgs":
            step_norm, clipped, alignment = self.lbfgs_state.step(force)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Please do not pass forces to step\(\)\..*",
                    category=UserWarning,
                    module=r"ase\.optimize\.optimize",
                )
                self.fire_optimizer.step(f=force)
            after = np.asarray(self.optimizable.get_x(), dtype=float).reshape(-1)
            displacement = after - before
            step_norm = float(np.linalg.norm(displacement))
            # ASE FIRE clips the global generalized-coordinate norm; use its
            # actual displacement and maxstep for the diagnostic flag.
            clipped = bool(np.linalg.norm(displacement) >= self.maxstep)

        metrics = self.lbfgs_state.metrics()
        lbfgs_step = (
            dict(getattr(self.lbfgs_state, "last_step_metrics", {}) or {})
            if active == "lbfgs"
            else {}
        )
        self._diagnostic_serial += 1
        self.last_step_diagnostics = {
            "diagnostic_serial": self._diagnostic_serial,
            "optimizer_step": int(self.nsteps) + 1,
            "active_optimizer": active,
            "switch_event": decision.switch_event,
            "fmax": fmax,
            "step_norm": float(step_norm),
            "step_clipped": "" if clipped == "" else int(bool(clipped)),
            "direction_alignment": alignment,
            "raw_step_norm": lbfgs_step.get("raw_step_norm", ""),
            "raw_step_max": lbfgs_step.get("raw_step_max", ""),
            "actual_step_norm": (
                lbfgs_step.get("actual_step_norm", "") if active == "lbfgs" else step_norm
            ),
            "actual_step_max": lbfgs_step.get("actual_step_max", ""),
            "maxstep": lbfgs_step.get("maxstep", self.maxstep),
            "maxstep_rescaled": lbfgs_step.get("maxstep_rescaled", ""),
            "clip_scale": lbfgs_step.get("clip_scale", ""),
            "damping": lbfgs_step.get("damping", ""),
            "applied_scale": lbfgs_step.get("applied_scale", ""),
            "warm_start_history": int(self.warm_start_history),
            "history_pairs_at_switch": int(decision.history_pairs_at_switch),
            "lbfgs_history_size": int(metrics["history_size"]),
            "lbfgs_pairs_accepted_total": int(metrics["pairs_accepted_total"]),
            "lbfgs_pairs_rejected_total": int(metrics["pairs_rejected_total"]),
            "lbfgs_pairs_skipped_total": int(metrics.get("pairs_skipped_total", 0)),
            "lbfgs_pairs_damped_total": int(metrics.get("pairs_damped_total", 0)),
            "lbfgs_pairs_powell_damped_total": int(metrics.get("pairs_powell_damped_total", 0)),
            "lbfgs_worst_raw_s_dot_y": metrics.get("worst_sy", ""),
            "lbfgs_history_resets": int(metrics["reset_count"]),
            "lbfgs_last_reset_reason": metrics["last_reset_reason"],
            "lbfgs_curvature_guard": metrics.get("guard_mode", "off"),
            "lbfgs_curvature_floor": metrics.get("curvature_floor", ""),
            "lbfgs_powell_eta": metrics.get("powell_eta", ""),
            "lbfgs_latest_guard_action": metrics.get("latest_guard_action", ""),
            "lbfgs_latest_powell_theta": metrics.get("latest_powell_theta", ""),
            "lbfgs_latest_powell_s_dot_Bs": metrics.get("latest_powell_s_dot_Bs", ""),
            "lbfgs_alpha": metrics.get("alpha", ""),
            "lbfgs_initial_inverse_hessian_scale": metrics.get(
                "initial_inverse_hessian_scale", ""
            ),
            "lbfgs_memory": metrics.get("memory", ""),
            "lbfgs_use_line_search": int(bool(metrics.get("use_line_search", 0))),
            "lbfgs_force_calls": int(metrics.get("force_calls", 0)),
            "lbfgs_function_calls": int(metrics.get("function_calls", 0)),
            "lbfgs_step_force_calls": lbfgs_step.get("step_force_calls", ""),
            "lbfgs_step_function_calls": lbfgs_step.get(
                "step_function_calls", ""
            ),
            "lbfgs_alpha_k": lbfgs_step.get("alpha_k", ""),
            "lbfgs_latest_s_norm": metrics.get("latest_s_norm", ""),
            "lbfgs_latest_y_norm": metrics.get("latest_y_norm", ""),
            "lbfgs_latest_s_dot_y": metrics.get("latest_s_dot_y", ""),
            "lbfgs_latest_secant_curvature": metrics.get(
                "latest_secant_curvature", ""
            ),
            "lbfgs_latest_secant_cosine": metrics.get("latest_secant_cosine", ""),
            "lbfgs_latest_force_change_norm": metrics.get(
                "latest_force_change_norm", ""
            ),
            "lbfgs_latest_pair_damped": metrics.get("latest_pair_damped", ""),
            "lbfgs_latest_raw_s_dot_y": metrics.get("latest_raw_s_dot_y", ""),
            "lbfgs_latest_raw_secant_curvature": metrics.get(
                "latest_raw_secant_curvature", ""
            ),
            "lbfgs_latest_stored_s_dot_y": metrics.get(
                "latest_stored_s_dot_y", ""
            ),
            "lbfgs_latest_stored_secant_curvature": metrics.get(
                "latest_stored_secant_curvature", ""
            ),
            "fire_dt": float(self.fire_optimizer.dt),
        }

    def hybrid_summary(self):
        metrics = self.lbfgs_state.metrics()
        return {
            "final_active_optimizer": self.controller.state,
            "switch_count": int(self.switch_count),
            "lbfgs_history_size": int(metrics["history_size"]),
            "lbfgs_pairs_accepted_total": int(
                metrics["pairs_accepted_total"]
            ),
            "lbfgs_pairs_rejected_total": int(
                metrics["pairs_rejected_total"]
            ),
            "lbfgs_pairs_skipped_total": int(metrics.get("pairs_skipped_total", 0)),
            "lbfgs_pairs_damped_total": int(metrics.get("pairs_damped_total", 0)),
            "lbfgs_pairs_powell_damped_total": int(metrics.get("pairs_powell_damped_total", 0)),
            "lbfgs_use_line_search": int(bool(metrics.get("use_line_search", 0))),
            "lbfgs_curvature_guard": metrics.get("guard_mode", "off"),
            "lbfgs_curvature_floor": metrics.get("curvature_floor", ""),
            "lbfgs_powell_eta": metrics.get("powell_eta", ""),
            "lbfgs_latest_guard_action": metrics.get("latest_guard_action", ""),
            "lbfgs_latest_powell_theta": metrics.get("latest_powell_theta", ""),
            "lbfgs_latest_powell_s_dot_Bs": metrics.get("latest_powell_s_dot_Bs", ""),
            "lbfgs_alpha": metrics.get("alpha", ""),
            "lbfgs_initial_inverse_hessian_scale": metrics.get(
                "initial_inverse_hessian_scale", ""
            ),
            "lbfgs_memory": metrics.get("memory", ""),
            "maxstep": float(self.maxstep),
            "damping": metrics.get("damping", ""),
            "lbfgs_history_resets": int(metrics["reset_count"]),
            "lbfgs_last_reset_reason": metrics["last_reset_reason"],
            "lbfgs_force_calls": int(metrics.get("force_calls", 0)),
            "lbfgs_function_calls": int(metrics.get("function_calls", 0)),
            "warm_start_history": int(self.warm_start_history),
        }
