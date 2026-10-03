"""No-extra-PES capture of the inherited Dimer mode's entry rotational force.

This module is deliberately independent of ASE imports.  The mixin wraps the
search object's existing ``get_rotational_force()`` call and snapshots only the
*first* returned rotational-force vector for that real Dimer solve.  The search
algorithm still owns when/how often that method is called, so instrumentation
cannot add an endpoint or center force evaluation.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


class DimerEntryTorqueCaptureMixin:
    """Capture the first already-computed Dimer rotational force of a solve."""

    def get_rotational_force(self):
        force = super().get_rotational_force()
        if getattr(self, "_sm_entry_torque_captured", False):
            return force

        self._sm_entry_torque_captured = True
        payload: dict[str, object] = {
            "valid": False,
            "reason": "unavailable_or_invalid",
            "initial_torque_norm": None,
            "f_rot_max": None,
            "entry_mode_already_converged": False,
            "entry_mode": None,
            "additional_pes_calls": 0,
        }
        try:
            torque = np.asarray(force, dtype=float)
            getter = getattr(self, "get_eigenmode", None)
            mode = np.asarray(getter() if callable(getter) else self.eigenmode, dtype=float)
            f_rot_max = float(self.control.get_parameter("f_rot_max"))
            torque_norm = float(np.linalg.norm(torque))
            if (
                torque.shape != mode.shape
                or not np.all(np.isfinite(torque))
                or not np.all(np.isfinite(mode))
                or not np.isfinite(torque_norm)
                or torque_norm < 0.0
                or not np.isfinite(f_rot_max)
                or f_rot_max < 0.0
            ):
                raise ValueError("nonfinite or shape-incompatible entry torque/mode")
            payload.update({
                "valid": True,
                "reason": "captured_first_existing_rotational_force",
                "initial_torque_norm": torque_norm,
                "f_rot_max": f_rot_max,
                "entry_mode_already_converged": bool(torque_norm <= f_rot_max),
                "entry_mode": mode.copy(),
            })
        except Exception as exc:  # fail closed; scheduling decides what unavailable means.
            payload["reason"] = f"capture_failed:{type(exc).__name__}"
        self.sm_entry_torque_capture = payload
        return force


def entry_torque_capture_payload(search) -> dict[str, object]:
    """Return a detached payload from a completed real Dimer search."""
    raw = getattr(search, "sm_entry_torque_capture", None)
    if not isinstance(raw, Mapping):
        return {
            "valid": False,
            "reason": "entry_torque_not_observed",
            "initial_torque_norm": None,
            "f_rot_max": None,
            "entry_mode_already_converged": False,
            "entry_mode": None,
            "additional_pes_calls": 0,
        }
    out = dict(raw)
    mode = out.get("entry_mode")
    if mode is not None:
        out["entry_mode"] = np.asarray(mode, dtype=float).copy()
    return out


__all__ = ["DimerEntryTorqueCaptureMixin", "entry_torque_capture_payload"]
