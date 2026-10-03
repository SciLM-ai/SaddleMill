"""Stage-A runtime checkpoint state for canonical/rotational histories.

This module serializes already-existing state objects without changing their
numerical algorithms.  It is activated only for history-enabled Dimer/MMF runs.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping
import numpy as np

from saddlemill.dimertools.foundation_types import json_safe
from saddlemill.dimertools.force_evaluator import PhysicalForceEvaluator

RUNTIME_SCHEMA = "saddlemill_dimer_runtime_state_v1"
STATE_WINDOW_SCHEMA = "saddlemill_state_window_rotation_history_v1"
SEQUENTIAL_SCHEMA = "saddlemill_sequential_rotation_history_v1"


def active_coordinate_provenance(obj, positions):
    """Mirror the existing whole-atom fixed-coordinate Dimer convention."""
    mask=np.ones(np.asarray(positions).shape,dtype=bool)
    constraints=getattr(obj,"constraints",None)
    if constraints is None:
        wrapped=getattr(obj,"atoms",None)
        constraints=getattr(wrapped,"constraints",()) if wrapped is not None else ()
    constraints=list(constraints or ())
    if not constraints:
        return mask,"movable_cartesian_dofs"
    getter=getattr(constraints[0],"get_indices",None)
    if not callable(getter):
        return None,"unspecified"
    try:
        fixed=np.asarray(getter(),dtype=int).reshape(-1)
    except Exception:
        return None,"unspecified"
    if fixed.size and (np.any(fixed<0) or np.any(fixed>=mask.shape[0])):
        return None,"unspecified"
    mask[fixed,:]=False
    return mask,"movable_cartesian_dofs"


def _finite_array(value, *, name):
    array=np.asarray(value,dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array.copy()


def state_window_to_state_dict(history):
    pairs=[]
    for state_id, items in history._pairs_by_state.items():
        for item in items:
            pairs.append({"state_id": int(state_id), "serial": int(item.serial),
                          "kind": str(item.kind), "s": item.s.tolist(), "y": item.y.tolist()})
    return {
        "schema": STATE_WINDOW_SCHEMA,
        "settings": {
            "memory_states": history.memory_states,
            "initial_hessian": history.initial_hessian,
            "dynamic_h0": history.dynamic_h0,
            "curvature_epsilon": history.curvature_epsilon,
            "dense_bfgs_diagnostic": history.dense_bfgs_diagnostic,
        },
        "state_order": [int(v) for v in history._pairs_by_state.keys()],
        "current_state_id": history.current_state_id,
        "next_serial": int(history._next_serial),
        "pairs": pairs,
        "counters": {
            name: json_safe(getattr(history,name)) for name in (
                "accepted_pairs_total","rejected_pairs_total","trial_pairs_accepted_total",
                "trial_pairs_rejected_total","local_pairs_accepted_total","local_pairs_rejected_total",
                "states_dropped_total","pairs_dropped_with_states_total","reset_count","last_reset_reason",
                "last_pair_metrics","last_pair_accepted","apply_calls","apply_ns_total","last_apply_ns",
                "last_apply_pair_candidates","last_apply_pairs_used","last_apply_pairs_rejected_projection",
                "last_apply_states_contributing","last_apply_local_pairs_used","last_apply_trial_pairs_used",
                "last_dense_diagnostics",
            )
        },
    }


def state_window_from_state_dict(state, *, expected_settings=None):
    if state.get("schema") != STATE_WINDOW_SCHEMA:
        raise ValueError("unsupported StateWindowRotationHistory state schema")
    from saddlemill.dimertools.rotation_history import RotationSecant, StateWindowRotationHistory
    settings=dict(state.get("settings",{}))
    result=StateWindowRotationHistory(**settings)
    if expected_settings:
        for key,value in expected_settings.items():
            if key in settings and settings[key] != value:
                raise ValueError(f"rotation history resume setting mismatch for {key}")
    order=[int(v) for v in state.get("state_order",[])]
    if len(order)!=len(set(order)) or len(order)>result.memory_states:
        raise ValueError("invalid serialized rotation state order")
    current=state.get("current_state_id")
    if current is not None and int(current) not in order:
        raise ValueError("serialized current rotation state is not retained")
    result._pairs_by_state=OrderedDict((sid,[]) for sid in order)
    serials=set()
    for raw in state.get("pairs",[]):
        item=dict(raw); sid=int(item["state_id"]); serial=int(item["serial"])
        if sid not in result._pairs_by_state or serial in serials:
            raise ValueError("invalid or duplicate serialized rotation pair")
        serials.add(serial)
        s=_finite_array(item["s"],name="rotation s").reshape(-1)
        y=_finite_array(item["y"],name="rotation y").reshape(-1)
        if s.shape != y.shape:
            raise ValueError("serialized rotation pair shape mismatch")
        result._pairs_by_state[sid].append(RotationSecant(sid,serial,str(item["kind"]),s,y))
    next_serial=int(state.get("next_serial",0))
    if serials and next_serial <= max(serials):
        raise ValueError("serialized next rotation serial would duplicate a pair")
    result._next_serial=next_serial
    result.current_state_id=None if current is None else int(current)
    counters=dict(state.get("counters",{}))
    for name,value in counters.items():
        if hasattr(result,name): setattr(result,name,value)
    return result


def sequential_to_state_dict(history):
    return {
        "schema": SEQUENTIAL_SCHEMA,
        "policy": history.policy,
        "memory_states": history.memory_states,
        "current_mode": None if history.current_mode is None else history.current_mode.tolist(),
        "current_state_id": history.current_state_id,
        "state_order": [int(v) for v in history._state_order],
        "pairs": [{
            "anchor_mode": pair.anchor_mode.tolist(), "s": pair.s.tolist(), "y": pair.y.tolist(),
            "state_id": int(pair.state_id), "serial": int(pair.serial), "source": pair.source,
            "geometry": pair.geometry, "metadata": json_safe(pair.metadata),
        } for pair in history._pairs],
        "counters": {
            "advance_steps": history.advance_steps, "state_sync_steps": history.state_sync_steps,
            "vectors_transformed": history.vectors_transformed,
            "pairs_dropped_states": history.pairs_dropped_states,
            "s_step_ratios": list(history._s_step_ratios), "y_step_ratios": list(history._y_step_ratios),
            "cumulative_path_angle": history._cumulative_path_angle,
        },
    }


def sequential_from_state_dict(state, *, expected_policy=None, expected_memory_states=None):
    if state.get("schema") != SEQUENTIAL_SCHEMA:
        raise ValueError("unsupported SequentialRotationHistory state schema")
    from saddlemill.dimertools.riemannian_lbfgs import RotationSecant, SequentialRotationHistory
    policy=str(state["policy"]); memory=int(state["memory_states"])
    if expected_policy is not None and policy != str(expected_policy):
        raise ValueError("sequential rotation resume policy mismatch")
    if expected_memory_states is not None and memory != int(expected_memory_states):
        raise ValueError("sequential rotation resume memory mismatch")
    result=SequentialRotationHistory(policy=policy,memory_states=memory)
    order=[int(v) for v in state.get("state_order",[])]
    if len(order)!=len(set(order)) or len(order)>memory:
        raise ValueError("invalid serialized sequential state order")
    current_state=state.get("current_state_id")
    if current_state is not None and int(current_state) not in order:
        raise ValueError("serialized sequential current state is not retained")
    mode=state.get("current_mode")
    result.current_mode=None if mode is None else _finite_array(mode,name="sequential current mode")
    result.current_state_id=None if current_state is None else int(current_state)
    result._state_order=order
    serials=set(); pairs=[]
    for raw in state.get("pairs",[]):
        item=dict(raw); serial=int(item["serial"]); sid=int(item["state_id"])
        if serial in serials or sid not in order:
            raise ValueError("invalid or duplicate serialized sequential pair")
        serials.add(serial)
        pair=RotationSecant(
            anchor_mode=_finite_array(item["anchor_mode"],name="sequential anchor"),
            s=_finite_array(item["s"],name="sequential s"),
            y=_finite_array(item["y"],name="sequential y"), state_id=sid, serial=serial,
            source=str(item["source"]), geometry=str(item["geometry"]), metadata=dict(item.get("metadata",{})),
        )
        pairs.append(pair)
    result._pairs=pairs
    counters=dict(state.get("counters",{}))
    result.advance_steps=int(counters.get("advance_steps",0)); result.state_sync_steps=int(counters.get("state_sync_steps",0))
    result.vectors_transformed=int(counters.get("vectors_transformed",0)); result.pairs_dropped_states=int(counters.get("pairs_dropped_states",0))
    result._s_step_ratios=[float(v) for v in counters.get("s_step_ratios",[])]
    result._y_step_ratios=[float(v) for v in counters.get("y_step_ratios",[])]
    result._cumulative_path_angle=float(counters.get("cumulative_path_angle",0.0))
    return result


def capture_dimer_runtime_state(dimeratoms):
    evaluator=getattr(dimeratoms,"physical_force_evaluator",None)
    wave_runtime=getattr(dimeratoms,"wave_b_runtime",None)
    has_wave_state = any(hasattr(dimeratoms, name) for name in (
        "_wave_b_cg_by_root", "_wave_b_rotation_broyden_by_root", "_wave_b_translation_broyden_state"
    )) or wave_runtime is not None
    has_stage_c_state = any(hasattr(dimeratoms, name) for name in (
        "_partitioned_lbfgs_state", "_rfo_runtime_state"
    ))
    if evaluator is None and not has_wave_state and not has_stage_c_state:
        return None
    payload={"schema":RUNTIME_SCHEMA}
    if evaluator is not None:
        payload["force_evaluator"]=evaluator.to_state_dict()
    if wave_runtime is not None:
        payload["wave_b_runtime"]=wave_runtime.to_state_dict()
    for attr, key in (
        ("_wave_b_cg_by_root", "wave_b_cg_by_root"),
        ("_wave_b_rotation_broyden_by_root", "wave_b_rotation_broyden_by_root"),
        ("_wave_b_translation_broyden_state", "wave_b_translation_broyden_state"),
    ):
        value=getattr(dimeratoms,attr,None)
        if value is not None:
            payload[key]=json_safe(value)
    rotation=getattr(dimeratoms,"_canonical_rotation_history",None)
    if rotation is not None:
        payload["state_window_rotation_history"]=state_window_to_state_dict(rotation)
    partitioned=getattr(dimeratoms,"_partitioned_lbfgs_state",None)
    if partitioned is not None:
        payload["partitioned_lbfgs_state"]=json_safe(partitioned)
    rfo=getattr(dimeratoms,"_rfo_runtime_state",None)
    if rfo is not None:
        payload["rfo_runtime_state"]=json_safe(rfo)
    sequential=getattr(dimeratoms,"_sm_sequential_rotation_history",None)
    if sequential is not None:
        payload["sequential_rotation_history"]=sequential_to_state_dict(sequential)
    return payload


def restore_force_evaluator(runtime_state):
    if runtime_state is None: return None
    if runtime_state.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("unsupported Dimer runtime-state schema")
    state=runtime_state.get("force_evaluator")
    return None if state is None else PhysicalForceEvaluator.from_state_dict(dict(state))


def restore_rotation_histories(dimeratoms, runtime_state, *, history_options, rotation_lbfgs_options):
    if runtime_state is None: return
    if runtime_state.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("unsupported Dimer runtime-state schema")
    window=runtime_state.get("state_window_rotation_history")
    if window is not None:
        if not bool(history_options.get("rotation_reuse",False)):
            raise ValueError("resume state contains persistent rotation history but rotation_reuse is disabled")
        expected={
            "memory_states": int(history_options.get("memory_states",20)),
            "initial_hessian": float(rotation_lbfgs_options.get("initial_hessian",1.0)),
            "dynamic_h0": bool(rotation_lbfgs_options.get("dynamic_h0",False)),
            "curvature_epsilon": float(rotation_lbfgs_options.get("curvature_epsilon",1e-12)),
            "dense_bfgs_diagnostic": bool(rotation_lbfgs_options.get("dense_bfgs_diagnostic",False)),
        }
        dimeratoms._canonical_rotation_history=state_window_from_state_dict(dict(window),expected_settings=expected)
    sequential=runtime_state.get("sequential_rotation_history")
    if sequential is not None:
        from saddlemill.dimertools.riemannian_lbfgs import normalize_rotation_transport_policy
        policy=normalize_rotation_transport_policy(rotation_lbfgs_options.get("transport_policy","auto"), geometry=rotation_lbfgs_options.get("geometry","projected"))
        dimeratoms._sm_sequential_rotation_history=sequential_from_state_dict(
            dict(sequential), expected_policy=policy,
            expected_memory_states=int(history_options.get("memory_states",20)),
        )

__all__=[
    "active_coordinate_provenance","RUNTIME_SCHEMA","STATE_WINDOW_SCHEMA","SEQUENTIAL_SCHEMA","capture_dimer_runtime_state",
    "restore_force_evaluator","restore_rotation_histories","state_window_to_state_dict",
    "state_window_from_state_dict","sequential_to_state_dict","sequential_from_state_dict",
]
