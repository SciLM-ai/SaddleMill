"""Restart-safe publication for standalone Hessian artifacts.

The legacy Hessian output paths remain readable, but a result is reusable only
when a validated ``hessian_artifact_commit_v1`` record matches the full task
identity.  Shared trajectory/status files are materialized indexes; committed
per-job event trajectories plus commit records are the reconstruction source.
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase.constraints import FixAtoms
from ase.io import Trajectory

ARTIFACT_CONTRACT = "hessian_artifact_commit_v1"
TASK_IDENTITY_SCHEMA = "hessian_task_identity_v1"
ORDER_DEFINITION = "fixatoms_free_free_else_periodic_three_translations_v1"


class HessianArtifactError(RuntimeError):
    """Base class for managed Hessian artifact failures."""


class HessianArtifactIdentityMismatch(HessianArtifactError):
    """Raised when one fixed job index is already committed for another task."""


class HessianArtifactPublicationError(HessianArtifactError):
    """Raised after a computed result entered the publication transaction."""


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: str) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(os.path.abspath(path), threading.Lock())


@contextlib.contextmanager
def _exclusive_lock(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    local_lock = _thread_lock(path)
    with local_lock:
        with open(path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_dir(path: str) -> None:
    directory = os.path.dirname(path) or "."
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with open(temp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json(path: str, value: Mapping[str, Any]) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _atomic_bytes(path, data)


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    temp = os.path.join(
        os.path.dirname(dst),
        f".{os.path.basename(dst)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with open(src, "rb") as source, open(temp, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp, dst)
        _fsync_dir(dst)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _normal_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in Hessian task identity")
        return value
    if isinstance(value, np.generic):
        return _normal_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _normal_json(value[key]) for key in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (list, tuple)):
        return [_normal_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normal_json(item) for item in value.tolist()]
    raise TypeError(
        "Hessian task identity requires JSON-like calculator/config values; "
        f"got {type(value).__name__}"
    )


def _record_json(value: Any) -> Any:
    """JSON-safe artifact metadata while preserving CSV spellings for NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, np.generic):
        return _record_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _record_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _record_json(value.tolist())
    return _normal_json(value)


def _array_identity(array: Any, dtype: str) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))
    return {
        "shape": list(arr.shape),
        "dtype": arr.dtype.str,
        "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
    }


def _constraint_identity(atoms) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for constraint in getattr(atoms, "constraints", []) or []:
        if isinstance(constraint, FixAtoms):
            result.append({
                "type": "ase.constraints.FixAtoms",
                "indices": sorted(int(i) for i in constraint.get_indices()),
            })
            continue
        if hasattr(constraint, "todict"):
            result.append({
                "type": f"{type(constraint).__module__}.{type(constraint).__name__}",
                "value": _normal_json(constraint.todict()),
            })
            continue
        raise TypeError(
            "Hessian task identity cannot stably serialize constraint "
            f"{type(constraint).__name__}"
        )
    return result


def _get_tags(atoms) -> np.ndarray:
    if hasattr(atoms, "get_tags"):
        return np.asarray(atoms.get_tags(), dtype=np.int64)
    arrays = getattr(atoms, "arrays", {}) or {}
    if "tags" in arrays:
        return np.asarray(arrays["tags"], dtype=np.int64)
    return np.zeros(len(atoms), dtype=np.int64)


def build_task_identity(
    atoms,
    config_dict: Mapping[str, Any],
    source_info: Mapping[str, Any],
    *,
    fairchem_contract_version: str,
) -> tuple[str, dict[str, Any]]:
    """Return a stable identity for Hessian science and source provenance.

    Performance/output knobs (chunking, OOM retry, and ``store_hessian``) are not
    part of this identity.  The completion record separately states whether the
    full matrix artifact was requested/published.
    """
    hcfg = dict(config_dict.get("ourHessian", {}) or {})
    ccfg = dict(config_dict.get("FAIRChemCalculator", {}) or {})
    descriptor = {
        "schema": TASK_IDENTITY_SCHEMA,
        "source": {
            "numbers": _array_identity(getattr(atoms, "numbers"), "<i8"),
            "positions_A": _array_identity(getattr(atoms, "positions"), "<f8"),
            "cell_A": _array_identity(np.asarray(getattr(atoms, "cell")), "<f8"),
            "pbc": _array_identity(getattr(atoms, "pbc"), "|u1"),
            "tags": _array_identity(_get_tags(atoms), "<i8"),
            "constraints": _constraint_identity(atoms),
            "parent_source_idx": _normal_json(source_info.get("src_index", "")),
            "parent_attempt_id": _normal_json(source_info.get("attempt_id", "")),
        },
        "scientific_hessian": {
            "restrict_fixed_atoms": bool(hcfg.get("restrict_fixed_atoms", True)),
            "negative_eigenvalue_tolerance_eV_A2": float(
                hcfg.get("negative_eigenvalue_tolerance", 1.0e-6)
            ),
            "order_definition": ORDER_DEFINITION,
        },
        "calculator": {
            "calculator": str(config_dict.get("Main", {}).get("Calculator", "FAIRChemCalculator")),
            "fairchem_core_contract_version": str(fairchem_contract_version),
            "config": _normal_json(ccfg),
        },
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), descriptor


def _publication_boundary(name: str) -> None:
    """No-op hook monkeypatched by Task-06 interruption tests."""


def _write_summary_file(path: str, row: Mapping[str, Any], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        with open(temp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_event_file(path: str, atoms) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}.traj",
    )
    try:
        with Trajectory(temp, "w") as writer:
            writer.write(atoms)
        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _write_npz_file(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = os.path.join(
        os.path.dirname(path),
        f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz",
    )
    try:
        with open(temp, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _safe_token(value: Any) -> str:
    return str(value).replace(os.sep, "_")


class HessianArtifactStore:
    """Per-job restart-safe Hessian publication and reuse manager."""

    def __init__(
        self,
        job_id: Any,
        rank: Any,
        atoms,
        config_dict: Mapping[str, Any],
        source_info: Mapping[str, Any],
        *,
        summary_fields: list[str],
        fairchem_contract_version: str,
    ):
        self.job_id = job_id
        self.rank = rank
        self.atoms = atoms
        self.config_dict = config_dict
        self.source_info = dict(source_info)
        self.summary_fields = list(summary_fields)
        self.store_hessian = bool(config_dict.get("ourHessian", {}).get("store_hessian", True))
        self.task_identity, self.task_descriptor = build_task_identity(
            atoms,
            config_dict,
            self.source_info,
            fairchem_contract_version=fairchem_contract_version,
        )
        job = _safe_token(job_id)
        rank_token = _safe_token(rank)
        self.root = "Hessian_artifacts"
        self.lock_path = f"{self.root}/locks/hessian_{job}.lock"
        self.pending_path = f"{self.root}/pending/hessian_{job}.json"
        self.commit_path = f"{self.root}/commits/hessian_{job}.json"
        self.event_path = f"{self.root}/events/hessian_{job}.{self.task_identity}.traj"
        self.summary_path = f"Hessian_summary_csvs/hessian_{job}.csv"
        self.hessian_path = f"Hessian_hessians/hessian_{job}.npz"
        self.status_path = f"Hessian_status_csvs/status_rank_{rank_token}.csv"
        self.traj_path = f"Hessian_trajes/collected_hessian_rank_{rank_token}.traj"
        self.index_lock_path = f"{self.root}/locks/index_rank_{rank_token}.lock"
        self.legacy_status_snapshot = f"{self.root}/legacy_indexes/status_rank_{rank_token}.csv"
        self.legacy_traj_snapshot = f"{self.root}/legacy_indexes/collected_hessian_rank_{rank_token}.traj"
        self.publication_started = False

    @contextlib.contextmanager
    def locked(self):
        with _exclusive_lock(self.lock_path):
            yield self

    def _quarantine_dir(self, reason: str) -> str:
        stamp = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        path = f"{self.root}/quarantine/hessian_{_safe_token(self.job_id)}_{reason}_{stamp}"
        os.makedirs(path, exist_ok=False)
        return path

    def _move_preserving(self, src: str, directory: str) -> str | None:
        if not src or not os.path.exists(src):
            return None
        dst = os.path.join(directory, os.path.basename(src))
        os.replace(src, dst)
        _fsync_dir(dst)
        return dst

    def record_failure(self, exc: BaseException, *, kind: str) -> str:
        os.makedirs(f"{self.root}/quarantine", exist_ok=True)
        path = (
            f"{self.root}/quarantine/failure_hessian_{_safe_token(self.job_id)}_"
            f"{time.time_ns()}_{os.getpid()}.json"
        )
        _atomic_json(path, {
            "contract": ARTIFACT_CONTRACT,
            "kind": kind,
            "job_id": _record_json(self.job_id),
            "rank": _record_json(self.rank),
            "task_identity": self.task_identity,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return path

    def _preserve_unverified_fixed_outputs(self) -> None:
        if os.path.exists(self.commit_path) or os.path.exists(self.pending_path):
            return
        candidates = [path for path in (self.summary_path, self.hessian_path) if os.path.exists(path)]
        if not candidates:
            return
        directory = self._quarantine_dir("legacy_unverified")
        moved = [self._move_preserving(path, directory) for path in candidates]
        _atomic_json(os.path.join(directory, "legacy_unverified.json"), {
            "contract": ARTIFACT_CONTRACT,
            "validation": "legacy_unverified_not_reusable",
            "job_id": _record_json(self.job_id),
            "task_identity_requested": self.task_identity,
            "files": [path for path in moved if path],
        })

    def _ensure_legacy_index_snapshots(self) -> None:
        marker = f"{self.root}/legacy_indexes/rank_{_safe_token(self.rank)}.initialized.json"
        with _exclusive_lock(self.index_lock_path):
            if os.path.exists(marker):
                return
            if os.path.exists(self.status_path):
                _atomic_copy(self.status_path, self.legacy_status_snapshot)
            if os.path.exists(self.traj_path):
                _atomic_copy(self.traj_path, self.legacy_traj_snapshot)
            _atomic_json(marker, {
                "contract": ARTIFACT_CONTRACT,
                "rank": _record_json(self.rank),
                "legacy_status_snapshot": self.legacy_status_snapshot if os.path.exists(self.legacy_status_snapshot) else "",
                "legacy_traj_snapshot": self.legacy_traj_snapshot if os.path.exists(self.legacy_traj_snapshot) else "",
            })

    def _validate_npz(self, path: str, expected_sha: str) -> None:
        if not os.path.exists(path):
            raise ValueError(f"missing committed Hessian NPZ: {path}")
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"committed Hessian NPZ hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {
                "active_hessian_eV_A2", "order_hessian_eV_A2", "free_dofs",
                "eigenvalues", "lowest_eigenmode", "negative_mode_count",
                "negative_eigenvalue_tolerance", "hessian_artifact_contract",
                "hessian_task_identity",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"committed Hessian NPZ missing fields: {missing}")
            contract = str(np.asarray(data["hessian_artifact_contract"]).item())
            identity = str(np.asarray(data["hessian_task_identity"]).item())
            if contract != ARTIFACT_CONTRACT or identity != self.task_identity:
                raise ValueError("committed Hessian NPZ identity/contract mismatch")
            for key in ("active_hessian_eV_A2", "order_hessian_eV_A2", "eigenvalues", "lowest_eigenmode"):
                if not np.all(np.isfinite(np.asarray(data[key], dtype=float))):
                    raise ValueError(f"committed Hessian NPZ contains non-finite {key}")

    def _summary_bytes(self, row: Mapping[str, Any]) -> bytes:
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=self.summary_fields)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in self.summary_fields})
        return stream.getvalue().encode()

    def _repair_summary(self, commit: Mapping[str, Any]) -> None:
        expected = commit["files"]["summary"]["sha256"]
        if os.path.exists(self.summary_path) and _sha256_file(self.summary_path) == expected:
            return
        if os.path.exists(self.summary_path):
            directory = self._quarantine_dir("invalid_summary")
            self._move_preserving(self.summary_path, directory)
        _atomic_bytes(self.summary_path, self._summary_bytes(commit["summary_row"]))
        if _sha256_file(self.summary_path) != expected:
            raise ValueError("reconstructed Hessian summary hash mismatch")

    def _validate_event(self, commit: Mapping[str, Any]) -> None:
        event = commit["files"]["event"]
        path = event["path"]
        if not os.path.exists(path) or _sha256_file(path) != event["sha256"]:
            raise ValueError(f"missing/corrupt committed Hessian event: {path}")
        with Trajectory(path, "r") as traj:
            if len(traj) != 1:
                raise ValueError(f"committed Hessian event must contain one frame: {path}")
            info = getattr(traj[0], "info", {})
            if str(info.get("hessian_task_identity", "")) != self.task_identity:
                raise ValueError("committed Hessian event task identity mismatch")
            if str(info.get("hessian_artifact_contract", "")) != ARTIFACT_CONTRACT:
                raise ValueError("committed Hessian event contract mismatch")

    def _invalidate_commit_for_recompute(self, commit: Mapping[str, Any], reason: str) -> None:
        directory = self._quarantine_dir(reason)
        paths = [self.commit_path, self.summary_path]
        event = commit.get("files", {}).get("event", {}).get("path")
        if event:
            paths.append(event)
        hessian = commit.get("files", {}).get("hessian")
        if isinstance(hessian, dict) and hessian.get("path"):
            paths.append(hessian["path"])
        moved = []
        for path in paths:
            moved_path = self._move_preserving(path, directory)
            if moved_path:
                moved.append(moved_path)
        _atomic_json(os.path.join(directory, "invalidated.json"), {
            "contract": ARTIFACT_CONTRACT,
            "reason": reason,
            "job_id": _record_json(self.job_id),
            "task_identity": self.task_identity,
            "preserved_files": moved,
        })
        old_rank = commit.get("rank")
        if old_rank is not None:
            try:
                self.rebuild_indexes(commit_rank=old_rank)
            except Exception:
                # The invalid commit is already preserved. A later successful
                # publication/reuse will rebuild indexes again.
                pass

    def _load_commit(self) -> dict[str, Any] | None:
        if not os.path.exists(self.commit_path):
            return None
        try:
            commit = _read_json(self.commit_path)
        except Exception as exc:
            directory = self._quarantine_dir("invalid_commit_json")
            self._move_preserving(self.commit_path, directory)
            _atomic_json(os.path.join(directory, "error.json"), {"error": str(exc)})
            return None
        if commit.get("contract") != ARTIFACT_CONTRACT or commit.get("state") != "committed":
            self._invalidate_commit_for_recompute(commit, "invalid_commit_contract")
            return None
        committed_identity = str(commit.get("task_identity", ""))
        if committed_identity != self.task_identity:
            exc = HessianArtifactIdentityMismatch(
                "Hessian job index already has a validated commit for a different task identity: "
                f"job_id={self.job_id} committed={committed_identity} requested={self.task_identity}"
            )
            self.record_failure(exc, kind="identity_mismatch")
            raise exc
        return commit

    def _finish_pending(self) -> dict[str, Any] | None:
        if not os.path.exists(self.pending_path):
            return None
        try:
            pending = _read_json(self.pending_path)
        except Exception as exc:
            directory = self._quarantine_dir("invalid_pending_json")
            self._move_preserving(self.pending_path, directory)
            _atomic_json(os.path.join(directory, "error.json"), {"error": str(exc)})
            return None
        pending_identity = str(pending.get("task_identity", ""))
        if pending_identity != self.task_identity:
            exc = HessianArtifactIdentityMismatch(
                "Hessian job index has an interrupted publication for a different task identity: "
                f"job_id={self.job_id} pending={pending_identity} requested={self.task_identity}"
            )
            self.record_failure(exc, kind="pending_identity_mismatch")
            raise exc
        if pending.get("contract") != ARTIFACT_CONTRACT or pending.get("state") != "prepared":
            directory = self._quarantine_dir("invalid_pending_contract")
            self._move_preserving(self.pending_path, directory)
            return None
        try:
            for key in ("event", "summary", "hessian"):
                spec = pending.get("files", {}).get(key)
                if not isinstance(spec, dict):
                    continue
                final = spec["path"]
                stage = spec.get("stage", "")
                expected = spec["sha256"]
                if os.path.exists(final) and _sha256_file(final) == expected:
                    if stage and os.path.exists(stage):
                        os.unlink(stage)
                    continue
                if os.path.exists(final):
                    directory = self._quarantine_dir(f"pending_{key}_conflict")
                    self._move_preserving(final, directory)
                if not stage or not os.path.exists(stage) or _sha256_file(stage) != expected:
                    raise ValueError(f"pending Hessian publication cannot recover {key}")
                os.replace(stage, final)
                _fsync_dir(final)
                _publication_boundary(f"recover_{key}_published")
            commit = dict(pending)
            commit["state"] = "committed"
            for spec in commit.get("files", {}).values():
                if isinstance(spec, dict):
                    spec.pop("stage", None)
            _atomic_json(self.commit_path, commit)
            _publication_boundary("recover_commit_published")
            os.unlink(self.pending_path)
            _fsync_dir(self.pending_path)
            return commit
        except Exception:
            # Keep a valid prepared transaction in place for a later retry. If
            # some required stage was lost, preserve what remains and recompute.
            recoverable = True
            for spec in pending.get("files", {}).values():
                if not isinstance(spec, dict):
                    continue
                final_ok = os.path.exists(spec["path"]) and _sha256_file(spec["path"]) == spec["sha256"]
                stage = spec.get("stage", "")
                stage_ok = bool(stage and os.path.exists(stage) and _sha256_file(stage) == spec["sha256"])
                if not (final_ok or stage_ok):
                    recoverable = False
                    break
            if recoverable:
                raise
            directory = self._quarantine_dir("unrecoverable_pending")
            self._move_preserving(self.pending_path, directory)
            for spec in pending.get("files", {}).values():
                if isinstance(spec, dict):
                    for path in (spec.get("stage"), spec.get("path")):
                        if path and os.path.exists(path):
                            self._move_preserving(path, directory)
            return None

    def recover_or_reuse(self) -> bool:
        """Recover interrupted publication and return True for reusable completion."""
        commit = self._load_commit()
        if commit is not None:
            if os.path.exists(self.pending_path):
                # Commit publication won the crash boundary; discard only the
                # transaction's now-redundant staging files.
                try:
                    pending = _read_json(self.pending_path)
                    if str(pending.get("task_identity", "")) != self.task_identity:
                        raise HessianArtifactIdentityMismatch(
                            "committed and pending Hessian identities disagree"
                        )
                    for spec in pending.get("files", {}).values():
                        if isinstance(spec, dict):
                            stage = spec.get("stage")
                            if stage and os.path.exists(stage):
                                os.unlink(stage)
                    os.unlink(self.pending_path)
                except HessianArtifactIdentityMismatch:
                    raise
                except Exception:
                    pass
        else:
            try:
                recovered = self._finish_pending()
            except HessianArtifactIdentityMismatch:
                raise
            except Exception as exc:
                raise HessianArtifactPublicationError(
                    f"Hessian artifact recovery interrupted for job {self.job_id}: {exc}"
                ) from exc
            commit = recovered if recovered is not None else self._load_commit()
        if commit is None:
            return False

        try:
            self._validate_event(commit)
            self._repair_summary(commit)
            hessian_spec = commit.get("files", {}).get("hessian")
            if isinstance(hessian_spec, dict):
                self._validate_npz(hessian_spec["path"], hessian_spec["sha256"])
            elif self.store_hessian:
                # Same scientific task, but the earlier completion explicitly
                # omitted the full matrix. Recompute to satisfy the stronger
                # output contract; do not call the old completion reusable.
                self._invalidate_commit_for_recompute(commit, "store_hessian_upgrade")
                return False
        except Exception as exc:
            self.record_failure(exc, kind="invalid_committed_artifact")
            self._invalidate_commit_for_recompute(commit, "invalid_committed_artifact")
            return False

        try:
            self.rebuild_indexes(commit_rank=commit.get("rank"))
        except Exception as exc:
            raise HessianArtifactPublicationError(
                f"Hessian artifact index recovery failed for job {self.job_id}: {exc}"
            ) from exc
        return True

    def _stage_paths(self) -> dict[str, str]:
        token = f"{os.getpid()}.{uuid.uuid4().hex}"
        paths = {
            "event": os.path.join(os.path.dirname(self.event_path), f".{os.path.basename(self.event_path)}.stage.{token}.traj"),
            "summary": os.path.join(os.path.dirname(self.summary_path), f".{os.path.basename(self.summary_path)}.stage.{token}"),
        }
        if self.store_hessian:
            paths["hessian"] = os.path.join(os.path.dirname(self.hessian_path), f".{os.path.basename(self.hessian_path)}.stage.{token}.npz")
        return paths

    def publish(
        self,
        *,
        summary_row: Mapping[str, Any],
        output_atoms,
        npz_payload: Mapping[str, Any] | None,
        status: str,
    ) -> None:
        """Publish one computed Hessian result with recoverable transaction state."""
        self.publication_started = True
        self._ensure_legacy_index_snapshots()
        self._preserve_unverified_fixed_outputs()
        os.makedirs(os.path.dirname(self.event_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        if self.store_hessian:
            os.makedirs(os.path.dirname(self.hessian_path), exist_ok=True)
        stages = self._stage_paths()
        try:
            _write_event_file(stages["event"], output_atoms)
            _write_summary_file(stages["summary"], summary_row, self.summary_fields)
            if self.store_hessian:
                if npz_payload is None:
                    raise ValueError("store_hessian=True requires NPZ payload")
                _write_npz_file(stages["hessian"], npz_payload)
            files: dict[str, Any] = {
                "event": {"path": self.event_path, "stage": stages["event"], "sha256": _sha256_file(stages["event"])},
                "summary": {"path": self.summary_path, "stage": stages["summary"], "sha256": _sha256_file(stages["summary"])},
            }
            if self.store_hessian:
                files["hessian"] = {"path": self.hessian_path, "stage": stages["hessian"], "sha256": _sha256_file(stages["hessian"]) }
            pending = {
                "contract": ARTIFACT_CONTRACT,
                "state": "prepared",
                "job_id": _record_json(self.job_id),
                "rank": _record_json(self.rank),
                "task_identity": self.task_identity,
                "task_descriptor": self.task_descriptor,
                "store_hessian": self.store_hessian,
                "status": status,
                "summary_row": _record_json(dict(summary_row)),
                "files": files,
            }
            _atomic_json(self.pending_path, pending)
            _publication_boundary("pending_written")

            for key in ("event", "summary", "hessian"):
                spec = files.get(key)
                if not isinstance(spec, dict):
                    continue
                final = spec["path"]
                if os.path.exists(final):
                    # A fixed legacy file can remain if it appeared after the
                    # initial preservation check. Preserve it before replacement.
                    if _sha256_file(final) != spec["sha256"]:
                        directory = self._quarantine_dir(f"prepublish_{key}")
                        self._move_preserving(final, directory)
                    else:
                        if os.path.exists(spec["stage"]):
                            os.unlink(spec["stage"])
                        _publication_boundary(f"{key}_published")
                        continue
                os.replace(spec["stage"], final)
                _fsync_dir(final)
                _publication_boundary(f"{key}_published")

            commit = dict(pending)
            commit["state"] = "committed"
            for spec in commit["files"].values():
                if isinstance(spec, dict):
                    spec.pop("stage", None)
            _atomic_json(self.commit_path, commit)
            _publication_boundary("commit_published")
            os.unlink(self.pending_path)
            _fsync_dir(self.pending_path)
            self.rebuild_indexes(commit_rank=self.rank)
            _publication_boundary("indexes_published")
        except Exception as exc:
            raise HessianArtifactPublicationError(
                f"Hessian artifact publication interrupted for job {self.job_id}: {exc}"
            ) from exc
        finally:
            self.publication_started = False

    def _iter_commits_for_rank(self, rank: Any) -> list[dict[str, Any]]:
        commits: list[dict[str, Any]] = []
        directory = Path(f"{self.root}/commits")
        if not directory.exists():
            return commits
        for path in sorted(directory.glob("hessian_*.json")):
            try:
                commit = _read_json(str(path))
            except Exception:
                continue
            if commit.get("contract") != ARTIFACT_CONTRACT or commit.get("state") != "committed":
                continue
            if str(commit.get("rank")) != str(rank):
                continue
            event = commit.get("files", {}).get("event")
            if not isinstance(event, dict):
                continue
            if not os.path.exists(event.get("path", "")):
                continue
            if _sha256_file(event["path"]) != event.get("sha256"):
                continue
            commits.append(commit)
        commits.sort(key=lambda item: (str(item.get("job_id")), str(item.get("task_identity"))))
        return commits

    def rebuild_indexes(self, *, commit_rank: Any) -> None:
        """Reconstruct shared legacy status/trajectory indexes from commit events."""
        rank = commit_rank
        rank_token = _safe_token(rank)
        status_path = f"Hessian_status_csvs/status_rank_{rank_token}.csv"
        traj_path = f"Hessian_trajes/collected_hessian_rank_{rank_token}.traj"
        lock_path = f"{self.root}/locks/index_rank_{rank_token}.lock"
        legacy_status = f"{self.root}/legacy_indexes/status_rank_{rank_token}.csv"
        legacy_traj = f"{self.root}/legacy_indexes/collected_hessian_rank_{rank_token}.traj"
        with _exclusive_lock(lock_path):
            commits = self._iter_commits_for_rank(rank)
            legacy_status_bytes = b""
            if os.path.exists(legacy_status):
                legacy_status_bytes = Path(legacy_status).read_bytes()
                if legacy_status_bytes and not legacy_status_bytes.endswith(b"\n"):
                    legacy_status_bytes += b"\n"
            managed = "".join(
                f'{commit["job_id"]},{commit["rank"]},"{commit["status"]}"\n'
                for commit in commits
            ).encode()
            _atomic_bytes(status_path, legacy_status_bytes + managed)

            os.makedirs(os.path.dirname(traj_path), exist_ok=True)
            temp = os.path.join(
                os.path.dirname(traj_path),
                f".{os.path.basename(traj_path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}.traj",
            )
            try:
                with Trajectory(temp, "w") as writer:
                    if os.path.exists(legacy_traj):
                        with Trajectory(legacy_traj, "r") as legacy_reader:
                            for frame in legacy_reader:
                                writer.write(frame)
                    for commit in commits:
                        event_path = commit["files"]["event"]["path"]
                        with Trajectory(event_path, "r") as event_reader:
                            if len(event_reader) != 1:
                                raise ValueError(
                                    f"Hessian event {event_path} must contain exactly one frame"
                                )
                            writer.write(event_reader[0])
                with open(temp, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp, traj_path)
                _fsync_dir(traj_path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
