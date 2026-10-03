"""Internal worker-resource helpers for shared-GPU executorlib workers."""

import fcntl
import hashlib
import json
import operator
import os
import tempfile


STATE_VERSION = 1
DEFAULT_GPU_STATE_BASE = "/tmp/sm_gpu"


def _as_nonnegative_int(value, label, *, allow_zero=True):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got boolean {value!r}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}, got {value}")
    return int(value)


def _require_scope_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} is required for GPU worker-slot scoping")
    return value.strip()


def worker_scope_identity(hostname, environ=None):
    """Build the explicit allocation/broker/host identity for node-local state."""
    environ = os.environ if environ is None else environ
    hostname = _require_scope_string(hostname, "hostname")
    broker_id = _require_scope_string(environ.get("FLUX_URI"), "FLUX_URI")
    allocation_id = (environ.get("SLURM_JOB_ID") or
                     environ.get("SLURM_JOBID") or
                     environ.get("PBS_JOBID") or
                     "broker-only")
    return {
        "version": STATE_VERSION,
        "uid": os.getuid(),
        "allocation_id": str(allocation_id),
        "broker_id": broker_id,
        "hostname": hostname,
    }


def scope_dir(scope, base=DEFAULT_GPU_STATE_BASE):
    payload = json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return os.path.join(base, f"uid_{scope['uid']}", digest)


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".state_", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _read_state(path):
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise RuntimeError(f"Unable to read GPU worker-slot state at {path}: {exc}") from exc
    if not raw.strip():
        raise RuntimeError(f"Empty GPU worker-slot state at {path}")
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Corrupted GPU worker-slot state at {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"Corrupted GPU worker-slot state at {path}: expected JSON object")
    return state


def _new_state(scope):
    return {
        "version": STATE_VERSION,
        "scope": dict(scope),
        "next_claim": 0,
        "workers": {},
    }


def _validate_state(state, scope, path):
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(f"Unsupported GPU worker-slot state version at {path}: {state!r}")
    if state.get("scope") != scope:
        raise RuntimeError(
            f"GPU worker-slot scope mismatch at {path}: stored={state.get('scope')!r}, "
            f"expected={scope!r}")
    try:
        next_claim = _as_nonnegative_int(state.get("next_claim"), "next_claim")
    except ValueError as exc:
        raise RuntimeError(f"Corrupted GPU worker-slot state at {path}: {exc}") from exc
    workers = state.get("workers")
    if not isinstance(workers, dict):
        raise RuntimeError(f"Corrupted GPU worker-slot state at {path}: workers is not an object")
    if next_claim < len(workers):
        raise RuntimeError(
            f"Corrupted GPU worker-slot state at {path}: next_claim={next_claim} "
            f"but {len(workers)} worker assignments exist")
    return next_claim, workers


def claim_local_gpu_slot(worker_id, ngpus, *, scope, base=DEFAULT_GPU_STATE_BASE):
    """Claim a stable GPU slot inside one allocation/broker/host scope."""
    worker_id = _as_nonnegative_int(worker_id, "executorlib_worker_id")
    ngpus = _as_nonnegative_int(ngpus, "local GPU count", allow_zero=False)
    scope = dict(scope)
    directory = scope_dir(scope, base=base)
    os.makedirs(directory, mode=0o700, exist_ok=True)

    lock_fd = os.open(os.path.join(directory, "state.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        state_path = os.path.join(directory, "state.json")
        state = _read_state(state_path) if os.path.exists(state_path) else _new_state(scope)
        next_claim, workers = _validate_state(state, scope, state_path)

        key = str(worker_id)
        if key in workers:
            assignment = workers[key]
            if not isinstance(assignment, dict):
                raise RuntimeError(
                    f"Corrupted GPU worker-slot state at {state_path}: "
                    f"worker {worker_id} assignment is not an object")
            try:
                stored_ngpus = _as_nonnegative_int(
                    assignment.get("ngpus"), "stored ngpus", allow_zero=False)
                physical_gpu = _as_nonnegative_int(
                    assignment.get("physical_gpu"), "stored physical_gpu")
            except ValueError as exc:
                raise RuntimeError(
                    f"Corrupted GPU worker-slot state at {state_path}: {exc}") from exc
            if stored_ngpus != ngpus:
                raise RuntimeError(
                    f"GPU resource count changed inside one worker scope at {state_path}: "
                    f"stored {stored_ngpus}, current {ngpus}")
            if physical_gpu >= ngpus:
                raise RuntimeError(
                    f"Corrupted GPU worker-slot state at {state_path}: physical_gpu="
                    f"{physical_gpu} outside 0..{ngpus - 1}")
            return physical_gpu

        physical_gpu = next_claim % ngpus
        state["next_claim"] = next_claim + 1
        workers[key] = {"ngpus": ngpus, "physical_gpu": physical_gpu}
        _atomic_write_json(state_path, state)
        return physical_gpu
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _host_aliases(hostname):
    hostname = str(hostname).strip().lower()
    if not hostname:
        return set()
    return {hostname, hostname.split(".", 1)[0]}


def _flux_node_gpu_inventory(rset):
    try:
        nnodes = _as_nonnegative_int(rset.nnodes, "Flux node count", allow_zero=False)
    except (AttributeError, ValueError) as exc:
        raise RuntimeError(f"Invalid Flux resource set: {exc}") from exc

    inventory = []
    for rank in range(nnodes):
        try:
            node_rset = rset.copy_ranks(str(rank))
            node_name = str(node_rset.nodelist).strip()
            ngpus = _as_nonnegative_int(node_rset.ngpus, f"GPU count for Flux rank {rank}")
        except (AttributeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Flux resource data for rank {rank}: {exc}") from exc
        if not node_name:
            raise RuntimeError(f"Flux rank {rank} has an empty node name")
        inventory.append((rank, node_name, ngpus))
    return inventory


def resolve_local_gpu_count(rset, hostname):
    """Validate shared-MPS resources and return the local per-node GPU count."""
    inventory = _flux_node_gpu_inventory(rset)
    counts = {ngpus for _, _, ngpus in inventory}
    summary = ", ".join(f"rank {rank}:{node}={ngpus}"
                        for rank, node, ngpus in inventory)

    if len(counts) != 1:
        raise RuntimeError(
            "Shared-MPS worker placement requires homogeneous per-node GPU counts; "
            f"Flux reported heterogeneous resources: {summary}")
    ngpus = next(iter(counts))
    if ngpus <= 0:
        raise RuntimeError(
            f"Shared-MPS GPU job has zero GPUs per node according to Flux: {summary}")

    local_aliases = _host_aliases(hostname)
    matching = [entry for entry in inventory
                if local_aliases & _host_aliases(entry[1])]
    if len(matching) > 1:
        raise RuntimeError(
            f"Hostname {hostname!r} matched multiple Flux ranks: {matching!r}")
    # If Flux's nodelist spelling cannot be matched, homogeneity makes the count
    # unambiguous; heterogeneous allocations already failed above.
    return matching[0][2] if matching else ngpus
