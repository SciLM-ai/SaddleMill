import os
import socket
import traceback
import faulthandler
import signal
from saddlemill.config import load_config, load_calculator, load_optimizer
from saddlemill.worker_resources import (
    claim_local_gpu_slot as _claim_local_gpu_slot,
    resolve_local_gpu_count as _resolve_local_gpu_count,
    worker_scope_identity as _worker_scope_identity,
)


_GPU_STATE_BASE = "/tmp/sm_gpu"
_MPS_PIPE_ROOT = "/tmp"


def init_function(executorlib_worker_id=None):
    print(f"[init-enter] w={executorlib_worker_id}", flush=True)   # FIRST line
    try:
        log_file = open(f"frozen_trace_{executorlib_worker_id}.log", "w")
        faulthandler.register(signal.SIGUSR1, file=log_file)
        config_dict = load_config("config.ini")

        is_gpu_job = (config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive")
                      and config_dict[config_dict["Main"]["Calculator"]].get("device") == "cuda")

        if config_dict["Main"]["executorlib"] is True and config_dict["Main"]["jobs_per_gpu"] != 1:
            if is_gpu_job:
                from flux import Flux, resource
                handle = Flux()
                rset = resource.list.resource_list(handle).get().all
                hostname = socket.gethostname()
                ngpus = _resolve_local_gpu_count(rset, hostname)
                scope = _worker_scope_identity(hostname)
                physical_gpu = _claim_local_gpu_slot(
                    executorlib_worker_id,
                    ngpus,
                    scope=scope,
                    base=_GPU_STATE_BASE,
                )

                mps_pipe = os.path.join(_MPS_PIPE_ROOT, f"mps_{physical_gpu}")

                # Pipe dir selects the physical GPU; client always sees device "0".
                # Set BOTH before any CUDA/torch init, or the driver ignores them.
                control = os.path.join(mps_pipe, "control")
                if not os.path.exists(control):
                    # Do NOT silently fall back to plain GPU 0 — that's the collapse bug.
                    raise RuntimeError(
                        f"Worker {executorlib_worker_id}: MPS control socket missing at "
                        f"{control}; refusing to fall back to GPU 0. Check run_phase MPS startup.")
                os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

                import torch  # only AFTER env is set
                print(f"[assign] w={executorlib_worker_id} "
                      f"cuda_already_init={torch.cuda.is_initialized()} "
                      f"physical_gpu={physical_gpu} pipe={mps_pipe} "
                      f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

        hostname = socket.gethostname()
        cpus = sorted(os.sched_getaffinity(0))
        print(f"Worker {executorlib_worker_id} started on node {hostname}", flush=True)
        print(f"  CPUs: {cpus}", flush=True)
        if is_gpu_job:
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
                  f"  MPS: {os.environ.get('CUDA_MPS_PIPE_DIRECTORY', 'off')}", flush=True)

        calc = load_calculator(config_dict)
        if config_dict["Main"]["Calculator"] not in ("Vasp", "VaspInteractive"):
            calc = calc(**config_dict[config_dict["Main"]["Calculator"]])
        Optimizer = load_optimizer(config_dict)

        return {"calc": calc, "Optimizer": Optimizer, "consecutive_errors": [0]}

    except Exception as e:
        print(f"Worker {executorlib_worker_id} FAILED during init_function: {e}", flush=True)
        print(f"\nTraceback details:\n{traceback.format_exc()}", flush=True)
        raise
