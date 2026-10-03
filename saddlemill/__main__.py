import os
import concurrent.futures
from ase.io import Trajectory
from itertools import groupby
from contextlib import nullcontext
from saddlemill.init_function import init_function
from saddlemill.tools import (save_ordered_traj_names, read_ordered_traj_names,
                            clean_up_files, load_and_sanitize, passes_input_filter,
                            extract_previous_results)
from saddlemill.config import (load_config, load_method, get_trajes_and_indices,
                            create_results_directories, get_remaining_trajes,
                            get_flux_resources, archive_and_clean_csvs,
                            archive_and_clean_outputs, build_redo_info)


def check_and_print_status(futures, total, task_counts=None, failures=None):
    """Collect completed Futures without aborting executor cleanup.

    Launcher/executor exceptions are recorded separately from scientific method
    outcomes.  A Future that returns normally is a successful launcher task even
    if the method itself recorded a scientific ``not_converged`` status.
    """
    done, futures = concurrent.futures.wait(futures, timeout=0.1)
    for f in done:
        try:
            f.result()
        except Exception as e:
            print(f"[worker task died] {e}", flush=True)
            if task_counts is not None:
                task_counts["failed"] += 1
            if failures is not None:
                failures.append(f"worker task: {e}")
        else:
            if task_counts is not None:
                task_counts["successful"] += 1
    if done:
        completed = total - len(futures)
        if task_counts is None:
            print(f"{len(futures)} REMAINING --- {completed} COMPLETED --- {total} TOTAL")
        else:
            print(
                f"{len(futures)} REMAINING --- {completed} COMPLETED --- {total} TOTAL; "
                f"{task_counts['successful']} SUCCESSFUL --- "
                f"{task_counts['failed']} FAILED",
                flush=True,
            )
    return futures

def main():
    import faulthandler
    import signal

    master_log = open("frozen_trace_main.log", "w")
    faulthandler.register(signal.SIGUSR1, file=master_log)

    config_dict = load_config("config.ini")
    print(config_dict,"\n")

    method = load_method(config_dict)
    trajes_and_idxs = get_trajes_and_indices(config_dict)
    can_resume = os.path.exists('traj_files_ordered.json')
    previous_results = {}
    redo_info = {}

    from saddlemill.config import _expected_dimer_entries
    method_name = config_dict["Main"]["method"]
    chunk_size = config_dict["Main"]["attempt_chunk_size"]
    # Attempt chunking is a Dimer orchestration feature.  Other methods may use
    # entries_to_run for their own subunits (notably DoubleMin sides), but those
    # sets must remain together in one task.
    n_expected = _expected_dimer_entries(config_dict) if method_name == "Dimer" else None

    if can_resume:
        trajes_and_idxs_old = read_ordered_traj_names()
        if trajes_and_idxs != trajes_and_idxs_old:
            raise ValueError("Provided dirpath creates a different trajes_and_idxs. I can't resume.")
        job_IDs, trajes_and_idxs = get_remaining_trajes(trajes_and_idxs, config_dict)

        # Build per-job redo info (which subunits to redo)
        redo_info = build_redo_info(job_IDs, config_dict)

        # Extract previous output BEFORE archiving (output files still intact).
        # Always extract when redoing subunits: NEB needs sub-band endpoints,
        # DoubleMinimization needs the kept side for reaction check,
        # Dimer needs previous structures only when continue_from_result=True.
        if redo_info:
            print(f"Extracting previous results for {len(redo_info)} jobs...", flush=True)
            previous_results = extract_previous_results(list(redo_info.keys()), config_dict, redo_info)
            print(f"  Extracted {len(previous_results)} of {len(redo_info)} results.", flush=True)

        from saddlemill.config import _normalize_run_jobs
        categories_to_clean = _normalize_run_jobs(config_dict["Main"]["run_jobs"])
        cleaned = archive_and_clean_csvs(config_dict, job_IDs, categories_to_clean)
        archive_and_clean_outputs(config_dict, cleaned)
        clean_up_files(config_dict)
    else:
        job_IDs = list(range(len(trajes_and_idxs)))
        save_ordered_traj_names(trajes_and_idxs)
        create_results_directories(config_dict)

    if config_dict["Main"]["executorlib"]:
        from executorlib import FluxJobExecutor

        max_workers, cores, gpus_per_core, threads_per_core = get_flux_resources(config_dict)

        executor = FluxJobExecutor(
            flux_log_files = True,
            max_workers = max_workers,
            block_allocation = True,
            init_function = init_function,
            restart_limit = config_dict["Main"]["restart_limit"],
            resource_dict = {
                "cores": cores,
                "gpus_per_core": gpus_per_core,
                "threads_per_core": threads_per_core,
                "num_nodes": 1,
                "error_log_file": "error",
                "cwd": os.getcwd(),
            }
        )
        # 'exe' will be the executor instance
        get_submitter = lambda exe: exe.submit
    else:
        # Serial Mode: Use empty context and a dummy submitter that runs immediately
        init_data = init_function()
        # Executorlib injects its worker ID when it calls the task.  Serial
        # dispatch has no executor to do that, so use the sole serial worker's
        # stable identity instead of allowing status rows named/ranked None.
        init_data["executorlib_worker_id"] = 0
        executor = nullcontext()
        get_submitter = lambda _: lambda fn, *args, **kwargs: fn(*args, **init_data, **kwargs)


    # 2. Unified Execution Loop
    input_format = config_dict["Main"].get("input_format", "traj")
    if input_format == "lmdb":
        # SinglePoint + LMDB: read rows via ase.db (fairchem.core.datasets
        # registers the aselmdb backend).
        import fairchem.core.datasets  # noqa: F401
        from ase.db import connect as _lmdb_connect

        def open_src(path):
            return _lmdb_connect(path, type='aselmdb', readonly=True)

        def load_item(src, i, j):
            rows = [src.get(rid) for rid in range(i, j)]
            atoms_list = []
            extras = []
            for r in rows:
                atoms = r.toatoms()
                # row.toatoms() doesn't populate atoms.info from row.data,
                # so we lift it across explicitly. This keeps existing
                # orig_info / status conventions readable by passes_input_filter.
                atoms.info = dict(r.data.get("info", {}))
                atoms_list.append(atoms)
                extras.append({"kvp": dict(r.key_value_pairs),
                               "row_data": dict(r.data)})
            images = atoms_list if len(atoms_list) > 1 else atoms_list[0]
            return images, {"extras": extras}

        def close_src(_):
            pass
    else:
        def open_src(path):
            return Trajectory(path, 'r')

        def load_item(src, i, j):
            return load_and_sanitize(src, i, j), {}

        def close_src(src):
            src.close()

    task_counts = {
        "scanned": 0,
        "filtered": 0,
        "submitted": 0,
        "successful": 0,
        "failed": 0,
    }
    failures = []

    with executor as exe:
        submitter = get_submitter(exe)
        futures = []
        idx = 0

        for src_path, group in groupby(trajes_and_idxs, key=lambda x: x[0]):
            src = open_src(src_path)
            try:
                for _, i, j in group:
                    job_id = job_IDs[idx]
                    task_counts["scanned"] += 1
                    images, extra = load_item(src, i, j)
                    if not passes_input_filter(images, config_dict):
                        task_counts["filtered"] += 1
                        idx += 1
                        continue
                    try:
                        entries = redo_info.get(job_id)
                        if method_name != "Dimer" or chunk_size <= 0:
                            chunks = [entries]
                        elif entries is None:
                            # Fresh Dimer structure: chunk the full generated attempt range.
                            if n_expected is None:
                                chunks = [None]
                            else:
                                rng = list(range(n_expected))
                                chunks = [
                                    set(rng[k:k + chunk_size])
                                    for k in range(0, n_expected, chunk_size)
                                ]
                        else:
                            # Resumed Dimer: preserve the selected attempt IDs while
                            # splitting only this method's attempt-level work.
                            entries = sorted(entries)
                            chunks = [
                                set(entries[k:k + chunk_size])
                                for k in range(0, len(entries), chunk_size)
                            ] or [set()]

                        for ch in chunks:
                            # Count each launcher dispatch attempt before calling the
                            # submitter so serial and submit-time failures are visible.
                            task_counts["submitted"] += 1
                            try:
                                f = submitter(
                                    method, job_id, config_dict, images,
                                    continuation_data=previous_results.get(job_id),
                                    entries_to_run=ch,
                                    **extra,
                                )
                            except Exception as e:
                                task_counts["failed"] += 1
                                failures.append(f"job {job_id}: {e}")
                                print(
                                    f"CRITICAL ERROR on job {job_id} ({src_path}): {e}",
                                    flush=True,
                                )
                                continue

                            if config_dict["Main"]["executorlib"]:
                                futures.append(f)
                            else:
                                task_counts["successful"] += 1
                    finally:
                        idx += 1
            finally:
                close_src(src)

        if config_dict["Main"]["executorlib"]:
            while futures:
                futures = check_and_print_status(
                    futures, task_counts["submitted"], task_counts, failures
                )

    print(
        f"Scanned {task_counts['scanned']} input frame(s); "
        f"filtered {task_counts['filtered']}; "
        f"submitted {task_counts['submitted']} task(s); "
        f"successful {task_counts['successful']}; "
        f"failed {task_counts['failed']}.",
        flush=True,
    )
    if failures:
        print(
            f"Launcher detected {len(failures)} execution/submission failure(s); "
            "returning nonzero status after executor cleanup.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
