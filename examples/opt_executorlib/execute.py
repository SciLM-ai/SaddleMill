from executorlib import FluxJobExecutor
from flux import Flux, resource
import concurrent.futures
from tsearch.opt import main as optm
import os


def check_and_print_status(futures, total):
    done, futures = concurrent.futures.wait(futures, timeout=0.1)
    if len(done)!=0:
        print(f"{len(futures)} REMAINING  ---  {total-len(futures)} FINISHED  ---  {total} TOTAL")
    return futures


handle = Flux()
rs = resource.status.ResourceStatusRPC(handle).get()
rl = resource.list.resource_list(handle).get()
all_ncores = rl.all.ncores
all_ngpus = rl.all.ngpus
print(all_ncores,all_ngpus)

nstructures = 16

# with FluxJobExecutor(flux_log_files=True, max_workers=all_ngpus, block_allocation=True, resource_dict={"cores": 1, "gpus_per_core": 1, "threads_per_core":1, "num_nodes": 1}) as mps_exe:
with FluxJobExecutor(flux_log_files=True, max_workers=all_ngpus*4, block_allocation=True, resource_dict={"cores": 1, "gpus_per_core": 0, "threads_per_core":17, "num_nodes": 1}) as exe:  # pmi_mode="pmix"
   
    # command = "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER\n"\
    #             "export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER\n"\
    #             "mkdir -p $CUDA_MPS_PIPE_DIRECTORY\n"\
    #             "mkdir -p $CUDA_MPS_LOG_DIRECTORY\n"\
    #             "nvidia-cuda-mps-control -d"
    # # Starting daemons:
    # for i in range(all_ngpus):
    #     mps_exe.submit(os.system,command)
    
    futures = []
    for i in range(nstructures):
        futures.append(exe.submit(optm, i))

    while len(futures):
        futures = check_and_print_status(futures, nstructures)


