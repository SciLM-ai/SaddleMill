#!/bin/sh
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -p gh
#SBATCH -t 48:00:00
#SBATCH -o slurm_%j.out
#SBATCH -A YOUR_ALLOCATION
#SBATCH -J reconverge_oc22
# The three stages in one allocation: Sella reconvergence -> exact Hessian -> FIRE double minimization of the index-1 saddles.

pwd; hostname -f; date
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
srun -N $SLURM_NNODES -n $SLURM_NNODES mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
srun -N $SLURM_NNODES -n $SLURM_NNODES nvidia-cuda-mps-control -d

(cd 1_sella && srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m saddlemill)
(cd 2_hessian && srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m saddlemill)
python select_index1.py 2_hessian/SinglePoint_trajes 2_hessian/index1
(cd 3_doubleminimization && srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m saddlemill)

srun -N $SLURM_NNODES -n $SLURM_NNODES bash -c "echo quit | nvidia-cuda-mps-control"
date
