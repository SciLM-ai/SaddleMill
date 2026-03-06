#!/bin/sh
##SBATCH -N 64
##SBATCH -n 64
##SBATCH -p gh
##SBATCH -t 48:00:00
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -p gh-dev
#SBATCH -t 2:00:00
#SBATCH -o ll_out
#SBATCH -A CHE25011

export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
srun -N $SLURM_NNODES -n $SLURM_NNODES mkdir -p $CUDA_MPS_PIPE_DIRECTORY
srun -N $SLURM_NNODES -n $SLURM_NNODES mkdir -p $CUDA_MPS_LOG_DIRECTORY

srun -N $SLURM_NNODES -n $SLURM_NNODES nvidia-cuda-mps-control -d

srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m tsearch

srun -N $SLURM_NNODES -n $SLURM_NNODES bash -c "echo quit | nvidia-cuda-mps-control"