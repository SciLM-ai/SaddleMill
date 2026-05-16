#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH -t 02:00:00
#SBATCH -o slurm_%j.out
#SBATCH -A YOUR_ALLOCATION
#SBATCH -J sp_lmdb_mp20bat

pwd; hostname -f; date

export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$USER
srun -N $SLURM_NNODES -n $SLURM_NNODES mkdir -p $CUDA_MPS_PIPE_DIRECTORY
srun -N $SLURM_NNODES -n $SLURM_NNODES mkdir -p $CUDA_MPS_LOG_DIRECTORY

srun -N $SLURM_NNODES -n $SLURM_NNODES nvidia-cuda-mps-control -d

srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m saddlemill

srun -N $SLURM_NNODES -n $SLURM_NNODES bash -c "echo quit | nvidia-cuda-mps-control"

date
