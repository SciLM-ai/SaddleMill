#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=224
#SBATCH -q debug
#SBATCH -t 02:00:00
#SBATCH -J sp_vtstdimer_mp20bat
#SBATCH -o slurm_%j.log

pwd; hostname -f; date

ml list;

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate saddlemill

export VASP_PP_PATH=/home/graeme/vasp
#export PATH=/opt/ohpc/pub/libs/intel/openmpi5/vasp/6.6.0/bin:$PATH  # should already be loaded as module
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m saddlemill

date
