#!/bin/bash

#SBATCH -p development
#SBATCH -N 64
#SBATCH --ntasks-per-node=128
#SBATCH -t 24:00:00
#SBATCH -J oc20nebVASP
#SBATCH -A CHE23004


ml unload xalt python3
ml load impi cuda/12.8

# # 1. Create a local scratch directory on the compute node
# export TDIR=/tmp/ilgar/$SLURM_JOB_ID
# mkdir -p $TDIR

# # 2. Copy inputs to local scratch
# cp -r ./* $TDIR/
# cd $TDIR

# 3. Run Python here (on local disk)
srun -N $SLURM_NNODES -n $SLURM_NNODES --mpi=pmi2 flux start python -u -m tsearch
