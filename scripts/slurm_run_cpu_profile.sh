#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclude=della-r4c[1-4]n[1-16],della-r1c[3,4]n[1-16]
#SBATCH -t 00:10:00

module load intel-vtune/oneapi
module load anaconda3/2024.2

conda activate spike_find

export OMP_NUM_THREADS=32

vtune -r profiles/vtune_mcmc_omp -collect hpc-performance -- python tests/test_pgas.py