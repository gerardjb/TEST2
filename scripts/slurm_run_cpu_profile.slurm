#!/bin/bash
#SBATCH --job-name=infer_test2
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=06:00:00

#module load intel-vtune/oneapi
module load anaconda3/2024.2

conda activate spike_find

export OMP_NUM_THREADS=32

python TEST2_exc.py
python TEST2_inh.py