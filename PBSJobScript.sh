#!/bin/bash
#PBS -l nodes=1:ppn=30   # Match multiprocessor count from nvidia-smi
#PBS -l walltime=00:10:00 # Max runtime of 10 minutes
#PBS -N SparseMatrixOpt  # Job name

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=30 # Match ppn
export OMP_PROC_BIND=true # Thread pinning
./spmv_opt
