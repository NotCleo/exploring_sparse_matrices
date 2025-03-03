#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1  # Request 1 CPU core, 1 GPU
#PBS -l walltime=00:10:00      # Max runtime of 10 minutes
#PBS -N SparseMatrixCUDA       # Job name

cd $PBS_O_WORKDIR
./spmv_cuda
