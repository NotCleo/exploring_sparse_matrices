#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1  # Request 1 CPU core, 1 GPU
#PBS -l walltime=00:10:00      # Max runtime of 10 minutes
#PBS -N SparseMatrixCUDAOpt    # Job name

cd $PBS_O_WORKDIR
./spmv_cuda_opt


//(or) add on the below

//cd $PBS_O_WORKDIR
//export OMP_NUM_THREADS=30 # Match ppn
//export OMP_PROC_BIND=true # Thread pinning
//./spmv_opt
