#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define ROWS 125000     // Matrix rows (125K)
#define COLS 125000     // Matrix columns (125K)
#define SPARSITY 0.00005 // Adjusted to ~15,625 non-zeros
#define MAX_NNZ 31250LL // 15,625 * 2 as long long
#define THREADS_PER_BLOCK 256

void generate_sparse_matrix(double *val, int *col_idx, int *row_ptr, int *nnz) {
    *nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if ((rand() % 1000000) / 1000000.0 < SPARSITY) {
                if (*nnz >= MAX_NNZ) {
                    printf("Warning: Exceeded MAX_NNZ, truncating at %lld\n", (long long)MAX_NNZ);
                    row_ptr[i + 1] = *nnz;
                    for (int k = i + 1; k <= ROWS; k++) row_ptr[k] = *nnz;
                    return;
                }
                val[*nnz] = rand() % 10 + 1;
                col_idx[*nnz] = j;
                (*nnz)++;
            }
        }
        row_ptr[i + 1] = *nnz;
    }
}

__global__ void spmv_kernel(const double *val, const int *col_idx, const int *row_ptr,
                            const double *x, double *y, int rows) {
    extern __shared__ double shared_x[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadIdx.x; i < COLS; i += blockDim.x) {
        if (i < COLS) shared_x[i] = x[i];
    }
    __syncthreads();

    if (row < rows) {
        double sum = 0.0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += val[j] * shared_x[col_idx[j]];
        }
        y[row] = sum;
    }
}

int main() {
    int i, j, nnz;
    float kernel_time, total_time;

    double *h_val = (double*)malloc(MAX_NNZ * sizeof(double));
    int *h_col_idx = (int*)malloc(MAX_NNZ * sizeof(int));
    int *h_row_ptr = (int*)malloc((ROWS + 1) * sizeof(int));
    double *h_x = (double*)malloc(COLS * sizeof(double));
    double *h_y = (double*)malloc(ROWS * sizeof(double));

    if (!h_val || !h_col_idx || !h_row_ptr || !h_x || !h_y) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    srand(42);
    for (i = 0; i < COLS; i++) h_x[i] = rand() % 10;
    for (i = 0; i < ROWS; i++) h_y[i] = 0;
    generate_sparse_matrix(h_val, h_col_idx, h_row_ptr, &nnz);

    for (i = 0; i < ROWS; i++) {
        for (j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
            volatile double temp = h_val[j] * h_x[h_col_idx[j]];
            (void)temp;
        }
    }

    double start_seq = omp_get_wtime();
    for (i = 0; i < ROWS; i++) {
        h_y[i] = 0;
        for (j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
            h_y[i] += h_val[j] * h_x[h_col_idx[j]];
        }
    }
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;

    double *d_val, *d_x, *d_y;
    int *d_col_idx, *d_row_ptr;
    cudaMalloc(&d_val, MAX_NNZ * sizeof(double));
    cudaMalloc(&d_col_idx, MAX_NNZ * sizeof(int));
    cudaMalloc(&d_row_ptr, (ROWS + 1) * sizeof(int));
    cudaMalloc(&d_x, COLS * sizeof(double));
    cudaMalloc(&d_y, ROWS * sizeof(double));

    if (cudaGetLastError() != cudaSuccess) {
        printf("CUDA memory allocation failed\n");
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(d_val, h_val, MAX_NNZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, MAX_NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, COLS * sizeof(double), cudaMemcpyHostToDevice);

    int grid_size = (ROWS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start);
    spmv_kernel<<<grid_size, THREADS_PER_BLOCK, COLS * sizeof(double)>>>(d_val, d_col_idx, d_row_ptr, d_x, d_y, ROWS);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

    cudaMemcpy(h_y, d_y, ROWS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);

    printf("Sequential CPU Time: %.6f seconds\n", seq_time);
    printf("CUDA Kernel Time: %.6f seconds\n", kernel_time / 1000.0);
    printf("Total CUDA Time (including transfers): %.6f seconds\n", total_time / 1000.0);
    printf("Speedup (Kernel vs Sequential): %.2fx\n", seq_time / (kernel_time / 1000.0));
    printf("Speedup (Total vs Sequential): %.2fx\n", seq_time / (total_time / 1000.0));
    printf("Non-zero Elements: %d (%.4f%% sparsity)\n", nnz, (float)nnz / ((long long)ROWS * COLS) * 100);
    printf("Grid Size: %d blocks, Block Size: %d threads\n", grid_size, THREADS_PER_BLOCK);

    cudaFree(d_val); cudaFree(d_col_idx); cudaFree(d_row_ptr);
    cudaFree(d_x); cudaFree(d_y);
    free(h_val); free(h_col_idx); free(h_row_ptr);
    free(h_x); free(h_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_stop);

    return 0;
}
