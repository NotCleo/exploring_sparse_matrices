#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ROWS 1000    // Matrix rows
#define COLS 1000    // Matrix columns
#define SPARSITY 0.01 // 1% non-zero elements
#define THREADS_PER_BLOCK 256

// Generate a sparse matrix in CSR format (same as before)
void generate_sparse_matrix(double *val, int *col_idx, int *row_ptr, int *nnz) {
    *nnz = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if ((rand() % 10000) / 10000.0 < SPARSITY) {
                val[*nnz] = rand() % 10 + 1;
                col_idx[*nnz] = j;
                (*nnz)++;
            }
        }
        row_ptr[i + 1] = *nnz;
    }
}

// CUDA kernel for SpMV
__global__ void spmv_kernel(const double *val, const int *col_idx, const int *row_ptr, 
                            const double *x, double *y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += val[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

int main() {
    int i, nnz;
    float kernel_time, total_time;

    // Host arrays
    double *h_val = (double*)malloc(ROWS * COLS * sizeof(double));
    int *h_col_idx = (int*)malloc(ROWS * COLS * sizeof(int));
    int *h_row_ptr = (int*)malloc((ROWS + 1) * sizeof(int));
    double *h_x = (double*)malloc(COLS * sizeof(double));
    double *h_y = (double*)malloc(ROWS * sizeof(double));

    // Initialize data
    for (i = 0; i < COLS; i++) h_x[i] = rand() % 10;
    for (i = 0; i < ROWS; i++) h_y[i] = 0;
    generate_sparse_matrix(h_val, h_col_idx, h_row_ptr, &nnz);

    // Device arrays
    double *d_val, *d_x, *d_y;
    int *d_col_idx, *d_row_ptr;
    cudaMalloc(&d_val, ROWS * COLS * sizeof(double));
    cudaMalloc(&d_col_idx, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_row_ptr, (ROWS + 1) * sizeof(int));
    cudaMalloc(&d_x, COLS * sizeof(double));
    cudaMalloc(&d_y, ROWS * sizeof(double));

    // Copy data to device
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(d_val, h_val, ROWS * COLS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, COLS * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((ROWS + block.x - 1) / block.x);
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventRecord(kernel_start);
    spmv_kernel<<<grid, block>>>(d_val, d_col_idx, d_row_ptr, d_x, d_y, ROWS);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);

    // Copy result back
    cudaMemcpy(h_y, d_y, ROWS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);

    // Output metrics
    printf("CUDA Kernel Time: %f seconds\n", kernel_time / 1000.0);
    printf("Total Time (including transfers): %f seconds\n", total_time / 1000.0);
    printf("Non-zero Elements: %d (%.2f%% sparsity)\n", nnz, (float)nnz / (ROWS * COLS) * 100);
    printf("Grid Size: %d blocks, Block Size: %d threads\n", grid.x, block.x);

    // Sequential CPU for speedup comparison
    double start_seq = omp_get_wtime();
    for (i = 0; i < ROWS; i++) {
        h_y[i] = 0;
        for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
            h_y[i] += h_val[j] * h_x[h_col_idx[j]];
        }
    }
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;
    printf("Sequential CPU Time: %f seconds\n", seq_time);
    printf("Speedup (Total vs Sequential): %.2fx\n", seq_time / (total_time / 1000.0));

    // Cleanup
    cudaFree(d_val); cudaFree(d_col_idx); cudaFree(d_row_ptr);
    cudaFree(d_x); cudaFree(d_y);
    free(h_val); free(h_col_idx); free(h_row_ptr);
    free(h_x); free(h_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_stop);

    return 0;
} 
