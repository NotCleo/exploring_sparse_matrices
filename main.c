#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 1000    // Matrix rows
#define COLS 1000    // Matrix columns
#define SPARSITY 0.01 // 1% non-zero elements

// Generate a sparse matrix in CSR format
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

int main() {
    int i, j, nnz;
    double start, end;

    // Allocate CSR arrays and vectors
    double *val = (double*)malloc(ROWS * COLS * sizeof(double));
    int *col_idx = (int*)malloc(ROWS * COLS * sizeof(int));
    int *row_ptr = (int*)malloc((ROWS + 1) * sizeof(int));
    double *x = (double*)malloc(COLS * sizeof(double));
    double *y_seq = (double*)malloc(ROWS * sizeof(double));
    double *y_par = (double*)malloc(ROWS * sizeof(double));

    // Initialize vector x and generate sparse matrix
    for (i = 0; i < COLS; i++) x[i] = rand() % 10;
    generate_sparse_matrix(val, col_idx, row_ptr, &nnz);

    // Sequential SpMV
    start = omp_get_wtime();
    for (i = 0; i < ROWS; i++) {
        y_seq[i] = 0;
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y_seq[i] += val[j] * x[col_idx[j]];
        }
    }
    end = omp_get_wtime();
    double seq_time = end - start;
    printf("Sequential SpMV Time: %f seconds\n", seq_time);

    // Optimized Parallel SpMV with OpenMP
    start = omp_get_wtime();
    #pragma omp parallel for private(i, j) schedule(dynamic)
    for (i = 0; i < ROWS; i++) {
        y_par[i] = 0;
        // Loop unrolling (factor of 4)
        for (j = row_ptr[i]; j < row_ptr[i + 1] - 3; j += 4) {
            y_par[i] += val[j] * x[col_idx[j]] +
                        val[j + 1] * x[col_idx[j + 1]] +
                        val[j + 2] * x[col_idx[j + 2]] +
                        val[j + 3] * x[col_idx[j + 3]];
        }
        // Handle remaining elements
        for (; j < row_ptr[i + 1]; j++) {
            y_par[i] += val[j] * x[col_idx[j]];
        }
    }
    end = omp_get_wtime();
    double par_time = end - start;
    int num_threads = omp_get_max_threads();

    // Output detailed metrics
    printf("Parallel SpMV Time: %f seconds\n", par_time);
    printf("Number of Threads: %d\n", num_threads);
    printf("Speedup: %.2fx\n", seq_time / par_time);
    printf("Efficiency: %.2f%%\n", (seq_time / par_time) / num_threads * 100);
    printf("Non-zero Elements: %d (%.2f%% sparsity)\n", nnz, (float)nnz / (ROWS * COLS) * 100);

    // Free memory
    free(val); free(col_idx); free(row_ptr);
    free(x); free(y_seq); free(y_par);

    return 0;
}
