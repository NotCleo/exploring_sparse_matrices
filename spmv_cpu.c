#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 125000     // Matrix rows (125K)
#define COLS 125000     // Matrix columns (125K)
#define SPARSITY 0.00005 // Adjusted to ~15,625 non-zeros
#define MAX_NNZ 31250LL // 15,625 * 2 as long long

// Generate a sparse matrix in CSR format
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

int main() {
    int i, j, nnz;
    double start, end;

    double *val = (double*)malloc(MAX_NNZ * sizeof(double));
    int *col_idx = (int*)malloc(MAX_NNZ * sizeof(int));
    int *row_ptr = (int*)malloc((ROWS + 1) * sizeof(int));
    double *x = (double*)malloc(COLS * sizeof(double));
    double *y_seq = (double*)malloc(ROWS * sizeof(double));
    double *y_par = (double*)malloc(ROWS * sizeof(double));

    if (!val || !col_idx || !row_ptr || !x || !y_seq || !y_par) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    srand(42);
    for (i = 0; i < COLS; i++) x[i] = rand() % 10;
    generate_sparse_matrix(val, col_idx, row_ptr, &nnz);

    for (i = 0; i < ROWS; i++) {
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            volatile double temp = val[j] * x[col_idx[j]];
            (void)temp;
        }
    }

    start = omp_get_wtime();
    for (i = 0; i < ROWS; i++) {
        y_seq[i] = 0;
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y_seq[i] += val[j] * x[col_idx[j]];
        }
    }
    end = omp_get_wtime();
    double seq_time = end - start;

    start = omp_get_wtime();
    #pragma omp parallel for private(i, j) schedule(static)
    for (i = 0; i < ROWS; i++) {
        y_par[i] = 0;
        for (j = row_ptr[i]; j < row_ptr[i + 1] - 3; j += 4) {
            y_par[i] += val[j] * x[col_idx[j]] +
                        val[j + 1] * x[col_idx[j + 1]] +
                        val[j + 2] * x[col_idx[j + 2]] +
                        val[j + 3] * x[col_idx[j + 3]];
        }
        for (; j < row_ptr[i + 1]; j++) {
            y_par[i] += val[j] * x[col_idx[j]];
        }
    }
    end = omp_get_wtime();
    double par_time = end - start;
    int num_threads = omp_get_max_threads();

    printf("Sequential CPU Time: %.6f seconds\n", seq_time);
    printf("Parallel CPU Time: %.6f seconds\n", par_time);
    printf("Speedup (Parallel vs Sequential): %.2fx\n", seq_time / par_time);
    printf("Number of Threads: %d\n", num_threads);
    printf("Non-zero Elements: %d (%.4f%% sparsity)\n", nnz, (float)nnz / ((long long)ROWS * COLS) * 100);

    free(val); free(col_idx); free(row_ptr);
    free(x); free(y_seq); free(y_par);

    return 0;
}
