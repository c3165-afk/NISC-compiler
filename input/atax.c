/* ATAX: Matrix Transpose and Vector Multiplication */
void atax(float A[][32], float x[], float y[], float tmp[], int n) {
    for (int i = 0; i < n; i = i + 1)
        tmp[i] = 0.0f;
    for (int i = 0; i < n; i = i + 1) {
        for (int j = 0; j < n; j = j + 1)
            tmp[i] += A[i][j] * x[j];
        for (int j = 0; j < n; j = j + 1)
            y[j] += A[i][j] * tmp[i];
    }
}
