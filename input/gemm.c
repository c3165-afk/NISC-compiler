void gemm(float C[][4], float A[][4], float B[][4], float alpha, float beta, int n) {
    for (int i = 0; i < n; i = i + 1)
        for (int j = 0; j < n; j = j + 1) {
            C[i][j] *= beta;
            for (int k = 0; k < n; k = k + 1)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
}
