/* MVT: Matrix Vector Product and Transpose */
void mvt(float A[][32], float x1[], float x2[], float y1[], float y2[], int n) {
    for (int i = 0; i < n; i = i + 1)
        for (int j = 0; j < n; j = j + 1)
            x1[i] += A[i][j] * y1[j];
    for (int i = 0; i < n; i = i + 1)
        for (int j = 0; j < n; j = j + 1)
            x2[i] += A[j][i] * y2[j];
}
