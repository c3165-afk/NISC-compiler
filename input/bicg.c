/* BiCG: BiCG Sub Kernel of BiCGStab Linear Solver */
void bicg(float A[][32], float s[], float q[], float p[], float r[], int n) {
    for (int i = 0; i < n; i = i + 1) {
        for (int j = 0; j < n; j = j + 1) {
            s[j] += r[i] * A[i][j];
            q[i] += A[i][j] * p[j];
        }
    }
}
