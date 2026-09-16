int main() {
    int A[4][4];
    int B[4][4];
    int C[4][4];

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0;
        }
    }

    for (int i = 0; i < 4; i++){
        for (int j =0; j < 4; j++){
            for (int k = 0; k < 4; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return 0;
}