int test_complex(int a[], int n) {
    int total_sum = 0;
    
    for (int i = 0; i < n; i = i + 1) {
        int x = a[i];
        
        // 1. 同型の加算命令（ALU_ADD）が連続して生成される区間
        int val1 = x + 10;
        int val2 = x + 20;
        int val3 = x + 30;
        int val4 = x + 40;
        
        // 2. 加算結果の集計（ここでも加算命令が連続する）
        int sum_val = val1 + val2 + val3 + val4;
        
        // 3. 乗算（MUL）と加算（ALU）の複合処理
        total_sum = total_sum + (sum_val * 2);
    }
    
    return total_sum;
}