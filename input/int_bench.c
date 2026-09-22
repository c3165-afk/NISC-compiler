int complex_calc(int n) {
    int total = 0;
    
    for (int i = 0; i < n; i = i + 1) {
        // 第1段：基礎値の計算
        int a = i * 3 + 7;
        int b = i * 5 - 2;
        int c = i * 11 + 13;
        int d = i * 17 - 5;

        // 第2段：相互演算（データ依存を発生させる）
        int x1 = a * b + c;
        int x2 = b * c - d;
        int x3 = c * d + a;
        int x4 = d * a - b;

        // 第3段：さらに依存関係を深める
        int y1 = x1 * x2 + x3;
        int y2 = x2 * x3 - x4;
        int y3 = x3 * x4 + x1;
        int y4 = x4 * x1 - x2;

        // 第4段：集計計算
        int step1 = y1 + y2;
        int step2 = y3 - y4;
        int step3 = step1 * step2;

        total = total + step3;
    }
    
    return total;
}