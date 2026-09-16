void scale_offset(int src[], int dst[], int n, int offset) {
    for (int i = 0; i < n; i = i + 1) {
        int addr = i + offset;
        dst[addr] = src[i];
    }
}