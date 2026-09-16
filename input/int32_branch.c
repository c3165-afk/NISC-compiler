int int32_branch(int a, int b) {
    int result = 0;
    if (a <= b) {
        result = a + 10;
    } else {
        result = b * 2;
    }
    return result;
}
