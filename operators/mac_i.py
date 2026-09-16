from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH   = WIDTH   if 'WIDTH'   in dir() else 32
COUNT   = COUNT   if 'COUNT'   in dir() else 1
LATENCY = LATENCY if 'LATENCY' in dir() else 3

typ = f"i{WIDTH}"
operators = [
    Operator(
        f"mac_i{WIDTH}", LATENCY,
        _pat((0, "arith.muli", typ), (1, "arith.addi", typ), edges=[(0, 1)]),
        typ=typ, count=COUNT,
        description=f"{WIDTH}bit int multiply-accumulate",
    )
]
