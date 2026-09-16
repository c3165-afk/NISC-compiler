from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH   = WIDTH   if 'WIDTH'   in dir() else 32
COUNT   = COUNT   if 'COUNT'   in dir() else 1
LATENCY = LATENCY if 'LATENCY' in dir() else 3

typ = f"f{WIDTH}"
operators = [
    Operator(
        f"mac_f{WIDTH}", LATENCY,
        _pat((0, "arith.mulf", typ), (1, "arith.addf", typ), edges=[(0, 1)]),
        typ=typ, count=COUNT,
        description=f"{WIDTH}bit float multiply-accumulate",
    )
]
