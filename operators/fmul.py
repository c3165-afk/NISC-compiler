from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH = WIDTH if 'WIDTH' in dir() else 32
COUNT = COUNT if 'COUNT' in dir() else 1

typ = f"f{WIDTH}"
operators = [
    Operator(f"mulf{WIDTH}", 2, _pat((0, "arith.mulf", typ)), typ=typ, count=COUNT, description=f"{WIDTH}bit float multiplier"),
]
