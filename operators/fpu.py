from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH = WIDTH if 'WIDTH' in dir() else 32
COUNT = COUNT if 'COUNT' in dir() else 1

typ = f"f{WIDTH}"
operators = [
    Operator(f"addf{WIDTH}", 1, _pat((0, "arith.addf", typ)), typ=typ, count=COUNT, description=f"{WIDTH}bit float add"),
    Operator(f"subf{WIDTH}", 1, _pat((0, "arith.subf", typ)), typ=typ, count=COUNT, description=f"{WIDTH}bit float sub"),
]
