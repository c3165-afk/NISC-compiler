from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH = WIDTH if 'WIDTH' in dir() else 32
COUNT = COUNT if 'COUNT' in dir() else 1

typ = f"i{WIDTH}"
operators = [
    Operator(f"muli{WIDTH}", 2, _pat((0, "arith.muli", typ)), typ=typ, count=COUNT, description=f"{WIDTH}bit int multiplier"),
]
