# operators/cmp.py
from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH = WIDTH if 'WIDTH' in dir() else 32
COUNT = COUNT if 'COUNT' in dir() else 1

operators = [
    Operator("cmpi", 1, _pat((0, "arith.cmpi", f"i{WIDTH}")), typ="i1", count=COUNT, description=f"{WIDTH}bit int comparator"),
    Operator("cmpf", 1, _pat((0, "arith.cmpf", f"f{WIDTH}")), typ="i1", count=COUNT, description=f"{WIDTH}bit float comparator"),
]