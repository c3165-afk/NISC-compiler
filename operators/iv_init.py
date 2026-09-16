from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH = WIDTH if 'WIDTH' in dir() else 32
COUNT = COUNT if 'COUNT' in dir() else 1

operators = [
    Operator("iv_init", 1, _pat((0, "nisc.iv_init")),
             typ=f"i{WIDTH}", count=COUNT, resource="alu",
             description="loop index variable initialization (iv = lb)"),
]
