from nisc_compiler.passes.operator_assign.dp import Operator, _pat

operators = [
    Operator("addf64", 1, _pat((0, "arith.addf")), typ="f64",
             description="64bit float add"),
]
