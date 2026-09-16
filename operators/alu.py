from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH = WIDTH if 'WIDTH' in dir() else 32
COUNT = COUNT if 'COUNT' in dir() else 1

typ = f"i{WIDTH}"
operators = [
    Operator("addi",  1, _pat((0, "arith.addi",  typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int add"),
    Operator("subi",  1, _pat((0, "arith.subi",  typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int sub"),
    Operator("andi",  1, _pat((0, "arith.andi",  typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int and"),
    Operator("ori",   1, _pat((0, "arith.ori",   typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int or"),
    Operator("xori",  1, _pat((0, "arith.xori",  typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int xor"),
    Operator("shli",  1, _pat((0, "arith.shli",  typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int shift left"),
    Operator("shrsi", 1, _pat((0, "arith.shrsi", typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int shift right signed"),
    Operator("shrui", 1, _pat((0, "arith.shrui", typ)), typ=typ, count=COUNT, resource="alu", description=f"{WIDTH}bit int shift right unsigned"),
]