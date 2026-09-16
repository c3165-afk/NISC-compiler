# operators/mem.py
from nisc_compiler.passes.operator_assign.dp import Operator, _pat

LOAD_COUNT  = LOAD_COUNT  if 'LOAD_COUNT'  in dir() else 1
STORE_COUNT = STORE_COUNT if 'STORE_COUNT' in dir() else 1

operators = [
    # Operator("load",  1, _pat((0, "memref.load")),  typ="", count=LOAD_COUNT,  description="memory load"),
    # Operator("store", 1, _pat((0, "memref.store")), typ="", count=STORE_COUNT, description="memory store"),
    # Operator("alloca", 1, _pat((0, "memref.alloca")), typ="", count=1, description="stack allocation"),
    Operator("load",  2, _pat((0, "memref.load")),  typ="", count=LOAD_COUNT,  description="memory load"),
    Operator("store", 2, _pat((0, "memref.store")), typ="", count=STORE_COUNT, description="memory store"),
    Operator("alloca", 2, _pat((0, "memref.alloca")), typ="", count=1, description="stack allocation"),
]