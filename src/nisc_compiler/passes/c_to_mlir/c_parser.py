"""
c_parser.py: C → MLIRパス
"""
from __future__ import annotations
import networkx as nx

from nisc_compiler.passes.base import CToMLIRBase
from nisc_compiler.context import CompileContext
from nisc_compiler.passes.c_to_mlir.core import parse_c


class CToMLIRPass(CToMLIRBase):
    """C → MLIRパス。"""
    name = "c_to_mlir"

    def run(self, cdfg: None, context: CompileContext) -> None:
        if context.profile == "int32":
            from .int32 import validate_source, Int32MLIRGen
            context.extra['abi'] = validate_source(context.c_source)
            if context.extra['abi']['return_address'] >= context.memory_words:
                raise ValueError("Argument and return area exceeds SRAM capacity")
            context.mlir_text = Int32MLIRGen().generate(context.c_source)
        else:
            context.mlir_text = parse_c(context.c_source)
        return None
