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
        context.mlir_text = parse_c(context.c_source)
        return None