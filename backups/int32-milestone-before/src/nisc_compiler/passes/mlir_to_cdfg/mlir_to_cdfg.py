"""
mlir_to_cdfg.py: MLIR → CDFGパス
"""
from __future__ import annotations
import networkx as nx

from nisc_compiler.passes.base import MLIRToCDFGBase
from nisc_compiler.context import CompileContext
from nisc_compiler.passes.mlir_to_cdfg.core import lower_mlir


class MLIRToCDFGPass(MLIRToCDFGBase):
    """MLIR → CDFGパス。"""
    name = "mlir_to_cdfg"

    def run(self, cdfg: None, context: CompileContext) -> nx.DiGraph:
        return lower_mlir(context.mlir_text)