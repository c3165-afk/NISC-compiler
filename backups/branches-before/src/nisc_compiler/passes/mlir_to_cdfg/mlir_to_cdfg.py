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
        graph = lower_mlir(context.mlir_text, stable_names=context.profile == 'int32')
        if context.profile == "int32":
            from .int32 import add_abi
            graph = add_abi(graph, context.extra['abi'])
        return graph
