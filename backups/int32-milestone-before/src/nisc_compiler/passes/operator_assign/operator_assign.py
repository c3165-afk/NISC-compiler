"""
operator_assign.py: 演算器割り当てパス
"""
from __future__ import annotations
import networkx as nx

from nisc_compiler.passes.base import OperatorAssignBase
from nisc_compiler.context import CompileContext
from nisc_compiler.passes.operator_assign.core import match_all


class VF2OperatorAssignPass(OperatorAssignBase):
    name = "operator_assign"

    def run(self, cdfg, context):
        cdfg, results, unmatched = match_all(cdfg, context.operators)
        if unmatched:
            raise RuntimeError(
                f"Unmatched operations: {unmatched}\n"
                f"These operations have no matching operator in the DP."
            )
        context.unmatched_ops = unmatched
        return cdfg
