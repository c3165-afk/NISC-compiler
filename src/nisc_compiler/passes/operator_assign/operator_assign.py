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
        operators = context.operators
        if context.profile == "int32":
            from nisc_compiler.passes.state_assign.int32 import select_operators
            operators = select_operators(operators)
            context.extra['int32_operators'] = operators
        cdfg, results, unmatched = match_all(cdfg, operators)
        context.unmatched_ops = unmatched
        if unmatched:
            raise RuntimeError(
                f"Unmatched operations: {unmatched}\n"
                f"These operations have no matching operator in the DP."
            )
        context.unmatched_ops = unmatched
        return cdfg
