"""
state_assign.py: ASAPスケジューリングパス
"""
from __future__ import annotations
import networkx as nx
 
from nisc_compiler.passes.base import StateAssignBase
from nisc_compiler.context import CompileContext
from nisc_compiler.passes.state_assign.core import iterative_schedule
 
 
class ASAPStateAssignPass(StateAssignBase):
    """ASAPスケジューリングによるステート割り当てパス。"""
    name = "state_assign"
 
    def run(self, cdfg: nx.DiGraph, context: CompileContext) -> nx.DiGraph:
        cdfg, reg_map, imm_map, spill_map, spill_base_reg, iters = iterative_schedule(
            cdfg,
            context.operators,
            num_registers=context.num_registers,
            num_imm_registers=context.num_imm_registers,
        )
        context.reg_map = reg_map
        context.imm_map = imm_map
        context.spill_map = spill_map
        context.spill_base_reg = spill_base_reg
        context.iterations = iters
        return cdfg