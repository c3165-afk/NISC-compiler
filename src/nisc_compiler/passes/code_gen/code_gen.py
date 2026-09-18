"""
code_gen.py: Chiselコード生成パス
"""
from __future__ import annotations
import os
import networkx as nx
from nisc_compiler.passes.base import CodeGenBase
from nisc_compiler.context import CompileContext
from nisc_compiler.passes.code_gen.emitter import emit


class ChiselCodeGenPass(CodeGenBase):
    """Chisel FSM/DECコード生成パス。"""
    name = "code_gen"

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir

    def run(self, cdfg: nx.DiGraph, context: CompileContext) -> nx.DiGraph:
        program_name = context.source_file.replace('.c', '').replace('/', '_')
        package = f"nisc.program.{program_name}"

        program, init_txt = emit(
            cdfg,
            context.reg_map,
            context.imm_map,
            package=package,
            spill_map=context.spill_map,
            spill_base_reg=context.spill_base_reg,
            operators = context.operators,
        )
        context.program_scala = program
        context.init_txt = init_txt

        # output/{program_name}/ にファイルを出力
        out_dir = os.path.join(self.output_dir, program_name)
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, 'Program.scala'), 'w').write(program + '\n')
        open(os.path.join(out_dir, 'init.txt'), 'w').write(init_txt)

        return cdfg
