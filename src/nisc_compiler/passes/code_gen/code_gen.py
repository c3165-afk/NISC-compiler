"""
code_gen.py: Chiselコード生成パス
"""
from __future__ import annotations
import os
import re
import json
import networkx as nx
from nisc_compiler.passes.base import CodeGenBase
from nisc_compiler.context import CompileContext
from nisc_compiler.passes.code_gen.emitter import emit


class ChiselCodeGenPass(CodeGenBase):
    """Chisel FSM/DECコード生成パス。"""
    name = "code_gen"

    def __init__(self, output_dir: str | None = "output"):
        self.output_dir = output_dir

    def run(self, cdfg: nx.DiGraph, context: CompileContext) -> nx.DiGraph:
        stem = context.source_file.replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
        program_name = 'program_' + re.sub(r'[^A-Za-z0-9_]', '_', stem)
        package = f"nisc.program.{program_name}"

        if context.profile == "int32":
            from .int32 import emit_int32
            program, init_txt = emit_int32(cdfg, context, package)
        else:
            program, init_txt = emit(
                cdfg,
                context.reg_map,
                context.imm_map,
                package=package,
                spill_map=context.spill_map,
                spill_base_reg=context.spill_base_reg,
                operators=context.operators,
            )
        context.program_scala = program
        context.init_txt = init_txt

        # output/{program_name}/ にファイルを出力
        if self.output_dir is None:
            return cdfg
        out_dir = os.path.join(self.output_dir, program_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'Program.scala'), 'w', encoding='utf-8') as f:
            f.write(program + '\n')
        with open(os.path.join(out_dir, 'init.txt'), 'w', encoding='utf-8') as f:
            f.write(init_txt)
        if context.profile == 'int32':
            with open(os.path.join(out_dir, 'controls.json'), 'w', encoding='utf-8') as f:
                json.dump(context.extra['control_image'], f, ensure_ascii=False, indent=2)

        return cdfg
