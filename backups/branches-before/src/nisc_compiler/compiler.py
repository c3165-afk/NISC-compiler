"""
compiler.py: NISCコンパイラのメインクラス

パスを組み立てて順番に実行する。
各パスは差し替え可能。
"""
from __future__ import annotations
import time
import networkx as nx

from nisc_compiler.context import CompileContext
from nisc_compiler.passes.base import CompilerPass


class NISCCompiler:
    """
    NISCコンパイラ。

    パスを順番に実行してCからChiselコードを生成する。
    各パスは差し替え可能。

    使い方:
        compiler = NISCCompiler.default(num_registers=32)
        ctx = compiler.compile("test.c", c_source)

    パスの差し替え:
        compiler.replace_pass('state_assign', ALAPStateAssign())P
    """

    def __init__(self, passes: list[CompilerPass], context: CompileContext):
        self._passes = passes
        self.context = context

    @classmethod
    def default(
        cls,
        num_registers: int = 32,
        num_imm_registers: int = 32,
        reg_width: int = 32,       # ← 追加
        profile: str = "int32",
        memory_words: int = 256,
    ) -> "NISCCompiler":
        """デフォルト設定でコンパイラを作成する。"""
        from nisc_compiler.passes.c_to_mlir.c_parser import CToMLIRPass
        from nisc_compiler.passes.mlir_to_cdfg.mlir_to_cdfg import MLIRToCDFGPass
        from nisc_compiler.passes.operator_assign.operator_assign import VF2OperatorAssignPass
        from nisc_compiler.passes.state_assign.state_assign import ASAPStateAssignPass
        from nisc_compiler.passes.code_gen.code_gen import ChiselCodeGenPass

        context = CompileContext(
            operators=[],
            num_registers=num_registers,
            num_imm_registers=num_imm_registers,
            reg_width=reg_width,   # ← 追加
            profile=profile,
            memory_words=memory_words,
        )

        passes = [
            CToMLIRPass(),
            MLIRToCDFGPass(),
            VF2OperatorAssignPass(),
            ASAPStateAssignPass(),
            ChiselCodeGenPass(),
        ]

        return cls(passes, context)

    def replace_pass(self, name: str, new_pass: CompilerPass):
        """指定した名前のパスを差し替える。"""
        for i, p in enumerate(self._passes):
            if p.name == name:
                self._passes[i] = new_pass
                return
        raise ValueError(
            f"Pass '{name}' not found. "
            f"Available: {[p.name for p in self._passes]}"
        )

    def insert_pass(self, after: str, new_pass: CompilerPass):
        """指定したパスの後に新しいパスを挿入する。"""
        for i, p in enumerate(self._passes):
            if p.name == after:
                self._passes.insert(i + 1, new_pass)
                return
        raise ValueError(f"Pass '{after}' not found.")
    
    def remove_pass(self, name: str):
        """指定した名前のパスを削除する。"""
        for i, p in enumerate(self._passes):
            if p.name == name:
                self._passes.pop(i)
                return
        raise ValueError(
            f"Pass '{name}' not found. "
            f"Available: {[p.name for p in self._passes]}"
        )

    def compile(self, source_file: str, c_source: str) -> CompileContext:
        """
        Cソースをコンパイルする。

        Returns:
            コンパイル結果が入ったCompileContext
        """
        # A failed or repeated compilation must not expose stale success data.
        from dataclasses import replace
        self.context = replace(
            self.context, mlir_text="", reg_map={}, imm_map={}, spill_map={},
            spill_base_reg=-1, iterations=0, program_scala="", init_txt="",
            unmatched_ops=[], pass_times={}, extra={},
        )
        if self.context.profile not in ("int32", "legacy"):
            raise ValueError("Unknown compilation profile")
        if self.context.profile == "int32":
            if self.context.reg_width != 32:
                raise ValueError("int32 profile requires reg_width=32")
            for name in ("num_registers", "num_imm_registers", "memory_words"):
                value = getattr(self.context, name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
        self.context.source_file = source_file
        self.context.c_source = c_source

        cdfg = None
        for pass_ in self._passes:
            t0 = time.perf_counter()
            cdfg = pass_.run(cdfg, self.context)
            t1 = time.perf_counter()
            self.context.pass_times[pass_.name] = round(t1 - t0, 4)

        self.context.extra['cdfg'] = cdfg

        return self.context
    
    def add_operators(self, operators: list):
        """演算器リストを追加する。

        使い方:
            from nisc_compiler.passes.operator_assign.dp import Operator, _pat
            ops = [Operator("mac_f32", latency=2, ...)]
            compiler.add_operators(ops)
        """
        self.context.operators += operators

    def load_operators(self, path: str, **kwargs):
        """演算器定義ファイルを読み込んで追加する。"""
        with open(path, encoding="utf-8") as f:
            source = f.read()
        
        # kwargsをグローバル変数として定義してから実行
        namespace = {k.upper(): v for k, v in kwargs.items()}
        exec(source, namespace)
        
        if 'operators' not in namespace:
            raise ValueError(f"'{path}' does not define 'operators' list.")
        self.context.operators += namespace['operators']

    def print_passes(self):
        """登録されているパスを表示する。"""
        print("Compiler passes:")
        for i, p in enumerate(self._passes):
            print(f"  {i+1}. {p}")
