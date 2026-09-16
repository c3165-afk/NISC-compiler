"""
context.py: コンパイルコンテキスト

パス間で共有される状態を管理する。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class CompileContext:
    """
    コンパイル全体で共有されるコンテキスト。

    各パスはこのオブジェクトを通じて情報を共有する。
    """
    # 入力
    source_file: str = ""
    c_source: str = ""

    # DP設定
    operators: list = field(default_factory=list)
    num_registers: int = 32
    num_imm_registers: int = 32
    reg_width: int = 32      # ← 追加（GPRのビット幅）
    profile: str = "int32"
    memory_words: int = 256

    # MLIR中間表現
    mlir_text: str = ""

    # レジスタ割り当て結果
    reg_map: dict[str, int] = field(default_factory=dict)
    imm_map: dict[str, int] = field(default_factory=dict)

    # スピル情報
    spill_map: dict[str, int] = field(default_factory=dict)
    spill_base_reg: int = -1

    # スケジューリング情報
    iterations: int = 0

    # 出力
    program_scala: str = ""
    init_txt: str = ""

    # レポート用
    unmatched_ops: list = field(default_factory=list)
    pass_times: dict[str, float] = field(default_factory=dict)

    # 任意の追加データ
    extra: dict[str, Any] = field(default_factory=dict)
