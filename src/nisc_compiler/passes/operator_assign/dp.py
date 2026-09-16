"""
dp.py: NISCのDP演算器定義

構成:
  固定演算器（必ず載る）:
    ALU: addi, subi, andi, ori, xori, shli, shrsi, shrui

  オプション演算器（ユーザが選択）:
    MUL:   muli
    CMP:   cmpi, cmpf
    FPU:   addf, subf
    FMUL:  mulf
    MAC_I: muli → addi
    MAC_F: mulf → addf
    SQRT, SIN, COS, EXP, LOG

使い方:
  from dp import load_dp
  operators = load_dp()                             # ALUのみ
  operators = load_dp(mul=True, cmp=True, fpu=True) # ALU+MUL+CMP+FPU
"""
from __future__ import annotations
from dataclasses import dataclass
import networkx as nx


@dataclass
class Operator:
    name: str
    latency: int
    pattern: nx.DiGraph
    count: int = 1
    pipelined: bool = False
    typ: str = ""
    description: str = ""
    resource: str = ""  # ← 追加（同じresourceは同じ演算器を使う）


def _pat(*ops: tuple, edges: list[tuple[int, int]] = None) -> nx.DiGraph:
    """DFGパターンを作るヘルパー。
    
    ops: (node_id, op_name) または (node_id, op_name, typ)
         typを省略した場合は型チェックなし
    """
    g = nx.DiGraph()
    for op in ops:
        node_id, op_name = op[0], op[1]
        typ = op[2] if len(op) > 2 else ""
        g.add_node(node_id, op_name=op_name, typ=typ)
    for src, dst in (edges or []):
        g.add_edge(src, dst)
    return g


# ================================================================
# 固定演算器: ALU
# ================================================================
ALU_OPERATORS = [
    Operator("addi",  1, _pat((0, "arith.addi")),  typ="i32", description="i32 add"),
    Operator("subi",  1, _pat((0, "arith.subi")),  typ="i32", description="i32 sub"),
    Operator("andi",  1, _pat((0, "arith.andi")),  typ="i32", description="i32 bitwise and"),
    Operator("ori",   1, _pat((0, "arith.ori")),   typ="i32", description="i32 bitwise or"),
    Operator("xori",  1, _pat((0, "arith.xori")),  typ="i32", description="i32 bitwise xor"),
    Operator("shli",  1, _pat((0, "arith.shli")),  typ="i32", description="i32 shift left"),
    Operator("shrsi", 1, _pat((0, "arith.shrsi")), typ="i32", description="i32 shift right signed"),
    Operator("shrui", 1, _pat((0, "arith.shrui")), typ="i32", description="i32 shift right unsigned"),
]

# ================================================================
# オプション演算器
# ================================================================
OPTIONAL_OPERATORS: dict[str, list[Operator]] = {
    "mul": [
        Operator("muli", 2, _pat((0, "arith.muli")), typ="i32", description="i32 multiplier"),
    ],
    "cmp": [
        Operator("cmpi", 1, _pat((0, "arith.cmpi")), typ="i1",  description="i32 comparator"),
        Operator("cmpf", 1, _pat((0, "arith.cmpf")), typ="i1",  description="f32 comparator"),
    ],
    "fpu": [
        Operator("addf", 1, _pat((0, "arith.addf")), typ="f32", description="f32 add"),
        Operator("subf", 1, _pat((0, "arith.subf")), typ="f32", description="f32 sub"),
    ],
    "fmul": [
        Operator("mulf", 2, _pat((0, "arith.mulf")), typ="f32", description="f32 multiplier"),
    ],
    "mac_i": [
        Operator("mac_i32", 3,
            _pat((0, "arith.muli"), (1, "arith.addi"), edges=[(0, 1)]),
            typ="i32", description="i32 multiply-accumulate"),
    ],
    "mac_f": [
        Operator("mac_f32", 3,
            _pat((0, "arith.mulf"), (1, "arith.addf"), edges=[(0, 1)]),
            typ="f32", description="f32 multiply-accumulate"),
    ],
    "sqrt": [
        Operator("sqrt", 4, _pat((0, "math.sqrt")), typ="f32", description="f32 square root"),
    ],
    "sin": [
        Operator("sin", 8, _pat((0, "math.sin")), typ="f32", description="f32 sine"),
    ],
    "cos": [
        Operator("cos", 8, _pat((0, "math.cos")), typ="f32", description="f32 cosine"),
    ],
    "exp": [
        Operator("exp", 8, _pat((0, "math.exp")), typ="f32", description="f32 exponential"),
    ],
    "log": [
        Operator("log", 8, _pat((0, "math.log")), typ="f32", description="f32 logarithm"),
    ],
    "mem": [
        Operator("load",  1, _pat((0, "memref.load")),  typ="i32",  description="memory load"),
        Operator("store", 1, _pat((0, "memref.store")), typ="void", description="memory store"),
    ],
}


# ================================================================
# 公開API
# ================================================================
def load_dp(
    mul:   bool = False,
    cmp:   bool = False,
    fpu:   bool = False,
    fmul:  bool = False,
    mac_i: bool = False,
    mac_f: bool = False,
    sqrt:  bool = False,
    sin:   bool = False,
    cos:   bool = False,
    exp:   bool = False,
    log:   bool = False,
    mem:   bool = False,
) -> list[Operator]:
    """
    DP演算器定義を返す。
    ALUは常に含まれる。その他はフラグで選択する。
    複合演算器（MAC）を先に返す（大きいパターンを優先マッチング）。
    """
    ops: list[Operator] = []

    # 複合演算器を先に（大きいパターンを優先マッチング）
    flags = dict(mac_i=mac_i, mac_f=mac_f, sqrt=sqrt,
                 sin=sin, cos=cos, exp=exp, log=log,
                 mul=mul, cmp=cmp, fpu=fpu, fmul=fmul,
                 mem=mem)
    for key, enabled in flags.items():
        if enabled:
            ops.extend(OPTIONAL_OPERATORS[key])

    # ALUは常に末尾に追加（単体演算器として最後にマッチング）
    ops.extend(ALU_OPERATORS)
    return ops


if __name__ == "__main__":
    print("=== ALUのみ ===")
    ops = load_dp()
    for op in ops:
        n = op.pattern.number_of_nodes()
        e = op.pattern.number_of_edges()
        print(f"  {op.name:<12} latency={op.latency}  count={op.count}  ({n}nodes,{e}edges)  {op.description}")

    print()
    print("=== ALU + MUL + CMP + FPU + FMUL + MAC ===")
    ops = load_dp(mul=True, cmp=True, fpu=True, fmul=True, mac_i=True, mac_f=True)
    for op in ops:
        n = op.pattern.number_of_nodes()
        e = op.pattern.number_of_edges()
        print(f"  {op.name:<12} latency={op.latency}  count={op.count}  ({n}nodes,{e}edges)  {op.description}")