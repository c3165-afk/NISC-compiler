"""
representation.py: FSM/DEC中間表現

CDFGからChiselコードへの変換の中間表現。
ハードウェアの配線名に依存しない抽象的な表現。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class OpKind(Enum):
    """演算の種類。"""
    ALU     = "alu"
    CMP     = "cmp"
    MUL     = "mul"
    FPU     = "fpu"
    FMUL    = "fmul"
    LOAD    = "load"
    STORE   = "store"
    DONE    = "done"
    NOP     = "nop"


@dataclass
class Transition:
    """FSMの遷移。"""
    next_state: int
    condition: str | None = None  # None=無条件, "true"=cmp真, "false"=cmp偽


@dataclass
class FSMState:
    """FSMの1ステート。"""
    state_id: int
    transitions: list[Transition] = field(default_factory=list)
    is_done: bool = False

    def add_transition(self, next_state: int, condition: str | None = None):
        self.transitions.append(Transition(next_state, condition))


@dataclass
class DECOp:
    """DECの1ステートの演算。"""
    state_id: int
    kind: OpKind
    inputs: list[str] = field(default_factory=list)   # 入力レジスタ名のリスト
    outputs: list[str] = field(default_factory=list)  # 出力レジスタ名のリスト
    mem_addr: str | None = None   # メモリアドレス
    mem_data: str | None = None   # メモリデータ
    mem_wen: bool = False         # メモリ書き込みイネーブル
    comment: str | None = None    # コメント

    # 使い方:
    # ALU:   inputs=[in1, in2], outputs=[out]
    # CMP:   inputs=[in1, in2], outputs=[]
    # MUL:   inputs=[in1, in2], outputs=[out]
    # MAC:   inputs=[in1, in2, acc], outputs=[out]
    # LOAD:  inputs=[addr], outputs=[dest]
    # STORE: inputs=[addr, data], outputs=[]


@dataclass
class RegInfo:
    """レジスタ情報。"""
    reg_id: int
    name: str
    is_imm: bool = False
    imm_value: int | None = None


@dataclass
class FSMDECProgram:
    """FSM/DEC中間表現。"""
    package: str = "nisc.program.generated"
    source_file: str = ""

    # FSM
    fsm_states: list[FSMState] = field(default_factory=list)

    # DEC
    dec_ops: list[DECOp] = field(default_factory=list)

    # レジスタ情報
    gpr: list[RegInfo] = field(default_factory=list)
    imm: list[RegInfo] = field(default_factory=list)

    def get_fsm_state(self, state_id: int) -> FSMState | None:
        return next((s for s in self.fsm_states if s.state_id == state_id), None)

    def get_dec_op(self, state_id: int) -> DECOp | None:
        return next((op for op in self.dec_ops if op.state_id == state_id), None)

    def add_fsm_state(self, state_id: int) -> FSMState:
        state = FSMState(state_id=state_id)
        self.fsm_states.append(state)
        return state

    def add_dec_op(self, state_id: int, kind: OpKind, **kwargs) -> DECOp:
        op = DECOp(state_id=state_id, kind=kind, **kwargs)
        self.dec_ops.append(op)
        return op