"""
cdfg_to_fsmrep.py: CDFG → FSM/DEC中間表現への変換

emitter.pyのgen_fsm/gen_decのロジックを中間表現生成に置き換える。
ハードウェアの配線名に依存しない抽象的な表現を生成する。
"""
from __future__ import annotations
from collections import defaultdict
import networkx as nx

from nisc_compiler.passes.code_gen.representation import (
    FSMDECProgram, FSMState, DECOp, RegInfo, OpKind, Transition
)


# ----------------------------------------------------------------
# 演算名 → OpKind のマッピング
# ----------------------------------------------------------------
OP_KIND_MAP = {
    'arith.addi':  OpKind.ALU,
    'arith.subi':  OpKind.ALU,
    'arith.andi':  OpKind.ALU,
    'arith.ori':   OpKind.ALU,
    'arith.xori':  OpKind.ALU,
    'arith.shli':  OpKind.ALU,
    'arith.shrsi': OpKind.ALU,
    'arith.shrui': OpKind.ALU,
    'arith.muli':  OpKind.MUL,
    'arith.cmpi':  OpKind.CMP,
    'arith.cmpf':  OpKind.CMP,
    'arith.addf':  OpKind.FPU,
    'arith.subf':  OpKind.FPU,
    'arith.mulf':  OpKind.FMUL,
    'memref.load':  OpKind.LOAD,
    'memref.store': OpKind.STORE,
    'done':         OpKind.DONE,
}


# ----------------------------------------------------------------
# CDFGユーティリティ（emitter.pyから移植）
# ----------------------------------------------------------------
def _get_all_bbs(cdfg: nx.DiGraph) -> list[int]:
    return [n for n, d in cdfg.nodes(data=True) if d.get('type') == 'bb']


def _get_bb_ops(cdfg: nx.DiGraph, bb_id: int) -> list[int]:
    return [dst for _, dst, edata in cdfg.out_edges(bb_id, data=True)
            if edata.get('label') == 'contains'
            and cdfg.nodes[dst].get('type') == 'op']


def _get_cfg_edges(cdfg: nx.DiGraph) -> list[tuple[int, int, str]]:
    bb_set = set(_get_all_bbs(cdfg))
    edges = []
    for src, dst, edata in cdfg.edges(data=True):
        if src in bb_set and dst in bb_set and edata.get('type') == 'ctrl':
            label = edata.get('label', '')
            if label != 'contains':
                edges.append((src, dst, label))
    return edges


def _state_of_bb(cdfg: nx.DiGraph, bb_id: int) -> int | None:
    ops = _get_bb_ops(cdfg, bb_id)
    states = [cdfg.nodes[n].get('state') for n in ops
              if cdfg.nodes[n].get('state') is not None]
    return min(states) if states else None


def _last_state_of_bb(cdfg: nx.DiGraph, bb_id: int) -> int | None:
    ops = _get_bb_ops(cdfg, bb_id)
    states = [cdfg.nodes[n].get('state') for n in ops
              if cdfg.nodes[n].get('state') is not None]
    return max(states) if states else None


def _get_done_state(cdfg: nx.DiGraph) -> int | None:
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'done':
            return data.get('state')
    return None


def _resolve_next_state(cdfg: nx.DiGraph, bb_id: int, visited: set = None) -> int | None:
    if visited is None:
        visited = set()
    if bb_id in visited:
        return None
    visited.add(bb_id)

    state = _state_of_bb(cdfg, bb_id)
    if state is not None:
        return state

    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') in ('cond', 'body', 'init'):
            result = _resolve_next_state(cdfg, dst, visited)
            if result is not None:
                return result

    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') == 'contains':
            dst_data = cdfg.nodes[dst]
            if dst_data.get('op_name') == 'scf.for':
                bb_cond = dst_data.get('bb_cond')
                if bb_cond is not None:
                    result = _resolve_next_state(cdfg, bb_cond, visited)
                    if result is not None:
                        return result

    return None


def _resolve_exit_state(cdfg: nx.DiGraph, bb_id: int, visited: set = None) -> int | None:
    if visited is None:
        visited = set()
    if bb_id in visited:
        return None
    visited.add(bb_id)

    state = _state_of_bb(cdfg, bb_id)
    if state is not None:
        return state

    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') == 'next':
            result = _resolve_exit_state(cdfg, dst, visited)
            if result is not None:
                return result

    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') == 'contains':
            if cdfg.nodes[dst].get('op_name') == 'done':
                return cdfg.nodes[dst].get('state')

    return None


# ----------------------------------------------------------------
# レジスタ名解決
# ----------------------------------------------------------------
def _build_reg_to_name(reg_map: dict[str, int]) -> dict[int, str]:
    """レジスタ番号 → 変数名のマップを生成する。"""
    from nisc_compiler.passes.code_gen.emitter import _build_reg_to_name as _orig
    return _orig(reg_map)


def _imm_name(idx: int) -> str:
    return f"CONST_{idx}"


# ----------------------------------------------------------------
# FSM中間表現生成
# ----------------------------------------------------------------
def build_fsm_states(cdfg: nx.DiGraph) -> list[FSMState]:
    """CDFGからFSMStateのリストを生成する。"""
    bbs = _get_all_bbs(cdfg)
    cfg_edges = _get_cfg_edges(cdfg)

    transitions: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for src, dst, label in cfg_edges:
        transitions[src].append((dst, label))

    fsm_states: list[FSMState] = []
    processed_states = set()

    for bb_id in bbs:
        last_state = _last_state_of_bb(cdfg, bb_id)
        if last_state is None:
            continue
        if last_state in processed_states:
            continue
        processed_states.add(last_state)

        dsts = transitions.get(bb_id, [])
        if not dsts:
            continue

        # BB内の中間stateの連続遷移
        first_state = _state_of_bb(cdfg, bb_id)
        if first_state is not None and first_state < last_state:
            for s in range(first_state, last_state):
                if s not in processed_states:
                    processed_states.add(s)
                    state = FSMState(state_id=s)
                    state.add_transition(s + 1)
                    fsm_states.append(state)

        state = FSMState(state_id=last_state)

        if len(dsts) == 1:
            dst_bb, label = dsts[0]
            dst_state = _state_of_bb(cdfg, dst_bb)
            if dst_state is not None:
                state.add_transition(dst_state)

        elif len(dsts) == 2:
            body_dst = next((dst for dst, lbl in dsts if lbl in ('body', 'then')), None)
            exit_dst = next((dst for dst, lbl in dsts if lbl in ('exit', 'else')), None)
            back_dst = next((dst for dst, lbl in dsts if lbl == 'back'), None)

            if body_dst is not None and exit_dst is not None:
                body_state = _resolve_next_state(cdfg, body_dst)
                exit_state = _resolve_exit_state(cdfg, exit_dst)
                if body_state is not None:
                    state.add_transition(body_state, condition="true")
                    if exit_state is not None:
                        state.add_transition(exit_state, condition="false")
                    else:
                        state.is_done = True
            elif back_dst is not None:
                back_state = _state_of_bb(cdfg, back_dst)
                if back_state is not None:
                    state.add_transition(back_state)
            else:
                cond_dst = next((dst for dst, lbl in dsts if lbl == 'cond'), None)
                if cond_dst is not None:
                    cond_state = _resolve_next_state(cdfg, cond_dst)
                    if cond_state is not None:
                        state.add_transition(cond_state)

        fsm_states.append(state)

    # doneステート
    done_state_id = _get_done_state(cdfg)
    if done_state_id is not None:
        done_state = FSMState(state_id=done_state_id, is_done=True)
        fsm_states.append(done_state)

    return sorted(fsm_states, key=lambda s: s.state_id)


# ----------------------------------------------------------------
# DEC中間表現生成
# ----------------------------------------------------------------
def build_dec_ops(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    imm_map: dict[str, int],
    spill_map: dict[str, int] = None,
    spill_base_reg: int = -1,
) -> list[DECOp]:
    """CDFGからDECOpのリストを生成する。"""
    reg_to_name = _build_reg_to_name(reg_map)

    def reg_str(ssa: str) -> str:
        r = reg_map.get(ssa, -1)
        if r >= 0:
            return reg_to_name.get(r, f"R{r}")
        if spill_map and ssa in spill_map:
            return "SPILL_BASE"
        return "???"

    def imm_str(ssa: str) -> str:
        i = imm_map.get(ssa, -1)
        if i >= 0:
            return _imm_name(i)
        return "???"

    def is_imm(ssa: str) -> bool:
        return ssa in imm_map

    def operand_str(ssa: str) -> str:
        if is_imm(ssa):
            return imm_str(ssa)
        return reg_str(ssa)

    # stateごとにopノードをグループ化
    state_nodes: dict[int, list[int]] = defaultdict(list)
    for nid, data in cdfg.nodes(data=True):
        state = data.get('state')
        if state is not None and data.get('type') == 'op':
            state_nodes[state].append(nid)

    dec_ops: list[DECOp] = []

    for state in sorted(state_nodes.keys()):
        for nid in state_nodes[state]:
            data = cdfg.nodes[nid]
            op_name = data.get('op_name', '')
            operands = data.get('operands', [])
            results = data.get('results', [])

            kind = OP_KIND_MAP.get(op_name, OpKind.NOP)

            if kind == OpKind.ALU:
                inputs = [operand_str(op) for op in operands[:2]]
                outputs = [reg_str(results[0])] if results else []
                dec_ops.append(DECOp(
                    state_id=state,
                    kind=kind,
                    inputs=inputs,
                    outputs=outputs,
                    comment=op_name,
                ))

            elif kind == OpKind.CMP:
                inputs = [operand_str(op) for op in operands[:2]]
                dec_ops.append(DECOp(
                    state_id=state,
                    kind=kind,
                    inputs=inputs,
                    outputs=[],
                    comment=op_name,
                ))

            elif kind in (OpKind.MUL, OpKind.FMUL, OpKind.FPU):
                inputs = [operand_str(op) for op in operands[:2]]
                outputs = [reg_str(results[0])] if results else []
                dec_ops.append(DECOp(
                    state_id=state,
                    kind=kind,
                    inputs=inputs,
                    outputs=outputs,
                    comment=op_name,
                ))

            elif kind == OpKind.LOAD:
                spill_offset = data.get('spill_offset')
                if spill_offset is not None:
                    outputs = [reg_str(results[0])] if results else []
                    dec_ops.append(DECOp(
                        state_id=state,
                        kind=OpKind.LOAD,
                        inputs=[reg_to_name.get(spill_base_reg, "???"),
                                _imm_name(imm_map.get(f"%spill_offset_{operands[0].lstrip('%')}", -1))],
                        outputs=outputs,
                        comment=f"spill load",
                    ))
                else:
                    outputs = [reg_str(results[0])] if results else []
                    dec_ops.append(DECOp(
                        state_id=state,
                        kind=OpKind.LOAD,
                        inputs=["???"],  # 2次元配列アドレスは未対応
                        outputs=outputs,
                        comment=f"load",
                    ))

            elif kind == OpKind.STORE:
                spill_offset = data.get('spill_offset')
                if spill_offset is not None:
                    src = reg_str(operands[0]) if operands else "???"
                    dec_ops.append(DECOp(
                        state_id=state,
                        kind=OpKind.STORE,
                        inputs=[reg_to_name.get(spill_base_reg, "???"),
                                _imm_name(imm_map.get(f"%spill_offset_{operands[0].lstrip('%')}", -1)),
                                src],
                        mem_wen=True,
                        comment=f"spill store",
                    ))
                else:
                    src = reg_str(operands[0]) if operands else "???"
                    dec_ops.append(DECOp(
                        state_id=state,
                        kind=OpKind.STORE,
                        inputs=["???", src],  # 2次元配列アドレスは未対応
                        mem_wen=True,
                        comment=f"store",
                    ))

    return dec_ops


# ----------------------------------------------------------------
# レジスタ情報生成
# ----------------------------------------------------------------
def build_reg_info(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    imm_map: dict[str, int],
) -> tuple[list[RegInfo], list[RegInfo]]:
    """GPRとIMMのRegInfoリストを生成する。"""
    reg_to_name = _build_reg_to_name(reg_map)

    gpr = [
        RegInfo(reg_id=reg_id, name=name)
        for reg_id, name in sorted(reg_to_name.items())
    ]

    imm_to_value: dict[int, int] = {}
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arith.constant':
            for r in data.get('results', []):
                if r in imm_map:
                    idx = imm_map[r]
                    imm_to_value[idx] = data.get('const_value', 0)

    imm = [
        RegInfo(reg_id=idx, name=_imm_name(idx), is_imm=True, imm_value=val)
        for idx, val in sorted(imm_to_value.items())
    ]

    return gpr, imm


# ----------------------------------------------------------------
# 公開API
# ----------------------------------------------------------------
def build_fsmrep(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    imm_map: dict[str, int],
    package: str = "nisc.program.generated",
    source_file: str = "",
    spill_map: dict[str, int] = None,
    spill_base_reg: int = -1,
) -> FSMDECProgram:
    """CDFGからFSMDECProgramを生成する。"""
    program = FSMDECProgram(package=package, source_file=source_file)

    program.fsm_states = build_fsm_states(cdfg)
    program.dec_ops = build_dec_ops(cdfg, reg_map, imm_map, spill_map, spill_base_reg)
    program.gpr, program.imm = build_reg_info(cdfg, reg_map, imm_map)

    return program