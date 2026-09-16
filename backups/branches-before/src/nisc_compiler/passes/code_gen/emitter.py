"""
emitter.py: CDFG → Chisel FSM/DEC生成

allocator.pyの結果を受けてChiselコードを生成する。

生成するもの:
  VarName:    GPRアドレス定数
  ConstTable: IMMレジスタ値テーブル
  FSM:        状態遷移
  DEC:        各stateの演算器制御信号
"""
from __future__ import annotations
from collections import defaultdict
import networkx as nx


# ----------------------------------------------------------------
# MLIRのop_name → Defs.scalaの定数マッピング
# ----------------------------------------------------------------

# ALU演算コード
ALU_CODE = {
    "arith.addi":  "ALU_ADD",
    "arith.addf":  "ALU_ADD",
    "arith.subi":  "ALU_SUB",
    "arith.subf":  "ALU_SUB",
    "arith.andi":  "ALU_AND",
    "arith.ori":   "ALU_OR",
    "arith.xori":  "ALU_XOR",
    "arith.shli":  "ALU_SLL",
    "arith.shrsi": "ALU_SRA",
    "arith.shrui": "ALU_SRL",
    "nisc.iv_init": "ALU_COPY",  # ← 追加
}

# CMP演算コード（MLIRのpredicate → Defs定数）
CMP_CODE = {
    "slt": "BR_BLT",
    "sle": "BR_BGE",
    "sgt": "BR_BLT",  # 反転
    "sge": "BR_BGE",  # 反転
    "eq":  "BR_BEQ",
    "ne":  "BR_BNE",
    "olt": "BR_BLT",
    "ole": "BR_BGE",
    "ogt": "BR_BLT",
    "oge": "BR_BGE",
    "oeq": "BR_BEQ",
    "one": "BR_BNE",
}

# 演算器名 → writeback信号名（wen, waddr）
WB_SIGNALS = {
    'arith.muli':  ('MUL_RF_WB', 'MUL_RF_WB'),
    'arith.mulf':  ('MUL_RF_WB', 'MUL_RF_WB'),
    'arith.addf':  ('FPU_RF_WB', 'FPU_RF_WB'),
    'arith.subf':  ('FPU_RF_WB', 'FPU_RF_WB'),
    'arith.mulf':  ('MUL_RF_WB', 'MUL_RF_WB'),
}

# MAC演算（mulf→addf）
MAC_OPS = {"mac_f32", "mac_i32"}

# rf_raddr インデックス → muxインデックス（IMMを使う場合）
MUX_IDX = {
    'ALU_IN1': 0,
    'ALU_IN2': 1,
    'CMP_IN1': 2,
    'CMP_IN2': 3,
    'MUL_IN1': 4,
    'MUL_IN2': 5,
}

# rf_raddr インデックス → const_reg_addrポート名
CONST_REG_PORT = {
    'ALU_IN1': 'ALU_IN1_ConstReg',
    'ALU_IN2': 'ALU_IN2_ConstReg',
    'CMP_IN1': 'CMP_IN1_ConstReg',
    'CMP_IN2': 'CMP_IN2_ConstReg',
    'MUL_IN1': 'MUL_IN1_ConstReg',
    'MUL_IN2': 'MUL_IN2_ConstReg',
}

def _reg_name(ssa: str) -> str:
    """SSA変数名をVarName用の定数名に変換する。"""
    # %n → N, %arg_d00 → ARG_D00, %iv_i → IV_I
    name = ssa.lstrip('%')
    name = name.upper()
    name = name.replace('-', '_').replace('.', '_')
    # 数字始まりの場合はVAR_プレフィックス
    if name and name[0].isdigit():
        name = "VAR_" + name
    return name


def _imm_name(idx: int) -> str:
    """IMMレジスタのConstTable定数名を返す。"""
    return f"CONST_{idx}"


def _get_done_state(cdfg: nx.DiGraph) -> int | None:
    """doneノードのstateを返す。"""
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'done':
            return data.get('state')
    return None


def _get_bb_ops(cdfg: nx.DiGraph, bb_id: int) -> list[int]:
    """BBノードに含まれるopノードを返す。"""
    ops = []
    for _, dst, data in cdfg.out_edges(bb_id, data=True):
        if data.get('label') == 'contains' and cdfg.nodes[dst].get('type') == 'op':
            ops.append(dst)
    return ops


def _get_all_bbs(cdfg: nx.DiGraph) -> list[int]:
    return [n for n, d in cdfg.nodes(data=True) if d.get('type') == 'bb']


def _get_cfg_edges(cdfg: nx.DiGraph) -> list[tuple[int, int, str]]:
    """BB間のCFGエッジを返す。(src_bb, dst_bb, label)のリスト。"""
    edges = []
    bbs = set(_get_all_bbs(cdfg))
    for src, dst, data in cdfg.edges(data=True):
        if src in bbs and dst in bbs and data.get('type') == 'ctrl':
            label = data.get('label', '')
            if label not in ('contains',):
                edges.append((src, dst, label))
    return edges


def _state_of_bb(cdfg: nx.DiGraph, bb_id: int) -> int | None:
    """BBの最初のstateを返す。演算がなければNone。"""
    ops = _get_bb_ops(cdfg, bb_id)
    states = [cdfg.nodes[n].get('state') for n in ops if cdfg.nodes[n].get('state') is not None]
    return min(states) if states else None


def _resolve_next_state(cdfg: nx.DiGraph, bb_id: int, visited: set = None) -> int | None:
    """
    BBのstateを返す。演算がない場合はcond/body/initエッジを辿る。
    bb_bodyが空の場合（ネストしたループ）に内側のbb_condへ辿るために使う。
    containsエッジでscf.forを見つけてそのbb_condも辿る。
    """
    if visited is None:
        visited = set()
    if bb_id in visited:
        return None
    visited.add(bb_id)

    state = _state_of_bb(cdfg, bb_id)
    if state is not None:
        return state

    # cond/body/initエッジを辿る
    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') in ('cond', 'body', 'init'):
            result = _resolve_next_state(cdfg, dst, visited)
            if result is not None:
                return result

    # containsエッジでscf.forを見つけてそのbb_condを辿る
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


def _resolve_exit_state(cdfg, bb_id, visited=None):
    if visited is None:
        visited = set()
    if bb_id in visited:
        return None
    visited.add(bb_id)
    state = _state_of_bb(cdfg, bb_id)
    if state is not None:
        return state
    # nextエッジを辿る（bb_exit → bb_init → bb_cond等）
    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') in ('next', 'cond'):  # ← condも辿る
            result = _resolve_exit_state(cdfg, dst, visited)
            if result is not None:
                return result
    # doneノードを最後のフォールバック
    for _, dst, edata in cdfg.out_edges(bb_id, data=True):
        if edata.get('label') == 'contains':
            if cdfg.nodes[dst].get('op_name') == 'done':
                return cdfg.nodes[dst].get('state')
    return None


def _last_state_of_bb(cdfg: nx.DiGraph, bb_id: int) -> int | None:
    """BBの最後のstateを返す。"""
    ops = _get_bb_ops(cdfg, bb_id)
    states = [cdfg.nodes[n].get('state') for n in ops if cdfg.nodes[n].get('state') is not None]
    return max(states) if states else None


# ----------------------------------------------------------------
# VarName生成
# ----------------------------------------------------------------
def _best_ssa_name(ssas: list[str]) -> str:
    def priority(ssa: str) -> int:
        if ssa.startswith('%iv_'):
            return 0  # ループインデックス（最優先）
        if ssa.startswith('%addr_'):
            return 1  # アドレス（高優先）
        if ssa.startswith('%id_') or \
           ssa.startswith('%cmp_for_') or \
           ssa.startswith('%incr_for_') or \
           ssa.startswith('%mul_') or \
           ssa.startswith('%cols_') or \
           ssa.startswith('%init_iv_') or \
           ssa.startswith('%offset_'):
            return 3  # 内部ID・中間変数（優先度最低）
        return 2  # 引数・その他
    return min(ssas, key=priority)


def _build_reg_to_name(reg_map: dict[str, int]) -> dict[int, str]:
    """レジスタ番号 → VarName定数名のマッピングを構築する。"""
    # レジスタ番号 → SSA変数名リスト
    reg_to_ssas: dict[int, list[str]] = {}
    for ssa, reg in reg_map.items():
        if reg >= 0:
            if reg not in reg_to_ssas:
                reg_to_ssas[reg] = []
            reg_to_ssas[reg].append(ssa)

    # 各レジスタに最もわかりやすい名前を選ぶ
    reg_to_name: dict[int, str] = {}
    for reg, ssas in reg_to_ssas.items():
        best = _best_ssa_name(ssas)
        reg_to_name[reg] = _reg_name(best)

    return reg_to_name


def gen_varname(reg_map: dict[str, int], package: str) -> str:
    """VarNameオブジェクトを生成する。"""
    reg_to_name = _build_reg_to_name(reg_map)

    lines = []
    lines.append(f"package {package}")
    lines.append("")
    lines.append("import chisel3._")
    lines.append("import nisc.Defs._")
    lines.append("")
    lines.append("object VarName {")
    for reg in sorted(reg_to_name.keys()):
        name = reg_to_name[reg]
        lines.append(f"    val {name:<20} = {reg}.U(RF_LEN_BIT.W)")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------
# ConstTable生成
# ----------------------------------------------------------------
def gen_consttable(imm_map: dict[str, int], cdfg: nx.DiGraph, package: str) -> str:
    """ConstTableオブジェクトを生成する。"""
    # IMMレジスタ番号 → 実際の定数値
    imm_to_value: dict[int, int] = {}
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arith.constant':
            for r in data.get('results', []):
                if r in imm_map:
                    idx = imm_map[r]
                    # mlir_typとoperandsから値を取得
                    # operandsは空なのでノードのop情報から取得できない
                    # → lowering時に値を保存しておく必要がある
                    # 今は仮に0を入れる
                    imm_to_value[idx] = data.get('const_value', 0)

    lines = []
    lines.append(f"package {package}")
    lines.append("")
    lines.append("import chisel3._")
    lines.append("import nisc.Defs._")
    lines.append("")
    lines.append("object ConstTable {")

    # 定数名の定義
    for idx in sorted(imm_to_value.keys()):
        name = _imm_name(idx)
        lines.append(f"    val {name:<20} = {idx}.U(ConstReg_LEN_BIT.W)")

    lines.append("")
    lines.append("    private val table = Map(")
    for idx, val in sorted(imm_to_value.items()):
        lines.append(f"        {idx} -> {val},")
    lines.append("    )")
    lines.append("")
    lines.append("    val constInit = (0 until ConstReg_LEN).map(i => table.getOrElse(i, 0))")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------
# FSM生成
# ----------------------------------------------------------------
def gen_fsm(cdfg: nx.DiGraph, package: str) -> str:
    """FSMクラスを生成する。"""
    lines = []
    lines.append(f"package {package}")
    lines.append("")
    lines.append("import chisel3._")
    lines.append("import chisel3.util._")
    lines.append("import nisc._")
    lines.append("import nisc.Defs._")
    lines.append("")
    lines.append("class FSM extends FsmBase {")
    lines.append("    switch(state) {")

    # BBごとのCFGエッジから状態遷移を生成
    bbs = _get_all_bbs(cdfg)
    bb_set = set(bbs)
    cfg_edges = _get_cfg_edges(cdfg)

    # src_bb → [(dst_bb, label)] のマッピング
    transitions: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for src, dst, label in cfg_edges:
        transitions[src].append((dst, label))

    # 各BBの最終stateで遷移を生成
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
            # dsts=[]の場合はdoneへ遷移
            done_state = _get_done_state(cdfg)
            if done_state is not None:
                lines.append(f"        is({last_state}.U) {{")
                lines.append(f"            state := {done_state}.U")
                lines.append(f"        }}")
            continue

        # cmpノードがあれば条件分岐
        ops = _get_bb_ops(cdfg, bb_id)
        has_cmp = any(
            cdfg.nodes[n].get('op_name') in ('arith.cmpi', 'arith.cmpf')
            for n in ops
        )

        # BB内の中間stateの連続遷移を生成（state N → state N+1）
        first_state = _state_of_bb(cdfg, bb_id)
        if first_state is not None and first_state < last_state:
            for s in range(first_state, last_state):
                if s not in processed_states:
                    processed_states.add(s)
                    lines.append(f"        is({s}.U) {{")
                    lines.append(f"            state := {s+1}.U")
                    lines.append(f"        }}")

        lines.append(f"        is({last_state}.U) {{")

        if len(dsts) == 1:
            # 無条件遷移（ループバック含む）
            dst_bb, label = dsts[0]
            dst_state = _state_of_bb(cdfg, dst_bb)
            if dst_state is not None:
                lines.append(f"            state := {dst_state}.U")
        elif len(dsts) == 2:
            body_dst = next((dst for dst, lbl in dsts if lbl in ('body', 'then')), None)
            exit_dst = next((dst for dst, lbl in dsts if lbl in ('exit', 'else')), None)
            back_dst = next((dst for dst, lbl in dsts if lbl == 'back'), None)

            if body_dst is not None and exit_dst is not None:
                # 条件分岐（body/exit, then/else）
                body_state = _resolve_next_state(cdfg, body_dst)
                exit_state = _resolve_exit_state(cdfg, exit_dst)
                if body_state is not None:
                    lines.append(f"            when(io.dp_cmp_out === true.B) {{")
                    lines.append(f"                state := {body_state}.U")
                    lines.append(f"            }}.otherwise {{")
                    if exit_state is not None:
                        lines.append(f"                state := {exit_state}.U")
                    else:
                        lines.append(f"                io.done := true.B")
                    lines.append(f"            }}")
            elif back_dst is not None:
                # ループバック
                back_state = _state_of_bb(cdfg, back_dst)
                if back_state is not None:
                    lines.append(f"            state := {back_state}.U")
            else:
                # incr + cond パターン
                cond_dst = next((dst for dst, lbl in dsts if lbl == 'cond'), None)
                if cond_dst is not None:
                    cond_state = _resolve_next_state(cdfg, cond_dst)
                    if cond_state is not None:
                        lines.append(f"            state := {cond_state}.U")
        lines.append("        }")

    # doneステートを追加
    done_state = _get_done_state(cdfg)
    if done_state is not None:
        lines.append(f"        is({done_state}.U) {{")
        lines.append(f"            io.done := true.B")
        lines.append(f"            state := {done_state}.U")
        lines.append(f"        }}")

    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------------
# DEC生成
# ----------------------------------------------------------------

def gen_init_map(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    spill_map: dict[str, int] = None,
    spill_base_addr: int = 240,
) -> str:
    """
    SRAMの初期化マップを生成する。

    引数はSRAM[0]から順番に配置する。
    スピルエリアはSRAM[240]から。

    Returns:
        init.txtの内容
    """
    lines = []
    lines.append("# SRAM initialization map")
    lines.append("# Format: SRAM[addr] = variable_name (type)")
    lines.append("")

    # argノードを順番に処理
    arg_addr = 0
    arg_map = {}  # SSA変数名 → SRAMアドレス
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arg':
            for ssa in data.get('results', []):
                lines.append(f"SRAM[{arg_addr}] = {ssa}  # argument")
                arg_map[ssa] = arg_addr
                arg_addr += 1

    lines.append("")
    lines.append("# Spill area")
    if spill_map:
        for ssa, offset in spill_map.items():
            addr = spill_base_addr + offset
            lines.append(f"SRAM[{addr}] = {ssa}  # spill")
    else:
        lines.append("# (no spills)")

    lines.append("")
    lines.append("# Prologue: load arguments into GPR")
    reg_to_name = _build_reg_to_name(reg_map)
    for ssa, addr in arg_map.items():
        r = reg_map.get(ssa, -1)
        reg_name = reg_to_name.get(r, f"R{r}") if r >= 0 else "SPILL"
        lines.append(f"# LOAD SRAM[{addr}] → {reg_name}  ({ssa})")

    return "\n".join(lines)


def gen_prologue_dec(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    imm_map: dict[str, int],
    spill_map: dict[str, int] = None,
    spill_base_reg: int = -1,
    prologue_start_state: int = 0,
) -> list[str]:
    """
    プロローグのDEC（引数ロード）を生成する。

    各引数をSRAM[0], SRAM[1], ...からGPRにロードする。

    Returns:
        DECのstateブロックのリスト
    """
    lines = []
    reg_to_name = _build_reg_to_name(reg_map)

    # IMMに固定アドレスを割り当てる
    # arg_addr_0, arg_addr_1, ... としてConstTableに追加
    arg_nodes = []
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arg':
            for ssa in data.get('results', []):
                arg_nodes.append((ssa, data))

    for i, (ssa, data) in enumerate(arg_nodes):
        state = prologue_start_state + i
        r = reg_map.get(ssa, -1)
        if r >= 0:
            reg_name = reg_to_name.get(r, f"R{r}")
            lines.append(f"        is({state}.U) {{")
            lines.append(f"            // prologue: load arg {ssa} from SRAM[{i}]")
            lines.append(f"            io.dp.rf_raddr(MEM_WADDR) := ARG_ADDR_{i}")
            lines.append(f"            io.dp.rf_wen(MEM_RF_WB)   := true.B")
            lines.append(f"            io.dp.rf_waddr(MEM_RF_WB) := {reg_name}")
            lines.append(f"        }}")
        elif spill_map and ssa in spill_map:
            # スピルされた引数はプロローグでスピルエリアに直接書き込む
            # → テストベンチが直接スピルエリアに書いておく方が簡単
            lines.append(f"        is({state}.U) {{")
            lines.append(f"            // prologue: arg {ssa} is spilled to SRAM[{240 + spill_map[ssa]}]")
            lines.append(f"            // (written by testbench directly)")
            lines.append(f"        }}")

    return lines

def gen_dec(cdfg: nx.DiGraph, reg_map: dict[str, int],
            imm_map: dict[str, int], package: str,
            spill_map: dict[str, int] = None,
            spill_base_reg: int = -1,
            operators: list = None) -> str:
    
    op_map = {op.name: op for op in (operators or [])}
    """DECクラスを生成する。"""
    lines = []
    lines.append(f"package {package}")
    lines.append("")
    lines.append("import chisel3._")
    lines.append("import chisel3.util._")
    lines.append("import nisc._")
    lines.append("import nisc.Defs._")
    lines.append(f"import {package}.VarName._")
    lines.append(f"import {package}.ConstTable._")
    lines.append("")
    lines.append("class DEC extends DecBase {")
    lines.append("    switch(io.state) {")

    # stateごとにopノードをグループ化
    state_nodes: dict[int, list[int]] = defaultdict(list)
    for nid, data in cdfg.nodes(data=True):
        state = data.get('state')
        if state is not None and data.get('type') == 'op':
            state_nodes[state].append(nid)

    def get_reg(ssa: str) -> int:
        return reg_map.get(ssa, -1)

    def get_imm(ssa: str) -> int:
        return imm_map.get(ssa, -1)

    reg_to_name = _build_reg_to_name(reg_map)

    def reg_str(ssa: str) -> str:
        r = get_reg(ssa)
        if r >= 0:
            return reg_to_name.get(r, f"R{r}")
        # スピルされた変数はspill_base_regを参照
        if spill_map and ssa in spill_map:
            return f"SPILL_BASE"  # emitter上の仮の名前
        return "???"

    spill_base_name = reg_to_name.get(spill_base_reg, f"R{spill_base_reg}") if spill_base_reg >= 0 else "???"

    def imm_str(ssa: str) -> str:
        i = get_imm(ssa)
        if i >= 0:
            return _imm_name(i)
        return "???"

    def is_imm(ssa: str) -> bool:
        return ssa in imm_map

    # latency>1の演算器のwriteback待ちを管理
    # {state: [('mul', result_ssa), ('fmul', result_ssa), ...]}
    pending_writeback: dict = defaultdict(list)

    for state in sorted(state_nodes.keys()):
        lines.append(f"        is({state}.U) {{")
        # pending_writebackの出力
        for wb in pending_writeback.get(state, []):
            r = get_reg(wb['result_ssa'])
            if r >= 0:
                wen   = wb['wen_signal']
                waddr = wb['waddr_signal']
                lines.append(f"            io.dp.rf_wen({wen})   := true.B")
                lines.append(f"            io.dp.rf_waddr({waddr}) := {reg_to_name.get(r, f'R{r}')}")
        for nid in state_nodes[state]:
            data = cdfg.nodes[nid]
            op_name = data.get('op_name', '')
            assigned_op = data.get('assigned_op', op_name)
            operands = data.get('operands', [])
            results = data.get('results', [])
            operand_regs = data.get('operand_regs', [])
            operand_types = data.get('operand_reg_types', [])
            result_regs = data.get('result_regs', [])
            result_types = data.get('result_reg_types', [])

            # ALU演算
            if op_name in ALU_CODE:
                alu_code = ALU_CODE[op_name]
                lines.append(f"            io.dp.alu_code           := {alu_code}")

                # 入力1（ALU_IN1）: 常にGPR
                if len(operands) > 0:
                    op0 = operands[0]
                    if is_imm(op0):
                        lines.append(f"            io.dp.mux({MUX_IDX['ALU_IN1']})                       := true.B")
                        lines.append(f"            io.const_reg_addr({CONST_REG_PORT['ALU_IN1']}) := {imm_str(op0)}")
                    else:
                        lines.append(f"            io.dp.rf_raddr(ALU_IN1)  := {reg_str(op0)}")

                # 入力2（ALU_IN2）: GPRかIMM
                if len(operands) > 1:
                    op1 = operands[1]
                    if is_imm(op1):
                        lines.append(f"            io.dp.mux({MUX_IDX['ALU_IN2']})                       := true.B")
                        lines.append(f"            io.const_reg_addr({CONST_REG_PORT['ALU_IN2']}) := {imm_str(op1)}")
                    else:
                        lines.append(f"            io.dp.rf_raddr(ALU_IN2)  := {reg_str(op1)}")

                # 出力（ALU_RF_WB）
                if results:
                    res = results[0]
                    r = get_reg(res)
                    if r >= 0:
                        lines.append(f"            io.dp.rf_wen(ALU_RF_WB)  := true.B")
                        lines.append(f"            io.dp.rf_waddr(ALU_RF_WB) := {reg_str(res)}")

            # CMP演算
            elif op_name in ('arith.cmpi', 'arith.cmpf'):
                # predicateはop_nameから取れないのでデフォルトでBR_BLTを使用
                # TODO: predicateをCDFGに保存する
                lines.append(f"            io.dp.cmp_code           := BR_BLT")

                if len(operands) > 0:
                    op0 = operands[0]
                    if is_imm(op0):
                        lines.append(f"            io.dp.mux({MUX_IDX['CMP_IN1']})                       := true.B")
                        lines.append(f"            io.const_reg_addr({CONST_REG_PORT['CMP_IN1']}) := {imm_str(op0)}")
                    else:
                        lines.append(f"            io.dp.rf_raddr(CMP_IN1)  := {reg_str(op0)}")

                if len(operands) > 1:
                    op1 = operands[1]
                    if is_imm(op1):
                        lines.append(f"            io.dp.mux({MUX_IDX['CMP_IN2']})                       := true.B")
                        lines.append(f"            io.const_reg_addr({CONST_REG_PORT['CMP_IN2']}) := {imm_str(op1)}")
                    else:
                        lines.append(f"            io.dp.rf_raddr(CMP_IN2)  := {reg_str(op1)}")

            # MUL演算
            elif op_name in ('arith.muli', 'arith.mulf'):
                if len(operands) > 0:
                    op0 = operands[0]
                    if is_imm(op0):
                        lines.append(f"            io.dp.mux({MUX_IDX['MUL_IN1']})                       := true.B")
                        lines.append(f"            io.const_reg_addr({CONST_REG_PORT['MUL_IN1']}) := {imm_str(op0)}")
                    else:
                        lines.append(f"            io.dp.rf_raddr(MUL_IN1)  := {reg_str(op0)}")
                if len(operands) > 1:
                    op1 = operands[1]
                    if is_imm(op1):
                        lines.append(f"            io.dp.mux({MUX_IDX['MUL_IN2']})                       := true.B")
                        lines.append(f"            io.const_reg_addr({CONST_REG_PORT['MUL_IN2']}) := {imm_str(op1)}")
                    else:
                        lines.append(f"            io.dp.rf_raddr(MUL_IN2)  := {reg_str(op1)}")
                # MUL_RF_WBはlatency後のstateで出力
                if results:
                    res = results[0]
                    operator = op_map.get(assigned_op)
                    latency = operator.latency if operator else 1
                    wb_state = state + latency - 1
                    wen, waddr = WB_SIGNALS.get(op_name, ('MUL_RF_WB', 'MUL_RF_WB'))
                    pending_writeback[wb_state].append({
                        'wen_signal':  wen,
                        'waddr_signal': waddr,
                        'result_ssa':  res,
                    })

            # math演算（sqrt, sin, cosなど）
            elif op_name.startswith('math.'):
                func_name = op_name.split('.')[1].upper()
                lines.append(f"            // {op_name}")
                if len(operands) > 0:
                    op0 = operands[0]
                    lines.append(f"            io.dp.rf_raddr(ALU_IN1)  := {reg_str(op0)}")
                if results:
                    res = results[0]
                    r = get_reg(res)
                    if r >= 0:
                        lines.append(f"            io.dp.rf_wen(ALU_RF_WB)  := true.B")
                        lines.append(f"            io.dp.rf_waddr(ALU_RF_WB) := {reg_str(res)}")

            # メモリLOAD: SRAM[addr] → GPR
            elif op_name == 'memref.load':
                spill_offset = data.get('spill_offset')
                if spill_offset is not None:
                    # スピルLOAD: ALU(r_spill_base + imm_offset) → SRAM_ADDR
                    #             SRAM[ALU_OUT] → GPR
                    # IMMにスピルオフセット値が入っている
                    imm_key = f"%spill_offset_{operands[0].lstrip('%')}" if operands else None
                    imm_idx = imm_map.get(imm_key, -1) if imm_key else -1
                    imm_off_name = _imm_name(imm_idx) if imm_idx >= 0 else f"SPILL_OFF_{spill_offset}"
                    lines.append(f"            // spill load: SRAM[{spill_base_name} + {spill_offset}]")
                    lines.append(f"            io.dp.alu_code           := ALU_ADD")
                    lines.append(f"            io.dp.rf_raddr(ALU_IN1)  := {spill_base_name}")
                    lines.append(f"            io.dp.rf_raddr(ALU_IN2)  := {imm_off_name}")
                    # 書き込み先: 使用ノードのオペランドレジスタ
                    for _, use_nid, edata in cdfg.out_edges(nid, data=True):
                        if edata.get('type') == 'data':
                            use_data = cdfg.nodes[use_nid]
                            use_operand_regs = use_data.get('operand_regs', [])
                            use_operands = use_data.get('operands', [])
                            reload_ssa = edata.get('ssa', '')
                            for i, op in enumerate(use_operands):
                                if op == reload_ssa and i < len(use_operand_regs):
                                    dest_reg = use_operand_regs[i]
                                    if dest_reg >= 0:
                                        lines.append(f"            io.dp.rf_wen(MEM_RF_WB)   := true.B")
                                        lines.append(f"            io.dp.rf_waddr(MEM_RF_WB) := {reg_to_name.get(dest_reg, f'R{dest_reg}')}")
                                    break
                else:
                    # 通常LOAD: SRAM[addr] → GPR
                    if len(operands) == 1:
                        # アドレス計算済み（lowering.pyで計算）
                        lines.append(f"            io.dp.rf_raddr(MEM_WADDR) := {reg_str(operands[0])}")
                    elif len(operands) > 1:
                        # 旧来の形式
                        lines.append(f"            // load base={reg_str(operands[0])}")
                        lines.append(f"            io.dp.rf_raddr(MEM_WADDR) := {reg_str(operands[1])}")
                    if results:
                        res = results[0]
                        r = get_reg(res)
                        if r >= 0:
                            lines.append(f"            io.dp.rf_wen(MEM_RF_WB)   := true.B")
                            lines.append(f"            io.dp.rf_waddr(MEM_RF_WB) := {reg_to_name.get(r, f'R{r}')}")

            # メモリSTORE: GPR → SRAM[addr]
            elif op_name == 'memref.store':
                spill_offset = data.get('spill_offset')
                if spill_offset is not None:
                    # スピルSTORE: ALU(r_spill_base + imm_offset) → SRAM_ADDR
                    #              MEM_WDATA → SRAM[ALU_OUT]
                    if len(operands) > 0:
                        src_op = operands[0]
                        src_reg = get_reg(src_op)
                        src_name = reg_to_name.get(src_reg, f'R{src_reg}') if src_reg >= 0 else '???'
                        imm_key = f"%spill_offset_{src_op.lstrip('%')}" if src_op else None
                        imm_idx = imm_map.get(imm_key, -1) if imm_key else -1
                        imm_off_name = _imm_name(imm_idx) if imm_idx >= 0 else f"SPILL_OFF_{spill_offset}"
                        lines.append(f"            // spill store: {src_name} → SRAM[{spill_base_name} + {spill_offset}]")
                        lines.append(f"            io.dp.alu_code           := ALU_ADD")
                        lines.append(f"            io.dp.rf_raddr(ALU_IN1)  := {spill_base_name}")
                        lines.append(f"            io.dp.rf_raddr(ALU_IN2)  := {imm_off_name}")
                        lines.append(f"            io.dp.rf_raddr(MEM_WDATA) := {src_name}")
                        lines.append(f"            io.dp.mem_wen             := true.B")
                else:
                    # 通常STORE: GPR → SRAM[base + index]
                    if len(operands) > 0:
                        src_op = operands[0]
                        if is_imm(src_op):
                            lines.append(f"            io.dp.rf_raddr(MEM_WDATA) := {imm_str(src_op)}")
                        else:
                            lines.append(f"            io.dp.rf_raddr(MEM_WDATA) := {reg_str(src_op)}")
                    if len(operands) > 1:
                        addr_op = operands[1]
                        lines.append(f"            io.dp.rf_raddr(MEM_WADDR) := {reg_str(addr_op)}")
                    lines.append(f"            io.dp.mem_wen             := true.B")
        lines.append("        }")

    # state_nodesにないpending_writebackのstateも出力
    for wb_state in sorted(pending_writeback.keys()):
        if wb_state not in state_nodes:
            lines.append(f"        is({wb_state}.U) {{")
            for wb in pending_writeback[wb_state]:
                r = get_reg(wb['result_ssa'])
                if r >= 0:
                    wen   = wb['wen_signal']
                    waddr = wb['waddr_signal']
                    lines.append(f"            io.dp.rf_wen({wen})   := true.B")
                    lines.append(f"            io.dp.rf_waddr({waddr}) := {reg_to_name.get(r, f'R{r}')}")
            lines.append("        }")
    lines.append("    }")
    lines.append("}\n")


    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------
# 公開API
# ----------------------------------------------------------------
def emit(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    imm_map: dict[str, int],
    package: str = "nisc.program.generated",
    spill_map: dict[str, int] = None,
    spill_base_reg: int = -1,
    operators: list = None,
) -> str:
    """
    CDFGからChiselコードを生成する。
    全オブジェクト/クラスを1つのProgram.scalaにまとめて返す。
    """
    lines = []

    # パッケージ宣言とimport
    lines.append(f"package {package}")
    lines.append("")
    lines.append("import chisel3._")
    lines.append("import chisel3.util._")
    lines.append("import nisc._")
    lines.append("import nisc.Defs._")
    lines.append("")

    # VarName
    reg_to_name = _build_reg_to_name(reg_map)

    lines.append("object VarName {")
    for reg in sorted(reg_to_name.keys()):
        name = reg_to_name[reg]
        lines.append(f"    val {name:<20} = {reg}.U(RF_LEN_BIT.W)")
    lines.append("}")
    lines.append("")

    # ConstTable
    imm_to_value: dict[int, int] = {}
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arith.constant':
            for r in data.get('results', []):
                if r in imm_map:
                    idx = imm_map[r]
                    imm_to_value[idx] = data.get('const_value', 0)

    lines.append("object ConstTable {")
    for idx in sorted(imm_to_value.keys()):
        name = _imm_name(idx)
        lines.append(f"    val {name:<20} = {idx}.U(ConstReg_LEN_BIT.W)")
    lines.append("")
    lines.append("    private val table = Map(")
    for idx, val in sorted(imm_to_value.items()):
        lines.append(f"        {idx} -> {val},")
    lines.append("    )")
    lines.append("")
    lines.append("    val constInit = (0 until ConstReg_LEN).map(i => table.getOrElse(i, 0))")
    lines.append("}")
    lines.append("")

    # import VarName/ConstTable
    lines.append(f"import VarName._")
    lines.append(f"import ConstTable._")
    lines.append("")

    # FSM
    fsm = gen_fsm(cdfg, package)
    # パッケージ宣言とimportを除いた部分だけ取得
    fsm_body = "\n".join(
        l for l in fsm.split("\n")
        if not l.startswith("package") and not l.startswith("import")
    ).strip()
    # state番号を全て+1（state=0を待機ステートにするため）
    import re as _re
    def _shift_states(text):
        def _rep(m):
            prefix = m.group(1)
            num = int(m.group(2)) + 1
            return f"{prefix}{num}.U"
        text = _re.sub(r"(is\()([0-9]+)(\.U\))", lambda m: f"is({int(m.group(2))+1}.U)", text)
        text = _re.sub(r"(state := )([0-9]+)(\.U)", lambda m: f"state := {int(m.group(2))+1}.U", text)
        return text
    fsm_body = _shift_states(fsm_body)
    # shift後にstate=0の待機ステートを追加
    fsm_body = fsm_body.replace(
        "switch(state) {",
        "switch(state) {\n        is(0.U) {\n            state := 1.U\n        }",
        1
    )
    lines.append(fsm_body)
    lines.append("")

    # DEC
    dec = gen_dec(cdfg, reg_map, imm_map, package, spill_map=spill_map, spill_base_reg=spill_base_reg,operators=operators)
    dec_body = "\n".join(
        l for l in dec.split("\n")
        if not l.startswith("package") and not l.startswith("import")
    ).strip()
    dec_body = _shift_states(dec_body)
    lines.append(dec_body)

    # init.txtを生成
    init_txt = gen_init_map(cdfg, reg_map, spill_map=spill_map)

    return "\n".join(lines), init_txt


def print_emit(content: str, init_txt: str = None):
    """生成されたChiselコードを表示する。"""
    print("=== Program.scala ===")
    print(content)
    if init_txt:
        print()
        print("=== init.txt ===")
        print(init_txt)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from parser import parse_c
    from lowering import lower_mlir
    from dp import load_dp
    from matcher import match_all
    from scheduler import schedule
    from allocator import allocate

    if len(sys.argv) < 2:
        print("使い方: python emitter.py <Cファイル>")
        sys.exit(1)

    c_src = open(sys.argv[1]).read()
    mlir_text = parse_c(c_src)
    cdfg = lower_mlir(mlir_text)

    operators = load_dp(mul=True, cmp=True, fpu=True, fmul=True,
                        mac_i=True, mac_f=True, sqrt=True)
    cdfg, results, unmatched = match_all(cdfg, operators)
    cdfg = schedule(cdfg, operators)
    cdfg, reg_map, imm_map = allocate(cdfg, num_registers=32)

    program = emit(cdfg, reg_map, imm_map)
    print_emit(program)