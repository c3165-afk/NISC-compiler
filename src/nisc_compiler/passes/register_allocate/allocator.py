"""
allocator.py: ライフタイム解析によるレジスタ割り当て

手順:
  1. 各SSA変数の生存区間（定義state〜最終使用state）を計算
  2. 干渉グラフを構築（同時に生存している変数はエッジで繋ぐ）
  3. グラフ彩色でレジスタ番号を割り当て（最小レジスタ数）
  4. ループ変数（block_arg ↔ yield）は同じレジスタに統合

CDFGノードに追加される属性:
  result_regs: list[int]  結果SSA変数のレジスタ番号リスト
  operand_regs: list[int] オペランドSSA変数のレジスタ番号リスト
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx


class AllocationError(Exception):
    pass


@dataclass
class LiveRange:
    """SSA変数の生存区間。"""
    ssa: str           # SSA変数名
    def_state: int     # 定義されたstate
    last_use: int      # 最後に使われたstate
    reg: int = -1      # 割り当てられたレジスタ番号（-1は未割り当て）

    def overlaps(self, other: 'LiveRange') -> bool:
        """生存区間が重なるか判定。"""
        return not (self.last_use < other.def_state or
                    other.last_use < self.def_state)


def _get_all_op_nodes(cdfg: nx.DiGraph) -> list[int]:
    """演算ノード（type=op または type=arg）を全て返す。"""
    return [n for n, d in cdfg.nodes(data=True)
            if d.get('type') in ('op', 'arg', 'ctrl')]


def _max_state(cdfg: nx.DiGraph) -> int:
    """CDFGの全opノードの最大stateを返す。"""
    all_states = [
        d.get('state') for _, d in cdfg.nodes(data=True)
        if d.get('state') is not None
    ]
    return max(all_states) if all_states else 0


def _resolve_state(cdfg: nx.DiGraph, nid: int, ssa_uses: dict[str, list[int]],
                   visited: set = None) -> int:
    """
    ノードのstateを返す。state=Noneの場合はそのノードの結果を
    使うopノードの最小stateを返す（CTRL_OPS対応）。
    使用先がない終端ノード（func.returnなど）は全opノードの最大stateを返す。
    """
    if visited is None:
        visited = set()
    if nid in visited:
        return 0
    visited.add(nid)

    state = cdfg.nodes[nid].get('state')
    if state is not None:
        return state

    # state=Noneの場合: このノードの結果を使うopノードのstateを探す
    results = cdfg.nodes[nid].get('results', [])
    use_states = []
    for r in results:
        for use_nid in ssa_uses.get(r, []):
            s = cdfg.nodes[use_nid].get('state')
            if s is not None:
                use_states.append(s)
            else:
                s = _resolve_state(cdfg, use_nid, ssa_uses, visited)
                use_states.append(s)

    if use_states:
        return min(use_states)

    # 使用先がない終端ノード（func.returnなど）
    # → 全opノードの最大stateを返す
    return _max_state(cdfg)


def _compute_live_ranges(cdfg: nx.DiGraph) -> dict[str, LiveRange]:
    """
    各SSA変数の生存区間を計算する。

    定義state: そのSSA変数を出力するノードのstate
               state=Noneの場合はそのSSA変数を使う最初のopノードのstate
    最終使用state: そのSSA変数を入力とするノードの最大state
    """
    # SSA変数名 → 定義ノードID
    ssa_def: dict[str, int] = {}
    # SSA変数名 → 使用ノードIDのリスト
    ssa_uses: dict[str, list[int]] = defaultdict(list)

    for nid, data in cdfg.nodes(data=True):
        for r in data.get('results', []):
            ssa_def[r] = nid
        for op in data.get('operands', []):
            ssa_uses[op].append(nid)

    live_ranges: dict[str, LiveRange] = {}

    for ssa, def_nid in ssa_def.items():
        op_name = cdfg.nodes[def_nid].get('op_name', '')

        # arith.index_castでindex型の結果はNISCではハードウェア配線なのでスキップ
        # ただしindex→i32の変換結果（%iv_i_i32など）はGPRが必要なのでスキップしない
        if op_name == 'arith.index_cast':
            # 結果の型がindexならスキップ（i32→index変換）
            # 結果の型がi32ならスキップしない（index→i32変換）
            result_typ = cdfg.nodes[def_nid].get('mlir_typ', '')
            if result_typ == 'index' or result_typ.startswith('index'):
                continue

        # スピルノード（memref.load/store）はライフタイム計算から除外
        if op_name in ('memref.load', 'memref.store') and (
            cdfg.nodes[def_nid].get('spill_addr') is not None or
            cdfg.nodes[def_nid].get('spill_offset') is not None
        ):
            continue

        # %scratch_/%reload_/%spill_load_で始まるSSA変数はスピルのreload変数なので除外
        if (ssa.startswith('%scratch_') or ssa.startswith('%reload_') or
                ssa.startswith('%spill_load_') or ssa.startswith('%spill_store_')):
            continue

        # arith.constantはIMMに入るのでGPRのライフタイムから除外
        if op_name == 'arith.constant':
            continue

        # stateを解決（CTRL_OPSはstate=Noneなので使用先から解決）
        def_state = _resolve_state(cdfg, def_nid, ssa_uses, set())

        # 使用ノードのstateの最大値
        use_states = []
        for use_nid in ssa_uses.get(ssa, []):
            s = cdfg.nodes[use_nid].get('state')
            if s is not None:
                use_states.append(s)
            else:
                # CTRL_OPSの場合はその使用先から解決
                s = _resolve_state(cdfg, use_nid, ssa_uses, set())
                use_states.append(s)

        if not use_states:
            last_use = def_state
        else:
            last_use = max(use_states)

        live_ranges[ssa] = LiveRange(
            ssa=ssa,
            def_state=def_state,
            last_use=last_use,
        )

    return live_ranges


def _unify_loop_vars(
    cdfg: nx.DiGraph,
    live_ranges: dict[str, LiveRange],
) -> dict[str, str]:
    """
    ループ変数の統合マッピングを返す。

    scf.while/forのblock_argとyieldの値は
    本質的に同じレジスタなので統合する。

    returns: {エイリアスSSA: 正規SSA} のマッピング
    """
    alias: dict[str, str] = {}

    for nid, data in cdfg.nodes(data=True):
        op_name = data.get('op_name', '')

        # scf.yieldのオペランドとblock_argを対応付ける
        if op_name == 'scf.yield':
            # yieldのオペランド = 次イテレーションのblock_arg
            # CDFGのctrlエッジを辿ってblock_argを探す
            yield_operands = data.get('operands', [])

            # yieldの親ctrl（scf.for/while）を探す
            for src, _, edata in cdfg.in_edges(nid, data=True):
                if edata.get('type') == 'ctrl':
                    ctrl_nid = src
                    # ctrl → block_arg のエッジを探す
                    block_args = []
                    for _, dst, edata2 in cdfg.out_edges(ctrl_nid, data=True):
                        if (edata2.get('type') == 'ctrl' and
                                cdfg.nodes[dst].get('op_name') == 'block_arg'):
                            block_args.extend(cdfg.nodes[dst].get('results', []))

                    # yieldのi番目のオペランドとblock_argのi番目を統合
                    # block_argsの末尾len(yield_operands)個がiter_args
                    # （先頭はループカウンタ%iv_iなので除外）
                    iter_block_args = block_args[-len(yield_operands):]
                    for i, (yop, barg) in enumerate(zip(yield_operands, iter_block_args)):
                        if yop in live_ranges and barg in live_ranges:
                            # yieldのオペランドをblock_argのエイリアスにする
                            # block_argを正規SSAとして扱う
                            alias[yop] = barg

    # %incr_for_で始まる変数（ループカウンタのインクリメント）を
    # 対応するblock_arg（%iv_i）のエイリアスにする
    for nid, data in cdfg.nodes(data=True):
        for ssa in data.get('results', []):
            if ssa.startswith('%incr_for_') and ssa in live_ranges:
                # このノードのオペランドからblock_argを探す
                for op in data.get('operands', []):
                    for nid2, data2 in cdfg.nodes(data=True):
                        if (data2.get('op_name') == 'block_arg' and
                                op in data2.get('results', []) and
                                op in live_ranges):
                            alias[ssa] = op
                            break

    return alias


def _build_interference_graph(
    live_ranges: dict[str, LiveRange],
    alias: dict[str, str],
) -> nx.Graph:
    """
    干渉グラフを構築する。
    同時に生存している変数間にエッジを追加する。
    """
    # エイリアスを正規化
    def canonical(ssa: str) -> str:
        return alias.get(ssa, ssa)

    vars_list = [ssa for ssa in live_ranges if ssa not in alias]
    G = nx.Graph()
    G.add_nodes_from(vars_list)

    for i, ssa1 in enumerate(vars_list):
        for ssa2 in vars_list[i+1:]:
            lr1 = live_ranges[ssa1]
            lr2 = live_ranges[ssa2]
            if lr1.overlaps(lr2):
                G.add_edge(ssa1, ssa2)

    return G


def _graph_color(
    G: nx.Graph,
    num_registers: int,
    live_ranges: dict[str, "LiveRange"] = None,
) -> tuple[dict[str, int], list[str]]:
    """
    干渉グラフをグリーディ彩色してレジスタ番号を割り当てる。

    レジスタが足りない場合はスピル対象を選んでSRAMに追い出す。
    スピル対象: 生存区間が最も長い変数

    Returns:
        (coloring, spilled)
        coloring: SSA変数名 → レジスタ番号
        spilled:  スピルされたSSA変数名のリスト
    """
    coloring = {}
    spilled = []

    # 次数の降順でノードをソート（多く干渉する変数を先に処理）
    nodes_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

    for node in nodes_sorted:
        if node in spilled:
            continue

        # 隣接ノードが使っているレジスタを除外
        # R0はゼロレジスタなので予約
        used = {coloring[nb] for nb in G.neighbors(node) if nb in coloring}
        used.add(0)  # ← R0を予約
        # 使われていない最小のレジスタ番号を割り当て
        reg = 0
        while reg in used:
            reg += 1

        if reg >= num_registers:
            # レジスタが足りない → スピル対象を選ぶ
            # 生存区間が最も長い隣接変数をスピル
            candidates = [
                nb for nb in G.neighbors(node)
                if nb in coloring and nb not in spilled
            ]
            if live_ranges and candidates:
                # 生存区間が最も長い変数をスピル
                spill_target = max(
                    candidates,
                    key=lambda n: (
                        live_ranges[n].last_use - live_ranges[n].def_state
                        if n in live_ranges else 0
                    )
                )
            elif candidates:
                spill_target = candidates[0]
            else:
                # 隣接変数がなければ自分自身をスピル
                spill_target = node

            spilled.append(spill_target)
            # スピルした変数のレジスタを解放して再試行
            freed_reg = coloring.pop(spill_target, None)

            # 再度レジスタを探す
            used = {coloring[nb] for nb in G.neighbors(node) if nb in coloring}
            reg = 0
            while reg in used:
                reg += 1

            if reg >= num_registers:
                raise AllocationError(
                    f"Register overflow even after spilling. "
                    f"Need more than {num_registers} registers."
                )

        coloring[node] = reg

    return coloring, spilled


def _insert_spill_code(
    cdfg: nx.DiGraph,
    spilled: list[str],
    live_ranges: dict[str, "LiveRange"],
    scratch_regs: list[int] = None,
    spill_base_addr: int = 240,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    スピルされた変数に対してload/storeノードをCDFGに挿入する。

    scratch registerを永続確保せず、load/storeノードのレジスタは
    emitter.pyが動的に決定する（spill_reg=-1で示す）。

    Returns:
        spill_addr_map: SSA変数名 → スピルアドレス
        spill_reg_map:  SSA変数名 → scratch register番号（-1=動的決定）
    """
    spill_addr_map = {}
    spill_reg_map = {}
    node_counter = max(cdfg.nodes()) + 1

    def new_node_id():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    for i, ssa in enumerate(spilled):
        addr = spill_base_addr + i
        spill_addr_map[ssa] = addr
        spill_reg_map[ssa] = -1  # 動的に決定
        print(f"  SPILL: {ssa} → SRAM[{addr}]")

        # 定義ノードを探す
        def_nid = None
        for nid, data in cdfg.nodes(data=True):
            if ssa in data.get('results', []):
                def_nid = nid
                break

        if def_nid is None:
            continue

        def_state = cdfg.nodes[def_nid].get('state', 0)

        # STOREノードを追加（定義の直後）
        store_nid = new_node_id()
        cdfg.add_node(store_nid,
            type='op',
            op_name='memref.store',
            operands=[ssa],
            results=[],
            mlir_typ='void',
            state=def_state,
            spill_addr=addr,
            spill_reg=-1,
            assigned_op='store',
            latency=1,
        )
        cdfg.add_edge(def_nid, store_nid, type='data', ssa=ssa)

        # 使用ノードを探してLOADノードを挿入
        use_nodes = []
        for src, dst, edata in list(cdfg.edges(data=True)):
            if edata.get('type') == 'data' and edata.get('ssa') == ssa:
                if dst != store_nid:
                    use_nodes.append(dst)

        for use_nid in use_nodes:
            use_state = cdfg.nodes[use_nid].get('state', 0)

            reload_ssa = f"%reload_{ssa.lstrip('%')}_{use_nid}"
            load_nid = new_node_id()
            cdfg.add_node(load_nid,
                type='op',
                op_name='memref.load',
                operands=[ssa],
                results=[reload_ssa],
                mlir_typ=cdfg.nodes[def_nid].get('mlir_typ', 'i32'),
                state=max(0, use_state - 1),
                spill_addr=addr,
                spill_reg=-1,  # 動的に決定
                assigned_op='load',
                latency=1,
            )

            if cdfg.has_edge(def_nid, use_nid):
                cdfg.remove_edge(def_nid, use_nid)
            cdfg.add_edge(load_nid, use_nid, type='data', ssa=reload_ssa)

            operands = cdfg.nodes[use_nid].get('operands', [])
            new_operands = [reload_ssa if op == ssa else op for op in operands]
            cdfg.nodes[use_nid]['operands'] = new_operands

    return spill_addr_map, spill_reg_map


def allocate(
    cdfg: nx.DiGraph,
    num_registers: int = 32,
    num_imm_registers: int = 32,
    existing_spill_map: dict[str, int] = None,
) -> tuple[nx.DiGraph, dict[str, int], dict[str, int]]:
    """
    レジスタ割り当てを行い、CDFGノードにレジスタ情報を追加する。

    汎用レジスタ（r0〜r{num_registers-1}）:
      演算結果・引数・ループ変数

    即値レジスタ（imm0〜imm{num_imm_registers-1}）:
      arith.constantの定数値

    Args:
        cdfg:              対象のCDFG
        num_registers:     汎用レジスタ数
        num_imm_registers: 即値レジスタ数

    Returns:
        (更新されたCDFG, SSA→汎用レジスタマッピング, SSA→即値レジスタマッピング)
    """
# 1. 即値レジスタの割り当て（arith.constant）
    # 同じ値のarith.constantは同じIMMレジスタを使い回す
    imm_map: dict[str, int] = {}
    imm_counter = 0
    value_to_imm: dict = {}  # const_value → imm番号

    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arith.constant':
            const_value = data.get('const_value', 0)
            for r in data.get('results', []):
                if const_value in value_to_imm:
                    imm_map[r] = value_to_imm[const_value]
                else:
                    if imm_counter >= num_imm_registers:
                        raise AllocationError(
                            f"Immediate register overflow: need more than {num_imm_registers} imm registers."
                        )
                    imm_map[r] = imm_counter
                    value_to_imm[const_value] = imm_counter
                    imm_counter += 1

    # 2. 生存区間を計算
    live_ranges = _compute_live_ranges(cdfg)

    # 既にスピル済みの変数はGPRライフタイムから除外
    if existing_spill_map:
        for ssa in list(existing_spill_map.keys()):
            live_ranges.pop(ssa, None)

    # 3. ループ変数の統合
    alias = _unify_loop_vars(cdfg, live_ranges)

    # 4. 干渉グラフを構築
    interference = _build_interference_graph(live_ranges, alias)

    # 5. グラフ彩色（スピルあり）
    # GPR 1本をスピルベースアドレス用に確保
    # → 実際に割り当てに使えるレジスタは num_registers - 1 本
    effective_registers = num_registers - 1  # r(n-1)をスピルベース用に予約
    spill_base_reg = num_registers - 1       # スピルベースアドレスのGPR番号

    coloring, spilled = _graph_color(interference, effective_registers, live_ranges)

    # IMMに割り当て済みの変数はスピル対象から除外
    spilled = [s for s in spilled if s not in imm_map]

    if spilled:
        print(f"WARNING: {len(spilled)} variable(s) spilled to SRAM:")
        # スピル変数のオフセットをIMMに割り当て
        spill_addr_map = {}
        for i, ssa in enumerate(spilled):
            offset = i  # SRAMオフセット（r_spill_base + offset）
            spill_addr_map[ssa] = offset
            # IMMにオフセット値を追加
            imm_key = f"%spill_offset_{ssa.lstrip('%')}"
            imm_map[imm_key] = imm_counter
            imm_counter += 1
            print(f"  SPILL: {ssa} → SRAM[r{spill_base_reg} + imm{imm_map[imm_key]}(={offset})]")
    else:
        spill_addr_map = {}
        spill_base_reg = -1  # スピルなしならベースレジスタ不要

    # エイリアスも解決
    def get_reg(ssa: str) -> int:
        canonical = alias.get(ssa, ssa)
        # スピルされた変数はspill_base_reg経由でアクセス
        # → emitter.pyがload/storeを生成するのでreg=-1
        if ssa in spill_addr_map or canonical in spill_addr_map:
            return -1
        # existing_spill_mapに含まれる変数も-1
        if existing_spill_map:
            if ssa in existing_spill_map or canonical in existing_spill_map:
                return -1
        return coloring.get(canonical, -1)

    # 6. CDFGノードにレジスタ情報を追加
    for nid, data in cdfg.nodes(data=True):
        results = data.get('results', [])
        operands = data.get('operands', [])

        # 即値レジスタか汎用レジスタか判定
        result_regs = []
        result_reg_types = []
        for r in results:
            if r in imm_map:
                result_regs.append(imm_map[r])
                result_reg_types.append('imm')
            else:
                result_regs.append(get_reg(r))
                result_reg_types.append('gpr')

        operand_regs = []
        operand_reg_types = []
        for op in operands:
            if op in imm_map:
                operand_regs.append(imm_map[op])
                operand_reg_types.append('imm')
            else:
                operand_regs.append(get_reg(op))
                operand_reg_types.append('gpr')

        cdfg.nodes[nid]['result_regs'] = result_regs
        cdfg.nodes[nid]['result_reg_types'] = result_reg_types
        cdfg.nodes[nid]['operand_regs'] = operand_regs
        cdfg.nodes[nid]['operand_reg_types'] = operand_reg_types

    # SSA変数名→レジスタ番号の全マッピング
    reg_map = {ssa: get_reg(ssa) for ssa in live_ranges}

    return cdfg, reg_map, imm_map, spill_addr_map, spill_base_reg


def _reg_str(reg: int, reg_type: str, ssa: str = None, spill_map: dict[str, int] = None) -> str:
    """レジスタ番号と種類を文字列で返す。"""
    if reg < 0:
        if ssa and spill_map and ssa in spill_map:
            return f"SPILL[{spill_map[ssa]}]"
        return "?"
    return f"imm{reg}" if reg_type == 'imm' else f"r{reg}"


def print_allocation(reg_map: dict[str, int], cdfg: nx.DiGraph, imm_map: dict[str, int] = None,
                     spill_map: dict[str, int] = None, spill_base_reg: int = -1):
    """レジスタ割り当て結果を表示する。"""
    gpr_count = max((v for v in reg_map.values() if v >= 0), default=-1) + 1
    imm_count = max(imm_map.values(), default=-1) + 1 if imm_map else 0
    print(f"Register allocation: {gpr_count} GPR, {imm_count} IMM registers used")
    print()

    # スピル情報の表示
    if spill_map and spill_base_reg >= 0:
        print(f"  Spill base register: r{spill_base_reg}")
        for ssa, offset in spill_map.items():
            print(f"    {ssa} → SRAM[r{spill_base_reg} + {offset}]")
        print()

    # 即値レジスタの表示
    if imm_map:
        print("  Immediate registers:")
        for nid, data in cdfg.nodes(data=True):
            if data.get('op_name') == 'arith.constant':
                results = data.get('results', [])
                result_regs = data.get('result_regs', [])
                result_types = data.get('result_reg_types', [])
                for r, rt, res in zip(result_regs, result_types, results):
                    print(f"    imm{r} ← {res}")
        print()

    # SSA変数名→オペランド名のマッピング（spill表示用）
    ssa_to_operand: dict[str, str] = {}
    for nid, data in cdfg.nodes(data=True):
        for r in data.get('results', []):
            ssa_to_operand[r] = r

    # stateごとにまとめて表示
    state_nodes: dict[int, list[int]] = defaultdict(list)
    for nid, data in cdfg.nodes(data=True):
        state = data.get('state')
        if state is not None and data.get('type') == 'op':
            state_nodes[state].append(nid)

    for state in sorted(state_nodes.keys()):
        print(f"  state {state}:")
        for nid in state_nodes[state]:
            data = cdfg.nodes[nid]
            assigned = data.get('assigned_op', '?')
            result_regs = data.get('result_regs', [])
            result_types = data.get('result_reg_types', [])
            operand_regs = data.get('operand_regs', [])
            operand_types = data.get('operand_reg_types', [])
            results = data.get('results', [])
            operands = data.get('operands', [])

            ops_str = ", ".join(
                _reg_str(r, rt, op, spill_map)
                for r, rt, op in zip(operand_regs, operand_types, operands)
            )
            res_str = ", ".join(
                _reg_str(r, rt, res, spill_map)
                for r, rt, res in zip(result_regs, result_types, results)
            )
            print(f"    [{nid}] {assigned}: {ops_str} → {res_str}")
    print()

    # 引数のレジスタ
    print("  Arguments:")
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'arg':
            results = data.get('results', [])
            result_regs = data.get('result_regs', [])
            result_types = data.get('result_reg_types', [])
            for r, rt, res in zip(result_regs, result_types, results):
                print(f"    {res} → {_reg_str(r, rt, res, spill_map)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from parser import parse_c
    from lowering import lower_mlir
    from dp import load_dp
    from matcher import match_all
    from scheduler import schedule

    if len(sys.argv) < 2:
        print("使い方: python allocator.py <Cファイル>")
        sys.exit(1)

    c_src = open(sys.argv[1]).read()
    mlir_text = parse_c(c_src)
    cdfg = lower_mlir(mlir_text)

    operators = load_dp(mul=True, cmp=True, fpu=True, fmul=True,
                        mac_i=True, mac_f=True, sqrt=True)

    cdfg, results, unmatched = match_all(cdfg, operators)
    cdfg = schedule(cdfg, operators)
    cdfg, reg_map, imm_map = allocate(cdfg, num_registers=32)
    print_allocation(reg_map, cdfg, imm_map)