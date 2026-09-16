"""
scheduler.py: CDFGへのASAPスケジューリング

各演算ノードにステート番号を割り当てる。

制約:
  1. データ依存: A→BのデータエッジがあればAのステート < Bのステート
  2. 演算器数:   同じステートに同じ演算器はcount個まで
  3. BBの境界:   BBが違えば必ず別のステートグループ

ノードに追加される属性:
  state: ステート番号（0始まり）
"""
from __future__ import annotations
from collections import defaultdict
import networkx as nx

from nisc_compiler.passes.operator_assign.dp import Operator, load_dp


class ScheduleError(Exception):
    pass


def verify_completion(graph, context):
    """A09: check end-of-cycle commits before transfer, use and final done.

    For launch s and latency L, commit is at the end of s+L-1; a consumer
    can launch at s+L. Across blocks, state labels are not elapsed time.
    """
    done = graph.graph.get('done_state')
    if type(done) is not int or done < 1:
        raise ScheduleError('Invalid done state')
    blocks = graph.graph.get('blocks', {})
    owners = {}
    for bb, data in graph.nodes(data=True):
        if data['type'] == 'bb':
            for _, n, edge in graph.out_edges(bb, data=True):
                if edge.get('label') == 'contains':
                    if n in owners:
                        raise ScheduleError('Operation belongs to multiple basic blocks')
                    owners[n] = bb
    occupied = set()
    for bb, meta in blocks.items():
        start, end = meta.get('start'), meta.get('end')
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < done:
            raise ScheduleError('Invalid basic block completion range')
        states = set(range(start, end + 1))
        if states & occupied:
            raise ScheduleError('Overlapping basic block states')
        occupied |= states
    if blocks and (blocks[graph.graph['entry']]['start'] != 0
                   or blocks[graph.graph['exit']]['end'] + 1 != done):
        raise ScheduleError('Entry/exit state does not match execution boundaries')
    op_map = {op.name: op for op in context.extra['int32_operators']}
    completions = {}
    producers = {}
    for n, data in graph.nodes(data=True):
        for s in data.get('results', []):
            producers[s] = n
        if data['type'] != 'op':
            continue
        start, latency = data.get('state'), data.get('latency')
        operator = op_map.get(data.get('assigned_op'))
        if (type(start) is not int or start < 0 or type(latency) is not int or latency < 1
                or operator is None or latency != operator.latency):
            raise ScheduleError('Invalid operation launch time or latency')
        if n not in owners or (blocks and owners[n] not in blocks):
            raise ScheduleError('Scheduled operation has no basic block')
        finish = start + latency
        if finish > done:
            raise ScheduleError('done precedes operation completion')
        if blocks:
            meta = blocks[owners[n]]
            if start < meta['start'] or finish > meta['end'] + 1:
                raise ScheduleError('Control transfer before operation completion')
        completions[n] = finish
    stores = [n for n, d in graph.nodes(data=True) if d['op_name'] == 'memref.store']
    store = graph.graph.get('return_store')
    if stores != [store] or store not in completions:
        raise ScheduleError('Expected exactly one identified return STORE')
    if completions[store] != done:
        raise ScheduleError('done must immediately follow return STORE completion')
    if blocks and owners[store] != graph.graph['exit']:
        raise ScheduleError('Return STORE must belong to the common exit')
    # Reconstruct operand timing even if a data edge was accidentally lost.
    for n, data in graph.nodes(data=True):
        if data['type'] != 'op':
            continue
        for s in data['operands']:
            p = producers.get(s)
            if p is None:
                raise ScheduleError(f'Undefined scheduled operand: {s}')
            if p in completions and (not blocks or owners[p] == owners[n]):
                if completions[p] > data['state']:
                    raise ScheduleError(f'Operand used before completion: {s}')


def schedule_cfg(graph, context):
    """Lay out cyclic CFG blocks once; FSM back edges perform the iterations."""
    from nisc_compiler.passes.register_allocate.allocator import allocate_cfg, AllocationError
    blocks = graph.graph['blocks']
    cfg = nx.DiGraph((bb, s) for bb, m in blocks.items() for s in m['successors'])
    cfg.add_nodes_from(blocks)
    order = list(nx.dfs_preorder_nodes(cfg, graph.graph['entry']))
    if set(order) != set(blocks):
        raise ScheduleError('Unreachable CFG block')
    order.remove(graph.graph['exit'])
    order.append(graph.graph['exit'])
    owners = {n: bb for bb in blocks for _, n, e in graph.out_edges(bb, data=True)
              if e.get('label') == 'contains'}
    # State labels are locations, not time across loop iterations. Only local
    # dependencies constrain a block's schedule; dominance checks the others.
    local = graph.copy()
    local.remove_edges_from([(u, v) for u, v, e in graph.edges(data=True)
                             if e['type'] in ('data', 'order') and owners.get(u) != owners.get(v)])
    offset = 0
    for bb in order:
        members = [n for n, owner in owners.items() if owner == bb]
        condition = blocks[bb]['condition']
        if condition:
            cmp = next(n for n in members if condition in graph.nodes[n]['results'])
            for n in members:
                if n != cmp and graph.nodes[n]['type'] == 'op':
                    local.add_edge(n, cmp, type='order')
                    graph.add_edge(n, cmp, type='order')
        for n in members:
            if graph.nodes[n]['op_name'] == 'nisc.phi':
                local.nodes[n].update(state=offset, latency=0)
        end = max(offset, schedule_bb(local, bb, context.extra['int32_operators'], offset))
        blocks[bb].update(start=offset, end=end)
        for n in members:
            graph.nodes[n].update(local.nodes[n])
        offset = end + 1
    graph.graph['done_state'] = offset
    immediates, imm_map = [], {}
    for _, d in graph.nodes(data=True):
        if d['op_name'] == 'arith.constant':
            value = d['const_value']
            if not -(1 << 31) <= value < (1 << 31):
                raise AllocationError('Constant outside int32 range')
            if value not in immediates:
                immediates.append(value)
            imm_map.update((s, immediates.index(value)) for s in d['results'])
    if len(immediates) > context.num_imm_registers:
        raise AllocationError('Immediate register capacity exceeded')
    context.imm_map = imm_map
    context.extra['imm_values'] = immediates
    context.reg_map = allocate_cfg(graph, context)
    context.iterations = 1
    for _, d in graph.nodes(data=True):
        for field, prefix in (('operands', 'operand'), ('results', 'result')):
            d[prefix + '_regs'] = [imm_map[s] if s in imm_map else context.reg_map.get(s) for s in d[field]]
            d[prefix + '_reg_types'] = ['imm' if s in imm_map else 'gpr' if s in context.reg_map else 'flag' for s in d[field]]
    verify_cfg_schedule(graph, context)
    return graph


def verify_cfg_schedule(graph, context):
    from .int32 import RESOURCE, verify_cfg
    from nisc_compiler.passes.register_allocate.allocator import cfg_interference, AllocationError
    verify_completion(graph, context)
    verify_cfg(graph, context)
    blocks = graph.graph['blocks']
    owners = {n: bb for bb in blocks for _, n, e in graph.out_edges(bb, data=True)
              if e.get('label') == 'contains'}
    op_map = {o.name: o for o in context.extra['int32_operators']}
    reservations = set()
    occupied_blocks = set()
    for bb, meta in blocks.items():
        if not 0 <= meta['start'] <= meta['end'] < graph.graph['done_state']:
            raise ScheduleError('Invalid CFG block state range')
        states = set(range(meta['start'], meta['end'] + 1))
        if occupied_blocks & states:
            raise ScheduleError('Overlapping CFG block states')
        occupied_blocks |= states
    for n, d in graph.nodes(data=True):
        if d['type'] != 'op':
            continue
        if d['latency'] != op_map[d['assigned_op']].latency or d['latency'] < 1:
            raise ScheduleError('Invalid scheduled latency')
        for t in range(d['state'], d['state'] + d['latency']):
            key = (t, RESOURCE[d['op_name']])
            if key in reservations:
                raise ScheduleError('Physical resource conflict')
            reservations.add(key)
    for u, v, edge in graph.edges(data=True):
        a, b = graph.nodes[u], graph.nodes[v]
        if (edge['type'] in ('order', 'data') and owners.get(u) == owners.get(v)
                and a['type'] == b['type'] == 'op'
                and a['state'] + a['latency'] > b['state']):
            raise ScheduleError('Local dependency violated')
    interference, values = cfg_interference(graph, context.imm_map)
    aliases = graph.graph.get('aliases', {})
    for s in values:
        r = context.reg_map.get(s, -1)
        if not 1 <= r < context.num_registers or r != context.reg_map.get(aliases.get(s, s)):
            raise AllocationError('Invalid CFG register or edge-copy alias')
    if any(context.reg_map[a] == context.reg_map[b] for a, b in interference.edges):
        raise AllocationError('Live CFG values share a register')


def _get_bb_subgraph(cdfg: nx.DiGraph, bb_id: int) -> list[int]:
    """BBノードに含まれるopノードのリストを返す。"""
    op_nodes = []
    for _, dst, data in cdfg.out_edges(bb_id, data=True):
        if data.get('label') == 'contains' and cdfg.nodes[dst].get('type') == 'op':
            op_nodes.append(dst)
    return op_nodes


def _get_all_bbs(cdfg: nx.DiGraph) -> list[int]:
    """CDFGの全BBノードIDを返す。"""
    return [n for n, d in cdfg.nodes(data=True) if d.get('type') == 'bb']


def _data_predecessors(cdfg: nx.DiGraph, node_id: int) -> list[int]:
    """データエッジで繋がった前段ノードのリストを返す。"""
    preds = []
    for src, _, data in cdfg.in_edges(node_id, data=True):
        if data.get('type') in ('data', 'order'):
            preds.append(src)
    return preds


def schedule_bb(
    cdfg: nx.DiGraph,
    bb_id: int,
    operators: list[Operator],
    state_offset: int = 0,
) -> int:
    """
    1つのBBをASAPスケジューリングする。

    Args:
        cdfg:         対象のCDFG
        bb_id:        スケジューリング対象のBBノードID
        operators:    DP演算器リスト
        state_offset: このBBのステート番号の開始オフセット

    Returns:
        このBBが使った最後のステート番号
    """
    op_nodes = _get_bb_subgraph(cdfg, bb_id)
    if not op_nodes:
        return state_offset - 1

    # 演算器名 → Operatorのマッピング
    op_map: dict[str, Operator] = {op.name: op for op in operators}

    # ステートごとの演算器使用数: {state: {op_name: count}}
    state_usage: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # トポロジカル順序でノードを処理（データ依存順）
    # op_nodes内でのトポロジカルソート
    subgraph = cdfg.subgraph(op_nodes)
    try:
        topo_order = list(nx.topological_sort(subgraph))
    except nx.NetworkXUnfeasible:
        raise ScheduleError(f"Cycle detected in BB {bb_id}")

    # match_groupごとにノードをグループ化
    # 同じmatch_groupのノードは同じステートに割り当てる
    group_to_nodes: dict[int, list[int]] = defaultdict(list)
    ungrouped: list[int] = []
    for node_id in topo_order:
        data = cdfg.nodes[node_id]
        if data.get('assigned_op') is None:
            continue
        group = data.get('match_group')
        if group is not None:
            group_to_nodes[group].append(node_id)
        else:
            ungrouped.append(node_id)

    # グループ単位でスケジューリング
    # トポロジカル順序でグループの代表ノード（先頭）を処理
    scheduled_groups: set[int] = set()

    for node_id in topo_order:
        data = cdfg.nodes[node_id]
        assigned_op = data.get('assigned_op')

        if assigned_op is None:
            continue

        operator = op_map.get(assigned_op)
        if operator is None:
            raise ScheduleError(f"Operator '{assigned_op}' not found in dp")
        if type(operator.latency) is not int or operator.latency < 1 or operator.count < 1:
            raise ScheduleError(f"Invalid timing/capacity for {assigned_op}")

        group = data.get('match_group')

        # グループ済みならスキップ
        if group is not None and group in scheduled_groups:
            continue

        # このノードが属するグループの全ノードを取得
        group_nodes = group_to_nodes[group] if group is not None else [node_id]

        # グループ全体のデータ依存から最早ステートを計算
        earliest = state_offset
        for gnode_id in group_nodes:
            for pred_id in _data_predecessors(cdfg, gnode_id):
                pred_state = cdfg.nodes[pred_id].get('state')
                # 同じグループ内のノードは除外
                pred_group = cdfg.nodes[pred_id].get('match_group')
                if pred_group is not None and pred_group == group:
                    continue
                if pred_state is not None:
                    pred_latency = cdfg.nodes[pred_id].get('latency', 1)
                    earliest = max(earliest, pred_state + pred_latency)

        # 演算器数制約を満たす最早ステートを探す
        state = earliest
        # resource が設定されていればresourceでカウント、なければassigned_opで
        resource_key = operator.resource if operator.resource else assigned_op
        occupy_cycles = 1 if operator.pipelined else operator.latency
        while True:
            if all(state_usage[s][resource_key] < operator.count
                   for s in range(state, state + occupy_cycles)):
                break
            state += 1
        # グループ内の全ノードに同じステートを割り当て
        for gnode_id in group_nodes:
            cdfg.nodes[gnode_id]['state'] = state
        # 演算器の占有を記録
        occupy_cycles = 1 if operator.pipelined else operator.latency
        for s in range(state, state + occupy_cycles):
            state_usage[s][resource_key] += 1

        if group is not None:
            scheduled_groups.add(group)

    # このBBの最終ステートを返す
    states = [
        cdfg.nodes[n]['state'] + cdfg.nodes[n].get('latency', 1) - 1
        for n in op_nodes
        if cdfg.nodes[n].get('state') is not None
    ]
    return max(states) if states else state_offset - 1


def _get_ctrl_children(cdfg: nx.DiGraph, bb_id: int) -> dict[str, list[int]]:
    """BBノードの制御エッジの子BBをlabelごとに返す。"""
    children: dict[str, list[int]] = {}
    for _, dst, data in cdfg.out_edges(bb_id, data=True):
        if data.get('type') == 'ctrl' and cdfg.nodes[dst].get('type') == 'bb':
            label = data.get('label', '')
            if label not in children:
                children[label] = []
            children[label].append(dst)
    return children


def _get_parent_ctrl(cdfg: nx.DiGraph, bb_id: int):
    """BBノードの親制御ノード（scf.if/for/whileなど）を返す。"""
    for src, _, data in cdfg.in_edges(bb_id, data=True):
        if data.get('type') == 'ctrl' and cdfg.nodes[src].get('type') in ('ctrl', 'op'):
            return src
    return None


def schedule(
    cdfg: nx.DiGraph,
    operators: list[Operator] = None,
) -> nx.DiGraph:
    """
    CDFGの全BBをASAPスケジューリングする。

    CFGのlabelを見てthen/elseブロックには同じstate_offsetを与える。
    ループのbodyブロックはcondブロックの後から始まる。

    Args:
        cdfg:      対象のCDFG
        operators: DP演算器リスト（Noneでデフォルト）

    Returns:
        state属性が追加されたCDFG
    """
    if operators is None:
        operators = load_dp()

    # BBをトポロジカル順序で処理
    # ただしthen/elseは同じoffsetから始める
    bb_offset: dict[int, int] = {}  # bb_id → state_offset
    bb_last: dict[int, int] = {}    # bb_id → last_state

    # エントリBBを探す（入力エッジがないBB）
    all_bbs = _get_all_bbs(cdfg)
    bb_set = set(all_bbs)

    # BBのトポロジカルソート
    bb_subgraph = cdfg.subgraph(all_bbs)
    try:
        topo_bbs = [n for n in nx.topological_sort(bb_subgraph) if n in bb_set]
    except nx.NetworkXUnfeasible:
        topo_bbs = all_bbs

    current_offset = 0

    for bb_id in topo_bbs:
        # このBBのoffsetを決定
        # 既に設定済み（then/elseで同じoffsetを使う）なら使う
        if bb_id not in bb_offset:
            bb_offset[bb_id] = current_offset

        offset = bb_offset[bb_id]
        last_state = schedule_bb(cdfg, bb_id, operators, state_offset=offset)
        bb_last[bb_id] = last_state

        # 子BBのoffsetを設定
        children = _get_ctrl_children(cdfg, bb_id)

        # scf.ifのthen/else → 同じoffsetから始まる
        then_bbs = children.get('then', [])
        else_bbs = children.get('else', [])
        if then_bbs or else_bbs:
            branch_offset = last_state + 1
            for child in then_bbs + else_bbs:
                if child not in bb_offset:
                    bb_offset[child] = branch_offset
            # 合流後のoffsetは後でmax計算
            continue

        # scf.forのbody → forの後から始まる
        body_bbs = children.get('body', [])
        cond_bbs = children.get('cond', [])
        for child in body_bbs + cond_bbs:
            if child not in bb_offset:
                bb_offset[child] = last_state + 1

        # contains → 通常の連番
        contains_bbs = children.get('contains', [])
        for child in contains_bbs:
            if child not in bb_offset:
                bb_offset[child] = last_state + 1

        # offsetを更新
        if not (then_bbs or else_bbs or body_bbs or cond_bbs or contains_bbs):
            current_offset = last_state + 1

    # then/else合流後のoffsetを計算
    # （簡易的にbb_lastの最大値+1を使う）
    if bb_last:
        current_offset = max(bb_last.values()) + 1

    # doneステートを追加
    cdfg = _add_done_state(cdfg)

    return cdfg


def print_schedule(cdfg: nx.DiGraph):
    """スケジューリング結果を表示する。"""
    bbs = _get_all_bbs(cdfg)

    for bb_id in bbs:
        bb_name = cdfg.nodes[bb_id].get('op_name', 'bb')
        op_nodes = _get_bb_subgraph(cdfg, bb_id)
        if not op_nodes:
            continue

        print(f"[{bb_id}] {bb_name}:")

        # ステートごとにグループ化
        state_groups: dict[int, list[int]] = defaultdict(list)
        for nid in op_nodes:
            state = cdfg.nodes[nid].get('state')
            if state is not None:
                state_groups[state].append(nid)

        for state in sorted(state_groups.keys()):
            print(f"  state {state}:")
            for nid in state_groups[state]:
                data = cdfg.nodes[nid]
                assigned = data.get('assigned_op', '?')
                op_name = data.get('op_name', '?')
                latency = data.get('latency', '?')
                print(f"    node[{nid}] {op_name} → {assigned} (latency={latency})")
        print()




def _add_done_state(cdfg: nx.DiGraph) -> nx.DiGraph:
    """
    bb_exitが空の場合にdoneノードを追加してstateを割り当てる。
    emitter.pyがそのstateでio.done := true.Bを生成する。
    """
    # 全opノードの最大stateを取得
    all_states = [
        d.get('state') for _, d in cdfg.nodes(data=True)
        if d.get('state') is not None
    ]
    if not all_states:
        return cdfg

    done_state = max(all_states) + 1

    # bb_exitを探す
    exit_bbs = [
        n for n, d in cdfg.nodes(data=True)
        if d.get('op_name') == 'bb_exit'
    ]

    for bb_id in exit_bbs:
        # 既にdoneノードがあればスキップ
        has_done = any(
            cdfg.nodes[dst].get('op_name') == 'done'
            for _, dst, edata in cdfg.out_edges(bb_id, data=True)
            if edata.get('label') == 'contains'
        )
        if has_done:
            continue

        # doneノードを追加
        done_nid = max(cdfg.nodes()) + 1
        cdfg.add_node(done_nid,
            type='ctrl',
            op_name='done',
            operands=[],
            results=[],
            mlir_typ='void',
            state=done_state,
        )
        cdfg.add_edge(bb_id, done_nid, type='ctrl', label='contains')

    # lowering.pyで追加済みのdoneノードにstateを割り当てる
    for nid, data in cdfg.nodes(data=True):
        if data.get('op_name') == 'done' and data.get('state') is None:
            cdfg.nodes[nid]['state'] = done_state

    return cdfg


def reset_states(cdfg: nx.DiGraph) -> nx.DiGraph:
    """全ノードのstate属性をリセットする。スピルノード挿入後の再スケジューリング用。"""
    for nid in cdfg.nodes():
        if 'state' in cdfg.nodes[nid]:
            del cdfg.nodes[nid]['state']
    return cdfg


def _get_mem_ops_at_state(cdfg: nx.DiGraph, state: int) -> list[int]:
    """指定stateのメモリアクセスノードを返す。"""
    return [
        nid for nid, data in cdfg.nodes(data=True)
        if data.get('state') == state and
        data.get('op_name') in ('memref.load', 'memref.store')
    ]


def insert_spill_nodes(
    cdfg: nx.DiGraph,
    spill_map: dict[str, int],
    operators: list[Operator],
) -> nx.DiGraph:
    """
    スピル変数のload/storeノードをCDFGに挿入してスケジューリングする。

    STORE: 定義ノードと同じstateに配置（メモリポートが空いていれば）
    LOAD:  使用ノードのstateに配置（空いていれば同じstate、塞がっていれば1つ前）

    Args:
        cdfg:      スケジューリング済みCDFG
        spill_map: SSA変数名 → スピルオフセット
        operators: DP演算器リスト

    Returns:
        load/storeノードが追加されたCDFG
    """
    node_counter = max(cdfg.nodes()) + 1

    def new_nid():
        nonlocal node_counter
        nid = node_counter
        node_counter += 1
        return nid

    # BBノードのマッピング: ノードID → 所属BB
    node_to_bb: dict[int, int] = {}
    for bb_id, data in cdfg.nodes(data=True):
        if data.get('type') == 'bb':
            for _, dst, edata in cdfg.out_edges(bb_id, data=True):
                if edata.get('label') == 'contains':
                    node_to_bb[dst] = bb_id

    for ssa, offset in spill_map.items():
        # 定義ノードを探す
        def_nid = None
        for nid, data in cdfg.nodes(data=True):
            if ssa in data.get('results', []):
                def_nid = nid
                break
        if def_nid is None:
            continue

        def_state = cdfg.nodes[def_nid].get('state', 0)
        def_bb = node_to_bb.get(def_nid)

        # argノードはBBに属していないので
        # 最初のBB（bb_entry/bb_init/bb_cond）にSTOREを追加
        if def_bb is None:
            all_bbs = [n for n, d in cdfg.nodes(data=True) if d.get('type') == 'bb']
            if all_bbs:
                def_bb = all_bbs[0]
                # 最初のBBの最初のstateを使う
                bb_ops = _get_bb_subgraph(cdfg, def_bb)
                if bb_ops:
                    def_state = min(
                        cdfg.nodes[n].get('state', 0)
                        for n in bb_ops
                        if cdfg.nodes[n].get('state') is not None
                    )

        # STOREノード: 定義直後のstateに配置
        # メモリポートが塞がっていれば次のstateに
        store_state = def_state
        if _get_mem_ops_at_state(cdfg, store_state):
            store_state += 1

        store_nid = new_nid()
        cdfg.add_node(store_nid,
            type='op',
            op_name='memref.store',
            operands=[ssa],
            results=[],
            mlir_typ='void',
            state=store_state,
            spill_offset=offset,
            assigned_op='store',
            latency=1,
        )
        cdfg.add_edge(def_nid, store_nid, type='data', ssa=ssa)
        # BBに追加
        if def_bb is not None:
            cdfg.add_edge(def_bb, store_nid, type='ctrl', label='contains')

        # 使用ノードを探してLOADノードを挿入
        use_nodes = []
        for src, dst, edata in list(cdfg.edges(data=True)):
            if edata.get('type') == 'data' and edata.get('ssa') == ssa:
                if dst != store_nid:
                    use_nodes.append(dst)

        for use_nid in use_nodes:
            use_state = cdfg.nodes[use_nid].get('state', 0)
            use_bb = node_to_bb.get(use_nid)

            # LOADノード: 使用stateに配置
            # メモリポートが塞がっていれば1つ前のstateに
            load_state = use_state
            if _get_mem_ops_at_state(cdfg, load_state):
                load_state = max(0, use_state - 1)

            reload_ssa = f"%spill_load_{ssa.lstrip('%')}_{use_nid}"
            load_nid = new_nid()
            cdfg.add_node(load_nid,
                type='op',
                op_name='memref.load',
                operands=[ssa],
                results=[reload_ssa],
                mlir_typ=cdfg.nodes[def_nid].get('mlir_typ', 'i32'),
                state=load_state,
                spill_offset=offset,
                assigned_op='load',
                latency=1,
            )
            cdfg.add_edge(load_nid, use_nid, type='data', ssa=reload_ssa)

            # オペランドを書き換え
            operands = cdfg.nodes[use_nid].get('operands', [])
            cdfg.nodes[use_nid]['operands'] = [
                reload_ssa if op == ssa else op for op in operands
            ]

            # BBに追加
            if use_bb is not None:
                cdfg.add_edge(use_bb, load_nid, type='ctrl', label='contains')

    return cdfg


def iterative_schedule(
    cdfg: nx.DiGraph,
    operators: list[Operator],
    num_registers: int = 32,
    num_imm_registers: int = 32,
    max_iter: int = 10,
):
    """
    スケジューリングとレジスタ割り当てをスピルがなくなるまで繰り返す。

    Returns:
        (cdfg, reg_map, imm_map, spill_map)
    """
    from nisc_compiler.passes.register_allocate.allocator import allocate

    # スピルアドレス管理: 変数名 → 固定アドレス
    # 一度スピルされた変数は常に同じアドレスを使う
    global_spill_map: dict[str, int] = {}
    spill_counter = [240]  # 次に使うSRAMアドレス

    prev_spill_keys = None

    for i in range(max_iter):
        print(f"[iter {i+1}] scheduling...")
        cdfg = schedule(cdfg, operators)

        print(f"[iter {i+1}] allocating registers...")
        cdfg, reg_map, imm_map, spill_map, spill_base_reg = allocate(
            cdfg, num_registers, num_imm_registers,
            existing_spill_map=global_spill_map,
        )

        if not spill_map:
            print(f"[iter {i+1}] done. no spills.")
            # global_spill_mapが空でない場合はspill_base_regを保持
            if global_spill_map:
                final_spill_base_reg = num_registers - 1
            else:
                final_spill_base_reg = -1
            return cdfg, reg_map, imm_map, global_spill_map, final_spill_base_reg, i + 1

        # global_spill_mapを更新（新しいスピルのみ追加）
        for ssa, addr in spill_map.items():
            if ssa not in global_spill_map:
                global_spill_map[ssa] = addr

        # スピル変数が前回と同じなら収束しない → レジスタ不足
        spill_keys = set(spill_map.keys())
        if spill_keys == prev_spill_keys:
            raise ScheduleError(
                f"Register allocation cannot converge with {num_registers} registers. "
                f"Need more registers. Spilled: {spill_keys}"
            )

        print(f"[iter {i+1}] {len(spill_map)} spill(s) → inserting spill nodes...")
        # 新しくスピルされた変数のみload/storeノードを挿入
        # prev_spill_keysを更新する前に計算する
        new_spills = {k: v for k, v in global_spill_map.items()
                      if k not in (prev_spill_keys or set())}
        prev_spill_keys = spill_keys

        if new_spills:
            cdfg = insert_spill_nodes(cdfg, new_spills, operators)
        print(f"[iter {i+1}] re-scheduling...")
        cdfg = reset_states(cdfg)

    raise ScheduleError(f"Failed to allocate registers after {max_iter} iterations.")
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from parser import parse_c
    from lowering import lower_mlir
    from dp import load_dp
    from matcher import match_all

    if len(sys.argv) < 2:
        print("使い方: python scheduler.py <Cファイル>")
        sys.exit(1)

    c_src = open(sys.argv[1]).read()
    mlir_text = parse_c(c_src)
    cdfg = lower_mlir(mlir_text)

    operators = load_dp(mul=True, cmp=True, fpu=True, fmul=True,
                        mac_i=True, mac_f=True, sqrt=True,
                        sin=True, cos=True, exp=True, log=True)

    cdfg, results, unmatched = match_all(cdfg, operators)
    cdfg = schedule(cdfg, operators)
    print_schedule(cdfg)
