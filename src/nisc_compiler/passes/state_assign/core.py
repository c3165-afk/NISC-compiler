"""
scheduler.py: CDFGへのASAPスケジューリング

各演算ノードにステート番号を割り当てる。

制約:
  1. 依存関係:   別の配置単位へ渡す値は、生成元の完了後に使用する
  2. 演算器数:   占有期間の全サイクルで、同じ資源の使用数をcount以下にする
  3. BBの境界:   BBが違えば必ず別のステートグループ

ノードに追加される属性:
  local_state: BBの先頭を0とした演算の開始位置
  state: state_offset + local_stateで求める配置先のステート番号
         探索対象のbb_initでは、演算とは別の入口判定用ステート番号
"""
from __future__ import annotations
from collections import defaultdict
import networkx as nx

from nisc_compiler.passes.operator_assign.dp import Operator, load_dp


class ScheduleError(Exception):
    pass


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
        if data.get('type') == 'data':
            preds.append(src)
    return preds


def schedule_bb(
    cdfg: nx.DiGraph,
    bb_id: int,
    operators: list[Operator],
    state_offset: int = 0,
) -> int:
    """
    1つのBB内の相対スケジュールを求め、指定オフセットにASAP配置する。

    変更: BB内の依存関係・占有期間・優先順位の計算は維持し、
    BB外のstateを時刻として参照しない。外部入力はBB開始時に利用可能とする。
    この前提は、後段のBB間の制御遷移・完了待ち・値の保持によって保証する。

    Args:
        cdfg:         対象のCDFG
        bb_id:        スケジューリング対象のBBノードID
        operators:    DP演算器リスト
        state_offset: 相対スケジュールを配置する開始オフセット（BB内の配置には影響しない）

    Returns:
        このBBの演算が完了する最後のステート番号（開始 + latency - 1）
    """
    op_nodes = _get_bb_subgraph(cdfg, bb_id)
    if not op_nodes:
        return state_offset - 1

    op_map = {op.name: op for op in operators}

    # 変更: ノードの固定順ではなく、複合演算をまとめた依存グラフで候補を選ぶ。
    # match_groupのない演算は、それぞれ独立した配置単位として扱う。
    groups = defaultdict(list)
    owner = {}
    for nid in op_nodes:
        data = cdfg.nodes[nid]
        group = data.get('match_group')
        key = ('group', group) if group is not None else ('node', nid)
        groups[key].append(nid)
        owner[nid] = key

    # 変更: グループ化で内部の循環が隠れないよう、元の依存グラフも検査する。
    node_dependencies = nx.DiGraph()
    node_dependencies.add_nodes_from(op_nodes)
    node_dependencies.add_edges_from(
        (u, v) for u, v, edge in cdfg.edges(data=True)
        if u in owner and v in owner and edge.get('type') in ('data', 'order'))
    if not nx.is_directed_acyclic_graph(node_dependencies):
        raise ScheduleError(f"Cycle detected in BB {bb_id}")

    dependencies = nx.DiGraph()
    dependencies.add_nodes_from(groups)
    group_ops = {}
    capacity = {}
    rank = {key: i for i, key in enumerate(groups)}
    for key, members in groups.items():
        names = {cdfg.nodes[n].get('assigned_op') for n in members}
        if len(names) != 1 or None in names:
            raise ScheduleError(f"Invalid operator assignment in BB {bb_id}: {members}")
        name = next(iter(names))
        operator = op_map.get(name)
        if operator is None:
            raise ScheduleError(f"Operator '{name}' not found in dp")
        # 変更: 不正な占有期間・個数による無限探索や予約の不整合を防ぐ。
        if (type(operator.latency) is not int or operator.latency < 1
                or type(operator.count) is not int or operator.count < 1):
            raise ScheduleError(f"Invalid latency/count for operator '{name}'")
        if any(cdfg.nodes[n].get('latency') != operator.latency for n in members):
            raise ScheduleError(f"Latency mismatch for operator '{name}'")
        resource = operator.resource or name
        if resource in capacity and capacity[resource] != operator.count:
            raise ScheduleError(f"Inconsistent count for shared resource '{resource}'")
        capacity[resource] = operator.count
        group_ops[key] = operator
        for nid in members:
            for pred, _, edge in cdfg.in_edges(nid, data=True):
                # データ依存に加え、明示された実行順序の辺も保持する。
                if edge.get('type') not in ('data', 'order'):
                    continue
                if pred in owner:
                    if owner[pred] != key:
                        dependencies.add_edge(owner[pred], key)
                # 変更: BB外のstateは配置番号であり、このBBの相対時刻ではない。
                # 外部への依存エッジ自体は残し、BB開始前の完了確認を後段に任せる。
    try:
        topology = list(nx.topological_sort(dependencies))
    except nx.NetworkXUnfeasible:
        raise ScheduleError(f"Cycle between operation groups in BB {bb_id}")

    # 変更: 自分から後続の終端までの最長レイテンシを優先度とする。
    # 同じ時刻に配置できる候補では、残りの依存鎖が長い演算を先に選ぶ。
    priority = {}
    for key in reversed(topology):
        priority[key] = group_ops[key].latency + max(
            (priority[child] for child in dependencies.successors(key)), default=0)

    state_usage = defaultdict(lambda: defaultdict(int))
    remaining = dict(dependencies.in_degree())
    ready = {key for key in groups if remaining[key] == 0}
    finish = {}
    while ready:
        candidates = []
        for key in ready:
            operator = group_ops[key]
            resource = operator.resource or operator.name
            # 既存のpipelined=Trueは開始間隔1という契約を維持する。
            # 結果の利用可能時刻は、パイプラインでもlatency後とする。
            duration = 1 if operator.pipelined else operator.latency
            # 変更: 開始位置はBB内の相対時刻で計算し、外部の配置番号を混ぜない。
            state = max(
                (finish[pred] for pred in dependencies.predecessors(key)),
                default=0)
            # 変更: 開始時点だけでなく占有期間の全サイクルで空きを確認する。
            # 将来の予約と重なる場合は、期間全体が入る位置まで進める。
            while any(state_usage[t][resource] >= capacity[resource]
                      for t in range(state, state + duration)):
                state += 1
            candidates.append((state, -priority[key], rank[key], key))

        # 変更: 最も早い空きに入る候補を選び、同時刻なら最長依存鎖を優先。
        # rankを最後の比較条件にし、同条件での配置を再現可能にする。
        state, _, _, key = min(candidates)
        operator = group_ops[key]
        resource = operator.resource or operator.name
        duration = 1 if operator.pipelined else operator.latency
        for nid in groups[key]:
            # 変更: 相対位置を保存し、既存の呼び出し側にはオフセット付きstateを渡す。
            cdfg.nodes[nid]['local_state'] = state
            cdfg.nodes[nid]['state'] = state_offset + state
        for t in range(state, state + duration):
            state_usage[t][resource] += 1
        finish[key] = state + operator.latency
        ready.remove(key)
        for child in dependencies.successors(key):
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.add(child)

    # 変更: BBの使用期間には、最後の演算の開始だけでなく完了までを含める。
    # 変更: 戻り値は従来どおり配置先の最終ステート。空BBは冒頭でoffset - 1を返す。
    return state_offset + max(finish.values()) - 1


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


def _assign_entry_states(cdfg: nx.DiGraph) -> int:
    # 変更: 関数直下の入口だけを、loweringが記録したプログラム順で先頭に配置する。
    # BBの生成順や演算の依存順では並べない。入れ子のinitは探索対象に含めない。
    entries = []
    for scope, data in cdfg.nodes(data=True):
        if not data.get('function_entry'):
            continue
        for index, bb_id in enumerate(data.get('dispatch_entries', [])):
            if bb_id not in cdfg:
                raise ScheduleError(f"Unknown dispatch entry: {bb_id}")
            entry = cdfg.nodes[bb_id]
            if (entry.get('type') != 'bb' or entry.get('op_name') != 'bb_init'
                    or not entry.get('dispatch_candidate')
                    or entry.get('dispatch_scope') != scope
                    or entry.get('dispatch_index') != index
                    or bb_id in entries):
                raise ScheduleError(f"Invalid dispatch entry metadata: {bb_id}")
            entries.append(bb_id)

    candidates = {n for n, d in cdfg.nodes(data=True) if d.get('dispatch_candidate')}
    if set(entries) != candidates:
        raise ScheduleError("Dispatch entries do not cover the dispatch candidates")
    for state, bb_id in enumerate(entries):
        # forのiv初期化など、init内の演算のstateとは区別してBB自身に記録する。
        # このstateでは準備判定を行い、準備済みの場合だけ初期化演算へ進む想定。
        cdfg.nodes[bb_id]['state'] = state
    return len(entries)


def _order_bbs_by_resources(cdfg: nx.DiGraph, bb_ids: list[int]) -> list[int]:
    # 変更: 同じ資源集合のBBをまとめ、共通資源の割合が大きい集合を隣へ置く。
    # 同率の場合は入力順（CDFGのBB生成順）を維持し、配置を再現可能にする。
    groups = {}
    empty_bbs = []
    for bb_id in bb_ids:
        resources = frozenset(cdfg.nodes[bb_id]['resources'])
        if not resources:
            empty_bbs.append(bb_id)
        else:
            groups.setdefault(resources, []).append(bb_id)
    ordered = []
    remaining = list(groups)
    if remaining:
        current = remaining.pop(0)
        while True:
            ordered.extend(groups[current])
            if not remaining:
                break
            # Jaccard類似度。maxは同率なら先に現れた候補を選ぶ。
            current = max(remaining, key=lambda candidate:
                          len(current & candidate) / len(current | candidate))
            remaining.remove(current)
    # 演算器を使わない出口・return等は、演算BBの後ろにまとめる。
    return ordered + empty_bbs


def schedule(
    cdfg: nx.DiGraph,
    operators: list[Operator] = None,
) -> nx.DiGraph:
    """
    BB内を相対スケジュールし、入口・初期化・資源が似たBBの順に区間を配置する。

    変更: 配置番号の順序と実行順序を分離する。実行順序を表すCFGは変更しない。
    各BBのstart_state/end_stateは実行区間の両端（終端を含む）。
    探索対象のbb_init自身のstateは、これとは別の入口判定用ステートである。
    FSMの遷移と、配置番号の大小に依存しない生存区間への対応は後段で必要。
    """
    if operators is None:
        operators = load_dp()
    entry_state_count = _assign_entry_states(cdfg)
    all_bbs = _get_all_bbs(cdfg)
    op_map = {op.name: op for op in operators}

    # 変更: 既存のschedule_bbの演算選択・占有期間の計算をそのまま使い、
    # まず全BBについて0始まりの相対位置と、演算が完了するまでの長さを求める。
    for bb_id in all_bbs:
        local_end = schedule_bb(cdfg, bb_id, operators, state_offset=0)
        resources = set()
        for nid in _get_bb_subgraph(cdfg, bb_id):
            operator = op_map[cdfg.nodes[nid]['assigned_op']]
            resources.add(operator.resource or operator.name)
        cdfg.nodes[bb_id]['resources'] = sorted(resources)
        # 空BBにも制御遷移先として1ステートを確保する。演算を追加するわけではない。
        cdfg.nodes[bb_id]['state_count'] = max(1, local_end + 1)

    # 入口判定は先頭に置き、その後に探索対象initの初期化区間をプログラム順で置く。
    entry_bbs = sorted(
        (bb for bb in all_bbs if cdfg.nodes[bb].get('dispatch_candidate')),
        key=lambda bb: cdfg.nodes[bb]['state'],
    )
    remaining = [bb for bb in all_bbs if not cdfg.nodes[bb].get('dispatch_candidate')]
    layout = entry_bbs + _order_bbs_by_resources(cdfg, remaining)
    current_offset = entry_state_count
    for bb_id in layout:
        data = cdfg.nodes[bb_id]
        data['start_state'] = current_offset
        data['end_state'] = current_offset + data['state_count'] - 1
        for nid in _get_bb_subgraph(cdfg, bb_id):
            cdfg.nodes[nid]['state'] = current_offset + cdfg.nodes[nid]['local_state']
        current_offset = data['end_state'] + 1

    # 変更: 後段が空BBも含めて配置を参照できるよう、区間順と終了位置を記録する。
    cdfg.graph['bb_layout'] = layout
    cdfg.graph['entry_state_count'] = entry_state_count
    cdfg.graph['done_state'] = current_offset
    cdfg.graph['state_count'] = current_offset + 1
    return _add_done_state(cdfg, done_state=current_offset)


def print_schedule(cdfg: nx.DiGraph):
    """スケジューリング結果を表示する。"""
    bbs = _get_all_bbs(cdfg)

    for bb_id in bbs:
        bb_name = cdfg.nodes[bb_id].get('op_name', 'bb')
        # 変更: 演算を持たない入口でも、判定用ステートの配置を確認できるよう表示する。
        if cdfg.nodes[bb_id].get('dispatch_candidate'):
            entry_state = cdfg.nodes[bb_id].get('state')
            print(f"[{bb_id}] {bb_name}: entry-check state {entry_state}")
        # 変更: 空BBの制御区間も含め、配置先と使用資源を表示する。
        bb_data = cdfg.nodes[bb_id]
        if 'start_state' in bb_data:
            print(f"[{bb_id}] {bb_name}: states {bb_data['start_state']}..{bb_data['end_state']}"
                  f" resources={bb_data['resources']}")
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




def _add_done_state(cdfg: nx.DiGraph, done_state: int | None = None) -> nx.DiGraph:
    """関数の終了ノードへ、全区間の後ろのdoneステートを設定する。"""
    if done_state is None:
        # 変更: 再配置前のdoneを数えず、演算の完了と空BBの区間末尾も考慮する。
        last_states = []
        for _, data in cdfg.nodes(data=True):
            if data.get('op_name') == 'done':
                continue
            if data.get('end_state') is not None:
                last_states.append(data['end_state'])
            if data.get('state') is not None:
                latency = data.get('latency', 1) if data.get('type') == 'op' else 1
                last_states.append(data['state'] + latency - 1)
        if not last_states:
            return cdfg
        done_state = max(last_states) + 1

    # 変更: 通常演算やループのexitは関数終了ではないため、doneを追加しない。
    # 新形式はfunction_exitを使い、旧形式でも次のBBがない出口だけを対象にする。
    function_exits = [n for n, d in cdfg.nodes(data=True) if d.get('function_exit')]
    exit_bbs = function_exits or [
        n for n, d in cdfg.nodes(data=True)
        if d.get('op_name') == 'bb_exit' and not _get_ctrl_children(cdfg, n)
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
        if data.get('op_name') == 'done':
            cdfg.nodes[nid]['state'] = done_state

    return cdfg


def reset_states(cdfg: nx.DiGraph) -> nx.DiGraph:
    """全ノードのstate属性をリセットする。スピルノード挿入後の再スケジューリング用。"""
    # 変更: 区間と相対位置も破棄し、スピル挿入後に古い配置情報を参照させない。
    for nid in cdfg.nodes():
        for attr in ('state', 'local_state', 'start_state', 'end_state', 'state_count', 'resources'):
            cdfg.nodes[nid].pop(attr, None)
    for attr in ('bb_layout', 'entry_state_count', 'done_state', 'state_count'):
        cdfg.graph.pop(attr, None)
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
