"""
reporter.py: NISCコンパイラのコンパイルレポート生成

3種類のユーザー向けにレポートを生成する:
  1. アプリケーション開発者: 総ステート数、レジスタ使用数、スピル数
  2. DP設計者: 演算器ごとの使用率、必要な演算器の種類と数
  3. コンパイラ設計者: スケジューリングのイテレーション数、最適化の影響
"""
from __future__ import annotations
from collections import defaultdict
import networkx as nx


def generate_report(
    cdfg: nx.DiGraph,
    reg_map: dict[str, int],
    imm_map: dict[str, int],
    spill_map: dict[str, int],
    spill_base_reg: int,
    num_registers: int,
    iterations: int,
    source_file: str = "",
) -> dict:
    """
    コンパイル結果のレポートデータを生成する。

    Returns:
        レポートデータの辞書
    """
    # ステート数を集計
    state_info = defaultdict(list)  # state → [(nid, op_name, assigned_op)]
    for nid, data in cdfg.nodes(data=True):
        state = data.get('state')
        if state is not None and data.get('type') in ('op', 'ctrl'):
            op_name = data.get('op_name', '?')
            assigned_op = data.get('assigned_op', op_name)
            state_info[state].append({
                'nid': nid,
                'op_name': op_name,
                'assigned_op': assigned_op,
                'is_spill': data.get('spill_offset') is not None or data.get('spill_addr') is not None,
                'is_done': op_name == 'done',
            })

    total_states = max(state_info.keys()) + 1 if state_info else 0

    # 演算器ごとの使用率
    op_usage = defaultdict(int)
    spill_states = 0
    for state, nodes in state_info.items():
        for node in nodes:
            if node['is_spill']:
                spill_states += 1
                op_usage['spill_mem'] += 1
            elif node['is_done']:
                pass
            else:
                op_usage[node['assigned_op']] += 1

    # レジスタ使用数
    gpr_count = max((v for v in reg_map.values() if v >= 0), default=-1) + 1
    imm_count = max(imm_map.values(), default=-1) + 1 if imm_map else 0

    # スピル情報
    spill_count = len(spill_map) if spill_map else 0

    # ステートの内訳
    state_breakdown = []
    for state in sorted(state_info.keys()):
        nodes = state_info[state]
        for node in nodes:
            category = 'spill' if node['is_spill'] else ('done' if node['is_done'] else 'compute')
            state_breakdown.append({
                'state': state,
                'op': node['assigned_op'],
                'category': category,
            })

    return {
        'source_file': source_file,
        'num_registers': num_registers,
        'gpr_used': gpr_count,
        'imm_used': imm_count,
        'spill_count': spill_count,
        'spill_map': spill_map or {},
        'spill_base_reg': spill_base_reg,
        'total_states': total_states,
        'iterations': iterations,
        'op_usage': dict(op_usage),
        'state_breakdown': state_breakdown,
    }


def print_report(report: dict):
    """レポートを表示する。"""
    sep = '=' * 52

    print(sep)
    print('  NISC Compilation Report')
    print(sep)

    if report['source_file']:
        print(f"  Program   : {report['source_file']}")

    print(f"  Registers : {report['gpr_used']} GPR / {report['num_registers']} available"
          f"   +   {report['imm_used']} IMM")
    print(f"  Spills    : {report['spill_count']}", end="")
    if report['spill_count'] > 0:
        spill_vars = ', '.join(report['spill_map'].keys())
        print(f"  ({spill_vars})", end="")
    print()
    print(f"  States    : {report['total_states']} total")
    print(f"  Iterations: {report['iterations']}")
    print()

    # ステートの内訳
    print('  State breakdown:')
    for entry in report['state_breakdown']:
        tag = ''
        if entry['category'] == 'spill':
            tag = '  [spill]'
        elif entry['category'] == 'done':
            tag = '  [done]'
        print(f"    state {entry['state']:2d}: {entry['op']:<12}{tag}")
    print()

    # 演算器使用率
    print('  Operator usage:')
    op_map = {
        'addi': 'ALU', 'subi': 'ALU', 'andi': 'ALU',
        'ori': 'ALU', 'xori': 'ALU', 'shli': 'ALU',
        'shrsi': 'ALU', 'shrui': 'ALU',
        'muli': 'MUL', 'mulf': 'MUL',
        'cmpi': 'CMP', 'cmpf': 'CMP',
        'load': 'MEM', 'store': 'MEM', 'spill_mem': 'MEM(spill)',
    }
    grouped = defaultdict(int)
    for op, count in report['op_usage'].items():
        group = op_map.get(op, op.upper())
        grouped[group] += count

    for group, count in sorted(grouped.items()):
        bar = '█' * count
        print(f"    {group:<12}: {bar} ({count})")
    print()

    print(sep)