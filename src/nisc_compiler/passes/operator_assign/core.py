"""
matcher.py: CDFGへのDP演算器パターンマッチング

networkxの部分グラフ同型探索でCDFGからDP演算器パターンを検出し、
マッチしたノードにassigned_op属性を追加する。

マッチング結果としてCDFGノードに追加される属性:
  assigned_op: 演算器名（例: "mac_f32"）
  latency:     レイテンシ（サイクル数）
  match_group: 同じ演算器に割り当てられたノードのグループID
"""
from __future__ import annotations
import networkx as nx
from networkx.algorithms import isomorphism

from nisc_compiler.passes.operator_assign.dp import Operator, load_dp


class MatchResult:
    """マッチング結果を保持するクラス。"""
    def __init__(self, operator: Operator, cdfg_nodes: list[int]):
        self.operator = operator      # マッチしたDP演算器
        self.cdfg_nodes = cdfg_nodes  # CDFGのノードIDリスト

    def __repr__(self):
        return f"MatchResult({self.operator.name}, nodes={self.cdfg_nodes})"

def _node_match(n1: dict, n2: dict) -> bool:
    if n1.get('type') != 'op':
        return False
    if n1.get('op_name') != n2.get('op_name'):
        return False
    pat_typ = n2.get('typ', '')
    if pat_typ:
        # cmpi/cmpfはresultがi1なのでoperand_typで照合
        if n1.get('op_name') in ('arith.cmpi', 'arith.cmpf'):
            return n1.get('operand_typ') == pat_typ
        return n1.get('mlir_typ') == pat_typ
    return True


def find_matches(
    cdfg: nx.DiGraph,
    operator: Operator,
    already_assigned: set[int] = None,
) -> list[dict[int, int]]:
    """
    CDFGからDP演算器パターンにマッチする部分グラフを全て探す。

    Args:
        cdfg:             対象のCDFG
        operator:         マッチするDP演算器
        already_assigned: 既に割り当て済みのCDFGノードID集合

    Returns:
        マッチのリスト。各マッチは {パターンノードID: CDFGノードID} の辞書。
    """
    if already_assigned is None:
        already_assigned = set()

    matcher = isomorphism.DiGraphMatcher(
        cdfg,
        operator.pattern,
        node_match=_node_match,
    )

    matches = []
    for iso in matcher.subgraph_isomorphisms_iter():
        # iso: {CDFGノードID: パターンノードID}
        cdfg_nodes = list(iso.keys())

        # 既に割り当て済みのノードが含まれていればスキップ
        if any(n in already_assigned for n in cdfg_nodes):
            continue

        # パターンノードID → CDFGノードID の辞書に変換
        match = {pat_id: cdfg_id for cdfg_id, pat_id in iso.items()}
        matches.append(match)

    return matches


def match_all(
    cdfg: nx.DiGraph,
    operators: list[Operator] = None,
) -> tuple[nx.DiGraph, list[MatchResult]]:
    """
    全DP演算器パターンをCDFGにマッチングし、
    マッチしたノードにassigned_op属性を追加する。

    大きいパターン（複合演算器）を先にマッチングし、
    マッチ済みノードは再マッチングしない（貪欲マッチング）。

    Args:
        cdfg:      対象のCDFG（変更される）
        operators: DP演算器リスト（Noneでデフォルト）

    Returns:
        (更新されたCDFG, マッチ結果リスト)
    """
    if operators is None:
        operators = load_dp()

    already_assigned: set[int] = set()
    results: list[MatchResult] = []
    group_id = 0

    for operator in operators:
        matches = find_matches(cdfg, operator, already_assigned)

        for match in matches:
            # match: {パターンノードID: CDFGノードID}
            cdfg_nodes = list(match.values())

            # CDFGノードに演算器割当情報を追加
            for cdfg_node_id in cdfg_nodes:
                cdfg.nodes[cdfg_node_id]['assigned_op'] = operator.name
                cdfg.nodes[cdfg_node_id]['latency'] = operator.latency
                cdfg.nodes[cdfg_node_id]['match_group'] = group_id

            already_assigned.update(cdfg_nodes)
            results.append(MatchResult(operator, cdfg_nodes))
            group_id += 1

    # マッチングできなかったopノードを収集
    unmatched = [
        (nid, data['op_name'])
        for nid, data in cdfg.nodes(data=True)
        if data.get('type') == 'op' and 'assigned_op' not in data
    ]
    if unmatched:
        for nid, op_name in unmatched:
            cdfg.nodes[nid]['unmatched'] = True

    return cdfg, results, unmatched


class MatchError(Exception):
    """マッチングできない演算があった場合のエラー。"""
    def __init__(self, unmatched: list[tuple[int, str]]):
        self.unmatched = unmatched
        ops = ", ".join(f"{op}(node{nid})" for nid, op in unmatched)
        super().__init__(
            f"Unmatched operations: {ops}\n"
            f"These operations have no corresponding DP operator.\n"
            f"Consider adding the operator to dp.py or implement expander.py."
        )


def check_unmatched(unmatched: list[tuple[int, str]], strict: bool = False):
    """
    unmatchedノードがある場合の処理。
    strict=True: 例外を発生させる
    strict=False: 警告を表示するだけ
    """
    if not unmatched:
        return
    if strict:
        raise MatchError(unmatched)
    else:
        print(f"WARNING: {len(unmatched)} unmatched operation(s):")
        for nid, op_name in unmatched:
            print(f"  node[{nid}] {op_name} ← no DP operator assigned")
        print("  → Run expander.py to expand these operations.")


def print_match_results(cdfg: nx.DiGraph, results: list[MatchResult], unmatched: list[tuple[int, str]] = None):
    """マッチング結果を表示する。"""
    if not results:
        print("No matches found.")
    else:
        print(f"Matched {len(results)} operator(s):")
        print()
        for r in results:
            op = r.operator
            print(f"  [{op.name}] latency={op.latency}")
            for nid in r.cdfg_nodes:
                data = cdfg.nodes[nid]
                print(f"    node[{nid}] {data['op_name']}")

    if unmatched:
        print()
        check_unmatched(unmatched, strict=False)
    else:
        print()
        print("All op nodes matched.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from parser import parse_c
    from lowering import lower_mlir
    from dp import load_dp

    if len(sys.argv) < 2:
        print("使い方: python matcher.py <Cファイル>")
        sys.exit(1)

    c_src = open(sys.argv[1]).read()
    mlir_text = parse_c(c_src)
    cdfg = lower_mlir(mlir_text)

    # ALU + MUL + CMP + FPU + FMUL + MAC
    operators = load_dp(mul=True, cmp=True, fpu=True, fmul=True, mac_i=True, mac_f=True,
                        sqrt=True, sin=True, cos=True, exp=True, log=True)

    cdfg, results, unmatched = match_all(cdfg, operators)
    print_match_results(cdfg, results, unmatched)