"""
lowering.py: MLIRテキスト → networkx CDFG（Control Data Flow Graph）変換

ノードの種類:
  type="arg"  関数引数・ブロック引数
  type="op"   演算ノード（arith.*, math.*）← 演算器を使う
  type="ctrl" 制御ノード（scf.*, func.return, arith.constant）
  type="bb"   BasicBlockノード

エッジの種類:
  type="data"  SSA変数の依存関係
  type="ctrl"  制御フロー（BB間の遷移・包含関係）
  label:       "then"/"else"/"body"/"cond"/"exit"/"back" で遷移の種類を表す
"""
from __future__ import annotations
import networkx as nx

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects import arith, scf, memref, func, math, builtin
from xdsl.ir import Operation, Block


# 演算器を使わない命令
CTRL_OPS = {
    "scf.for", "scf.while", "scf.if",
    "scf.condition", "scf.yield",
    "func.return",
    "arith.constant",
    "arith.index_cast",             # i32→index変換（演算器不要）
    "memref.alloca",
    # memref.load/storeはopノードとして扱う
}


def _ssa_name(value) -> str:
    if hasattr(value, 'name_hint') and value.name_hint is not None:
        return f"%{value.name_hint}"
    return f"%id_{id(value)}"


def _type_str(value) -> str:
    try:
        return str(value.type)
    except Exception:
        return "unknown"


class MLIRToCDFG:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_counter = 0
        self._value_to_node: dict[int, int] = {}
        self._sram_offset: int = 0          # ← 追加
        self._alloca_map: dict[str, int] = {}  # ← 追加 ssa名→SRAMアドレス
        self._alias_map: dict[str, str] = {}   # ← 追加 index_castエイリアス

    def lower(self, mlir_text: str) -> nx.DiGraph:
        ctx = Context()
        ctx.load_dialect(builtin.Builtin)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(scf.Scf)
        ctx.load_dialect(memref.MemRef)
        ctx.load_dialect(math.Math)

        parser = Parser(ctx, mlir_text)
        module = parser.parse_module()

        for op in module.body.block.ops:
            if op.name == "func.func":
                self._process_func(op)

        # bb_exitから外側ループのbb_incrへのエッジを後から張る
        # （bb_incrはforの処理が終わってから作られるため）
        self._fix_exit_edges()
        self._fix_body_edges()

        return self._graph

    def _fix_exit_edges(self):
        """
        bb_exitから外側ループのbb_incrへのエッジを張る。
        全forの処理が終わった後に呼ぶ。
        """
        # forノードのリストを収集（bb_incrを持つもの）
        for_nodes = [
            nid for nid, data in self._graph.nodes(data=True)
            if data.get('op_name') == 'scf.for' and 'bb_incr' in data
        ]

        for for_nid in for_nodes:
            bb_exit = self._graph.nodes[for_nid].get('bb_exit')
            bb_incr = self._graph.nodes[for_nid].get('bb_incr')
            if bb_exit is None or bb_incr is None:
                continue

            # このforを含む外側のforを探す
            # bb_bodyが親forのbb_bodyに含まれているかを確認
            parent_for = self._find_parent_for(for_nid, for_nodes)
            if parent_for is not None:
                parent_incr = self._graph.nodes[parent_for].get('bb_incr')
                if parent_incr is not None and not self._graph.has_edge(bb_exit, parent_incr):
                    self._graph.add_edge(bb_exit, parent_incr, type="ctrl", label="next")
            else:
                # 最外ループのbb_exit
                # 同じ親BBに次のforループがあればそのbb_initに繋ぐ
                # なければdoneに繋ぐ
                next_for = self._find_next_for(for_nid)
                if next_for is not None:
                    next_init = self._graph.nodes[next_for].get('bb_init')
                    if next_init is not None and not self._graph.has_edge(bb_exit, next_init):
                        self._graph.add_edge(bb_exit, next_init, type="ctrl", label="next")
                else:
                    # doneノードへ
                    done_nodes = [
                        nid for nid, d in self._graph.nodes(data=True)
                        if d.get('op_name') == 'done'
                    ]
                    for done_nid in done_nodes:
                        if not self._graph.has_edge(bb_exit, done_nid):
                            self._graph.add_edge(bb_exit, done_nid, type="ctrl", label="next")

    def _fix_body_edges(self):
        """
        bb_bodyの中にネストしたforがある場合、
        bb_bodyからネストしたforのbb_initへのエッジを張る。
        （bb_initでivがリセットされる）
        """
        edges_to_add = []
        for bb_id, data in self._graph.nodes(data=True):
            if data.get('op_name') != 'bb_body':
                continue
            for _, dst, edata in self._graph.out_edges(bb_id, data=True):
                if edata.get('label') == 'contains':
                    dst_data = self._graph.nodes[dst]
                    if dst_data.get('op_name') == 'scf.for':
                        bb_init = dst_data.get('bb_init')  # ← bb_condをbb_initに変更
                        if bb_init is not None:
                            if not self._graph.has_edge(bb_id, bb_init):
                                edges_to_add.append((bb_id, bb_init))
        for src, dst in edges_to_add:
            self._graph.add_edge(src, dst, type="ctrl", label="body")  # ← containsをbodyに変更

    def _find_parent_for(self, for_nid: int, for_nodes: list) -> int | None:
        """
        指定したforノードを含む外側のforノードを返す。
        bb_bodyのcontainsエッジを辿って探す。
        """
        bb_cond = self._graph.nodes[for_nid].get('bb_cond')
        if bb_cond is None:
            return None

        # bb_condに入ってくるエッジの元をたどって外側forを探す
        for candidate in for_nodes:
            if candidate == for_nid:
                continue
            candidate_body = self._graph.nodes[candidate].get('bb_body')
            if candidate_body is None:
                continue
            # candidate_bodyがfor_nidのbb_condの祖先かを確認
            # bb_bodyからcontainsエッジを辿ってbb_condを探す
            for _, dst, edata in self._graph.out_edges(candidate_body, data=True):
                if edata.get('label') == 'contains' and dst == for_nid:
                    return candidate

        return None
    
    def _find_next_for(self, for_nid: int) -> int | None:
        """
        同じ親BBに含まれる次のforループを返す。
        """
        # このforを含む親BBを探す
        parent_bb = None
        for nid, data in self._graph.nodes(data=True):
            if data.get('type') != 'bb':
                continue
            for _, dst, edata in self._graph.out_edges(nid, data=True):
                if edata.get('label') == 'contains' and dst == for_nid:
                    parent_bb = nid
                    break
            if parent_bb is not None:
                break

        if parent_bb is None:
            return None

        # 親BBに含まれるforノードを順番に取得
        sibling_fors = []
        for _, dst, edata in self._graph.out_edges(parent_bb, data=True):
            if edata.get('label') == 'contains':
                dst_data = self._graph.nodes[dst]
                if dst_data.get('op_name') == 'scf.for':
                    sibling_fors.append(dst)

        # for_nidの次のforを返す
        sibling_fors.sort()  # ノードIDでソート（生成順）
        for i, nid in enumerate(sibling_fors):
            if nid == for_nid and i + 1 < len(sibling_fors):
                return sibling_fors[i + 1]

        return None

    # ----------------------------------------------------------------
    # 関数処理
    # ----------------------------------------------------------------
    def _process_func(self, func_op: Operation):
        for region in func_op.regions:
            for block in region.blocks:
                for arg in block.args:
                    node_id = self._new_node()
                    self._graph.add_node(node_id,
                        type="arg",
                        op_name="arg",
                        operands=[],
                        results=[_ssa_name(arg)],
                        mlir_typ=_type_str(arg),
                    )
                    self._value_to_node[id(arg)] = node_id

                bb_id = self._new_bb("entry")
                self._process_block(block, bb_id)

    # ----------------------------------------------------------------
    # ブロック処理
    # ----------------------------------------------------------------
    def _process_block(self, block: Block, bb_id: int):
        for op in block.ops:
            self._process_op(op, bb_id)

    # ----------------------------------------------------------------
    # 命令処理
    # ----------------------------------------------------------------
    def _process_op(self, op: Operation, bb_id: int):
        # memref.allocaはSRAMアドレスを割り当ててスキップ
        if op.name == "memref.alloca":
            results = [_ssa_name(v) for v in op.results]
            mlir_typ = _type_str(op.results[0]) if op.results else ""
            size = self._get_memref_size(mlir_typ)
            if results and size:
                ssa = results[0]
                self._alloca_map[ssa] = self._sram_offset
                self._sram_offset += size
                # SSA名→定数ノードとして登録（アドレスをGPRにロードするため）
                const_id = self._new_node()
                self._graph.add_node(const_id,
                    type="ctrl",
                    op_name="arith.constant",
                    operands=[],
                    results=[ssa],
                    mlir_typ="i32",
                    const_value=self._alloca_map[ssa],
                )
                self._graph.add_edge(bb_id, const_id, type="ctrl", label="contains")
                self._value_to_node[ssa] = const_id
            return  # スキップ
        
        node_id = self._new_node()
        op_name = op.name

        operands = [self._alias_map.get(_ssa_name(v), _ssa_name(v)) for v in op.operands]
        results  = [_ssa_name(v) for v in op.results]

        if op.results:
            mlir_typ = _type_str(op.results[0])
        elif op.operands:
            mlir_typ = _type_str(op.operands[0])
        else:
            mlir_typ = "void"

        # operandsの型を保存（cmpi等でinputの型を識別するため）
        # operands[0]がindex型等で取れない場合は他のoperandを試す
        operand_typ = ""
        for v in op.operands:
            t = _type_str(v)
            if t and t not in ("", "none") and not t.startswith("memref") and not t.startswith("index"):
                operand_typ = t
                break

        node_type = "ctrl" if op_name in CTRL_OPS else "op"

        # arith.constantの値を取得
        const_value = None
        if op_name == "arith.constant":
            try:
                attr = op.value  # xdsl ConstantOpはop.valueで値を取得
                if hasattr(attr, 'value'):
                    inner = attr.value
                    if hasattr(inner, 'data'):
                        const_value = int(inner.data)
                    else:
                        const_value = int(inner)
                else:
                    const_value = int(attr)
            except Exception:
                const_value = 0

        # arith.index_castはエイリアスとして処理（ノードを作らない）
        if op_name == 'arith.index_cast':
            if operands and results:
                src_ssa = operands[0]
                dst_ssa = results[0]
                self._alias_map[dst_ssa] = src_ssa
                for v in op.operands:
                    src_node = self._value_to_node.get(id(v))
                    if src_node is not None:
                        for r in op.results:
                            self._value_to_node[id(r)] = src_node
            return

        node_attrs = dict(
            type=node_type,
            op_name=op_name,
            operands=operands,
            results=results,
            mlir_typ=mlir_typ,
            operand_typ=operand_typ,  # ← 追加
        )
        if const_value is not None:
            node_attrs['const_value'] = const_value

        self._graph.add_node(node_id, **node_attrs)

        for r in op.results:
            self._value_to_node[id(r)] = node_id

        # データエッジ
        for v in op.operands:
            src = self._value_to_node.get(id(v))
            if src is not None:
                self._graph.add_edge(src, node_id, type="data", ssa=_ssa_name(v))

        # BB → op の包含エッジ
        self._graph.add_edge(bb_id, node_id, type="ctrl", label="contains")

        # 制御構造ごとにCFGエッジを正しく構築
        if op_name == "scf.while":
            self._process_scf_while(node_id, op)
        elif op_name == "scf.for":
            self._process_scf_for(node_id, op)
        elif op_name == "scf.if":
            self._process_scf_if(node_id, op)
        elif op_name == "func.return":
            # func.returnをdoneノードに変換
            done_id = self._new_node()
            self._graph.add_node(done_id,
                type="ctrl",
                op_name="done",
                operands=[],
                results=[],
                mlir_typ="void",
            )
            self._graph.add_edge(bb_id, done_id, type="ctrl", label="contains")
        elif op_name in ("memref.load", "memref.store"):
            # 2次元配列のアドレス計算ノードを挿入
            # operands: load=[base, idx0, idx1], store=[val, base, idx0, idx1]
            base_offset = 1 if op_name == "memref.store" else 0
            indices = list(op.operands[base_offset + 1:])
            base_val = op.operands[base_offset]

            # memrefの型からcolsを取得
            base_typ = _type_str(base_val)
            cols = self._get_memref_cols(base_typ)

            if cols is not None and len(indices) == 2:
                # idx0（行）のSSA名とノードを取得
                idx0_val = indices[0]
                idx1_val = indices[1]

                # index_castを辿ってi32の値を取得
                idx0_i32 = self._resolve_index_to_i32(idx0_val)
                idx1_i32 = self._resolve_index_to_i32(idx1_val)
                # エイリアスマップで解決
                idx0_i32 = self._alias_map.get(idx0_i32, idx0_i32)
                idx1_i32 = self._alias_map.get(idx1_i32, idx1_i32)
                cols_ssa = f"%cols_{node_id}"
                offset_ssa = f"%offset_{node_id}"
                addr_ssa = f"%addr_{node_id}"

                # arith.constant cols
                cols_id = self._new_node()
                self._graph.add_node(cols_id,
                    type="ctrl",
                    op_name="arith.constant",
                    operands=[],
                    results=[cols_ssa],
                    mlir_typ="i32",
                    const_value=cols,
                )
                self._graph.add_edge(bb_id, cols_id, type="ctrl", label="contains")
                self._value_to_node[cols_ssa] = cols_id

                # arith.muli idx0 * cols → offset1
                mul_ssa = f"%mul_{node_id}"
                mul_id = self._new_node()
                idx0_src = self._value_to_node.get(id(idx0_val))
                self._graph.add_node(mul_id,
                    type="op",
                    op_name="arith.muli",
                    operands=[idx0_i32, cols_ssa],
                    results=[mul_ssa],
                    mlir_typ="i32",
                    operand_typ="i32",
                )
                self._graph.add_edge(bb_id, mul_id, type="ctrl", label="contains")
                if idx0_src:
                    self._graph.add_edge(idx0_src, mul_id, type="data", ssa=idx0_i32)
                self._graph.add_edge(cols_id, mul_id, type="data", ssa=cols_ssa)
                self._value_to_node[mul_ssa] = mul_id

                # arith.addi mul + idx1 → offset
                idx1_src = self._value_to_node.get(id(idx1_val))
                add1_id = self._new_node()
                self._graph.add_node(add1_id,
                    type="op",
                    op_name="arith.addi",
                    operands=[mul_ssa, idx1_i32],
                    results=[offset_ssa],
                    mlir_typ="i32",
                    operand_typ="i32",
                )
                self._graph.add_edge(bb_id, add1_id, type="ctrl", label="contains")
                self._graph.add_edge(mul_id, add1_id, type="data", ssa=mul_ssa)
                if idx1_src:
                    self._graph.add_edge(idx1_src, add1_id, type="data", ssa=idx1_i32)
                self._value_to_node[offset_ssa] = add1_id

                # arith.addi base + offset → addr
                base_src = self._value_to_node.get(id(base_val))
                base_i32 = _ssa_name(base_val)
                add2_id = self._new_node()
                self._graph.add_node(add2_id,
                    type="op",
                    op_name="arith.addi",
                    operands=[base_i32, offset_ssa],
                    results=[addr_ssa],
                    mlir_typ="i32",
                    operand_typ="i32",
                )
                self._graph.add_edge(bb_id, add2_id, type="ctrl", label="contains")
                if base_src:
                    self._graph.add_edge(base_src, add2_id, type="data", ssa=base_i32)
                self._graph.add_edge(add1_id, add2_id, type="data", ssa=offset_ssa)
                self._value_to_node[addr_ssa] = add2_id

                # load/storeノードのoperandsを更新
                if op_name == "memref.load":
                    self._graph.nodes[node_id]['operands'] = [addr_ssa]
                    self._graph.add_edge(add2_id, node_id, type="data", ssa=addr_ssa)
                else:  # store
                    val_ssa = _ssa_name(op.operands[0])
                    self._graph.nodes[node_id]['operands'] = [val_ssa, addr_ssa]
                    self._graph.add_edge(add2_id, node_id, type="data", ssa=addr_ssa)

    def _get_memref_cols(self, typ: str) -> int | None:
        """memref<?x?xi32>等からcols（列数）を取得する。"""
        # memref<4x4xi32> → 4
        # memref<?x?xi32> → None（動的サイズ）
        import re
        m = re.match(r'memref<(\d+)x(\d+)x\w+>', typ)
        if m:
            return int(m.group(2))
        return None

    def _get_memref_size(self, typ: str) -> int | None:
        """memref<4x4xi32>等から総要素数を返す。"""
        import re
        m = re.match(r'memref<(\d+)x(\d+)x\w+>', typ)
        if m:
            return int(m.group(1)) * int(m.group(2))
        # 1次元の場合
        m = re.match(r'memref<(\d+)x\w+>', typ)
        if m:
            return int(m.group(1))
        return None

    def _resolve_index_to_i32(self, val) -> str:
        """index型の値をi32のSSA名に解決する。"""
        # index_castを辿ってi32の値を取得
        if hasattr(val, 'op') and val.op.name == 'arith.index_cast':
            inner = list(val.op.operands)[0]
            return _ssa_name(inner)
        return _ssa_name(val)

    # ----------------------------------------------------------------
    # scf.while の CFG構造
    #
    #   while → BB_init → BB_cond → BB_do → BB_incr → BB_cond（ループバック）
    #                             ↘ BB_exit（条件が偽）
    # ----------------------------------------------------------------
    def _process_scf_while(self, while_id: int, op: Operation):
        regions = list(op.regions)
        if len(regions) < 2:
            return

        # BB_init: iter_argsの初期値をGPRにロード
        bb_init = self._new_bb("init")
        self._graph.add_edge(while_id, bb_init, type="ctrl", label="init")
        # iter_argsのオペランド（初期値）をinit BBに包含
        for v in op.operands:
            src = self._value_to_node.get(id(v))
            if src is not None:
                self._graph.add_edge(bb_init, src, type="ctrl", label="contains")

        # BB_cond: 条件ブロック（region[0]）
        bb_cond = self._new_bb("cond")
        self._graph.add_edge(bb_init, bb_cond, type="ctrl", label="cond")
        self._add_block_args(regions[0].blocks[0], bb_cond)
        self._process_block(regions[0].blocks[0], bb_cond)

        # BB_do: doブロック（region[1]）
        bb_do = self._new_bb("do")
        self._add_block_args(regions[1].blocks[0], bb_do)
        self._process_block(regions[1].blocks[0], bb_do)

        # BB_exit: ループ後
        bb_exit = self._new_bb("exit")

        # CFGエッジ
        self._graph.add_edge(bb_cond, bb_do,   type="ctrl", label="body")   # 条件が真
        self._graph.add_edge(bb_cond, bb_exit,  type="ctrl", label="exit")   # 条件が偽
        self._graph.add_edge(bb_do,   bb_cond,  type="ctrl", label="back")   # ループバック

        self._graph.nodes[while_id]['bb_init'] = bb_init
        self._graph.nodes[while_id]['bb_cond'] = bb_cond
        self._graph.nodes[while_id]['bb_do']   = bb_do
        self._graph.nodes[while_id]['bb_exit'] = bb_exit

    # ----------------------------------------------------------------
    # scf.for の CFG構造
    #
    #   for → BB_init → BB_cond → BB_body → BB_incr → BB_cond（ループバック）
    #                           ↘ BB_exit（ループ終了）
    # ----------------------------------------------------------------
    def _process_scf_for(self, for_id: int, op: Operation):
        regions = list(op.regions)
        if len(regions) < 1:
            return

        # ネストしたforのスタックに積む
        if not hasattr(self, '_for_stack'):
            self._for_stack = []
        self._for_stack.append(for_id)

        # BB_init: lb/ub/stepをGPRにロード・ivを初期化
        bb_init = self._new_bb("init")
        self._graph.add_edge(for_id, bb_init, type="ctrl", label="init")
        # forのオペランド（lb, ub, step）をinit BBに包含
        for v in op.operands:
            src = self._value_to_node.get(id(v))
            if src is not None:
                self._graph.add_edge(bb_init, src, type="ctrl", label="contains")

        # scf.forのオペランドを取得
        # operands: [lb(index), ub(index), step(index), iter_args...]
        operands_list = list(op.operands)
        lb_val   = operands_list[0] if len(operands_list) > 0 else None
        ub_val   = operands_list[1] if len(operands_list) > 1 else None
        step_val = operands_list[2] if len(operands_list) > 2 else None

        # block args: [iv(index), iter_arg0, ...]
        block_args_list = list(regions[0].blocks[0].args)
        iv_arg = block_args_list[0] if block_args_list else None

        # BB_init: ivをlbで初期化するノードを追加
        if iv_arg is not None and lb_val is not None:
            iv_name = _ssa_name(iv_arg)
            lb_i32 = None
            lb_src_node = None
            if hasattr(lb_val, 'op') and lb_val.op.name == 'arith.index_cast':
                lb_i32_val = list(lb_val.op.operands)[0]
                lb_i32 = _ssa_name(lb_i32_val)
                lb_src_node = self._value_to_node.get(id(lb_i32_val))
            else:
                lb_i32 = _ssa_name(lb_val)
                lb_src_node = self._value_to_node.get(id(lb_val))

            if lb_i32 is not None:
                init_id = self._new_node()
                self._graph.add_node(init_id,
                    type="op",
                    op_name="nisc.iv_init",
                    operands=[lb_i32],
                    results=[iv_name],
                    mlir_typ="i32",
                    operand_typ="i32",
                )
                self._value_to_node[iv_name] = init_id
                self._graph.add_edge(bb_init, init_id, type="ctrl", label="contains")
                if lb_src_node is not None:
                    self._graph.add_edge(lb_src_node, init_id, type="data", ssa=lb_i32)

        # BB_cond: 条件チェック（iv < ub）
        bb_cond = self._new_bb("cond")
        self._graph.add_edge(bb_init, bb_cond, type="ctrl", label="cond")

        # bb_condにcmpiノードを追加（iv < ub）
        # ub_valはindex型（index_cast経由）なので元のi32値を使う
        if iv_arg is not None and ub_val is not None:
            iv_name = _ssa_name(iv_arg)
            # index_castの入力（i32型の元の値）を取得
            ub_i32 = None
            if hasattr(ub_val, 'op') and ub_val.op.name == 'arith.index_cast':
                ub_i32_val = list(ub_val.op.operands)[0]
                ub_i32 = _ssa_name(ub_i32_val)
                ub_src_node = self._value_to_node.get(id(ub_i32_val))
            else:
                ub_i32 = _ssa_name(ub_val)
                ub_src_node = self._value_to_node.get(id(ub_val))

            cmp_id = self._new_node()
            cmp_result = f"%cmp_for_{for_id}"
            self._graph.add_node(cmp_id,
                type="op",
                op_name="arith.cmpi",
                operands=[iv_name, ub_i32],
                results=[cmp_result],
                mlir_typ="i1",
                operand_typ="i32",  # ← 追加
            )
            self._value_to_node[cmp_result] = cmp_id
            self._graph.add_edge(bb_cond, cmp_id, type="ctrl", label="contains")

            # iv_argのノードがあればデータエッジを追加
            iv_src = self._value_to_node.get(id(iv_arg))
            if iv_src is not None:
                self._graph.add_edge(iv_src, cmp_id, type="data", ssa=iv_name)
            if ub_src_node is not None:
                self._graph.add_edge(ub_src_node, cmp_id, type="data", ssa=ub_i32)

        # BB_body: ループ本体
        bb_body = self._new_bb("body")
        self._graph.add_edge(bb_cond, bb_body, type="ctrl", label="body")  # 条件が真
        self._add_block_args(regions[0].blocks[0], bb_body)
        self._process_block(regions[0].blocks[0], bb_body)

        # BB_incr: ivのインクリメント（iv = iv + step）
        bb_incr = self._new_bb("incr")
        self._graph.add_edge(bb_body, bb_incr, type="ctrl", label="incr")
        self._graph.add_edge(bb_incr, bb_cond, type="ctrl", label="back")  # ループバック

        # bb_incrにaddiノードを追加（iv = iv + step）
        # step_valはindex型（index_cast経由）なので元のi32値を使う
        if iv_arg is not None and step_val is not None:
            iv_name = _ssa_name(iv_arg)
            # index_castの入力（i32型の元の値）を取得
            step_i32 = None
            if hasattr(step_val, 'op') and step_val.op.name == 'arith.index_cast':
                step_i32_val = list(step_val.op.operands)[0]
                step_i32 = _ssa_name(step_i32_val)
                step_src_node = self._value_to_node.get(id(step_i32_val))
            else:
                step_i32 = _ssa_name(step_val)
                step_src_node = self._value_to_node.get(id(step_val))

            incr_id = self._new_node()
            incr_result = f"%incr_for_{for_id}"
            self._graph.add_node(incr_id,
                type="op",
                op_name="arith.addi",
                operands=[iv_name, step_i32],
                results=[incr_result],
                mlir_typ="i32",
            )
            self._value_to_node[incr_result] = incr_id
            self._graph.add_edge(bb_incr, incr_id, type="ctrl", label="contains")

            iv_src = self._value_to_node.get(id(iv_arg))
            if iv_src is not None:
                self._graph.add_edge(iv_src, incr_id, type="data", ssa=iv_name)
            if step_src_node is not None:
                self._graph.add_edge(step_src_node, incr_id, type="data", ssa=step_i32)

        # BB_exit: ループ後
        bb_exit = self._new_bb("exit")
        self._graph.add_edge(bb_cond, bb_exit, type="ctrl", label="exit")  # 条件が偽

        self._graph.nodes[for_id]['bb_init'] = bb_init
        self._graph.nodes[for_id]['bb_cond'] = bb_cond
        self._graph.nodes[for_id]['bb_body'] = bb_body
        self._graph.nodes[for_id]['bb_incr'] = bb_incr
        self._graph.nodes[for_id]['bb_exit'] = bb_exit

        # スタックから自分を取り出す
        self._for_stack.pop()

        # 外側ループのbb_incrへのエッジを張る
        if self._for_stack:
            parent_for_id = self._for_stack[-1]
            parent_incr = self._graph.nodes[parent_for_id].get('bb_incr')
            if parent_incr is not None:
                self._graph.add_edge(bb_exit, parent_incr, type="ctrl", label="next")

    # ----------------------------------------------------------------
    # scf.if の CFG構造
    #
    #   if → BB_then → BB_exit
    #      ↘ BB_else → BB_exit
    # ----------------------------------------------------------------
    def _process_scf_if(self, if_id: int, op: Operation):
        regions = list(op.regions)

        # BB_then（region[0]）
        bb_then = self._new_bb("then")
        self._graph.add_edge(if_id, bb_then, type="ctrl", label="then")
        self._process_block(regions[0].blocks[0], bb_then)

        # BB_else（region[1]があれば）
        if len(regions) >= 2:
            bb_else = self._new_bb("else")
            self._graph.add_edge(if_id, bb_else, type="ctrl", label="else")
            self._process_block(regions[1].blocks[0], bb_else)
            self._graph.nodes[if_id]['bb_else'] = bb_else

        self._graph.nodes[if_id]['bb_then'] = bb_then

    # ----------------------------------------------------------------
    # ブロック引数を登録
    # ----------------------------------------------------------------
    def _add_block_args(self, block: Block, bb_id: int):
        for arg in block.args:
            arg_id = self._new_node()
            self._graph.add_node(arg_id,
                type="arg",
                op_name="block_arg",
                operands=[],
                results=[_ssa_name(arg)],
                mlir_typ=_type_str(arg),
            )
            self._value_to_node[id(arg)] = arg_id
            self._graph.add_edge(bb_id, arg_id, type="ctrl", label="contains")

    # ----------------------------------------------------------------
    # ユーティリティ
    # ----------------------------------------------------------------
    def _new_node(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _new_bb(self, label: str = "") -> int:
        nid = self._node_counter
        self._node_counter += 1
        self._graph.add_node(nid,
            type="bb",
            op_name=f"bb_{label}" if label else "bb",
            operands=[],
            results=[],
            mlir_typ="",
        )
        return nid


# ----------------------------------------------------------------
# 公開API
# ----------------------------------------------------------------
def lower_mlir(mlir_text: str) -> nx.DiGraph:
    lowerer = MLIRToCDFG()
    cdfg = lowerer.lower(mlir_text)
    cdfg.graph['alloca_map'] = lowerer._alloca_map  # ← グラフ属性として保存
    return cdfg


def print_cdfg(g: nx.DiGraph):
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
    print()
    print("--- Nodes ---")
    for nid, data in g.nodes(data=True):
        t = data['type']
        marker = {"op": "★", "ctrl": "○", "bb": "□", "arg": "◇"}.get(t, "?")
        print(f"  {marker} [{nid}] {data['op_name']} ({t}) : {data['mlir_typ']}")
        if data['results']:
            print(f"       results:  {data['results']}")
        if data['operands']:
            print(f"       operands: {data['operands']}")
    print()
    print("--- Edges ---")
    for src, dst, data in g.edges(data=True):
        edge_type = data['type']
        label = data.get('label') or ''
        ssa = data.get('ssa') or ''
        info = f"{label} {ssa}".strip()
        print(f"  {src} -> {dst}  [{edge_type}] {info}")