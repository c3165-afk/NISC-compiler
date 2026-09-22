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
  label:       "next"/"then"/"else"/"body"/"cond"/"incr"/"exit"/"back" はBB間の遷移
               "contains" はBBへの所属（実行順序を表す遷移とは区別する）

変更: 各処理単位の入口にunit_kindとbb_init/bb_body/bb_exit等を記録する。
      空の入口・出口もCDFGに残す。ステート割り当てやFSM生成は後段で扱う。
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

        # 変更: 出口の接続は各ブロックを読む時点で行う。
        # 後から「次のfor」を探すと、その間の通常演算やifを飛び越えるため廃止した。
        return self._graph

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

                # 変更: 外側の演算を一つのbb_entryに集めず、処理単位ごとに分割する。
                bb_init, bb_exit = self._process_block(block, allow_return=True)
                self._graph.nodes[bb_init]['function_entry'] = True
                self._graph.nodes[bb_exit]['function_exit'] = True
                # 関数引数は実行開始時に利用できる値として、最初の入口に所属させる。
                for arg in block.args:
                    arg_id = self._value_to_node[id(arg)]
                    self._graph.add_edge(bb_init, arg_id, type="ctrl", label="contains")

    # ----------------------------------------------------------------
    # ブロック処理
    # ----------------------------------------------------------------
    def _process_block(
        self, block: Block, bb_id: int | None = None, *, allow_return: bool = False,
    ) -> tuple[int, int]:
        # 変更: 通常演算の連続区間と制御構造を分け、入口・出口をソース順につなぐ。
        # bb_idは親のbody/cond。入れ子がなければ従来どおりそこに演算を追加する。
        first_bb = bb_id
        last_bb = bb_id
        current_body = bb_id
        ops = list(block.ops)
        for index, op in enumerate(ops):
            if op.name in ("scf.for", "scf.while", "scf.if", "func.return"):
                # 変更: 今回は関数末尾のreturnのみ対応し、早期returnは実装しない。
                if op.name == "func.return" and (not allow_return or index != len(ops) - 1):
                    raise ValueError("途中のreturnには未対応です。関数末尾のreturnのみ使用できます。")
                bb_init = self._new_bb("init")
                node_id = self._process_op(op, bb_init)
                bb_exit = self._graph.nodes[node_id]['bb_exit']
                if last_bb is not None:
                    self._graph.add_edge(last_bb, bb_init, type="ctrl", label="next")
                if first_bb is None:
                    first_bb = bb_init
                last_bb = bb_exit
                current_body = None
            else:
                if current_body is None:
                    bb_init, current_body, bb_exit = self._new_plain_unit()
                    if last_bb is not None:
                        self._graph.add_edge(last_bb, bb_init, type="ctrl", label="next")
                    if first_bb is None:
                        first_bb = bb_init
                    last_bb = bb_exit
                self._process_op(op, current_body)

        # 空のブロックにも入口から出口までの経路を残す。
        if first_bb is None:
            first_bb, _, last_bb = self._new_plain_unit()
        return first_bb, last_bb

    def _new_plain_unit(self) -> tuple[int, int, int]:
        # 変更: 通常演算もinit → body → exitの共通形式にする。
        bb_init = self._new_bb("init")
        bb_body = self._new_bb("body")
        bb_exit = self._new_bb("exit")
        self._graph.add_edge(bb_init, bb_body, type="ctrl", label="body")
        self._graph.add_edge(bb_body, bb_exit, type="ctrl", label="exit")
        self._graph.nodes[bb_init].update(
            unit_kind="normal", bb_init=bb_init, bb_body=bb_body, bb_exit=bb_exit,
        )
        return bb_init, bb_body, bb_exit

    # ----------------------------------------------------------------
    # 命令処理
    # ----------------------------------------------------------------
    def _process_op(self, op: Operation, bb_id: int) -> int | None:
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

        # 変更: 制御ノードは専用入口に所属させ、BB同士の遷移を構築する。
        if op_name == "scf.while":
            self._process_scf_while(node_id, op, bb_id)
        elif op_name == "scf.for":
            self._process_scf_for(node_id, op, bb_id)
        elif op_name == "scf.if":
            self._process_scf_if(node_id, op, bb_id)
        elif op_name == "func.return":
            # 変更: returnはinit → body(return) → exit(done)として管理する。
            bb_body = self._new_bb("body")
            bb_exit = self._new_bb("exit")
            self._graph.remove_edge(bb_id, node_id)
            self._graph.add_edge(bb_body, node_id, type="ctrl", label="contains")
            self._graph.add_edge(bb_id, bb_body, type="ctrl", label="body")
            self._graph.add_edge(bb_body, bb_exit, type="ctrl", label="exit")
            done_id = self._new_node()
            self._graph.add_node(done_id,
                type="ctrl",
                op_name="done",
                operands=[],
                results=[],
                mlir_typ="void",
            )
            self._graph.add_edge(bb_exit, done_id, type="ctrl", label="contains")
            self._graph.nodes[node_id].update(bb_init=bb_id, bb_body=bb_body, bb_exit=bb_exit)
            self._graph.nodes[bb_id].update(
                unit_kind="return", bb_init=bb_id, bb_body=bb_body, bb_exit=bb_exit,
            )
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

        return node_id

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
    #   BB_init → BB_cond → BB_body → BB_cond（ループバック）
    #                ↘ BB_exit（条件が偽）
    # ----------------------------------------------------------------
    def _process_scf_while(self, while_id: int, op: Operation, bb_init: int):
        regions = list(op.regions)
        if len(regions) != 2:
            raise ValueError("scf.whileには条件と本体の2つの領域が必要です。")

        # 変更: 入力の生成元をinitに再所属させず、元のBBとデータ依存を保つ。
        bb_cond = self._new_bb("cond")
        self._graph.add_edge(bb_init, bb_cond, type="ctrl", label="cond")
        self._add_block_args(regions[0].blocks[0], bb_cond)
        _, cond_end = self._process_block(regions[0].blocks[0], bb_cond)

        # 変更: whileのbb_doを共通のbb_bodyに統一する。
        bb_body = self._new_bb("body")
        self._add_block_args(regions[1].blocks[0], bb_body)
        _, body_end = self._process_block(regions[1].blocks[0], bb_body)
        bb_exit = self._new_bb("exit")

        # 入れ子がある場合も、領域の最後から分岐・ループバックする。
        self._graph.add_edge(cond_end, bb_body, type="ctrl", label="body")
        self._graph.add_edge(cond_end, bb_exit, type="ctrl", label="exit")
        self._graph.add_edge(body_end, bb_cond, type="ctrl", label="back")
        condition = list(regions[0].blocks[0].ops)[-1]
        if condition.name == "scf.condition" and condition.operands:
            self._graph.nodes[cond_end]['condition'] = _ssa_name(condition.operands[0])
            self._graph.nodes[cond_end]['condition_node'] = self._value_to_node.get(id(condition.operands[0]))

        self._graph.nodes[while_id].update(
            bb_init=bb_init, bb_cond=bb_cond, bb_body=bb_body,
            bb_do=bb_body, bb_exit=bb_exit,
        )
        # bb_do属性のみ後段との互換用に残し、実際のBB名はbb_bodyとする。
        self._graph.nodes[bb_init].update(
            unit_kind="while", bb_init=bb_init, bb_cond=bb_cond,
            bb_body=bb_body, bb_exit=bb_exit,
        )

    # ----------------------------------------------------------------
    # scf.for の CFG構造
    #
    #   BB_init → BB_cond → BB_body → BB_incr → BB_cond（ループバック）
    #                ↘ BB_exit（ループ終了）
    # ----------------------------------------------------------------
    def _process_scf_for(self, for_id: int, op: Operation, bb_init: int):
        regions = list(op.regions)
        if len(regions) != 1:
            raise ValueError("scf.forには本体の領域が必要です。")

        # 変更: bb_initは呼び出し元が作成する専用入口を使用する。
        # lb/ub/stepの生成演算は元のBBに残し、ここではivの初期化だけを追加する。
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
        _, body_end = self._process_block(regions[0].blocks[0], bb_body)

        # BB_incr: ivのインクリメント（iv = iv + step）
        bb_incr = self._new_bb("incr")
        # 変更: 入れ子とその後の演算を終えた出口から更新へ進む。
        self._graph.add_edge(body_end, bb_incr, type="ctrl", label="incr")
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

        self._graph.nodes[bb_init].update(
            unit_kind="for", bb_init=bb_init, bb_cond=bb_cond,
            bb_body=bb_body, bb_incr=bb_incr, bb_exit=bb_exit,
        )
        # 変更: 次の処理への接続は_process_blockが担当する。
        # 内側forの直後にある通常演算を飛ばして外側incrへ接続しない。

    # ----------------------------------------------------------------
    # scf.if の CFG構造
    #
    #   BB_init → BB_cond → BB_body(then) → BB_exit
    #                ↘ BB_body(else) ──────↗
    # ----------------------------------------------------------------
    def _process_scf_if(self, if_id: int, op: Operation, bb_init: int):
        regions = list(op.regions)
        bb_cond = self._new_bb("cond")
        bb_exit = self._new_bb("exit")
        self._graph.add_edge(bb_init, bb_cond, type="ctrl", label="cond")

        # 変更: 分岐条件をcondに記録する。共有される条件の生成演算は移動しない。
        condition = op.operands[0]
        cond_src = self._value_to_node.get(id(condition))
        self._graph.nodes[bb_cond]['condition'] = _ssa_name(condition)
        self._graph.nodes[bb_cond]['condition_node'] = cond_src
        # 同じMLIRブロック内で、このifだけが使用する比較をcondに所属させる。
        # 外側で計算した条件をループ内部へ移し、反復ごとに再計算することは避ける。
        if (cond_src is not None
                and self._graph.nodes[cond_src].get('op_name') in ('arith.cmpi', 'arith.cmpf')
                and isinstance(condition.owner, Operation)
                and condition.owner.parent_block() is op.parent_block()
                and all(use.operation is op for use in condition.uses)):
            owners = [src for src, _, edge in self._graph.in_edges(cond_src, data=True)
                      if edge.get('label') == 'contains']
            for owner in owners:
                self._graph.remove_edge(owner, cond_src)
            self._graph.add_edge(bb_cond, cond_src, type="ctrl", label="contains")

        bb_then = self._new_bb("body")
        self._graph.nodes[bb_then]['branch'] = 'then'
        self._graph.add_edge(bb_cond, bb_then, type="ctrl", label="then")
        _, then_end = self._process_block(regions[0].blocks[0], bb_then)
        self._graph.add_edge(then_end, bb_exit, type="ctrl", label="next")

        # 変更: elseなしの場合も、偽側から共通出口へ進む経路を明示する。
        bb_else = None
        if len(regions) >= 2 and regions[1].blocks:
            bb_else = self._new_bb("body")
            self._graph.nodes[bb_else]['branch'] = 'else'
            self._graph.add_edge(bb_cond, bb_else, type="ctrl", label="else")
            _, else_end = self._process_block(regions[1].blocks[0], bb_else)
            self._graph.add_edge(else_end, bb_exit, type="ctrl", label="next")
        else:
            self._graph.add_edge(bb_cond, bb_exit, type="ctrl", label="else")

        self._graph.nodes[if_id].update(
            bb_init=bb_init, bb_cond=bb_cond, bb_body=bb_then,
            bb_then=bb_then, bb_else=bb_else, bb_exit=bb_exit,
        )
        self._graph.nodes[bb_init].update(
            unit_kind="if", bb_init=bb_init, bb_cond=bb_cond,
            bb_body=bb_then, bb_then=bb_then, bb_else=bb_else, bb_exit=bb_exit,
        )

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
