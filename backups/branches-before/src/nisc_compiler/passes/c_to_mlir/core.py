"""
parser.py: C ソース → MLIR テキスト生成

対応状況:
  [x] int / float の基本演算（+, -, *, /）
  [x] int/float 混合演算（sitofp）
  [x] 数学関数（sqrt, sin, cos, exp, log）
  [x] 変数宣言・代入
  [x] for ループ（scf.for）
  [x] while ループ（scf.while）
  [x] if 文（scf.if）
  [x] 多重ネスト
  [x] 配列アクセス（memref）
"""
from __future__ import annotations
import re
import pycparser
from pycparser import c_ast


class ParseError(Exception):
    pass


class MLIRGen:
    """pycparser の AST → MLIR テキストの変換器。1関数を対象とする。"""

    # C関数名 → MLIR math方言のOp
    FUNC_TO_MATH_OP = {
        "sqrt": "math.sqrt", "sqrtf": "math.sqrt",
        "sin":  "math.sin",  "sinf":  "math.sin",
        "cos":  "math.cos",  "cosf":  "math.cos",
        "exp":  "math.exp",  "expf":  "math.exp",
        "log":  "math.log",  "logf":  "math.log",
    }

    # 比較演算子 → predicate
    CMPI_PRED = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}
    CMPF_PRED = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "one"}

    def __init__(self):
        self._counter = 0                   # SSA変数カウンタ
        self._env: dict[str, str] = {}      # 変数名 → SSA変数名
        self._types: dict[str, str] = {}    # SSA変数名 → MLIR型
        self._body: list[str] = []          # 生成済み命令列
        self._for_depth = 0                 # forループのネスト深さ

    # ----------------------------------------------------------------
    # 公開API
    # ----------------------------------------------------------------
    def generate(self, c_source: str, func_name: str = None) -> str:
        c_source = self._strip_comments(c_source)
        parser = pycparser.CParser()
        try:
            ast = parser.parse(c_source, filename="<input>")
        except pycparser.c_parser.ParseError as e:
            raise ParseError(f"C parse error: {e}") from e
        func = self._find_func(ast, func_name)
        return self._gen_func(func)

    # ----------------------------------------------------------------
    # 関数
    # ----------------------------------------------------------------
    def _find_func(self, ast: c_ast.FileAST, name: str | None) -> c_ast.FuncDef:
        funcs = [n for n in ast.ext if isinstance(n, c_ast.FuncDef)]
        if not funcs:
            raise ParseError("No function found.")
        if name is None:
            return funcs[0]
        for f in funcs:
            if f.decl.name == name:
                return f
        raise ParseError(f"Function '{name}' not found.")

    def _gen_func(self, func: c_ast.FuncDef) -> str:
        name = func.decl.name

        # 引数
        params = []
        if func.decl.type.args:
            for param in func.decl.type.args.params:
                # f(void) has no parameter; void is not a named SSA value.
                if isinstance(param, c_ast.Typename):
                    t = param.type
                    if isinstance(t, c_ast.TypeDecl) and getattr(t.type, 'names', []) == ['void']:
                        continue
                ptype = self._extract_type(param)
                ssa = f"%{param.name}"
                self._env[param.name] = ssa
                self._types[ssa] = ptype
                params.append(f"{ssa}: {ptype}")

        ret_type = self._extract_ret_type(func)
        self._gen_compound(func.body)

        if ret_type == "void":
            sig = f"func.func @{name}({', '.join(params)})"
        else:
            sig = f"func.func @{name}({', '.join(params)}) -> {ret_type}"
        lines = [f"{sig} {{"]
        for inst in self._body:
            lines.append(f"    {inst}")
        lines.append("}")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # 文
    # ----------------------------------------------------------------
    def _gen_compound(self, compound: c_ast.Compound):
        if compound.block_items:
            for stmt in compound.block_items:
                self._gen_stmt(stmt)

    def _gen_stmt(self, stmt):
        if isinstance(stmt, c_ast.Return):
            if stmt.expr is None:
                self._emit("func.return")
            else:
                val = self._gen_expr(stmt.expr)
                self._emit(f"return {val} : {self._type(val)}")
        elif isinstance(stmt, c_ast.Decl):
            self._gen_decl(stmt)
        elif isinstance(stmt, c_ast.Assignment):
            self._gen_assignment(stmt)
        elif isinstance(stmt, c_ast.For):
            self._gen_for(stmt)
        elif isinstance(stmt, c_ast.While):
            self._gen_while(stmt)
        elif isinstance(stmt, c_ast.If):
            self._gen_if(stmt)
        else:
            raise ParseError(
                f"Unsupported statement: {type(stmt).__name__}. "
                f"Supported: return, decl, assignment, for, while, if."
            )

    def _gen_decl(self, decl: c_ast.Decl):
        typ = self._extract_type(decl)
        if typ.startswith("memref"):
            # 配列型 → memref.allocaでスタック確保
            val = self._new_ssa(typ)
            self._emit(f"{val} = memref.alloca() : {typ}")
            self._env[decl.name] = val
        elif decl.init is not None:
            val = self._gen_expr(decl.init)
            self._env[decl.name] = val
        else:
            # 初期化なし → 0で初期化
            val = self._new_ssa(typ)
            self._emit(f"{val} = arith.constant 0 : {typ}")
            self._env[decl.name] = val

    # 複合代入演算子 → 算術Op のマッピング
    COMPOUND_OPS = {
        "+=": {"+"},
        "-=": {"-"},
        "*=": {"*"},
        "/=": {"/"},
    }

    def _gen_assignment(self, assign: c_ast.Assignment):
        op = assign.op

        if op == "=":
            val = self._gen_expr(assign.rvalue)
        elif op in ("+=", "-=", "*=", "/="):
            # 複合代入: lvalue OP= rvalue → lvalue = lvalue OP rvalue
            arith_op = op[0]  # +, -, *, /
            rval = self._gen_expr(assign.rvalue)
            # lvalueの現在値を取得
            lval = self._gen_lvalue_read(assign.lvalue)
            # 演算
            ltyp = self._type(lval)
            rtyp = self._type(rval)
            if ltyp != rtyp:
                if ltyp == "i32":
                    conv = self._new_ssa("f32")
                    self._emit(f"{conv} = arith.sitofp {lval} : i32 to f32")
                    lval = conv
                else:
                    conv = self._new_ssa("f32")
                    self._emit(f"{conv} = arith.sitofp {rval} : i32 to f32")
                    rval = conv
                typ = "f32"
            else:
                typ = ltyp
            op_map = (
                {"+": "arith.addi", "-": "arith.subi", "*": "arith.muli", "/": "arith.divsi"}
                if typ == "i32" else
                {"+": "arith.addf", "-": "arith.subf", "*": "arith.mulf", "/": "arith.divf"}
            )
            val = self._new_ssa(typ)
            self._emit(f"{val} = {op_map[arith_op]} {lval}, {rval} : {typ}")
        else:
            raise ParseError(f"Unsupported assignment op: '{op}'")

        if isinstance(assign.lvalue, c_ast.ID):
            self._env[assign.lvalue.name] = val
        elif isinstance(assign.lvalue, c_ast.ArrayRef):
            arr_ssa, indices, memref_type = self._resolve_arrayref(assign.lvalue)
            # memref.storeのインデックスはindex型が必要
            idx_casted = []
            for idx in indices:
                if self._type(idx) != "index":
                    cast = self._new_ssa("index")
                    self._emit(f"{cast} = arith.index_cast {idx} : i32 to index")
                    idx_casted.append(cast)
                else:
                    idx_casted.append(idx)
            idx_str = ", ".join(idx_casted)
            self._emit(f"memref.store {val}, {arr_ssa}[{idx_str}] : {memref_type}")
        else:
            raise ParseError(f"Unsupported lvalue: {type(assign.lvalue).__name__}")

    def _gen_lvalue_read(self, lvalue) -> str:
        """lvalueの現在値をSSA変数として返す（複合代入用）。"""
        if isinstance(lvalue, c_ast.ID):
            if lvalue.name not in self._env:
                raise ParseError(f"Undefined variable: '{lvalue.name}'")
            return self._env[lvalue.name]
        elif isinstance(lvalue, c_ast.ArrayRef):
            return self._gen_arrayref(lvalue)
        else:
            raise ParseError(f"Unsupported lvalue: {type(lvalue).__name__}")

    # ----------------------------------------------------------------
    # for ループ（scf.for）
    # ----------------------------------------------------------------
    def _gen_for(self, stmt: c_ast.For):
        if stmt.init is None:
            raise ParseError("For loop must have an initializer.")

        iv_name = stmt.init.decls[0].name
        lb_raw = self._gen_expr(stmt.init.decls[0].init)
        ub_raw = self._gen_expr(stmt.cond.right)

        # xDSLはscf.forのlb/ub/stepにindex型を要求する
        lb = self._new_ssa("index")
        self._emit(f"{lb} = arith.index_cast {lb_raw} : i32 to index")
        ub = self._new_ssa("index")
        self._emit(f"{ub} = arith.index_cast {ub_raw} : i32 to index")

        # ステップ
        if stmt.next is None:
            step = self._new_ssa("index")
            self._emit(f"{step} = arith.constant 1 : index")
        elif isinstance(stmt.next, c_ast.Assignment):
            step_raw = self._gen_expr(stmt.next.rvalue.right)
            step = self._new_ssa("index")
            self._emit(f"{step} = arith.index_cast {step_raw} : i32 to index")
        else:
            step = self._new_ssa("index")
            self._emit(f"{step} = arith.constant 1 : index")

        # ループ内で変化する変数（iv以外）
        iter_vars = self._find_assigned_vars(stmt.stmt, exclude=iv_name)
        init_vals = [self._env[v] for v in iter_vars]
        iter_types = [self._type(v) for v in init_vals]
        n = len(iter_vars)

        # ループ変数はindex型
        iv_ssa = f"%iv_{iv_name}"
        self._types[iv_ssa] = "index"
        result_name = self._new_name()

        # ネスト深さでarg変数名をユニークに
        depth = self._for_depth
        arg_prefix = f"arg_d{depth}"

        if n == 0:
            self._emit(f"scf.for {iv_ssa} = {lb} to {ub} step {step} {{")
        elif n == 1:
            self._emit(
                f"%{result_name} = scf.for {iv_ssa} = {lb} to {ub} step {step} "
                f"iter_args(%{arg_prefix}0 = {init_vals[0]}) -> ({iter_types[0]}) {{"
            )
        else:
            type_str = ", ".join(iter_types)
            args_init = ", ".join(f"%{arg_prefix}{i} = {init_vals[i]}" for i in range(n))
            self._emit(
                f"%{result_name}:{n} = scf.for {iv_ssa} = {lb} to {ub} step {step} "
                f"iter_args({args_init}) -> ({type_str}) {{"
            )

        # 本体
        saved_env = dict(self._env)
        # ループ変数はindex型 → ループ本体ではi32にキャストして使う
        iv_i32 = f"%iv_{iv_name}_i32"
        self._types[iv_i32] = "i32"
        self._env[iv_name] = iv_ssa  # scf.forのヘッダではindex型
        for i, var in enumerate(iter_vars):
            self._env[var] = f"%{arg_prefix}{i}"
            self._types[f"%{arg_prefix}{i}"] = iter_types[i]

        # ループ本体内ではiv_nameをi32キャスト版として扱う
        saved_body = self._body
        self._body = []
        self._env[iv_name] = iv_i32
        self._for_depth += 1
        self._gen_block_body(stmt.stmt)
        self._for_depth -= 1
        body_insts = self._body
        self._body = saved_body

        # インデント付きで出力（先頭にキャスト命令を追加）
        self._emit(f"    {iv_i32} = arith.index_cast {iv_ssa} : index to i32")
        for inst in body_insts:
            self._emit(f"    {inst}")

        # scf.yield
        if n > 0:
            yield_vals = ", ".join(self._env[v] for v in iter_vars)
            type_str = ", ".join(iter_types)
            self._emit(f"    scf.yield {yield_vals} : {type_str}")

        self._emit("}")

        # env復元
        self._env = saved_env
        self._env[iv_name] = iv_ssa
        for i, var in enumerate(iter_vars):
            res = f"%{result_name}" if n == 1 else f"%{result_name}#{i}"
            self._env[var] = res
            self._types[res] = iter_types[i]
            self._types[f"%{arg_prefix}{i}"] = iter_types[i]

    # ----------------------------------------------------------------
    # while ループ（scf.while）
    # ----------------------------------------------------------------
    def _gen_while(self, stmt: c_ast.While):
        loop_vars = self._find_assigned_vars(stmt.stmt)
        if not loop_vars:
            raise ParseError("While loop has no loop variables.")

        init_vals = [self._env[v] for v in loop_vars]
        iter_types = [self._type(v) for v in init_vals]
        n = len(loop_vars)
        type_str = ", ".join(iter_types)
        result_name = self._new_name()

        args_init = ", ".join(f"%arg{i} = {init_vals[i]}" for i in range(n))
        self._emit(
            f"%{result_name}:{n} = scf.while ({args_init}) "
            f": ({type_str}) -> ({type_str}) {{"
        )

        # 条件ブロック（インデント付き）
        saved_env = dict(self._env)
        for i, var in enumerate(loop_vars):
            self._env[var] = f"%arg{i}"
            self._types[f"%arg{i}"] = iter_types[i]

        saved_body = self._body
        self._body = []
        cond = self._gen_expr(stmt.cond)
        cond_insts = self._body
        self._body = saved_body

        for inst in cond_insts:
            self._emit(f"    {inst}")
        args_str = ", ".join(f"%arg{i}" for i in range(n))
        self._emit(f"    scf.condition({cond}) {args_str} : {type_str}")

        # do ブロック
        self._emit("} do {")
        self._emit(f"^bb0({', '.join(f'%arg{i}: {iter_types[i]}' for i in range(n))}):")

        # 本体
        self._gen_block_indented(stmt.stmt)

        # scf.yield
        yield_vals = ", ".join(self._env[v] for v in loop_vars)
        self._emit(f"    scf.yield {yield_vals} : {type_str}")
        self._emit("}")

        # env復元
        self._env = saved_env
        for i, var in enumerate(loop_vars):
            res = f"%{result_name}#{i}"
            self._env[var] = res
            self._types[res] = iter_types[i]

    # ----------------------------------------------------------------
    # if 文（scf.if）
    # ----------------------------------------------------------------
    def _gen_if(self, stmt: c_ast.If):
        # 条件式（インデント付き）
        saved_body = self._body
        self._body = []
        cond = self._gen_expr(stmt.cond)
        cond_insts = self._body
        self._body = saved_body
        for inst in cond_insts:
            self._emit(inst)

        then_vars = self._find_assigned_vars(stmt.iftrue)
        else_vars = self._find_assigned_vars(stmt.iffalse) if stmt.iffalse else []
        result_vars = [v for v in then_vars if v in else_vars] if stmt.iffalse else then_vars
        n = len(result_vars)
        saved_env = dict(self._env)

        if n == 0:
            self._emit(f"scf.if {cond} {{")
            self._gen_block_indented(stmt.iftrue)
            if stmt.iffalse:
                self._emit("} else {")
                self._gen_block_indented(stmt.iffalse)
            self._emit("}")
            return

        init_types = [self._type(self._env[v]) for v in result_vars]
        type_str = ", ".join(init_types)
        ret_str = f"({type_str})"
        result_name = self._new_name()
        prefix = f"%{result_name}" if n == 1 else f"%{result_name}:{n}"

        self._emit(f"{prefix} = scf.if {cond} -> {ret_str} {{")
        self._gen_block_indented(stmt.iftrue)
        yield_vals = ", ".join(self._env[v] for v in result_vars)
        self._emit(f"    scf.yield {yield_vals} : {type_str}")

        self._env = dict(saved_env)
        if stmt.iffalse:
            self._emit("} else {")
            self._gen_block_indented(stmt.iffalse)
            yield_vals = ", ".join(self._env[v] for v in result_vars)
            self._emit(f"    scf.yield {yield_vals} : {type_str}")
        else:
            self._emit("} else {")
            yield_vals = ", ".join(saved_env[v] for v in result_vars)
            self._emit(f"    scf.yield {yield_vals} : {type_str}")
        self._emit("}")

        self._env = dict(saved_env)
        for i, var in enumerate(result_vars):
            res = f"%{result_name}" if n == 1 else f"%{result_name}#{i}"
            self._env[var] = res
            self._types[res] = init_types[i]

    # ----------------------------------------------------------------
    # 式
    # ----------------------------------------------------------------
    def _gen_expr(self, expr) -> str:
        if isinstance(expr, c_ast.BinaryOp):
            return self._gen_binop(expr)
        elif isinstance(expr, c_ast.UnaryOp):
            return self._gen_unaryop(expr)
        elif isinstance(expr, c_ast.FuncCall):
            return self._gen_funccall(expr)
        elif isinstance(expr, c_ast.ID):
            if expr.name not in self._env:
                raise ParseError(f"Undefined variable: '{expr.name}'")
            return self._env[expr.name]
        elif isinstance(expr, c_ast.Constant):
            return self._gen_constant(expr)
        elif isinstance(expr, c_ast.ArrayRef):
            return self._gen_arrayref(expr)
        elif isinstance(expr, c_ast.Cast):
            return self._gen_expr(expr.expr)
        else:
            raise ParseError(f"Unsupported expression: {type(expr).__name__}")

    def _gen_arrayref(self, expr: c_ast.ArrayRef) -> str:
        """配列の読み込み: memref.load"""
        arr_ssa, indices, memref_type = self._resolve_arrayref(expr)
        elem_type = memref_type.split("x")[-1].rstrip(">")
        # memref.loadのインデックスはindex型が必要
        idx_casted = []
        for idx in indices:
            if self._type(idx) != "index":
                cast = self._new_ssa("index")
                self._emit(f"{cast} = arith.index_cast {idx} : i32 to index")
                idx_casted.append(cast)
            else:
                idx_casted.append(idx)
        idx_str = ", ".join(idx_casted)
        result = self._new_ssa(elem_type)
        self._emit(f"{result} = memref.load {arr_ssa}[{idx_str}] : {memref_type}")
        return result

    def _resolve_arrayref(self, expr: c_ast.ArrayRef) -> tuple[str, list[str], str]:
        """
        ArrayRefを再帰的に解析して（配列SSA名, インデックスリスト, memref型）を返す。
        A[i][j] → ("%A", ["%iv_i", "%iv_j"], "memref<?x?xf32>")
        """
        indices = []
        node = expr
        while isinstance(node, c_ast.ArrayRef):
            idx = self._gen_expr(node.subscript)
            indices.insert(0, idx)
            node = node.name
        # nodeはIDになったはず
        if not isinstance(node, c_ast.ID):
            raise ParseError(f"Unsupported array base: {type(node).__name__}")
        if node.name not in self._env:
            raise ParseError(f"Undefined variable: '{node.name}'")
        arr_ssa = self._env[node.name]
        memref_type = self._type(arr_ssa)
        return arr_ssa, indices, memref_type

    def _gen_binop(self, expr: c_ast.BinaryOp) -> str:
        if expr.op in self.CMPI_PRED:
            return self._gen_cmp(expr)
        left = self._gen_expr(expr.left)
        right = self._gen_expr(expr.right)
        ltyp = self._type(left)
        rtyp = self._type(right)

        # 型が違う場合はf32に揃える
        if ltyp != rtyp:
            if ltyp == "i32":
                conv = self._new_ssa("f32")
                self._emit(f"{conv} = arith.sitofp {left} : i32 to f32")
                left = conv
            else:
                conv = self._new_ssa("f32")
                self._emit(f"{conv} = arith.sitofp {right} : i32 to f32")
                right = conv
            typ = "f32"
        else:
            typ = ltyp

        op_map = (
            {"+": "arith.addi", "-": "arith.subi", "*": "arith.muli", "/": "arith.divsi"}
            if typ == "i32" else
            {"+": "arith.addf", "-": "arith.subf", "*": "arith.mulf", "/": "arith.divf"}
        )
        mlir_op = op_map.get(expr.op)
        if mlir_op is None:
            raise ParseError(f"Unsupported operator: '{expr.op}'")
        result = self._new_ssa(typ)
        self._emit(f"{result} = {mlir_op} {left}, {right} : {typ}")
        return result

    def _gen_unaryop(self, expr: c_ast.UnaryOp) -> str:
        if expr.op == "-":
            inner = self._gen_expr(expr.expr)
            typ = self._type(inner)
            zero = self._new_ssa(typ)
            op = "arith.subi" if typ == "i32" else "arith.subf"
            self._emit(f"{zero} = arith.constant 0 : {typ}")
            result = self._new_ssa(typ)
            self._emit(f"{result} = {op} {zero}, {inner} : {typ}")
            return result
        raise ParseError(f"Unsupported unary op: '{expr.op}'")

    def _gen_funccall(self, expr: c_ast.FuncCall) -> str:
        name = expr.name.name
        math_op = self.FUNC_TO_MATH_OP.get(name)
        if math_op is None:
            raise ParseError(f"Unknown function '{name}'. Supported: {list(self.FUNC_TO_MATH_OP)}")
        args = [self._gen_expr(a) for a in (expr.args.exprs if expr.args else [])]
        result = self._new_ssa("f32")
        self._emit(f"{result} = {math_op} {', '.join(args)} : f32")
        return result

    def _gen_cmp(self, expr: c_ast.BinaryOp) -> str:
        left = self._gen_expr(expr.left)
        right = self._gen_expr(expr.right)
        typ = self._type(left)
        result = self._new_ssa("i1")
        if typ == "i32":
            self._emit(f"{result} = arith.cmpi {self.CMPI_PRED[expr.op]}, {left}, {right} : {typ}")
        else:
            self._emit(f"{result} = arith.cmpf {self.CMPF_PRED[expr.op]}, {left}, {right} : {typ}")
        return result

    def _gen_constant(self, expr: c_ast.Constant) -> str:
        if expr.type == "int":
            text = expr.value
            radix = 16 if text.lower().startswith('0x') else (8 if text.startswith('0') else 10)
            typ, val = "i32", str(int(text, radix))
        elif expr.type == "long":
            typ, val = "i32", expr.value
        else:
            typ = "f32"
            val = f"{float(expr.value.rstrip('fF')):.6e}"
        result = self._new_ssa(typ)
        self._emit(f"{result} = arith.constant {val} : {typ}")
        return result

    # ----------------------------------------------------------------
    # ブロック生成ヘルパー
    # ----------------------------------------------------------------
    def _gen_block_indented(self, stmt):
        """文をインデントして生成する（if/for/whileブロック用）。"""
        saved_body = self._body
        self._body = []
        self._gen_block_body(stmt)
        body = self._body
        self._body = saved_body
        for inst in body:
            self._emit(f"    {inst}")

    def _gen_block_body(self, stmt):
        """文を現在のbodyに生成する。"""
        if isinstance(stmt, c_ast.Compound):
            self._gen_compound(stmt)
        elif stmt is not None:
            self._gen_stmt(stmt)

    # ----------------------------------------------------------------
    # ループ変数の検出
    # ----------------------------------------------------------------
    def _find_assigned_vars(self, stmt, exclude: str = None) -> list[str]:
        """ブロック内（ネスト含む）で代入される変数を宣言順で返す。"""
        assigned = set()
        self._collect_assigned(stmt, assigned, exclude)
        return [v for v in self._env if v in assigned]

    def _collect_assigned(self, stmt, assigned: set, exclude: str = None):
        """再帰的に代入される変数名を収集する。"""
        if stmt is None:
            return
        if isinstance(stmt, c_ast.Compound):
            for item in (stmt.block_items or []):
                self._collect_assigned(item, assigned, exclude)
        elif isinstance(stmt, c_ast.Assignment):
            if isinstance(stmt.lvalue, c_ast.ID):
                if stmt.lvalue.name != exclude:
                    assigned.add(stmt.lvalue.name)
        elif isinstance(stmt, c_ast.For):
            self._collect_assigned(stmt.stmt, assigned, exclude)
        elif isinstance(stmt, c_ast.While):
            self._collect_assigned(stmt.stmt, assigned, exclude)
        elif isinstance(stmt, c_ast.If):
            self._collect_assigned(stmt.iftrue, assigned, exclude)
            self._collect_assigned(stmt.iffalse, assigned, exclude)

    # ----------------------------------------------------------------
    # ユーティリティ
    # ----------------------------------------------------------------
    def _new_ssa(self, typ: str = "i32") -> str:
        name = f"%{self._counter}"
        self._counter += 1
        self._types[name] = typ
        return name

    def _new_name(self) -> str:
        name = str(self._counter)
        self._counter += 1
        return name

    def _emit(self, inst: str):
        self._body.append(inst)

    def _type(self, ssa: str) -> str:
        return self._types.get(ssa, "i32")

    def _extract_type(self, node) -> str:
        try:
            t = node.type
            dims = []
            while isinstance(t, c_ast.ArrayDecl):
                # 固定サイズの場合はサイズを取得、動的は?
                if t.dim is not None and hasattr(t.dim, 'value'):
                    dims.append(t.dim.value)  # 例: "4"
                else:
                    dims.append("?")
                t = t.type
            while isinstance(t, c_ast.PtrDecl):
                dims.append("?")
                t = t.type
            while hasattr(t, "type") and not hasattr(t, "names"):
                t = t.type
            names = t.names if hasattr(t, "names") else []
            elem_type = "f32" if ("float" in names or "double" in names) else "i32"
            if not dims:
                return elem_type
            return "memref<" + "x".join(dims) + "x" + elem_type + ">"
        except Exception:
            return "i32"

    def _extract_ret_type(self, func: c_ast.FuncDef) -> str:
        try:
            t = func.decl.type.type
            while hasattr(t, "type"):
                t = t.type
            names = t.names if hasattr(t, "names") else []
            if "void" in names:
                return "void"
            return "f32" if ("float" in names or "double" in names) else "i32"
        except Exception:
            return "i32"

    def _strip_comments(self, source: str) -> str:
        source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
        source = re.sub(r'//[^\n]*', '', source)
        return source


# ----------------------------------------------------------------
# 公開API
# ----------------------------------------------------------------
def parse_c(c_source: str, func_name: str = None) -> str:
    """Cソース文字列 → MLIRテキスト。"""
    return MLIRGen().generate(c_source, func_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使い方: python parser.py <Cファイル>")
        sys.exit(1)
    print(parse_c(open(sys.argv[1]).read()))
