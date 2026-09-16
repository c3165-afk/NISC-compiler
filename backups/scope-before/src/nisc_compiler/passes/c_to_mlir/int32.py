"""Validation of the first, deliberately small, supported C subset."""
from pycparser import c_ast, c_parser

from .core import MLIRGen, ParseError

MIN_INT = -(1 << 31)
MAX_INT = (1 << 31) - 1


def validate_source(source: str) -> dict:
    """
    許容範囲の広いレガシーフロントエンドが実行される前に、サポートされていないセマンティクスを拒否する。

    ここでは、入力に依存する算術演算の評価は行われない。オーバーフローは許容される実行領域外であり、モデルランナーによってチェックされる。
    """
    try:
        ast = c_parser.CParser().parse(MLIRGen()._strip_comments(source))
    except Exception as exc:
        raise ParseError(f"C parse error: {exc}") from exc
    if len(ast.ext) != 1 or not isinstance(ast.ext[0], c_ast.FuncDef):
        raise ParseError("int32 profile requires exactly one function, without globals")
    func = ast.ext[0]

    def plain_int(t):
        return (isinstance(t, c_ast.TypeDecl) and not t.quals
                and isinstance(t.type, c_ast.IdentifierType)
                and t.type.names in (["int"], ["signed", "int"], ["signed"]))

    def declaration(d):
        if (not isinstance(d, c_ast.Decl) or not plain_int(d.type)
                or d.storage or d.quals or d.funcspec or d.bitsize or d.align):
            raise ParseError("Only unqualified, automatic signed int declarations are supported")

    if (not isinstance(func.decl.type, c_ast.FuncDecl)
            or not plain_int(func.decl.type.type) or func.decl.storage
            or func.decl.funcspec or func.param_decls):
        raise ParseError("Function must return int and use a prototype-style parameter list")
    params = list(func.decl.type.args.params) if func.decl.type.args else []
    if (len(params) == 1 and isinstance(params[0], c_ast.Typename)
            and isinstance(params[0].type, c_ast.TypeDecl)
            and isinstance(params[0].type.type, c_ast.IdentifierType)
            and params[0].type.type.names == ["void"]):
        params = []
    declared = set()
    initialized = set()
    arguments = []
    for param in params:
        declaration(param)
        if not param.name or param.name in declared:
            raise ParseError("Parameters must have distinct names")
        declared.add(param.name)
        initialized.add(param.name)
        arguments.append(param.name)
    scopes = [set(arguments)]

    def expression(expr):
        if isinstance(expr, c_ast.ID):
            if expr.name not in initialized:
                raise ParseError(f"Uninitialized or undeclared variable: {expr.name}")
        elif isinstance(expr, c_ast.Constant):
            if expr.type != "int":
                raise ParseError("Only int constants are supported")
            text = expr.value
            try:
                radix = 16 if text.lower().startswith("0x") else (8 if text.startswith("0") else 10)
                value = int(text, radix)
            except ValueError as exc:
                raise ParseError(f"Unsupported integer literal: {text}") from exc
            if not 0 <= value <= MAX_INT:
                raise ParseError(f"Integer literal outside signed int range: {text}")
        elif isinstance(expr, c_ast.BinaryOp) and expr.op in ("+", "-", "*", "<", "<=", ">", ">=", "==", "!="):
            expression(expr.left)
            expression(expr.right)
        elif isinstance(expr, c_ast.UnaryOp) and expr.op == "-":
            expression(expr.expr)
        else:
            raise ParseError(f"Unsupported int32 expression: {type(expr).__name__}")

    statements = func.body.block_items or []
    if not statements or not isinstance(statements[-1], c_ast.Return):
        raise ParseError("Function must end with one return expression")
    def statement(stmt, allow_return=False):
        nonlocal declared, initialized
        if isinstance(stmt, c_ast.Compound):
            outer_declared = set(declared)
            outer_initialized = set(initialized)
            scopes.append(set())
            for child in stmt.block_items or []:
                statement(child)
            local_names = scopes.pop()
            # Shadowing starts at the declaration, not at block entry.
            initialized = ((initialized & outer_declared) - local_names) | (outer_initialized & local_names)
            declared = outer_declared
        elif isinstance(stmt, c_ast.If):
            expression(stmt.cond)
            before_d, before_i = set(declared), set(initialized)
            statement(stmt.iftrue)
            then_i = set(initialized)
            declared, initialized = set(before_d), set(before_i)
            if stmt.iffalse is not None:
                statement(stmt.iffalse)
            initialized &= then_i
            declared = before_d
        elif isinstance(stmt, c_ast.EmptyStatement):
            return
        elif isinstance(stmt, c_ast.Decl):
            declaration(stmt)
            if stmt.name in scopes[-1]:
                raise ParseError(f"Duplicate declaration: {stmt.name}")
            scopes[-1].add(stmt.name)
            declared.add(stmt.name)
            initialized.discard(stmt.name)
            if stmt.init is not None:
                expression(stmt.init)
                initialized.add(stmt.name)
        elif isinstance(stmt, c_ast.Assignment):
            if not isinstance(stmt.lvalue, c_ast.ID) or stmt.lvalue.name not in declared:
                raise ParseError("Assignment requires a declared int variable")
            if stmt.op not in ("=", "+=", "-=", "*="):
                raise ParseError(f"Unsupported assignment: {stmt.op}")
            if stmt.op != "=":
                expression(stmt.lvalue)
            expression(stmt.rvalue)
            initialized.add(stmt.lvalue.name)
        elif isinstance(stmt, c_ast.Return) and allow_return:
            expression(stmt.expr)
        else:
            raise ParseError(f"Unsupported int32 statement: {type(stmt).__name__}")
    for index, stmt in enumerate(statements):
        statement(stmt, allow_return=index == len(statements) - 1)
    return {"profile": "int32", "function": func.decl.name,
            "arguments": arguments, "return_address": len(arguments),
            "word_bits": 32, "address_unit": "word"}


class Int32MLIRGen(MLIRGen):
    """Structured if lowering with scoped environments and complete joins."""

    def _gen_compound(self, compound):
        outer = dict(self._env)
        local = {s.name for s in (compound.block_items or []) if isinstance(s, c_ast.Decl)}
        super()._gen_compound(compound)
        # Keep assignments to enclosing bindings, discard/restore local bindings.
        self._env = {name: outer[name] if name in local else self._env[name] for name in outer}

    def _gen_stmt(self, stmt):
        if isinstance(stmt, c_ast.Compound):
            self._gen_compound(stmt)
        elif not isinstance(stmt, c_ast.EmptyStatement):
            super()._gen_stmt(stmt)

    def _condition(self, expr):
        if isinstance(expr, c_ast.BinaryOp) and expr.op in self.CMPI_PRED:
            return self._gen_cmp(expr)
        value = self._gen_expr(expr)
        zero = self._new_ssa('i32')
        condition = self._new_ssa('i1')
        self._emit(f'{zero} = arith.constant 0 : i32')
        self._emit(f'{condition} = arith.cmpi ne, {value}, {zero} : i32')
        return condition

    def _gen_expr(self, expr):
        if isinstance(expr, c_ast.BinaryOp) and expr.op in self.CMPI_PRED:
            # C comparisons produce int 0/1. The comparator drives a branch,
            # not a nonexistent comparator-to-GPR writeback port.
            condition = self._gen_cmp(expr)
            result, one, zero = self._new_ssa('i32'), self._new_ssa('i32'), self._new_ssa('i32')
            self._emit(f'{result} = scf.if {condition} -> (i32) {{')
            self._emit(f'    {one} = arith.constant 1 : i32')
            self._emit(f'    scf.yield {one} : i32')
            self._emit('} else {')
            self._emit(f'    {zero} = arith.constant 0 : i32')
            self._emit(f'    scf.yield {zero} : i32')
            self._emit('}')
            return result
        return super()._gen_expr(expr)

    def _gen_if(self, stmt):
        condition = self._condition(stmt.cond)
        saved_env, saved_body = dict(self._env), self._body
        arms = []
        for arm in (stmt.iftrue, stmt.iffalse):
            self._env, self._body = dict(saved_env), []
            self._gen_block_body(arm)
            arms.append((self._body, dict(self._env)))
        self._env, self._body = dict(saved_env), saved_body
        changed = [name for name, old in saved_env.items()
                   if any(env[name] != old for _, env in arms)]
        types = ', '.join('i32' for _ in changed)
        result = self._new_name() if changed else None
        prefix = (f'%{result}' if len(changed) == 1 else f'%{result}:{len(changed)}') if changed else ''
        self._emit(f'{prefix} = scf.if {condition} -> ({types}) {{' if changed else f'scf.if {condition} {{')
        for index, (body, env) in enumerate(arms):
            if index:
                self._emit('} else {')
            for inst in body:
                self._emit('    ' + inst)
            values = ', '.join(env[name] for name in changed)
            self._emit(f'    scf.yield {values} : {types}' if changed else '    scf.yield')
        self._emit('}')
        for i, name in enumerate(changed):
            value = f'%{result}' if len(changed) == 1 else f'%{result}#{i}'
            self._env[name], self._types[value] = value, 'i32'
