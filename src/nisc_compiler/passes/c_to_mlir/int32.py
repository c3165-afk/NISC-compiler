"""Validation of the first, deliberately small, supported C subset."""
from pycparser import c_ast, c_parser

from .core import MLIRGen, CFGMLIRGen, ParseError

MIN_INT = -(1 << 31)
MAX_INT = (1 << 31) - 1


def _prepare_source(source: str):
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
    # Lexical names resolve to declaration identities, not directly to values.
    # Rename the fresh AST so validation and MLIR generation use the same rule.
    scopes = [{}]
    next_binding = 0
    initialized = set()
    arguments = []

    def resolve(name):
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None

    def bind(decl):
        nonlocal next_binding
        name = decl.name
        if not name or name in scopes[-1]:
            raise ParseError(f"Duplicate or unnamed declaration: {name}")
        identity = f'nisc_binding{next_binding}'
        next_binding += 1
        scopes[-1][name] = identity
        decl.name = identity
        decl.type.declname = identity
        return identity

    for param in params:
        declaration(param)
        arguments.append(param.name)
        initialized.add(bind(param))

    def expression(expr):
        if isinstance(expr, c_ast.ID):
            identity = resolve(expr.name)
            if identity not in initialized:
                raise ParseError(f"Uninitialized or undeclared variable: {expr.name}")
            expr.name = identity
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
    def sequence(items):
        falls_through = True
        for child in items:
            child_falls_through = statement(child)
            falls_through = falls_through and child_falls_through
        return falls_through

    def statement(stmt):
        nonlocal initialized
        if isinstance(stmt, c_ast.Compound):
            scopes.append({})
            falls_through = sequence(stmt.block_items or [])
            # Drop only local identities; preserve updates to outer declarations.
            initialized.difference_update(scopes.pop().values())
            return falls_through
        elif isinstance(stmt, c_ast.If):
            expression(stmt.cond)
            before_i = set(initialized)
            then_falls = statement(stmt.iftrue)
            then_i = set(initialized)
            initialized = set(before_i)
            else_falls = statement(stmt.iffalse) if stmt.iffalse is not None else True
            # Only paths reaching the continuation participate in its join.
            if then_falls and else_falls:
                initialized &= then_i
            elif then_falls:
                initialized = then_i
            return then_falls or else_falls
        elif isinstance(stmt, c_ast.EmptyStatement):
            return True
        elif isinstance(stmt, (c_ast.While, c_ast.For)):
            scopes.append({})
            if isinstance(stmt, c_ast.For) and stmt.init is not None:
                statement(stmt.init)
            if stmt.cond is not None:
                expression(stmt.cond)
            before_loop = set(initialized)
            statement(stmt.stmt)
            if isinstance(stmt, c_ast.For) and stmt.next is not None:
                statement(stmt.next)
            # The body may execute zero times; only the initializer is definite.
            initialized = before_loop
            initialized.difference_update(scopes.pop().values())
        elif isinstance(stmt, (c_ast.DeclList, c_ast.ExprList)):
            sequence(stmt.decls if isinstance(stmt, c_ast.DeclList) else stmt.exprs)
        elif isinstance(stmt, c_ast.UnaryOp) and stmt.op in ('++', '--', 'p++', 'p--'):
            if not isinstance(stmt.expr, c_ast.ID):
                raise ParseError('Increment/decrement requires an int variable')
            expression(stmt.expr)
        elif isinstance(stmt, c_ast.Decl):
            declaration(stmt)
            # The new binding is visible in its own initializer. In particular,
            # int x = x must not read a shadowed, initialized outer x.
            identity = bind(stmt)
            if stmt.init is not None:
                expression(stmt.init)
                initialized.add(identity)
        elif isinstance(stmt, c_ast.Assignment):
            identity = resolve(stmt.lvalue.name) if isinstance(stmt.lvalue, c_ast.ID) else None
            if identity is None:
                raise ParseError("Assignment requires a declared int variable")
            if stmt.op not in ("=", "+=", "-=", "*="):
                raise ParseError(f"Unsupported assignment: {stmt.op}")
            if stmt.op != "=":
                expression(stmt.lvalue)
            expression(stmt.rvalue)
            stmt.lvalue.name = identity
            initialized.add(identity)
        elif isinstance(stmt, c_ast.Return):
            expression(stmt.expr)
            return False
        else:
            raise ParseError(f"Unsupported int32 statement: {type(stmt).__name__}")
        return True

    if sequence(statements):
        raise ParseError("Every reachable path must return an int value")
    abi = {"profile": "int32", "function": func.decl.name,
            "arguments": arguments, "return_address": len(arguments),
            "word_bits": 32, "address_unit": "word"}
    return abi, func


def validate_source(source: str) -> dict:
    """Validate with declaration identities, retaining source names in the ABI."""
    abi, _ = _prepare_source(source)
    return abi


class Int32MLIRGen(MLIRGen):
    """Structured if lowering with scoped environments and complete joins."""

    def generate(self, c_source: str, func_name: str = None) -> str:
        abi, func = _prepare_source(c_source)
        if func_name is not None and func_name != abi['function']:
            raise ParseError(f"Function '{func_name}' not found.")
        if CFGMLIRGen.contains_loop(func.body):
            return CFGMLIRGen().generate_resolved(func)
        return self._gen_func(func, normalize_returns=True)

    def _gen_compound(self, compound):
        outer = set(self._env)
        super()._gen_compound(compound)
        # Keys are declaration identities: remove locals, keep outer updates.
        self._env = {identity: self._env[identity] for identity in self._env if identity in outer}

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
