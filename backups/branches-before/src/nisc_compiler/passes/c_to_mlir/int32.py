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
        elif isinstance(expr, c_ast.BinaryOp) and expr.op in ("+", "-", "*"):
            expression(expr.left)
            expression(expr.right)
        elif isinstance(expr, c_ast.UnaryOp) and expr.op == "-":
            expression(expr.expr)
        else:
            raise ParseError(f"Unsupported int32 expression: {type(expr).__name__}")

    statements = func.body.block_items or []
    if not statements or not isinstance(statements[-1], c_ast.Return):
        raise ParseError("Function must end with one return expression")
    for index, stmt in enumerate(statements):
        if isinstance(stmt, c_ast.Decl):
            declaration(stmt)
            if stmt.name in declared:
                raise ParseError(f"Duplicate declaration: {stmt.name}")
            declared.add(stmt.name)
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
        elif isinstance(stmt, c_ast.Return) and index == len(statements) - 1:
            expression(stmt.expr)
        else:
            raise ParseError(f"Unsupported int32 statement: {type(stmt).__name__}")
    return {"profile": "int32", "function": func.decl.name,
            "arguments": arguments, "return_address": len(arguments),
            "word_bits": 32, "address_unit": "word"}
