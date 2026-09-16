"""
base.py: コンパイラパスの基底クラス定義

全パスはCompilerPassを継承してrunメソッドを実装する。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from nisc_compiler.context import CompileContext


class CompilerPass(ABC):
    """全コンパイラパスの基底クラス。"""

    name: str = "base"

    @abstractmethod
    def run(self, cdfg: nx.DiGraph | None, context: "CompileContext") -> nx.DiGraph:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class CToMLIRBase(CompilerPass):
    """C → MLIRテキストに変換するパスの基底クラス。"""
    name = "c_to_mlir"

    @abstractmethod
    def run(self, cdfg: None, context: "CompileContext") -> None:
        raise NotImplementedError


class MLIRToCDFGBase(CompilerPass):
    """MLIR → CDFGに変換するパスの基底クラス。"""
    name = "mlir_to_cdfg"

    @abstractmethod
    def run(self, cdfg: None, context: "CompileContext") -> nx.DiGraph:
        raise NotImplementedError


class OperatorAssignBase(CompilerPass):
    """演算器割り当てパスの基底クラス。"""
    name = "operator_assign"

    @abstractmethod
    def run(self, cdfg: nx.DiGraph, context: "CompileContext") -> nx.DiGraph:
        raise NotImplementedError


class StateAssignBase(CompilerPass):
    """ステート番号割り当てパスの基底クラス。"""
    name = "state_assign"

    @abstractmethod
    def run(self, cdfg: nx.DiGraph, context: "CompileContext") -> nx.DiGraph:
        raise NotImplementedError


class RegisterAllocateBase(CompilerPass):
    """レジスタ割り当てパスの基底クラス。"""
    name = "register_allocate"

    @abstractmethod
    def run(self, cdfg: nx.DiGraph, context: "CompileContext") -> nx.DiGraph:
        raise NotImplementedError


class CodeGenBase(CompilerPass):
    """コード生成パスの基底クラス。"""
    name = "code_gen"

    @abstractmethod
    def run(self, cdfg: nx.DiGraph, context: "CompileContext") -> nx.DiGraph:
        raise NotImplementedError