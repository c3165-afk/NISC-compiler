# NISC Compiler

NISCアーキテクチャ向けのコンパイラ。CソースコードからChisel FSM/DECを生成する。

## セットアップ

```bash
cd nisc_compiler_bench
source ./setup.sh
```

`source`で実行することで仮想環境が自動的に有効になる。
2回目以降は仮想環境の作成をスキップしてそのまま有効化される。

## 基本的な使い方

```python
from nisc_compiler import NISCCompiler

# 1. コンパイラを作る
compiler = NISCCompiler.default(
    num_registers=32,    # GPR数
    num_imm_registers=32, # IMM数
    reg_width=32,        # GPRのビット幅
)

# 2. 演算器を読み込む
compiler.load_operators('operators/alu.py', count=1)
compiler.load_operators('operators/cmp.py', count=1)
compiler.load_operators('operators/mem.py', load_count=1, store_count=1)
compiler.load_operators('operators/fpu.py', width=32, count=1)
compiler.load_operators('operators/fmul.py', width=32, count=1)

# 3. コンパイル
c_source = open('input/gemm.c').read()
ctx = compiler.compile('gemm.c', c_source)

# 4. 結果を使う
print(ctx.program_scala)  # Chiselコード（FSM + DEC）
print(ctx.init_txt)       # SRAMアドレスマップ
```

コンパイル結果は `output/{プログラム名}/` に自動的に出力される：

```
output/
  gemm/
    Program.scala   ← FSM/DECのChiselコード
    init.txt        ← SRAMアドレスマップ
```

## CompileContextの内容

```python
ctx.program_scala   # 生成されたChiselコード
ctx.init_txt        # SRAMアドレスマップ
ctx.reg_map         # 変数名 → GPR番号
ctx.imm_map         # 定数 → IMM番号
ctx.spill_map       # スピルされた変数 → SRAMオフセット
ctx.spill_base_reg  # スピルベースアドレスのGPR番号
ctx.iterations      # スケジューリングのイテレーション数
                    # （スピルが発生したとき再スケジューリングする回数）
ctx.pass_times      # 各パスの実行時間（秒）
```

## パイプライン構成

```
C source
  ↓ CToMLIRPass           (c_to_mlir)
MLIR
  ↓ MLIRToCDFGPass        (mlir_to_cdfg)
CDFG
  ↓ VF2OperatorAssignPass (operator_assign)
CDFG + 演算器割り当て
  ↓ ASAPStateAssignPass   (state_assign)
CDFG + ステート割り当て + レジスタ割り当て
  ↓ ChiselCodeGenPass     (code_gen)
output/{name}/Program.scala
output/{name}/init.txt
```

## 演算器の設定

演算器は`operators/`ディレクトリのファイルで定義して`load_operators()`で読み込む。
`src/`の中は編集しない。

### 標準演算器ファイル

| ファイル | 内容 | パラメータ |
|---|---|---|
| `operators/alu.py` | ALU（addi, subi等） | `width`, `count` |
| `operators/cmp.py` | CMP（cmpi, cmpf） | `width`, `count` |
| `operators/mem.py` | load/store | `width`, `load_count`, `store_count` |
| `operators/fpu.py` | FPU（addf, subf） | `width`, `count` |
| `operators/fmul.py` | FMUL（mulf） | `width`, `count` |
| `operators/mac_f.py` | float MAC（mulf+addf） | `width`, `count`, `latency` |
| `operators/mac_i.py` | int MAC（muli+addi） | `width`, `count`, `latency` |

### 読み込み例

```python
# 基本構成（32bit）
compiler.load_operators('operators/alu.py', count=1)
compiler.load_operators('operators/cmp.py', count=1)
compiler.load_operators('operators/mem.py', load_count=1, store_count=1)
compiler.load_operators('operators/fpu.py', width=32, count=1)
compiler.load_operators('operators/fmul.py', width=32, count=1)

# 64bit FPU
compiler.load_operators('operators/fpu.py', width=64, count=1)

# 演算器を2個搭載
compiler.load_operators('operators/alu.py', count=2)

# 32bitと64bitを両方搭載
compiler.load_operators('operators/fpu.py', width=32, count=1)
compiler.load_operators('operators/fpu.py', width=64, count=1)

# MACを追加（複合演算器）
compiler.load_operators('operators/mac_f.py', width=32, count=1, latency=3)
```

演算器がない場合はコンパイル時にエラーになる：

```
RuntimeError: Unmatched operations: [(34, 'arith.mulf'), ...]
These operations have no matching operator in the DP.
```

### カスタム演算器の作り方

```python
# operators/my_op.py
from nisc_compiler.passes.operator_assign.dp import Operator, _pat

WIDTH   = WIDTH   if 'WIDTH'   in dir() else 32
COUNT   = COUNT   if 'COUNT'   in dir() else 1
LATENCY = LATENCY if 'LATENCY' in dir() else 1

operators = [
    Operator(
        f"my_op{WIDTH}",   # 演算器名
        LATENCY,           # レイテンシ（サイクル数）
        _pat(
            (0, "arith.mulf"),  # ノード0
            (1, "arith.addf"),  # ノード1
            edges=[(0, 1)]      # ノード0の出力がノード1の入力
        ),
        typ=f"f{WIDTH}",   # データ型
        count=COUNT,       # 演算器の個数
        description="my custom operator",
    )
]
```

```python
# experiment.py
compiler.load_operators('operators/my_op.py', width=32, count=1, latency=2)
```

## パスの差し替え

各パスは対応する基底クラスを継承して`run`メソッドを実装することで差し替えられる。

### 基底クラスと対応するパス

```
CToMLIRBase          → c_to_mlir         （C → MLIR変換）
MLIRToCDFGBase       → mlir_to_cdfg      （MLIR → CDFG変換）
OperatorAssignBase   → operator_assign   （演算器割り当て）
StateAssignBase      → state_assign      （ステート割り当て）
RegisterAllocateBase → register_allocate （レジスタ割り当て）
CodeGenBase          → code_gen          （コード生成）
```

### 新しいパスの作り方

```python
from nisc_compiler.passes.base import StateAssignBase
from nisc_compiler.context import CompileContext
import networkx as nx

class ALAPStateAssignPass(StateAssignBase):
    name = "state_assign"  # 差し替えるパスと同じname

    def run(self, cdfg: nx.DiGraph, context: CompileContext) -> nx.DiGraph:
        # ALAPスケジューリングの実装
        ...
        return cdfg
```

### パスの操作

```python
# スケジューラをALAPに差し替え
from nisc_compiler.passes.state_assign.alap import ALAPStateAssignPass
compiler.replace_pass('state_assign', ALAPStateAssignPass())

# パスの挿入（指定したパスの後に追加）
compiler.insert_pass('operator_assign', MyOptimizationPass())

# パスの削除
compiler.remove_pass('code_gen')

# 現在のパスを確認
compiler.print_passes()
```

## ディレクトリ構成

```
nisc_compiler_bench/
├── README.md
├── setup.sh                       # セットアップスクリプト
├── pyproject.toml
├── input/                         # Cソースファイル
│   ├── gemm.c
│   ├── atax.c
│   ├── mvt.c
│   └── bicg.c
├── operators/                     # 演算器定義
│   ├── alu.py
│   ├── cmp.py
│   ├── mem.py
│   ├── fpu.py
│   ├── fmul.py
│   ├── mac_f.py
│   └── mac_i.py
├── output/                        # コンパイル結果（自動生成）
│   └── {program_name}/
│       ├── Program.scala
│       └── init.txt
├── experiment.py                  # 実験スクリプト
└── src/nisc_compiler/
    ├── compiler.py                # NISCCompilerクラス
    ├── context.py                 # CompileContextクラス
    └── passes/
        ├── base.py                # 各パスの基底クラス
        ├── c_to_mlir/
        │   ├── core.py            # C → MLIR変換（MLIRGen）
        │   └── c_parser.py        # CToMLIRPass
        ├── mlir_to_cdfg/
        │   ├── core.py            # MLIR → CDFG変換
        │   └── mlir_to_cdfg.py    # MLIRToCDFGPass
        ├── operator_assign/
        │   ├── dp.py              # 演算器の基底クラス定義
        │   ├── core.py            # VF2マッチング
        │   └── operator_assign.py # VF2OperatorAssignPass
        ├── state_assign/
        │   ├── core.py            # ASAPスケジューリング
        │   ├── state_assign.py    # ASAPStateAssignPass
        │   └── alap.py            # ALAPStateAssignPass
        ├── register_allocate/
        │   └── allocator.py       # グラフ彩色レジスタ割り当て
        └── code_gen/
            ├── emitter.py         # Chiselコード生成
            ├── representation.py  # FSM/DEC中間表現
            ├── cdfg_to_fsmrep.py  # CDFG → 中間表現変換
            └── code_gen.py        # ChiselCodeGenPass
```

## 既知の課題

```
未対応:
  DECのMEM_WADDR（2次元配列アドレス計算）
  プロローグFSM/DEC（引数のSRAMからGPRへの初期化）
  スピルのDEC出力

将来:
  fsmrep_to_chisel.py（中間表現→Chisel変換）
  複数FSMの設計
  ソフトウェア展開（MULなし環境でのロシア農民法）
```