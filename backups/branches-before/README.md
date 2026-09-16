# NISC コンパイラ（小田さん引継版）— ハンドオフ README

> **2026-09-16 更新**：既定のコンパイル対象を、検証可能な32ビットintの直線計算に限定しました。
> 採用仕様、引数・戻り値のSRAM配置、実行モデル、テスト方法は
> [int32到達点の仕様と検証](docs/int32-milestone.md)を参照してください。
> 下記のWSL手順以降にある第1章以降は引継時の説明です。ループ等の旧経路は`profile="legacy"`で明示的に選択しますが、今回の動作保証対象ではありません。

## WSLで現在のコンパイラの挙動を確認する

この手順は **Windows側の `C:\Users\c3165\compiler_01`** にある現在のコードを使います。
WSLでは同じフォルダが `/mnt/c/Users/c3165/compiler_01` に見えます。
OneDrive内にある別のコンパイラとは作業場所が異なる点に注意してください。

### 1. WSLを開き、作業フォルダへ移動する

WindowsのPowerShellからWSLを開きます。既にWSLターミナルを開いている場合は不要です。

```powershell
wsl.exe
```

以降のコマンドは **WSLのbash** で実行します。

```bash
cd /mnt/c/Users/c3165/compiler_01
pwd
.compiler/bin/python --version
```

既存の `.compiler` はLinux用仮想環境です。確認時のPythonは **3.14.4** でした。
以下では `.compiler/bin/python` を直接指定するため、仮想環境のactivateや
追加のパッケージインストールは不要です。`setup.sh`の再実行も不要です。

### 2. 修正の判定テストを実行する

```bash
.compiler/bin/python -B tools/test_int32.py
```

正常に完了すると、末尾に次のように表示されます。実行時間は環境によって異なります。

```text
Ran 11 tests in ...s

OK
```

テストには、定数返却、引数返却、加減乗算、負数、レイテンシ変更、
レジスタ不足・未対応入力の拒否、故意に壊した制御列の検出を含みます。
60個の生成プログラムの比較も、この11件の中に含まれます。
ファイル出力の試験は一時ディレクトリで行い、既存の `output/` は上書きしません。
`-B`はPythonのバイトコードキャッシュの作成を抑止する指定です。

### 3. サンプルをコンパイルして実行モデルで動かす

入力は [input/int32_milestone.c](input/int32_milestone.c) です。

```c
int int32_milestone(int a, int b) {
    int sum = a + b;
    int difference = a - b;
    return sum * difference;
}
```

引数 `a=9`, `b=4` を指定します。

```bash
.compiler/bin/python -B tools/run_int32.py input/int32_milestone.c 9 4
```

現在の既定設定では、次の結果になります。

```json
{
  "return_value": 65,
  "done_state": 12,
  "return_address": 2,
  "validation": "provisional datapath model; RTL not verified"
}
```

- `return_value`: `(9 + 4) * (9 - 4)` の計算結果。
- `done_state`: 終了状態の番号。状態0が初期待機で、状態12でdoneになります。
- `return_address`: 戻り値を格納するSRAMのワードアドレス。引数が2個なのでSRAM[2]です。
- `validation`: 暫定データパスモデルでの検証であり、RTL検証ではないことを示します。

このコマンドは、Cのコンパイルと生成した制御列のモデル実行を行います。
`--output`を指定しなければ、生成物をファイルへ保存しません。

### 4. 生成した制御列とChiselコードを保存・確認する

```bash
.compiler/bin/python -B tools/run_int32.py input/int32_milestone.c 9 4 --output output/wsl-int32-demo
ls output/wsl-int32-demo/program_int32_milestone
```

次の3ファイルが生成されます。同じ保存先で再実行すると、これらは上書きされます。

| ファイル | 内容 |
|---|---|
| `Program.scala` | 状態遷移を行うFSM、各状態の制御信号を出すDEC、定数テーブル |
| `init.txt` | 引数と戻り値のSRAM配置説明。数値入りの初期化イメージではありません |
| `controls.json` | 実行モデルへ渡した、サイクルごとの制御列 |

```bash
cat output/wsl-int32-demo/program_int32_milestone/init.txt
less output/wsl-int32-demo/program_int32_milestone/Program.scala
less output/wsl-int32-demo/program_int32_milestone/controls.json
```

`less`は`q`で終了します。このサンプルでは、SRAM[0]がa、SRAM[1]がb、
SRAM[2]が戻り値です。実行スクリプトが引数の数値をモデルのSRAMへ設定するため、
`init.txt`を手作業で編集する必要はありません。

### 5. 自分のCプログラムを試す

対応範囲の関数を1つだけ書いたCファイルを用意し、引数を宣言順に指定します。

```bash
# 例：input/my_int32.cにint引数が2個ある場合
.compiler/bin/python -B tools/run_int32.py input/my_int32.c 10 -3
# 引数なしの関数の場合
.compiler/bin/python -B tools/run_int32.py input/my_constant.c
```

現在の対応範囲は、32ビット符号付きint、ローカル変数、代入、加減乗算、
単項マイナス、末尾のreturnです。全中間結果がint範囲内に収まる入力を使用します。
分岐・ループ・配列・浮動小数点・除算などはエラーになります。
詳細は [採用仕様と修正判定基準](docs/int32-milestone.md) を参照してください。

### 実行できない場合・検証範囲

- **`.compiler/bin/python`が見つからない／起動できない**：まず`pwd`で作業フォルダを確認してください。
  この仮想環境はWSL側のPython 3.14を使用します。別のWSLディストリビューションでは
  そのPythonが存在せず、起動できない場合があります。`ls -l .compiler/bin/python*`と
  `cat .compiler/pyvenv.cfg`で参照先を確認してください。
- **`ModuleNotFoundError`**：Windows側のPythonではなく、上記の仮想環境のPythonを使っているか確認してください。
  依存関係は `pycparser`、`xdsl`、`networkx`です。手順は既存環境を前提にしています。
- **未対応構文・レジスタ不足・overflowのエラー**：現在の対応範囲または実行保証範囲を超えています。
  エラー内容に応じて入力や設定を確認してください。
- **`experiment.py`との違い**：既存の`experiment.py`はループを含むbicgを対象としており、
  既定のint32プロファイルでは拒否されます。今回の挙動確認には上記の`tools/`を使ってください。
- **RTLは未検証**：この手順ではScalaビルドやRTLシミュレーションは行いません。
  `nisc.Defs`、`FsmBase`、`DecBase`等の対象ハードウェアは別途必要です。

2026-09-16に、WSLの既存仮想環境（Python 3.14.4）でテスト11件の合格と、
サンプルの戻り値65・終了状態12を確認しました。

---

**対象**：後輩（RTL 少し触った程度、Python は普通に読める前提）
**バージョン**：小田さんから引き継いだ時点の baseline
**目的**：C コードから NISC 用の Chisel（Scala）コード（FSM + DEC）を生成する

---

## 1. これは何？

**NISC**（No-Instruction-Set Computer）は、命令フェッチをせず制御を直接ハード（FSM）で持つプロセッサアーキテクチャ。このコンパイラは：

- 入力：C ソース（1 関数）
- 出力：Chisel（Scala）で書かれた FSM + DEC のペア
  - **FSM**：状態遷移マシン（次にどの state に行くか）
  - **DEC**：デコーダ（各 state で何をするか＝ALU/MEM/etc の制御信号）

生成された Chisel を NISC ハード（別 repo の `nisc-hw`）にロードして実行する。

**ポイント**：普通のコンパイラが「命令列」を吐くのに対し、これは「ハードの状態機械」を吐く。だから CPU 側に命令デコーダが要らない（＝命令フェッチ不要）。

---

## 2. ディレクトリ構成

```
260511/                          ← リポジトリのルート
├── README.md                    ← 元の小田さん README（最小限）
├── setup.sh                     ← venv 作成＋依存インストール
├── pyproject.toml               ← Python パッケージ設定
├── requirements.txt
├── experiment.py                ← 動作確認用の実験スクリプト
│
├── input/                       ← 入力 C ソース群（ベンチマーク）
│   ├── a.c                        小さい scale_offset（動作確認向け）
│   ├── add_test.c
│   ├── atax.c, bicg.c, gemm.c, mat_mul.c, mvt.c   ← PolyBench 系
│   ├── mul_test.c
│   └── test_for.c
│
├── operators/                   ← 演算器定義（Python DSL）
│   ├── alu.py                     加減算・論理演算
│   ├── cmp.py                     比較・分岐
│   ├── mem.py                     ロード／ストア
│   ├── mul.py                     整数乗算
│   ├── fpu.py, fmul.py            浮動小数点系
│   ├── mac_i.py, mac_f.py         積和演算
│   ├── iv_init.py                 induction variable 初期化
│   └── test_fpu64.py
│
├── src/nisc_compiler/           ← コンパイラ本体
│   ├── __init__.py
│   ├── compiler.py                ← エントリ点 NISCCompiler クラス
│   ├── context.py                 ← パス間で共有される状態
│   ├── dp_config.py               ← DataPath 設定
│   └── passes/                    ← 各パス
│       ├── base.py                  パスの基底クラス
│       ├── c_to_mlir/               [Pass 1] C → MLIR
│       ├── mlir_to_cdfg/            [Pass 2] MLIR → CDFG
│       ├── operator_assign/         [Pass 3] 演算器割当
│       ├── register_allocate/       [Pass 4] レジスタ割当
│       ├── state_assign/            [Pass 5] スケジューリング
│       └── code_gen/                [Pass 6] Chisel コード生成
│
└── output/                      ← 生成結果（実行後に作られる）
    └── {プログラム名}/
        ├── Program.scala          ← FSM + DEC の Chisel コード
        └── init.txt               ← SRAM アドレスマップ
```

---

## 3. セットアップ

**前提**：Python 3.10 以上、Linux/macOS/WSL

```bash
cd 260511
source ./setup.sh
```

`source` で実行するのが重要（venv を活性化するため）。中でやってること：
1. `.compiler/` に venv 作成（初回のみ）
2. venv 活性化
3. `pip install -e .` でパッケージ登録
4. 依存（`pycparser`, `xdsl`, `networkx`）インストール

2 回目以降は venv 作成をスキップして活性化するだけ。

---

## 4. 動かしてみる（最初の一歩）

**手っ取り早く動作確認**：

```bash
python experiment.py
```

これで `input/a.c`（scale_offset の 5 行）がコンパイルされ、`output/a/Program.scala` と `output/a/init.txt` が生成される。ステート数などのレポートが標準出力に出る。

**期待される出力例**：
```
[iter 1] scheduling...
[iter 1] allocating registers...
[iter 1] done. no spills.
  states=8
  reg_map={...}
  imm_map={...}
```

**Program.scala を見てみる**：
```bash
cat output/a/Program.scala
```
`class FSM extends FsmBase { switch(state) { ... } }` と `class DEC extends DecBase { switch(io.state) { ... } }` が生成されているはず。

---

## 5. 使い方（自分の C ソースを試す）

Python REPL または独自スクリプトから：

```python
from nisc_compiler import NISCCompiler

# 1. コンパイラを作る
compiler = NISCCompiler.default(
    num_registers=32,      # GPR 数
    num_imm_registers=32,  # 即値レジスタ数
    reg_width=32,          # レジスタ幅（bit）
)

# 2. 演算器を読み込む（要る種類だけ）
compiler.load_operators('operators/alu.py', count=1)
compiler.load_operators('operators/cmp.py', count=1)
compiler.load_operators('operators/mem.py', load_count=1, store_count=1)
compiler.load_operators('operators/mul.py', width=32, count=1)
compiler.load_operators('operators/iv_init.py', width=32, count=1)

# 3. C ソースを読み込んでコンパイル
c_source = open('input/gemm.c').read()
ctx = compiler.compile('gemm.c', c_source)

# 4. 結果を取り出す
print(ctx.program_scala)  # Chisel コード（FSM + DEC）
print(ctx.init_txt)       # SRAM アドレスマップ
print(ctx.reg_map)        # 変数名 → レジスタ番号
print(ctx.imm_map)        # 即値 → 即値レジスタ番号
```

`output/{ソースファイル名}/` に自動的に書き出される。

---

## 6. コンパイルパイプライン（各パスが何をするか）

C → Chisel まで **6 段階のパス**を通る。順に：

```
input.c
   │
   ├─ [Pass 1] c_to_mlir       C → MLIR (中間表現、xdsl 利用)
   │   ソース：src/nisc_compiler/passes/c_to_mlir/
   │   使うライブラリ：pycparser（C パース）、xdsl（MLIR 生成）
   │
   ├─ [Pass 2] mlir_to_cdfg    MLIR → CDFG (制御データフローグラフ)
   │   ソース：src/nisc_compiler/passes/mlir_to_cdfg/
   │   ノード = 演算、エッジ = データ依存
   │   networkx で有向グラフとして表現
   │
   ├─ [Pass 3] operator_assign 各演算を利用可能な演算器 (ALU/MEM/etc) に割り当て
   │   ソース：src/nisc_compiler/passes/operator_assign/
   │   VF2 サブグラフマッチングを使用
   │   演算器定義は operators/*.py で読み込んだもの
   │
   ├─ [Pass 4] register_allocate  変数 → レジスタ番号
   │   ソース：src/nisc_compiler/passes/register_allocate/
   │   liveness 解析 + グリーディ or グラフ彩色
   │   レジスタ足りない場合はスピル（今回の baseline では未対応の可能性あり）
   │
   ├─ [Pass 5] state_assign    スケジューリング (どの演算をどの state で実行するか)
   │   ソース：src/nisc_compiler/passes/state_assign/
   │   デフォルトは ASAP (As Soon As Possible)
   │   ALAP (As Late As Possible) も差し替え可能：
   │     from nisc_compiler.passes.state_assign.alap import ALAPStateAssignPass
   │     compiler.replace_pass('state_assign', ALAPStateAssignPass())
   │
   └─ [Pass 6] code_gen        CDFG → Chisel コード生成
       ソース：src/nisc_compiler/passes/code_gen/
       emitter.py が Scala 文字列を組み立てる
       Program.scala と init.txt を出力
```

各パスは `CompilerPass` を継承した独立したクラスで、**差し替え可能**：

```python
compiler.replace_pass('state_assign', MyCustomStateAssignPass())
```

パスの一覧を確認：
```python
compiler.print_passes()
```

---

## 7. 主要ファイルの読み順（コード理解の順序）

**まず読むべき**（骨格を理解する）：
1. `src/nisc_compiler/compiler.py` — エントリ点、パスの並び
2. `src/nisc_compiler/context.py` — パス間で共有される状態
3. `src/nisc_compiler/passes/base.py` — パスの基底クラス

**次に読む**（1 パスずつ理解する）：
4. `src/nisc_compiler/passes/c_to_mlir/core.py` — C パース → MLIR 生成
5. `src/nisc_compiler/passes/mlir_to_cdfg/core.py` — CDFG 構築
6. `src/nisc_compiler/passes/operator_assign/core.py` — VF2 マッチング
7. `src/nisc_compiler/passes/state_assign/state_assign.py` — ASAP スケジューラ
8. `src/nisc_compiler/passes/code_gen/emitter.py` — Chisel 文字列組立

**演算器定義の読み方**：
9. `operators/alu.py` — 最も単純、他のテンプレート

---

## 8. デバッグ・実験のコツ

**中間結果を見る**：`ctx.mlir_text` で MLIR、`ctx.cdfg` で CDFG（networkx グラフ）が見える

```python
ctx = compiler.compile('test.c', c_source)
print(ctx.mlir_text)                        # Pass 1 の出力
import networkx as nx
print(nx.to_dict_of_lists(ctx.cdfg))        # Pass 2 の出力
```

**パスの実行時間**：
```python
print(ctx.pass_times)  # 各パスの秒数
```

**スケジューラを差し替えて比較**：
```python
# ASAP（デフォルト）
c1 = make_compiler()
ctx1 = c1.compile('a.c', src)

# ALAP
from nisc_compiler.passes.state_assign.alap import ALAPStateAssignPass
c2 = make_compiler()
c2.replace_pass('state_assign', ALAPStateAssignPass())
ctx2 = c2.compile('a.c', src)

# ステート数比較
def count_states(scala):
    return sum(1 for l in scala.split('\n') if l.strip().startswith('is(') and '.U)' in l)
print('ASAP:', count_states(ctx1.program_scala))
print('ALAP:', count_states(ctx2.program_scala))
```

**演算器を減らして帯域制約を試す**：
```python
compiler.load_operators('operators/alu.py', count=1)  # ALU 1 個 → ボトルネック
```

---

## 9. 制約・既知の限界（baseline 時点）

**現時点で対応していないこと**：

- **複数関数**：C ソースに複数関数書いても、最初の関数しかコンパイルされない
  - 1 ソース＝1 FSM モデル
  - 複数の FSM が要る場合は外側で関数ごとに `compile()` を呼び直す必要あり
- **関数呼び出し**：関数ポインタ・再帰は非対応
- **スタック**：ローカル変数は全部レジスタ or SRAM に静的割当
- **ポインタ**：配列アクセスは対応するが、ポインタ演算は限定的
- **浮動小数点**：`fpu.py` で対応するが、演算器を追加する必要あり

**サポートする C の範囲**：
- `for` ループ、`if/else`
- 算術・論理演算、比較
- 配列アクセス（1 次元）
- 整数（int）
- 一部の浮動小数点（float、演算器追加すれば）

**サポートしない C の範囲**（コンパイル通らない可能性大）：
- `while` ループ（`for` で代替）
- `switch` 文
- `struct` 大部分
- ダブルポインタ、関数ポインタ

**エラーが出やすいパターン**：
- `for (i = 0; ...)` は NG。**必ず `for (int i = 0; ...)`**（宣言付き）
- 変数を先に全部宣言してから使う（C89 スタイル）

---

## 10. トラブルシューティング

**`ModuleNotFoundError: No module named 'nisc_compiler'`**：
→ `source ./setup.sh` してない、または venv が activate されてない

**`AttributeError: 'Assignment' object has no attribute 'decls'`**：
→ `for (i=0;...)` になってる。`for (int i=0;...)` に修正

**演算器不足エラー（"no operator match"）**：
→ 使ってる C コードに必要な演算器を `load_operators()` で追加

**生成された Chisel が動かない**：
→ NISC ハード側の `Program.scala` として置き換え、Chisel シミュレーション（ChiselTest）で動作確認

---

## 11. 参考

**関連リポジトリ**：
- NISC ハード側（Chisel）：`nisc-hw` リポ
- 先輩の Ibex FSM 版：`ibex_src/custom_rtl_fsm_v2` 等

**主要ライブラリのドキュメント**：
- pycparser：C パーサ https://github.com/eliben/pycparser
- xdsl：MLIR framework in Python https://xdsl.dev/
- networkx：グラフライブラリ https://networkx.org/

**演算器 DSL の書き方**：
`operators/alu.py` を読んで真似るのが早い。パラメータ化して `load_operators(..., count=N, width=M)` で複数個・任意幅を作れる。

---

## 12. 質問があれば

- ソースコードのコメントが結構丁寧なので、まずコードを読んでみる
- 動かしながら中間結果（`ctx.*`）を print して感触つかむ
- `experiment.py` を fork して独自実験するのが理解の近道

引継元にも遠慮なく聞いて OK。頑張ってください。
