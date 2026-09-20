"""
experiment.py: NISCコンパイラの実験スクリプト

ASAP/ALAPスケジューラの比較実験。
"""
from nisc_compiler import NISCCompiler
from nisc_compiler.passes.state_assign.alap import ALAPStateAssignPass

# ================================================================
# 実験設定
# ================================================================

# ベンチマーク
BENCHMARKS = ['mul_test']

# DP設定
NUM_REGISTERS = 32
REG_WIDTH     = 32

# スケジューラ
SCHEDULERS = [
    ('ASAP', None),
    # ('ALAP', ALAPStateAssignPass()),
]


# ================================================================
# ヘルパー関数
# ================================================================

def make_compiler(sched_pass=None):
    """コンパイラを作成する。"""
    compiler = NISCCompiler.default(
        num_registers=NUM_REGISTERS,
        reg_width=REG_WIDTH,
    )
    compiler.load_operators('operators/alu.py',  count=1)
    compiler.load_operators('operators/cmp.py',  count=1)
    compiler.load_operators('operators/mul.py', width=32, count=1)
    compiler.load_operators('operators/mem.py',  load_count=1, store_count=1)
    # compiler.load_operators('operators/fpu.py',  width=REG_WIDTH, count=1)
    # compiler.load_operators('operators/fmul.py', width=REG_WIDTH, count=1)
    compiler.load_operators('operators/iv_init.py', width=REG_WIDTH, count=1)

    if sched_pass is not None:
        compiler.replace_pass('state_assign', sched_pass)

    return compiler


def count_fsm_states(program_scala):
    """FSMのステート数を数える。"""
    lines = program_scala.split('\n')
    in_fsm = False
    count = 0
    for line in lines:
        if 'class FSM' in line:
            in_fsm = True
        if 'class DEC' in line:
            in_fsm = False
        if in_fsm and line.strip().startswith('is(') and '.U)' in line:
            count += 1
    return count


# ================================================================
# パス構成の確認
# ================================================================

print('=== パス構成 ===')
for sched_name, sched_pass in SCHEDULERS:
    compiler = make_compiler(sched_pass)
    print(f'\n{sched_name}:')
    compiler.print_passes()


# ================================================================
# 実験
# ================================================================

print()
print(f"{'スケジューラ':10} {'ベンチマーク':10} {'ステート数':>10} {'GPR使用数':>10} {'IMM使用数':>10} {'スピル数':>8} {'反復回数':>8}")
print('-' * 65)

for sched_name, sched_pass in SCHEDULERS:
    for bench in BENCHMARKS:
        # スケジューラのインスタンスは毎回新しく作る
        if sched_pass is not None:
            sched_pass_instance = sched_pass.__class__()
        else:
            sched_pass_instance = None

        compiler = make_compiler(sched_pass_instance)
        c_source = open(f'input/{bench}.c').read()
        ctx = compiler.compile(f'{bench}.c', c_source)

        states = count_fsm_states(ctx.program_scala)
        gpr    = max((v for v in ctx.reg_map.values() if v >= 0), default=-1) + 1
        imm    = max(ctx.imm_map.values(), default=-1) + 1 if ctx.imm_map else 0
        spills = len(ctx.spill_map)

        print(f"{sched_name:10} {bench:10} {states:>10} {gpr:>10} {imm:>10} {spills:>8} {ctx.iterations:>8}")