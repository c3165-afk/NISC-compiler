import copy
import json
import random
import re
import tempfile
import unittest
from pathlib import Path

from nisc_compiler import NISCCompiler
from nisc_compiler.model import run, ModelError
from nisc_compiler.passes.c_to_mlir.core import ParseError
from nisc_compiler.passes.code_gen.code_gen import ChiselCodeGenPass
from nisc_compiler.passes.register_allocate.allocator import AllocationError
from nisc_compiler.passes.state_assign.core import ScheduleError, schedule_bb
from nisc_compiler.passes.state_assign.int32 import verify_schedule
from nisc_compiler.passes.operator_assign.dp import Operator, _pat
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]


def compiler(**kwargs):
    c = NISCCompiler.default(**kwargs)
    for name in ('alu', 'mul', 'mem'):
        c.load_operators(str(ROOT / 'operators' / (name + '.py')))
    c.replace_pass('code_gen', ChiselCodeGenPass(None))
    return c


class Int32Tests(unittest.TestCase):
    def check_program(self, source, args, expected, c=None):
        ctx = (c or compiler()).compile('case.c', source)
        result = run(ctx.extra['control_image'], args)
        self.assertEqual(result.return_value, expected)
        self.assertEqual(result.memory[len(args)], expected)
        self.assertEqual([result.memory[i] for i in range(len(args))], args)
        stores = [(row['state'], row['memory_writes']) for row in result.trace if row['memory_writes']]
        self.assertEqual(stores, [(result.done_state - 1, {len(args): expected})])
        self.assertNotIn('???', ctx.program_scala)
        self.assertNotIn(-1, ctx.reg_map.values())
        self.assertNotIn(0, ctx.reg_map.values())
        # Every emitted state exists, including waits and writeback-only cycles.
        fsm, dec = ctx.program_scala.split('class DEC')
        for text in (fsm, dec):
            states = list(map(int, re.findall(r'is\((\d+)\.U\)', text)))
            self.assertEqual(states, list(range(result.done_state + 1)))
        transitions = list(map(int, re.findall(r'state := (\d+)\.U', fsm)))
        self.assertEqual(transitions, list(range(1, result.done_state + 1)) + [result.done_state])
        self.assertEqual(dec.count('io.dp.mem_wen := true.B'), 1)
        self.assertIn('io.dp.rf_wen.foreach(_ := false.B)', dec)
        self.assertIn('io.dp.mux.foreach(_ := false.B)', dec)
        return ctx

    def test_required_cases(self):
        cases = [
            ('int f(void) { return 7; }', [], 7),
            ('int f() { return 0; }', [], 0),
            ('int f() { return -19; }', [], -19),
            ('int f(int a) { return a; }', [-17], -17),
            ('int f(int a) { return a + 2; }', [5], 7),
            ('int f(int a) { return (a + 2) + 3; }', [5], 10),
            ('int f(int a, int b) { int x=a+1; int y=b-2; return x*y; }', [4, 8], 30),
            ('int f(int a, int b) { return a*b+3; }', [-4, 5], -17),
            ('int f(int a, int b) { return a*b; }', [7, -6], -42),
            ('int f(int a) { int x; x=a; x+=3; x*=2; x-=1; return -x; }', [4], -13),
            ('int f(int a) { int x=a+1; return x*x+x; }', [3], 20),
            ('int f(int a, int b) { int unused=a*b; return 9; }', [2, 3], 9),
            ('int f(int tmp_0, int tmp_1) { return tmp_0+tmp_1; }', [4, 6], 10),
            ('int f() { return 010 + 0x10; }', [], 24),
            ('int f(int a) { return a; }', [-2147483648], -2147483648),
            ('int f() { return -2147483647 - 1; }', [], -2147483648),
            ('int f() { return 2147483647; }', [], 2147483647),
        ]
        for source, args, expected in cases:
            with self.subTest(source=source):
                self.check_program(source, args, expected)

    def test_configured_latencies_not_hardcoded(self):
        for alu, mul, load, store in [(1, 1, 1, 1), (2, 4, 3, 2), (3, 5, 2, 4)]:
            c = compiler()
            for op in c.context.operators:
                kind = next(iter(op.pattern.nodes(data=True)))[1]['op_name']
                if kind in ('arith.addi', 'arith.subi'):
                    op.latency = alu
                elif kind == 'arith.muli':
                    op.latency = mul
                elif kind == 'memref.load':
                    op.latency = load
                elif kind == 'memref.store':
                    op.latency = store
            self.check_program('int f(int a, int b) { return (a+b)*(a-b); }', [9, 4], 65, c)

    def test_rejects_unsupported_or_invalid_source(self):
        sources = [
            'float f(){return 1.5;}', 'unsigned int f(){return 1;}',
            'int f(long a){return a;}', 'int f(int *a){return 1;}',
            'int f(){int a[2]; return 0;}', 'int f(){static int a=1;return a;}',
            'int f(volatile int a){return a;}', 'int f(int a){return a/2;}',
            'int f(int a){return a%2;}', 'int f(int a){return a<<1;}',
            'int f(int a){return a>1;}', 'int f(){return (int)1.5;}',
            'int f(int a){if(a)return 1;return 0;}',
            'int f(){int a;return a;}', 'int f(){int a=a;return a;}',
            'int f(){a=1;return a;}', 'int f(){int a; a+=1;return a;}',
            'int f(){return 1;return 2;}', 'int f(){int a=1;}',
            'int f(){return 2147483648;}', 'int f(){return 0xffffffff;}',
            'int f(){return 1U;}', 'int f(){return 1L;}',
            'int f(){return g();}', 'int f(){return 1;} int g(){return 2;}',
            'int x; int f(){return 1;}', 'int f(int a, int a){return a;}',
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ParseError):
                compiler().compile('bad.c', source)

    def test_rejects_capacity_and_configuration(self):
        for kwargs in ({'reg_width': 64}, {'num_registers': 0}, {'memory_words': 0},
                       {'num_imm_registers': 0}, {'profile': 'typo'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compiler(**kwargs).compile('bad.c', 'int f(){return 1;}')
        with self.assertRaises(AllocationError):
            compiler(num_registers=2).compile('bad.c', 'int f(int a){return a+1;}')
        with self.assertRaises(AllocationError):
            compiler(num_imm_registers=1).compile('bad.c', 'int f(){return 19;}')
        with self.assertRaises(ValueError):
            compiler(memory_words=1).compile('bad.c', 'int f(int a){return a;}')
        for field, value in [('latency', 0), ('count', 2), ('pipelined', True)]:
            c = compiler()
            setattr(c.context.operators[0], field, value)
            with self.assertRaises(ScheduleError):
                c.compile('bad.c', 'int f(){return 1;}')
        c = compiler()
        c.context.operators = [o for o in c.context.operators if o.name != 'store']
        with self.assertRaises(RuntimeError):
            c.compile('bad.c', 'int f(){return 1;}')

    def test_reference_arithmetic_random_programs(self):
        rng = random.Random(20260916)
        c = compiler()
        for trial in range(60):
            args = [rng.randint(-8, 8) for _ in range(3)]
            values = dict(zip(('a', 'b', 'c'), args))
            statements = []
            for index in range(8):
                left = rng.choice(list(values))
                right = rng.randint(-3, 3)
                op = rng.choice(('+', '-', '*'))
                val = {'+': lambda: values[left] + right,
                       '-': lambda: values[left] - right,
                       '*': lambda: values[left] * right}[op]()
                name = f'x{index}'
                statements.append(f'int {name}={left}{op}({right});')
                values[name] = val
            source = 'int f(int a,int b,int c){' + ''.join(statements) + 'return x7;}'
            with self.subTest(trial=trial):
                self.check_program(source, args, values['x7'], c)

    def test_overflow_is_outside_execution_domain(self):
        for source, args in [('int f(int a){return a+1;}', [2147483647]),
                             ('int f(int a){return -a;}', [-2147483648]),
                             ('int f(int a){return a*a;}', [50000])]:
            ctx = compiler().compile('overflow.c', source)
            with self.assertRaisesRegex(ModelError, 'int32 execution domain'):
                run(ctx.extra['control_image'], args)

    def test_model_detects_corrupted_controls(self):
        ctx = compiler().compile('case.c', 'int f(int a){return a*a;}')
        original = ctx.extra['control_image']
        for fault in ('early_wb', 'hold', 'done', 'uninitialized', 'step_limit'):
            image = copy.deepcopy(original)
            mul_states = [i for i, s in enumerate(image['cycles']) if 'mul' in s['units']]
            if fault == 'early_wb':
                image['cycles'][mul_states[0]]['units']['mul']['write'] = 1
            elif fault == 'hold':
                image['cycles'][mul_states[-1]]['units']['mul']['inputs'][0] = ['gpr', 0]
            elif fault == 'done':
                image['cycles'][1]['done'] = True
            elif fault == 'uninitialized':
                image['cycles'][mul_states[0]]['units']['mul']['inputs'][0] = ['gpr', 31]
            with self.subTest(fault=fault), self.assertRaises(ModelError):
                run(image, [7], max_cycles=1 if fault == 'step_limit' else 100000)

    def test_verifier_detects_register_and_schedule_corruption(self):
        for fault in ('register', 'early', 'done'):
            ctx = compiler().compile('case.c', 'int f(int a,int b){return a*b+1;}')
            graph = ctx.extra['cdfg']
            if fault == 'register':
                ctx.reg_map = {s: 1 for s in ctx.reg_map}
            elif fault == 'early':
                for _, d in graph.nodes(data=True):
                    if d['op_name'] == 'arith.muli':
                        d['state'] = 0
            else:
                graph.graph['done_state'] = 1
            with self.subTest(fault=fault), self.assertRaises((ScheduleError, AllocationError)):
                verify_schedule(graph, ctx)

    def test_future_resource_reservation_collision(self):
        # A is placed at t=3 first; B's free launch at t=0 is insufficient:
        # its four-cycle occupancy would collide with A at t=3.
        graph = nx.DiGraph()
        graph.add_node(0, type='bb')
        graph.add_node(1, type='ctrl', state=0, latency=3)
        for n in (2, 3):
            graph.add_node(n, type='op', assigned_op='mul', latency=4)
            graph.add_edge(0, n, label='contains', type='ctrl')
        graph.add_edge(1, 2, type='data')
        op = Operator('mul', 4, _pat((0, 'arith.muli', 'i32')))
        final = schedule_bb(graph, 0, [op])
        self.assertEqual(graph.nodes[2]['state'], 3)
        self.assertEqual(graph.nodes[3]['state'], 7)
        self.assertEqual(final, 10)

    def test_output_roundtrip_and_failure_has_no_new_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = compiler()
            c.replace_pass('code_gen', ChiselCodeGenPass(tmp))
            ctx = c.compile(r'C:\dir with spaces\1-case.c', 'int f(){return 6;}')
            path = Path(tmp) / 'program_1_case'
            self.assertEqual((path / 'Program.scala').read_text(encoding='utf-8').strip(), ctx.program_scala.strip())
            image = json.loads((path / 'controls.json').read_text(encoding='utf-8'))
            self.assertEqual(run(image, []).return_value, 6)
            before = {p: p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            with self.assertRaises(ParseError):
                c.compile('bad.c', 'float f(){return 1.0;}')
            after = {p: p.read_bytes() for p in Path(tmp).rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(c.context.program_scala, '')
            self.assertNotIn('control_image', c.context.extra)
            self.assertTrue(ctx.program_scala)  # preceding result is a stable snapshot

    def test_deterministic_ssa_and_output(self):
        source = 'int f(int a_1,int a_2){int x=a_1+a_2;return x*x;}'
        a = compiler().compile('same.c', source)
        b = compiler().compile('same.c', source)
        self.assertEqual(a.program_scala, b.program_scala)
        self.assertEqual(a.extra['control_image'], b.extra['control_image'])


if __name__ == '__main__':
    unittest.main()
