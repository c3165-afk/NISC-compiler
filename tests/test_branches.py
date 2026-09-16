import copy
import operator
import random
import unittest
import networkx as nx

from test_int32 import compiler
from nisc_compiler.model import run, ModelError
from nisc_compiler.passes.c_to_mlir.core import ParseError
from nisc_compiler.passes.state_assign.int32 import verify_schedule
from nisc_compiler.passes.state_assign.core import ScheduleError
from nisc_compiler.passes.mlir_to_cdfg.core import lower_mlir
from nisc_compiler.passes.c_to_mlir.core import parse_c
from nisc_compiler.passes.code_gen.emitter import gen_dec, lower_integer_comparison


class BranchTests(unittest.TestCase):
    def test_common_predicate_preserved_in_both_lowering_paths(self):
        for predicate, symbol in [('eq', '=='), ('ne', '!='), ('slt', '<'),
                                  ('sle', '<='), ('sgt', '>'), ('sge', '>=')]:
            mlir = f'''func.func @f(%a: i32, %b: i32) -> i1 {{
                %condition = arith.cmpi {predicate}, %a, %b : i32
                return %condition : i1
            }}'''
            legacy = lower_mlir(mlir)
            source = f'int f(int a,int b){{return a{symbol}b;}}'
            current = self.compile(source).extra['cdfg']
            for graph in (legacy, current):
                node = next(d for _, d in graph.nodes(data=True) if d['op_name'] == 'arith.cmpi')
                self.assertEqual(node['predicate'], predicate)
        # The legacy loop builder creates its comparator itself, without a cmpi
        # operation in the input MLIR. It must also supply an explicit predicate.
        graph = lower_mlir(parse_c('int f(){int x=0;for(int i=0;i<3;i++){x+=i;}return x;}'))
        cmps = [d for _, d in graph.nodes(data=True) if d['op_name'] == 'arith.cmpi']
        self.assertTrue(cmps)
        self.assertTrue(all(d['predicate'] == 'slt' for d in cmps))

    def test_legacy_comparator_emission_uses_common_codes_and_operand_order(self):
        expected = {'eq': ('BR_BEQ', False), 'ne': ('BR_BNE', False),
                    'slt': ('BR_BLT', False), 'sle': ('BR_BGE', True),
                    'sgt': ('BR_BLT', True), 'sge': ('BR_BGE', False)}
        for predicate, (code, swap) in expected.items():
            for immediate in (False, True):
                graph = nx.DiGraph()
                node = dict(type='op', op_name='arith.cmpi', operand_typ='i32',
                            predicate=predicate, operands=['%a', '%b'], results=['%flag'], state=0)
                graph.add_node(0, **node)
                before = copy.deepcopy(dict(graph.nodes[0]))
                output = gen_dec(graph, {'%a': 1, '%b': 2}, {'%b': 0} if immediate else {}, 'test')
                self.assertIn(f'io.dp.cmp_code           := {code}', output)
                a_port, b_port = (2, 1) if swap else (1, 2)
                self.assertIn(f'io.dp.rf_raddr(CMP_IN{a_port})  := A', output)
                if immediate:
                    self.assertIn(f'io.const_reg_addr(CMP_IN{b_port}_ConstReg) := CONST_0', output)
                    self.assertIn(f'io.dp.mux({b_port + 1})', output)
                else:
                    self.assertIn(f'io.dp.rf_raddr(CMP_IN{b_port})  := B', output)
                self.assertEqual(dict(graph.nodes[0]), before)

    def test_common_comparator_rejects_missing_or_unsupported_conditions(self):
        valid = dict(type='op', op_name='arith.cmpi', operand_typ='i32', predicate='eq',
                     operands=['%a', '%b'], results=['%flag'], state=0)
        for changes in [{'predicate': None}, {'predicate': 'ult'}, {'predicate': 'invalid'},
                        {'op_name': 'arith.cmpf'}, {'operand_typ': 'f32'}, {'operand_typ': 'i64'},
                        {'operands': ['%a']}]:
            node = dict(valid, **changes)
            graph = nx.DiGraph()
            graph.add_node(0, **node)
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    lower_integer_comparison(node)
                with self.assertRaises(ValueError):
                    gen_dec(graph, {'%a': 1, '%b': 2}, {}, 'test')

    def compile(self, source, **kwargs):
        return compiler(**kwargs).compile('branch.c', source)

    def check_return(self, ctx, args, expected):
        result = run(ctx.extra['control_image'], args)
        self.assertEqual(result.return_value, expected)
        self.assertEqual([result.memory[i] for i in range(len(args))], args)
        stores = [(r['state'], r['memory_writes']) for r in result.trace if r['memory_writes']]
        self.assertEqual(stores, [(result.done_state - 1, {len(args): expected})])
        self.assertEqual(ctx.mlir_text.count('return '), 1)
        graph = ctx.extra['cdfg']
        if 'blocks' in graph.graph:
            blocks = graph.graph['blocks']
            cfg = nx.DiGraph()
            cfg.add_nodes_from(blocks)
            cfg.add_edges_from((b, s) for b, meta in blocks.items() for s in meta['successors'])
            if not graph.graph.get('explicit_cfg'):
                self.assertTrue(nx.is_directed_acyclic_graph(cfg))
            self.assertEqual({b for b in cfg if cfg.out_degree(b) == 0}, {graph.graph['exit']})
            self.assertEqual(nx.descendants(cfg, graph.graph['entry']) | {graph.graph['entry']}, set(cfg))
        return result

    def test_early_return_and_all_arms_return(self):
        for source, cases in [
            ('int f(int a){if(a)return 7;return 9;}', [(0, 9), (1, 7), (-1, 7)]),
            ('int f(int a){if(a<0)return -a;else return a+1;}', [(-3, 3), (0, 1), (4, 5)]),
            ('int f(int a){if(a<0){if(a<-2)return 5;return 6;}else if(a==0)return 7;else return 8;}',
             [(-3, 5), (-1, 6), (0, 7), (2, 8)]),
            ('int f(int a){{return a;}return 99;}', [(3, 3)]),
        ]:
            ctx = self.compile(source)
            for a, expected in cases:
                with self.subTest(source=source, a=a):
                    self.check_return(ctx, [a], expected)

    def test_return_continuation_and_shadowed_bindings(self):
        source = '''int f(int a,int b){int x=1;
            {x=2;int x=3;if(a)return x;}
            if(b){int x=5;if(b>1)return x;x=6;}else x=8;
            return x+10;}'''
        ctx = self.compile(source)
        for args, expected in [([1, 0], 3), ([0, 2], 5), ([0, 1], 12), ([0, 0], 18)]:
            self.check_return(ctx, args, expected)

    def test_return_join_uses_only_live_initialization_paths(self):
        for source in [
            'int f(int a){int x;if(a)return 7;else x=8;return x;}',
            'int f(int a){int x;if(a)x=7;else return 8;return x;}',
            'int f(int a){int x;{if(a)return 7;x=8;}return x;}',
        ]:
            ctx = self.compile(source)
            for a in (0, 1):
                self.check_return(ctx, [a], 7 if a else 8)
        for source in [
            'int f(int a){if(a)return 1;}',
            'int f(int a,int b){if(a){if(b)return 1;}else return 2;}',
            'int f(int a,int b){int x;if(a)return 1;else if(b)x=2;return x;}',
            'int f(int a){int x;if(a)return x;else x=2;return x;}',
            'int f(int a){if(a)return;return 2;}',
        ]:
            with self.subTest(source=source), self.assertRaises(ParseError):
                self.compile(source)

    def test_return_skips_following_arithmetic_and_conditions(self):
        ctx = self.compile('int f(int a){if(a)return 7;int x=2147483647+1;return x;}')
        self.check_return(ctx, [1], 7)
        with self.assertRaises(ModelError):
            run(ctx.extra['control_image'], [0])
        ctx = self.compile('int f(int a){return a;int x=2147483647+1;if(x*x)return 2;return 3;}')
        self.check_return(ctx, [4], 4)
        self.assertNotIn('arith.muli', ctx.mlir_text)
        self.assertNotIn('2147483647', ctx.mlir_text)

    def test_early_return_latency_and_single_exit_store(self):
        for latency in (1, 2, 4):
            c = compiler()
            for op in c.context.operators:
                op.latency = latency
            ctx = c.compile('return.c', 'int f(int a,int b){if(a*b>10)return a*b;if(a==b)return a<b;return a-b;}')
            for args, expected in [([4, 3], 12), ([2, 2], 0), ([1, 2], -1)]:
                self.check_return(ctx, args, expected)

    def test_generated_early_returns_against_reference(self):
        rng = random.Random(405)
        for _ in range(15):
            p, q, r = (rng.randint(-4, 4) for _ in range(3))
            source = f'''int f(int a,int b){{int x=a+({p});
                if(a<b){{if(x>{q})return x*b;x=x+2;}}
                else {{if(b<{r})return x-b;x=x-2;}}
                if(x==b)return a<=b;return x+b;}}'''
            ctx = self.compile(source)
            for a, b in [(-3, 2), (2, -3), (0, 0), (3, 3), (-2, -2)]:
                x = a + p
                if a < b and x > q:
                    expected = x * b
                elif a >= b and b < r:
                    expected = x - b
                else:
                    x += 2 if a < b else -2
                    expected = int(a <= b) if x == b else x + b
                with self.subTest(p=p, q=q, r=r, a=a, b=b):
                    self.check_return(ctx, [a, b], expected)

    def test_while_zero_one_many_and_recomputed_bound(self):
        for source, expected in [
            ('int f(int n){int i=0;int x=0;while(i<n){x+=i;i++;}return x;}',
             lambda n: sum(range(max(n, 0)))),
            ('int f(int n){int x=0;while(n>0){x+=n;n-=2;}return x;}',
             lambda n: sum(range(n, 0, -2))),
            ('int f(int n){int i=0;while(i<n){i++;n--;}return i;}',
             lambda n: (max(n, 0) + 1) // 2),
        ]:
            ctx = self.compile(source)
            for n in (-2, 0, 1, 2, 7, 20):
                with self.subTest(source=source, n=n):
                    result = self.check_return(ctx, [n], expected(n))
                    if n >= 7:
                        self.assertTrue(any(r['next_state'] < r['state'] for r in result.trace))

    def test_for_init_condition_step_and_scope(self):
        cases = [
            ('int f(int n){int x=0;for(int i=0;i<n;i++)x+=i;return x;}', lambda n: sum(range(n))),
            ('int f(int n){int x=0;for(int i=n;i>0;--i)x+=i;return x;}', lambda n: sum(range(1, n+1))),
            ('int f(int n){int x=0;for(int i=0;i<=n;i+=2)x+=i;return x;}', lambda n: sum(range(0, n+1, 2))),
            ('int f(int n){int i=0;for(;i!=n;)i++;return i;}', lambda n: n),
            ('int f(int n){int i=9;int x=0;for(int i=0,j=n;i<j;i++,j--)x+=i;return x+i;}',
             lambda n: sum(range((n+1)//2)) + 9),
            ('int f(int n){int i=0;for(i=n;i>0;i--);return i;}', lambda n: 0),
        ]
        for source, expected in cases:
            ctx = self.compile(source)
            for n in (0, 1, 2, 5, 10):
                with self.subTest(source=source, n=n):
                    self.check_return(ctx, [n], expected(n))

    def test_nested_loops_and_branch_updates(self):
        source = '''int f(int n){int x=0;
            for(int i=0;i<n;i++){int j=0;
                while(j<i){if(j<2)x+=i;else x-=j;j++;}}
            return x;}'''
        ctx = self.compile(source)
        for n in (0, 1, 2, 5, 8):
            expected = sum(i if j < 2 else -j for i in range(n) for j in range(i))
            self.check_return(ctx, [n], expected)

    def test_loop_return_skips_update_and_following_code(self):
        cases = [
            ('int f(int n){for(int i=0;i<n;i++){if(i==2)return i;}return 9;}',
             [(0, 9), (2, 9), (3, 2), (8, 2)]),
            ('int f(int n){for(int i=2147483647;i>n;i++){return i;}return n;}',
             [(0, 2147483647), (2147483647, 2147483647)]),
            ('int f(int n){while(n>0){while(n>2)return n;n--;}return 0;}',
             [(0, 0), (2, 0), (5, 5)]),
            ('int f(int n){for(;;){if(n==0)return 7;n--;}return 9;}', [(0, 7), (4, 7)]),
            ('int f(int n){if(n<0)return -n;while(n>0)n--;return n;}', [(-3, 3), (4, 0)]),
        ]
        for source, samples in cases:
            ctx = self.compile(source)
            for n, expected in samples:
                with self.subTest(source=source, n=n):
                    self.check_return(ctx, [n], expected)

    def test_loop_invariants_swaps_and_comparison_values(self):
        source = '''int f(int n,int a,int b){int bias=a*b;int i=0;
            while(i<n){int tmp=a;a=b;b=tmp;i++;}return a*3+b+bias;}'''
        ctx = self.compile(source, num_registers=24)
        for n in (0, 1, 2, 3, 12):
            a, b = (2, 5) if n % 2 == 0 else (5, 2)
            self.check_return(ctx, [n, 2, 5], a*3+b+10)
        ctx = self.compile('int f(int n){int i=0;int x=0;while((i<n)==1){x+=(i<2);i++;}return x;}')
        for n in (0, 1, 3, 7):
            self.check_return(ctx, [n], min(n, 2))

    def test_loop_latency_and_iteration_limit(self):
        for latency in (1, 2, 4):
            c = compiler()
            for op in c.context.operators:
                op.latency = latency
            ctx = c.compile('loop.c', 'int f(int n){int x=1;for(int i=0;i<n;i++)x*=2;return x;}')
            for n in (0, 1, 4):
                self.check_return(ctx, [n], 2**n)
        ctx = self.compile('int f(int n){while(n){}return 3;}')
        self.check_return(ctx, [0], 3)
        with self.assertRaisesRegex(ModelError, 'step limit'):
            run(ctx.extra['control_image'], [1], max_cycles=200)

    def test_loop_invalid_initialization_and_unsupported_control(self):
        for source in [
            'int f(int n){int x;while(n){x=1;n--;}return x;}',
            'int f(int n){for(int i=0;i<n;i++){}return i;}',
            'int f(int n){int i;for(;i<n;i++){}return 0;}',
            'int f(int n){while(n){break;}return 0;}',
            'int f(int n){for(int i=0;i<n;i++){continue;}return 0;}',
            'int f(int n){while(n){return n;}}',
            'int f(int n){int i=0;while(i++<n){}return i;}',
        ]:
            with self.subTest(source=source), self.assertRaises(ParseError):
                self.compile(source)

    def test_generated_loops_against_reference(self):
        rng = random.Random(411)
        for _ in range(12):
            threshold, step, bias = rng.randint(0, 4), rng.randint(1, 3), rng.randint(-3, 3)
            source = f'''int f(int n){{int x={bias};
                for(int i=0;i<n;i+={step}){{if(i<{threshold})x+=i;else x-=i;}}
                return x;}}'''
            ctx = self.compile(source)
            for n in (0, 1, 2, 5, 9):
                expected = bias + sum(i if i < threshold else -i for i in range(0, n, step))
                self.check_return(ctx, [n], expected)

    def test_loop_corruption_is_rejected(self):
        from nisc_compiler.passes.register_allocate.allocator import cfg_interference, AllocationError
        ctx = self.compile('int f(int n){int x=0;while(n>0){x+=n;n--;}return x;}')
        graph = ctx.extra['cdfg']
        broken = copy.deepcopy(graph)
        phi = next(d for _, d in broken.nodes(data=True) if d['op_name'] == 'nisc.phi' and len(d['incoming']) == 2)
        phi['incoming'].pop()
        with self.assertRaises(ScheduleError):
            verify_schedule(broken, ctx)
        with self.assertRaises(AllocationError):
            self.compile('int f(int n){int x=0;while(n>0){x+=n;n--;}return x;}', num_registers=3)
        interference, _ = cfg_interference(graph, ctx.imm_map)
        a, b = next(iter(interference.edges))
        saved = dict(ctx.reg_map)
        ctx.reg_map[a] = ctx.reg_map[b]
        with self.assertRaises(AllocationError):
            verify_schedule(graph, ctx)
        ctx.reg_map = saved
        broken = copy.deepcopy(graph)
        broken.graph['blocks'][broken.graph['entry']]['successors'] = [999999]
        with self.assertRaises(ScheduleError):
            verify_schedule(broken, ctx)

    def test_six_predicates_both_paths_and_boundaries(self):
        predicates = {'<': operator.lt, '<=': operator.le, '>': operator.gt,
                      '>=': operator.ge, '==': operator.eq, '!=': operator.ne}
        pairs = [(-3, 2), (2, -3), (7, 7), (-7, -7), (0, 0),
                 (-2147483648, 2147483647), (2147483647, -2147483648)]
        for symbol, compare in predicates.items():
            ctx = self.compile(f'int f(int a,int b){{int x=0;if(a{symbol}b)x=11;else x=22;return x;}}')
            for a, b in pairs:
                with self.subTest(symbol=symbol, a=a, b=b):
                    result = run(ctx.extra['control_image'], [a, b])
                    self.assertEqual(result.return_value, 11 if compare(a, b) else 22)
                    visited = [row['state'] for row in result.trace]
                    self.assertLess(len(visited), result.done_state)

    def test_comparison_values_are_int_zero_or_one(self):
        for symbol in ('<', '<=', '>', '>=', '==', '!='):
            ctx = self.compile(f'int f(int a,int b){{return a{symbol}b;}}')
            for a, b in [(1, 2), (2, 1), (2, 2)]:
                expected = int({'<': operator.lt, '<=': operator.le, '>': operator.gt,
                                '>=': operator.ge, '==': operator.eq, '!=': operator.ne}[symbol](a, b))
                self.assertEqual(run(ctx.extra['control_image'], [a, b]).return_value, expected)
        ctx = self.compile('int f(int a,int b){int x=(a<b)+3;return (x==4)*2;}')
        self.assertEqual(run(ctx.extra['control_image'], [1, 2]).return_value, 2)
        self.assertEqual(run(ctx.extra['control_image'], [2, 1]).return_value, 0)

    def test_one_arm_different_assignments_and_sequential_joins(self):
        source = '''int f(int a,int b){
            int x=10; int y=20;
            if(a<b) x=1; else y=2;
            if(a==b) x=x+3;
            return x+y;
        }'''
        ctx = self.compile(source)
        for args, expected in [([1, 2], 21), ([2, 1], 12), ([2, 2], 15)]:
            self.assertEqual(run(ctx.extra['control_image'], args).return_value, expected)

    def test_nested_conditions_empty_arms_and_integer_truth(self):
        source = '''int f(int a,int b){int x=5;
            if(a){if(b<0)x=7;else x=9;}else{;}
            if(b){}else{x=x+1;}
            return x;
        }'''
        ctx = self.compile(source)
        for args, expected in [([0, 0], 6), ([-2, -1], 7), ([1, 2], 9), ([3, 0], 10)]:
            self.assertEqual(run(ctx.extra['control_image'], args).return_value, expected)
        empty = self.compile('int f(int a){if(a){}else{}return 6;}')
        for a in (0, -1):
            self.assertEqual(run(empty.extra['control_image'], [a]).return_value, 6)

    def test_scopes_and_branch_initialization(self):
        cases = [
            ('int f(int a){int x=4;if(a){int x=8;x=x+1;}return x;}', [1], 4),
            ('int f(int a){int x;if(a)x=7;else x=8;return x;}', [0], 8),
            ('int f(int a){int x;if(a)x=7;else x=8;return x;}', [1], 7),
            ('int f(int a){int x=4;{int y=8;x=x+y;}return x;}', [0], 12),
            ('int f(int a){int x=4;int y=0;{y=x;int x=8;y=y+x;}return y+x;}', [0], 16),
        ]
        for source, args, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(run(self.compile(source).extra['control_image'], args).return_value, expected)
        for source in [
            'int f(int a){int x;if(a)x=1;return x;}',
            'int f(int a){if(a){int y=2;}return y;}',
            'int f(int a){int x=1;if(a){int x=x;}return x;}',
            'int f(int a){if(a)return 1;}',
            'int f(int a){do{a-=1;}while(a);return a;}',
        ]:
            with self.subTest(source=source), self.assertRaises(ParseError):
                self.compile(source)

    def test_unselected_arm_is_not_executed(self):
        ctx = self.compile('int f(int a){int x=0;if(a>0)x=7;else x=2147483647+1;return x;}')
        self.assertEqual(run(ctx.extra['control_image'], [1]).return_value, 7)
        with self.assertRaises(ModelError):
            run(ctx.extra['control_image'], [0])

    def test_compare_latency_and_branch_after_multicycle_operation(self):
        for latency in (1, 2, 4):
            c = compiler()
            for op in c.context.operators:
                if op.name == 'cmpi':
                    op.latency = latency
            ctx = c.compile('branch.c', 'int f(int a,int b){int x=0;if(a*b>=10)x=a*b;else x=a-b;return x;}')
            for args, expected in [([4, 3], 12), ([1, 2], -1)]:
                self.assertEqual(run(ctx.extra['control_image'], args).return_value, expected)

    def test_wrong_branch_and_missing_copy_rejected(self):
        ctx = self.compile('int f(int a){int x=1;if(a)x=2;return x;}')
        image = copy.deepcopy(ctx.extra['control_image'])
        branch = next(row for row in image['cycles'] if 'branch' in row)
        branch['units'].pop('cmp')
        with self.assertRaises(ModelError):
            run(image, [1])
        graph = copy.deepcopy(ctx.extra['cdfg'])
        phi = next(d for _, d in graph.nodes(data=True) if d['op_name'] == 'nisc.phi')
        phi['incoming'].pop()
        with self.assertRaises(ScheduleError):
            verify_schedule(graph, ctx)

    def test_emitted_fsm_targets_and_comparator_codes(self):
        codes = {'<': 'BR_BLT', '<=': 'BR_BGE', '>': 'BR_BLT',
                 '>=': 'BR_BGE', '==': 'BR_BEQ', '!=': 'BR_BNE'}
        for symbol, code in codes.items():
            ctx = self.compile(f'int f(int a,int b){{int x=0;if(a{symbol}b)x=1;return x;}}')
            image = ctx.extra['control_image']
            branch = next(c['branch'] for c in image['cycles'] if 'branch' in c)
            self.assertIn(f'io.dp.cmp_code := {code}', ctx.program_scala)
            self.assertIn(f"when(io.dp_cmp_out === true.B) {{\n                state := {branch['true']}.U", ctx.program_scala)
            self.assertIn(f"}}.otherwise {{\n                state := {branch['false']}.U", ctx.program_scala)
            cmp = next(c['units']['cmp'] for c in image['cycles'] if 'cmp' in c['units'])
            self.assertIsNone(cmp['write'])

    def test_generated_branch_programs_against_reference(self):
        rng = random.Random(20260916)
        for _ in range(30):
            threshold, left, right = [rng.randint(-5, 5) for _ in range(3)]
            source = f'''int f(int a,int b){{int x=a;int y=b;
                if(a<={threshold}){{x=x+({left});if(b)x=x*2;}}
                else{{y=y-({right});}}
                if(x==y)y=y+1;
                return x-y;
            }}'''
            ctx = self.compile(source)
            for a, b in [(rng.randint(-8, 8), rng.randint(-8, 8)) for _ in range(5)]:
                x, y = a, b
                if a <= threshold:
                    x += left
                    if b:
                        x *= 2
                else:
                    y -= right
                if x == y:
                    y += 1
                self.assertEqual(run(ctx.extra['control_image'], [a, b]).return_value, x - y)


if __name__ == '__main__':
    unittest.main()
