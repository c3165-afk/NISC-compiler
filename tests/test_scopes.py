"""Regression tests for declaration identity and definite initialization."""
import unittest

from test_int32 import compiler
from nisc_compiler.model import run
from nisc_compiler.passes.c_to_mlir.core import ParseError


class ScopeTests(unittest.TestCase):
    def check_result(self, source, expected, arguments=()):
        ctx = compiler().compile('scope.c', source)
        self.assertEqual(run(ctx.extra['control_image'], list(arguments)).return_value, expected)

    def test_write_before_shadow_survives_block_exit(self):
        self.check_result('int f(){int x=1;{x=2;int x=3;}return x;}', 2)

    def test_initialization_before_shadow_survives_block_exit(self):
        self.check_result('int f(){int x;{x=2;int x=3;}return x;}', 2)

    def test_nested_shadow_keeps_each_declarations_last_value(self):
        self.check_result('''int f(){int x=1;int result=0;
            {x=2;int x=3;{x=4;int x=5;result=x;}result=result+x;}
            return result+x;}''', 11)

    def test_branches_update_outer_binding_before_shadow(self):
        source = '''int f(int a){int x;
            if(a){x=2;int x=8;x+=1;}else{x=3;int x=9;x+=1;}
            return x;}'''
        self.check_result(source, 2, [1])
        self.check_result(source, 3, [0])

    def test_one_arm_preserves_previous_value(self):
        source = 'int f(int a){int x=7;if(a){x=2;int x=3;}return x;}'
        self.check_result(source, 2, [1])
        self.check_result(source, 7, [0])

    def test_parameter_and_generated_looking_names(self):
        self.check_result('int f(int x){{x+=2;int x=99;}return x;}', 5, [3])
        self.check_result('int f(int nisc_binding0){int nisc_binding1=2;'
                          '{nisc_binding0+=nisc_binding1;int nisc_binding0=99;}'
                          'return nisc_binding0;}', 5, [3])

    def test_sibling_scopes_and_multi_declarations(self):
        self.check_result('int f(){int x=1;{int x=2;x+=1;}'
                          '{x+=3;int x=4,y=x+1;x=y;}return x;}', 4)

    def test_invalid_reads_and_redeclarations_remain_rejected(self):
        sources = [
            'int f(){int x;{int x=2;}return x;}',
            'int f(){int x=1;{int x=x;}return x;}',
            'int f(int a){int x;if(a){x=2;int x=3;}return x;}',
            'int f(){int x=1;int x=2;return x;}',
            'int f(int x){int x=2;return x;}',
            'int f(){{int y=2;}return y;}',
            'int f(){int x=1;{int x;x+=1;}return x;}',
        ]
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ParseError):
                compiler().compile('bad-scope.c', source)


if __name__ == '__main__':
    unittest.main()
