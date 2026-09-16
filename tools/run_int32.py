"""Compile the supported subset and execute the provisional datapath model."""
import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
for site in sorted((root / '.compiler' / 'lib').glob('python*/site-packages')):
    sys.path.append(str(site))

from nisc_compiler import NISCCompiler
from nisc_compiler.model import run
from nisc_compiler.passes.code_gen.code_gen import ChiselCodeGenPass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('arguments', nargs='*', type=int)
    parser.add_argument('--output', type=Path, help='Optional output directory (default: no files)')
    args = parser.parse_args()
    compiler = NISCCompiler.default()
    for name in ('alu', 'mul', 'mem', 'cmp'):
        compiler.load_operators(str(root / 'operators' / (name + '.py')))
    # Validate execution before publishing artifacts for this invocation.
    compiler.replace_pass('code_gen', ChiselCodeGenPass(None))
    ctx = compiler.compile(str(args.source), args.source.read_text(encoding='utf-8'))
    result = run(ctx.extra['control_image'], args.arguments)
    if args.output is not None:
        ChiselCodeGenPass(str(args.output)).run(ctx.extra['cdfg'], ctx)
    print(json.dumps({'return_value': result.return_value, 'done_state': result.done_state,
                      'return_address': len(args.arguments),
                      'validation': 'provisional datapath model; RTL not verified'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
