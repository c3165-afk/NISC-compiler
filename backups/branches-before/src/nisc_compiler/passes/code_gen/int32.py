"""Lower scheduled int32 operations to explicit cycle controls and Chisel.

The same physical-register control image is consumed by the model runner.
The external nisc RTL/Base classes are not part of this repository; this backend
therefore specifies, but cannot certify, their clock and SRAM interface contract.
"""
from .emitter import MUX_IDX, CONST_REG_PORT
from nisc_compiler.passes.state_assign.int32 import verify_schedule

KINDS = {'arith.addi': ('alu', 'add'), 'arith.subi': ('alu', 'sub'),
         'arith.muli': ('mul', 'mul'), 'memref.load': ('mem', 'load'),
         'memref.store': ('mem', 'store')}


def build_image(graph, context):
    verify_schedule(graph, context)
    done = graph.graph['done_state'] + 1  # hardware state 0 is the reset/idle cycle
    cycles = [{"done": False, "units": {}} for _ in range(done + 1)]
    cycles[done]['done'] = True
    timings = {}
    for _, d in graph.nodes(data=True):
        if d['type'] != 'op':
            continue
        unit, operation = KINDS[d['op_name']]
        latency = d['latency']
        if operation in timings and timings[operation] != latency:
            raise ValueError("Different latencies for the same physical operation are unsupported")
        timings[operation] = latency
        inputs = [(['imm', context.imm_map[s]] if s in context.imm_map
                   else ['gpr', context.reg_map[s]]) for s in d['operands']]
        if unit == 'mem' and any(kind != 'gpr' for kind, _ in inputs):
            raise ValueError("Memory operands must be materialized into GPRs")
        start = d['state'] + 1
        for t in range(start, start + latency):
            if unit in cycles[t]['units']:
                raise ValueError("Two operations drive the same datapath controls")
            final = t == start + latency - 1
            cycles[t]['units'][unit] = {
                'op': operation, 'inputs': [list(src) for src in inputs],
                'write': context.reg_map[d['results'][0]] if final and d['results'] else None,
                'store': final and operation == 'store',
            }
    return {'schema': 'nisc-int32-controls-v1', 'abi': context.extra['abi'],
            'registers': context.num_registers, 'memory_words': context.memory_words,
            'immediates': context.extra['imm_values'], 'latencies': timings,
            'cycles': cycles, 'done_state': done}


def emit_int32(graph, context, package):
    image = build_image(graph, context)
    lines = [f'package {package}', '', 'import chisel3._', 'import chisel3.util._',
             'import nisc._', 'import nisc.Defs._', '',
             '// int32 contract: word-addressed SRAM; commit at the end of a state.',
             '// Non-pipelined inputs are held; reset starts one invocation.',
             'object VarName {']
    for reg in sorted(set(context.reg_map.values())):
        lines.append(f'    val R{reg} = {reg}.U(RF_LEN_BIT.W)')
    lines.extend(['}', 'object ConstTable {'])
    for index in range(len(image['immediates'])):
        lines.append(f'    val CONST_{index} = {index}.U(ConstReg_LEN_BIT.W)')
    lines.append('    private val table = Map[Int, BigInt](')
    lines.extend(f'        {i} -> BigInt("{v & 0xffffffff}"),' for i, v in enumerate(image['immediates']))
    lines.extend(['    )', '    val constInit = (0 until ConstReg_LEN).map(i => table.getOrElse(i, BigInt(0)))',
                  '}', 'import VarName._', 'import ConstTable._', '',
                  'class FSM extends FsmBase {', '    io.done := false.B', '    switch(state) {'])
    for state, cycle in enumerate(image['cycles']):
        lines.append(f'        is({state}.U) {{')
        if cycle['done']:
            lines.append('            io.done := true.B')
        lines.append(f"            state := {state if cycle['done'] else state + 1}.U")
        lines.append('        }')
    lines.extend(['    }', '}', '', 'class DEC extends DecBase {',
                  '    io.dp.rf_wen.foreach(_ := false.B)',
                  '    io.dp.mem_wen := false.B',
                  '    io.dp.mux.foreach(_ := false.B)',
                  '    io.dp.rf_raddr.foreach(_ := 0.U)',
                  '    io.dp.rf_waddr.foreach(_ := 0.U)',
                  '    io.const_reg_addr.foreach(_ := 0.U)',
                  '    io.dp.alu_code := ALU_ADD',
                  '    switch(io.state) {'])

    def read(port, source):
        kind, index = source
        if kind == 'imm':
            return [f'            io.dp.mux({MUX_IDX[port]}) := true.B',
                    f'            io.const_reg_addr({CONST_REG_PORT[port]}) := CONST_{index}']
        return [f'            io.dp.rf_raddr({port}) := R{index}']

    for state, cycle in enumerate(image['cycles']):
        lines.append(f'        is({state}.U) {{')
        for unit, control in sorted(cycle['units'].items()):
            op = control['op']
            if unit in ('alu', 'mul'):
                if unit == 'alu':
                    lines.append(f"            io.dp.alu_code := {'ALU_ADD' if op == 'add' else 'ALU_SUB'}")
                prefix = unit.upper()
                for i, src in enumerate(control['inputs'], 1):
                    lines.extend(read(f'{prefix}_IN{i}', src))
                wb = f'{prefix}_RF_WB'
            else:
                address = control['inputs'][0 if op == 'load' else 1]
                lines.extend(read('MEM_WADDR', address))
                if op == 'store':
                    lines.extend(read('MEM_WDATA', control['inputs'][0]))
                if control['store']:
                    lines.append('            io.dp.mem_wen := true.B')
                wb = 'MEM_RF_WB'
            if control['write'] is not None:
                lines.append(f'            io.dp.rf_wen({wb}) := true.B')
                lines.append(f"            io.dp.rf_waddr({wb}) := R{control['write']}")
        lines.append('        }')
    lines.extend(['    }', '}', ''])
    abi = image['abi']
    init = ['# int32 SRAM ABI; addresses are 32-bit WORD indices.',
            '# This is a layout, not an initialized numeric memory image.']
    init.extend(f'SRAM[{i}] = {name}  # input int32' for i, name in enumerate(abi['arguments']))
    init.append(f"SRAM[{abi['return_address']}] = return_value  # valid when done is true")
    context.extra['control_image'] = image
    context.extra['validation_level'] = 'structure'
    return '\n'.join(lines), '\n'.join(init) + '\n'
