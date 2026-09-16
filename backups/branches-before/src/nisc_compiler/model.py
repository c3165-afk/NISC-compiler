"""Independent cycle execution of emitted register/IMM/SRAM control words.

Does not read SSA, CDFG edges or the allocator's live intervals. It checks held
inputs, commit timing, uninitialized reads and signed overflow while executing.
This is the agreed provisional datapath model, not an RTL simulator.
"""
from dataclasses import dataclass


class ModelError(ValueError):
    pass


def int32(value):
    if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
        raise ModelError(f"Value outside the int32 execution domain: {value}")
    return value


@dataclass
class ModelResult:
    return_value: int
    memory: dict
    done_state: int
    trace: list


def run(image, arguments, max_cycles=100000):
    if image.get('schema') != 'nisc-int32-controls-v1':
        raise ModelError('Unsupported control image')
    if len(arguments) != len(image['abi']['arguments']):
        raise ModelError('Argument count mismatch')
    memory = {i: int32(v) for i, v in enumerate(arguments)}
    registers = {0: 0}
    pending = {}
    trace = []
    returned = False
    last_write = -1
    cycles = image['cycles']
    for state, cycle in enumerate(cycles):
        if state >= max_cycles:
            raise ModelError('Execution step limit exceeded')
        if cycle['done']:
            if pending or cycle['units'] or not returned or last_write >= state:
                raise ModelError('Premature done or controls active at done')
            if state != image['done_state'] or state != len(cycles) - 1:
                raise ModelError('Invalid terminal state')
            return ModelResult(memory[image['abi']['return_address']], memory, state, trace)
        writes = {}
        mem_writes = {}
        for unit in pending:
            if unit not in cycle['units']:
                raise ModelError('Operation/input hold interrupted')
        for unit, control in cycle['units'].items():
            op = control['op']
            if op not in {'alu': ('add', 'sub'), 'mul': ('mul',), 'mem': ('load', 'store')}.get(unit, ()):
                raise ModelError('Operation on incompatible physical unit')
            values = []
            for kind, index in control['inputs']:
                if kind == 'imm':
                    if not 0 <= index < len(image['immediates']):
                        raise ModelError('Invalid immediate register')
                    values.append(int32(image['immediates'][index]))
                elif kind == 'gpr':
                    if not 0 <= index < image['registers'] or index not in registers:
                        raise ModelError(f'Uninitialized/invalid register R{index}')
                    values.append(registers[index])
                else:
                    raise ModelError('Unknown operand source')
            expected_inputs = 1 if op == 'load' else 2
            if len(values) != expected_inputs:
                raise ModelError('Operand count mismatch')
            signature = (op, control['inputs'], values)
            if unit in pending:
                saved, elapsed = pending[unit]
                if saved != signature:
                    raise ModelError('Input or control changed before completion')
            else:
                elapsed = 0
            elapsed += 1
            latency = image['latencies'][op]
            commit = control['write'] is not None or control['store']
            if commit != (elapsed == latency) or elapsed > latency:
                raise ModelError('Writeback/commit at the wrong cycle')
            if not commit:
                pending[unit] = (signature, elapsed)
                continue
            pending.pop(unit, None)
            if control['store'] != (op == 'store'):
                raise ModelError('Invalid memory write enable')
            if op == 'add':
                result = int32(values[0] + values[1])
            elif op == 'sub':
                result = int32(values[0] - values[1])
            elif op == 'mul':
                result = int32(values[0] * values[1])
            else:
                address = values[0 if op == 'load' else 1]
                if not 0 <= address < image['memory_words']:
                    raise ModelError('SRAM address out of range')
                if op == 'load':
                    if address not in memory:
                        raise ModelError('Uninitialized SRAM read')
                    result = memory[address]
                else:
                    if address != image['abi']['return_address'] or returned:
                        raise ModelError('Unexpected or repeated return store')
                    if control['write'] is not None:
                        raise ModelError('STORE must not write a register')
                    mem_writes[address] = values[0]
                    returned = True
            if op != 'store':
                reg = control['write']
                if type(reg) is not int or not 1 <= reg < image['registers'] or reg in writes:
                    raise ModelError('Invalid/conflicting register write')
                writes[reg] = result
            last_write = state
        # Reads above see pre-edge contents. All commits occur together here.
        registers.update(writes)
        memory.update(mem_writes)
        trace.append({'state': state, 'register_writes': writes, 'memory_writes': mem_writes})
    raise ModelError('No done state reached')
