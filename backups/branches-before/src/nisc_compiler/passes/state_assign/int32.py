"""Single-BB scheduling and conservative, no-spill allocation for int32.

Uses the existing ASAP scheduler. Physical input paths are held for the full
latency, and register lifetimes include that hold time. R0 stays reserved.
"""
from dataclasses import replace
from collections import defaultdict

from .core import schedule_bb, ScheduleError
from nisc_compiler.passes.register_allocate.allocator import AllocationError

RESOURCE = {'arith.addi': 'alu', 'arith.subi': 'alu', 'arith.muli': 'mul',
            'memref.load': 'mem', 'memref.store': 'mem'}


def select_operators(operators):
    selected = []
    for op in operators:
        if len(op.pattern) != 1:
            continue  # fused patterns have no supported int32 backend lowering
        data = next(iter(op.pattern.nodes(data=True)))[1]
        kind = data.get('op_name')
        if (kind not in RESOURCE or data.get('typ', '') not in ('', 'i32')
                or op.typ not in ('', 'i32')):
            continue
        if type(op.latency) is not int or op.latency < 1:
            raise ScheduleError(f"Invalid latency for {op.name}")
        if type(op.count) is not int or op.count != 1 or op.pipelined:
            raise ScheduleError("int32 backend supports one non-pipelined unit per physical path")
        if any(o.name == op.name for o in selected):
            raise ScheduleError(f"Duplicate operator name: {op.name}")
        selected.append(replace(op, resource=RESOURCE[kind]))
    return selected


def schedule_allocate(graph, context):
    operators = context.extra['int32_operators']
    bb = next(n for n, d in graph.nodes(data=True) if d['type'] == 'bb')
    last = schedule_bb(graph, bb, operators)
    graph.graph['done_state'] = last + 1

    # Immediate table: same bit-pattern can share an IMM entry.
    imm_values = []
    imm_map = {}
    for _, d in graph.nodes(data=True):
        if d['op_name'] == 'arith.constant':
            value = d['const_value']
            if not -(1 << 31) <= value < (1 << 31):
                raise AllocationError("Constant outside int32 range")
            if value not in imm_values:
                imm_values.append(value)
            for ssa in d['results']:
                imm_map[ssa] = imm_values.index(value)
    if len(imm_values) > context.num_imm_registers:
        raise AllocationError("Immediate register capacity exceeded")

    # Reserve destination at launch; retain inputs through the last held cycle.
    # Inclusive endpoints deliberately forbid read/write reuse on the same edge.
    intervals = {}
    for _, d in graph.nodes(data=True):
        if d['type'] == 'op':
            for ssa in d['results']:
                intervals[ssa] = [d['state'], d['state'] + d['latency'] - 1]
    for _, d in graph.nodes(data=True):
        if d['type'] == 'op':
            for ssa in d['operands']:
                if ssa not in imm_map:
                    intervals[ssa][1] = max(intervals[ssa][1], d['state'] + d['latency'] - 1)
    reg_map = {}
    active = []
    for ssa, (start, end) in sorted(intervals.items(), key=lambda item: (item[1][0], item[0])):
        active = [(e, r) for e, r in active if e >= start]
        busy = {r for _, r in active}
        free = next((r for r in range(1, context.num_registers) if r not in busy), None)
        if free is None:
            raise AllocationError("int32 register capacity exceeded; spilling is not supported in this milestone")
        reg_map[ssa] = free
        active.append((end, free))
    context.reg_map = reg_map
    context.imm_map = imm_map
    context.iterations = 1
    context.extra['imm_values'] = imm_values
    for _, d in graph.nodes(data=True):
        for field in ('operands', 'results'):
            prefix = 'operand' if field == 'operands' else 'result'
            d[prefix + '_regs'] = [imm_map[s] if s in imm_map else reg_map[s] for s in d.get(field, [])]
            d[prefix + '_reg_types'] = ['imm' if s in imm_map else 'gpr' for s in d.get(field, [])]
    verify_schedule(graph, context)
    return graph


def verify_schedule(graph, context):
    """Reconstruct reservations and live ranges from the output, independently."""
    definitions = {}
    reservations = set()
    lifetimes = {}
    op_map = {o.name: o for o in context.extra['int32_operators']}
    for nid, d in graph.nodes(data=True):
        if d['type'] != 'op':
            continue
        start, latency = d['state'], d['latency']
        op = op_map[d['assigned_op']]
        if type(start) is not int or start < 0 or latency != op.latency:
            raise ScheduleError("Invalid scheduled time or latency")
        for t in range(start, start + latency):
            key = (t, RESOURCE[d['op_name']])
            if key in reservations:
                raise ScheduleError(f"Physical resource conflict at {key}")
            reservations.add(key)
        if start + latency > graph.graph['done_state']:
            raise ScheduleError("done precedes completion")
        for ssa in d['results']:
            definitions[ssa] = start + latency
            lifetimes[ssa] = [start, start + latency - 1]
    for _, d in graph.nodes(data=True):
        if d['type'] != 'op':
            continue
        for ssa in d['operands']:
            if ssa in context.imm_map:
                continue
            if ssa not in definitions or definitions[ssa] > d['state']:
                raise ScheduleError(f"Operand not ready: {ssa}")
            lifetimes[ssa][1] = max(lifetimes[ssa][1], d['state'] + d['latency'] - 1)
    for u, v, edge in graph.edges(data=True):
        a, b = graph.nodes[u], graph.nodes[v]
        if edge['type'] in ('order', 'data') and a['type'] == b['type'] == 'op':
            if a['state'] + a['latency'] > b['state']:
                raise ScheduleError("Dependency violated")
    by_register = defaultdict(list)
    for ssa, (start, end) in lifetimes.items():
        r = context.reg_map.get(ssa, -1)
        if not 1 <= r < context.num_registers:
            raise AllocationError(f"Invalid register for {ssa}")
        for lo, hi in by_register[r]:
            if start <= hi and lo <= end:
                raise AllocationError("Overlapping values share a register")
        by_register[r].append((start, end))
