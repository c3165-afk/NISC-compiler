"""Single-BB scheduling and conservative, no-spill allocation for int32.

Uses the existing ASAP scheduler. Physical input paths are held for the full
latency, and register lifetimes include that hold time. R0 stays reserved.
"""
from dataclasses import replace
from collections import defaultdict
import networkx as nx

from .core import schedule_bb, ScheduleError
from nisc_compiler.passes.register_allocate.allocator import AllocationError

RESOURCE = {'arith.addi': 'alu', 'arith.subi': 'alu', 'arith.muli': 'mul',
            'memref.load': 'mem', 'memref.store': 'mem', 'arith.cmpi': 'cmp'}


def select_operators(operators):
    selected = []
    for op in operators:
        if len(op.pattern) != 1:
            continue  # fused patterns have no supported int32 backend lowering
        data = next(iter(op.pattern.nodes(data=True)))[1]
        kind = data.get('op_name')
        if (kind not in RESOURCE or data.get('typ', '') not in ('', 'i32')
                or op.typ not in ('', 'i32', 'i1')
                or (op.typ == 'i1' and kind != 'arith.cmpi')):
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
    if graph.graph.get('explicit_cfg'):
        from .core import schedule_cfg
        return schedule_cfg(graph, context)
    operators = context.extra['int32_operators']
    blocks = graph.graph.get('blocks')
    if blocks:
        cfg = nx.DiGraph()
        cfg.add_nodes_from(blocks)
        for bb, meta in blocks.items():
            cfg.add_edges_from((bb, target) for target in meta['successors'])
        offset = 0
        for bb in nx.topological_sort(cfg):
            members = [v for _, v, edge in graph.out_edges(bb, data=True) if edge.get('label') == 'contains']
            condition = blocks[bb]['condition']
            if condition:
                cmp = next(n for n in members if condition in graph.nodes[n]['results'])
                for n in members:
                    if n != cmp and graph.nodes[n]['type'] == 'op':
                        graph.add_edge(n, cmp, type='order')
            for n in members:
                if graph.nodes[n]['op_name'] == 'nisc.phi':
                    graph.nodes[n].update(state=offset, latency=0)
            last = max(offset, schedule_bb(graph, bb, operators, offset))
            blocks[bb].update(start=offset, end=last)
            offset = last + 1
    else:
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
        if (d['type'] == 'op' and d['op_name'] != 'arith.cmpi') or d['op_name'] == 'nisc.phi':
            for ssa in d['results']:
                intervals[ssa] = [d['state'], d['state'] + d['latency'] - 1]
    for _, d in graph.nodes(data=True):
        if d['type'] == 'op':
            for ssa in d['operands']:
                if ssa not in imm_map:
                    intervals[ssa][1] = max(intervals[ssa][1], d['state'] + d['latency'] - 1)
    aliases = graph.graph.get('aliases', {})
    grouped = {}
    for ssa, (start, end) in intervals.items():
        canonical = aliases.get(ssa, ssa)
        lo, hi = grouped.get(canonical, (start, end))
        grouped[canonical] = [min(lo, start), max(hi, end)]
    colors = {}
    active = []
    for ssa, (start, end) in sorted(grouped.items(), key=lambda item: (item[1][0], item[0])):
        active = [(e, r) for e, r in active if e >= start]
        busy = {r for _, r in active}
        free = next((r for r in range(1, context.num_registers) if r not in busy), None)
        if free is None:
            raise AllocationError("int32 register capacity exceeded; spilling is not supported in this milestone")
        colors[ssa] = free
        active.append((end, free))
    reg_map = {ssa: colors[aliases.get(ssa, ssa)] for ssa in intervals}
    context.reg_map = reg_map
    context.imm_map = imm_map
    context.iterations = 1
    context.extra['imm_values'] = imm_values
    for _, d in graph.nodes(data=True):
        for field in ('operands', 'results'):
            prefix = 'operand' if field == 'operands' else 'result'
            d[prefix + '_regs'] = [imm_map[s] if s in imm_map else reg_map.get(s) for s in d.get(field, [])]
            d[prefix + '_reg_types'] = ['imm' if s in imm_map else ('gpr' if s in reg_map else 'flag') for s in d.get(field, [])]
    verify_schedule(graph, context)
    return graph


def verify_schedule(graph, context):
    """Reconstruct reservations and live ranges from the output, independently."""
    if graph.graph.get('explicit_cfg'):
        from .core import verify_cfg_schedule
        return verify_cfg_schedule(graph, context)
    from .core import verify_completion
    verify_completion(graph, context)
    definitions = {}
    reservations = set()
    lifetimes = {}
    op_map = {o.name: o for o in context.extra['int32_operators']}
    for nid, d in graph.nodes(data=True):
        if d['op_name'] == 'nisc.phi':
            for ssa in d['results']:
                definitions[ssa] = d['state']
                lifetimes[ssa] = [d['state'], d['state']]
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
        for ssa in ([] if d['op_name'] == 'arith.cmpi' else d['results']):
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
    aliases = graph.graph.get('aliases', {})
    merged = {}
    for ssa, (start, end) in lifetimes.items():
        r = context.reg_map.get(ssa, -1)
        if not 1 <= r < context.num_registers:
            raise AllocationError(f"Invalid register for {ssa}")
        canonical = aliases.get(ssa, ssa)
        if r != context.reg_map[canonical]:
            raise AllocationError('Branch copy and joined value must share a register')
        lo, hi = merged.get(canonical, (start, end))
        merged[canonical] = (min(lo, start), max(hi, end))
    by_register = defaultdict(list)
    for ssa, (start, end) in merged.items():
        r = context.reg_map[ssa]
        for lo, hi in by_register[r]:
            if start <= hi and lo <= end:
                raise AllocationError("Overlapping values share a register")
        by_register[r].append((start, end))
    if graph.graph.get('blocks'):
        verify_cfg(graph, context)


def verify_cfg(graph, context):
    blocks = graph.graph['blocks']
    cfg = nx.DiGraph()
    cfg.add_nodes_from(blocks)
    for bb, meta in blocks.items():
        cfg.add_edges_from((bb, target) for target in meta['successors'])
    if (set(cfg) != set(blocks) or graph.graph['entry'] not in cfg
            or graph.graph['exit'] not in cfg):
        raise ScheduleError('Invalid CFG target, entry or exit')
    if nx.descendants(cfg, graph.graph['entry']) | {graph.graph['entry']} != set(blocks):
        raise ScheduleError('Unreachable CFG block')
    if {bb for bb in cfg if cfg.out_degree(bb) == 0} != {graph.graph['exit']}:
        raise ScheduleError('CFG requires a single common exit')
    for bb, meta in blocks.items():
        expected = 2 if meta['condition'] else (0 if bb == graph.graph['exit'] else 1)
        if len(meta['successors']) != expected:
            raise ScheduleError('Invalid CFG terminator')
    if not graph.graph.get('explicit_cfg') and not nx.is_directed_acyclic_graph(cfg):
        raise ScheduleError('Only acyclic control flow is supported')
    dominators = nx.immediate_dominators(cfg, graph.graph['entry'])
    owners = {v: bb for bb in blocks for _, v, e in graph.out_edges(bb, data=True)
              if e.get('label') == 'contains'}
    producers = {s: n for n, d in graph.nodes(data=True) for s in d.get('results', [])}
    for n, d in graph.nodes(data=True):
        if d['type'] != 'op' and d['op_name'] != 'nisc.phi':
            continue
        bb = owners[n]
        if not blocks[bb]['start'] <= d['state'] <= blocks[bb]['end']:
            raise ScheduleError('Operation outside its basic block')
        if d['state'] + d['latency'] - 1 > blocks[bb]['end']:
            raise ScheduleError('Control transfer before operation completion')
        if d['op_name'] == 'nisc.phi':
            incoming = d['incoming']
            if (len(incoming) != cfg.in_degree(bb) or len(incoming) != len(set(incoming))
                    or {owners[x] for x in incoming} != set(cfg.predecessors(bb))):
                raise ScheduleError('Missing branch edge copy')
            for copy in incoming:
                name = graph.nodes[copy]['results'][0]
                if graph.graph['aliases'].get(name) != d['results'][0]:
                    raise ScheduleError('Incorrect branch edge copy')
        for ssa in d['operands']:
            if ssa in context.imm_map:
                continue
            defining_bb = owners[producers[ssa]]
            cursor = bb
            while cursor != defining_bb and dominators[cursor] != cursor:
                cursor = dominators[cursor]
            if cursor != defining_bb:
                raise ScheduleError('Value does not dominate its use')
    for bb, meta in blocks.items():
        if meta['condition']:
            cmp = graph.nodes[producers[meta['condition']]]
            if (cmp['op_name'] != 'arith.cmpi' or owners[producers[meta['condition']]] != bb
                    or cmp['state'] + cmp['latency'] - 1 != meta['end']):
                raise ScheduleError('Branch must sample the comparator on its final cycle')
