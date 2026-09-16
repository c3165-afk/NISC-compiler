"""Lower verified scf.if to explicit acyclic basic blocks and edge copies."""
import networkx as nx

from .core import integer_comparison_predicate


def lower_cfg(module):
    """Lower cf block arguments via explicit, parallel-safe edge copies."""
    graph = nx.DiGraph()
    blocks, aliases = {}, {}

    def add(**attrs):
        nid = len(graph)
        graph.add_node(nid, **attrs)
        return nid

    def block():
        bb = add(type='bb', op_name='bb', operands=[], results=[], mlir_typ='')
        blocks[bb] = {'successors': [], 'condition': None}
        return bb

    def value(v):
        return '%' + v.name_hint

    def operation(bb, kind, inputs, outputs, typ='i32', **attrs):
        nid = add(type='ctrl' if kind in ('arith.constant', 'nisc.phi', 'func.return') else 'op',
                  op_name=kind, operands=inputs, results=outputs, mlir_typ=typ,
                  operand_typ='i32', **attrs)
        graph.add_edge(bb, nid, type='ctrl', label='contains')
        return nid

    function = next(o for o in module.body.block.ops if o.name == 'func.func')
    bodies = list(function.regions[0].blocks)
    mapping = {body: block() for body in bodies}
    phis = {}
    for index, body in enumerate(bodies):
        for arg in body.args:
            if index == 0:
                add(type='arg', op_name='arg', operands=[], results=[value(arg)], mlir_typ='i32')
            else:
                phis[arg] = operation(mapping[body], 'nisc.phi', [], [value(arg)], incoming=[])

    def edge(target, arguments):
        if len(arguments) != len(target.args):
            raise ValueError('CFG edge argument count mismatch')
        bb = block()
        blocks[bb]['successors'] = [mapping[target]]
        zero = f'%cfgzero{len(graph)}'
        operation(bb, 'arith.constant', [], [zero], const_value=0)
        staged = []
        for source in arguments:
            name = f'%cfgtemp{len(graph)}'
            nid = operation(bb, 'arith.addi', [value(source), zero], [name])
            staged.append((nid, name))
        # Snapshot every source before overwriting any destination. This also
        # handles swaps and self-copies across loop back edges.
        for arg, (_, temp) in zip(target.args, staged):
            name = f'%cfgcopy{len(graph)}'
            nid = operation(bb, 'arith.addi', [temp, zero], [name])
            aliases[name] = value(arg)
            graph.nodes[phis[arg]]['incoming'].append(nid)
            for source, _ in staged:
                graph.add_edge(source, nid, type='order')
        return bb

    exits = []
    for body in bodies:
        bb = mapping[body]
        for op in body.ops:
            inputs, outputs = [value(v) for v in op.operands], [value(v) for v in op.results]
            if op.name == 'cf.br':
                blocks[bb]['successors'] = [edge(op.successor, op.arguments)]
            elif op.name == 'cf.cond_br':
                blocks[bb].update(condition=value(op.cond), successors=[
                    edge(op.then_block, op.then_arguments), edge(op.else_block, op.else_arguments)])
            elif op.name == 'arith.constant':
                operation(bb, op.name, [], outputs, const_value=int(op.value.value.data))
            elif op.name == 'arith.cmpi':
                operation(bb, op.name, inputs, outputs, typ='i1', predicate=integer_comparison_predicate(op))
            elif op.name in ('arith.addi', 'arith.subi', 'arith.muli', 'func.return'):
                operation(bb, op.name, inputs, outputs)
                if op.name == 'func.return':
                    exits.append(bb)
            else:
                raise ValueError(f'Unsupported CFG operation: {op.name}')
    if len(exits) != 1:
        raise ValueError('CFG requires one common return block')
    graph.graph.update(blocks=blocks, entry=mapping[bodies[0]], exit=exits[0],
                       aliases=aliases, explicit_cfg=True)
    return graph


def lower_structured(module):
    graph = nx.DiGraph()
    blocks, aliases = {}, {}
    counter = 0

    def add(**attrs):
        nonlocal counter
        nid = counter
        counter += 1
        graph.add_node(nid, **attrs)
        return nid

    def block():
        nid = add(type='bb', op_name='bb', operands=[], results=[], mlir_typ='')
        blocks[nid] = {'successors': [], 'condition': None}
        return nid

    def value(v):
        return '%' + v.name_hint

    def operation(bb, kind, inputs, outputs, typ='i32', **attrs):
        nid = add(type='ctrl' if kind in ('arith.constant', 'nisc.phi', 'func.return') else 'op',
                  op_name=kind, operands=inputs, results=outputs, mlir_typ=typ,
                  operand_typ='i32', **attrs)
        graph.add_edge(bb, nid, type='ctrl', label='contains')
        return nid

    entry = block()
    function = next(o for o in module.body.block.ops if o.name == 'func.func')
    body = function.regions[0].blocks[0]
    for arg in body.args:
        add(type='arg', op_name='arg', operands=[], results=[value(arg)], mlir_typ='i32')

    def lower(ops, current):
        for op in ops:
            inputs, outputs = [value(v) for v in op.operands], [value(v) for v in op.results]
            if op.name == 'scf.if':
                decision = current
                then, other = block(), block()
                blocks[decision].update(successors=[then, other], condition=inputs[0])
                ends, yielded = [], []
                for arm, region in zip((then, other), op.regions):
                    end, values = lower(region.blocks[0].ops, arm)
                    ends.append(end)
                    yielded.append(values)
                join = block()
                for end in ends:
                    blocks[end]['successors'] = [join]
                for i, result in enumerate(outputs):
                    incoming = []
                    for end, values in zip(ends, yielded):
                        zero = f'%edgezero{counter}'
                        operation(end, 'arith.constant', [], [zero], const_value=0)
                        copied = f'%edgecopy{counter}'
                        copy = operation(end, 'arith.addi', [values[i], zero], [copied])
                        aliases[copied] = result
                        incoming.append(copy)
                    operation(join, 'nisc.phi', [], [result], incoming=incoming)
                current = join
            elif op.name == 'scf.yield':
                return current, inputs
            elif op.name == 'arith.constant':
                operation(current, op.name, [], outputs, const_value=int(op.value.value.data))
            elif op.name == 'arith.cmpi':
                predicate = integer_comparison_predicate(op)
                operation(current, op.name, inputs, outputs, typ='i1', predicate=predicate)
            elif op.name in ('arith.addi', 'arith.subi', 'arith.muli', 'func.return'):
                operation(current, op.name, inputs, outputs)
            else:
                raise ValueError(f'Unsupported structured operation: {op.name}')
        return current, []

    exit_bb, _ = lower(body.ops, entry)
    graph.graph.update(blocks=blocks, entry=entry, exit=exit_bb, aliases=aliases)
    return graph
