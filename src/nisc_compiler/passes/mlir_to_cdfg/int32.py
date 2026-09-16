"""Make the SRAM calling convention explicit in the straight-line CDFG."""


def add_abi(graph, abi):
    bbs = [n for n, d in graph.nodes(data=True) if d['type'] == 'bb']
    if len(bbs) != 1 and 'blocks' not in graph.graph:
        raise ValueError("int32 profile requires a single basic block")
    bb = graph.graph.get('entry', bbs[0])
    returns = [n for n, d in graph.nodes(data=True) if d['op_name'] == 'func.return']
    if len(returns) != 1 or len(graph.nodes[returns[0]]['operands']) != 1:
        raise ValueError("Expected one int return value")
    result = graph.nodes[returns[0]]['operands'][0]
    original_ops = [n for n, d in graph.nodes(data=True) if d['type'] == 'op']
    args = [n for n, d in graph.nodes(data=True) if d['op_name'] == 'arg']
    if len(args) != len(abi['arguments']):
        raise ValueError("Argument count differs between C and CDFG")
    dead = returns + [n for n, d in graph.nodes(data=True) if d['op_name'] == 'done']
    graph.remove_nodes_from(dead)
    names = {s for _, d in graph.nodes(data=True) for s in d.get('results', [])}
    sequence = max(graph.nodes, default=-1) + 1

    def fresh():
        nonlocal sequence
        while f'%abi_{sequence}' in names:
            sequence += 1
        name = f'%abi_{sequence}'
        names.add(name)
        return name

    def node(op, operands=(), value=None, produces=True):
        nonlocal sequence
        name = fresh() if produces else None
        nid = sequence
        sequence += 1
        attrs = dict(type='ctrl' if op == 'arith.constant' else 'op',
                     op_name=op, operands=list(operands),
                     results=[name] if produces else [], mlir_typ='i32', operand_typ='i32')
        if value is not None:
            attrs['const_value'] = value
        graph.add_node(nid, **attrs)
        graph.add_edge(bb, nid, type='ctrl', label='contains')
        return nid, name

    _, zero = node('arith.constant', value=0)

    def materialize(value):
        _, literal = node('arith.constant', value=value)
        return node('arith.addi', [literal, zero])

    previous_load = None
    for address, arg in enumerate(args):
        _, addr = materialize(address)
        graph.nodes[arg].update(type='op', op_name='memref.load',
                                operands=[addr], abi_argument=address, operand_typ='i32')
        graph.add_edge(bb, arg, type='ctrl', label='contains')
        if previous_load is not None:
            graph.add_edge(previous_load, arg, type='order')
        previous_load = arg
    if previous_load is not None:
        for nid in original_ops:
            graph.add_edge(previous_load, nid, type='order')

    bb = graph.graph.get('exit', bb)

    constants = {s for _, d in graph.nodes(data=True) if d['op_name'] == 'arith.constant'
                 for s in d['results']}
    if result in constants:
        _, result = node('arith.addi', [result, zero])
    _, return_addr = materialize(abi['return_address'])
    store, _ = node('memref.store', [result, return_addr], produces=False)
    # Finish all computations (including unused source expressions) before return.
    for nid, data in list(graph.nodes(data=True)):
        if data['type'] == 'op' and nid != store:
            graph.add_edge(nid, store, type='order')

    # Rebuild dependencies from operands, not the legacy single-SSA edge label.
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'data'])
    producers = {}
    for nid, data in graph.nodes(data=True):
        for ssa in data.get('results', []):
            if ssa in producers:
                raise ValueError(f"Multiple SSA definitions: {ssa}")
            producers[ssa] = nid
    for nid, data in graph.nodes(data=True):
        for ssa in data.get('operands', []):
            if ssa not in producers:
                raise ValueError(f"Undefined SSA operand: {ssa}")
            graph.add_edge(producers[ssa], nid, type='data')
    graph.graph.update(abi=abi, return_store=store)
    return graph
