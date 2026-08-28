from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape='record',
        )
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot


if __name__ == '__main__':
    from engine import Value

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b
    c.label = 'c'

    draw(c).render('graph', view=True)