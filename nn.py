from engine import Value
import random



class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]



class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]




class MLP: #Multi-Layer Perceptron
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    layer = Layer(3, 4)
    x = [Value(1.0), Value(2.0), Value(3.0)]

    print("this layer has", len(layer.neurons), "neurons")
    print()

    for i, n in enumerate(layer.neurons):
        weights = [round(w.data, 3) for w in n.w]
        print(f"neuron {i}:  w = {weights}   b = {round(n.b.data, 3)}")

    print()
    out = layer(x)
    for i, o in enumerate(out):
        print(f"neuron {i} output: {round(o.data, 4)}")

    print()
    print("total parameters:", len(layer.parameters()))