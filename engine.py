

class Value:
    def __init__(self,data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None


    def __add__(self, other):
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad # c = a + b  --> it will make the grad of a equal to the grad of c and the grad of b equal to the grad of c because its addition
            other.grad += out.grad 


        out._backward = _backward
        return out

    def __mul__(self , other):
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += out.grad * other.data # c = a * b  --> it will make the grad of a equal to the grad of c multiplied by the value of b and the grad of b equal to the grad of c multiplied by the value of a because its multiplication
            other.grad += out.grad * self.data

        out._backward = _backward
        return out


    def backward(self):
        topo = []
        visited = set()
        
        def build_tupo (v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_tupo(child)
                topo.append(v)

        build_tupo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
            
        

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad} , prev={self._prev})"


if __name__ == '__main__':
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    e = Value(10.0, label='e')
    d = a * b + e
    d.label = 'd'

    d.backward()
    print(a.grad, b.grad, e.grad)