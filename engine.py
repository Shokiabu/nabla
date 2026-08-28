class Value:
    def __init__(self,data):
        self.data = data


    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self , other):
        return Value(self.data * other.data)

    def __repr__(self):
        return f"Value(data={self.data})"


    
a = Value(2.0)
b = Value(-3.0)
c = a * b

print(c)