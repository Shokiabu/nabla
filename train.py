from engine import Value
from nn import MLP

xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [-1.0, 1.0, 1.0, -1.0]

model = MLP(2, [4, 4, 1])

if __name__ == '__main__':
  for step in range(500):
    # 1. guess
    preds = [model(x) for x in xs]

    # 2. how wrong
    loss = sum(((p - y)**2 for p, y in zip(preds, ys)), Value(0.0))

    # 3. clear old gradients
    for p in model.parameters():
        p.grad = 0.0

    # 4. find which way each weight should move
    loss.backward()

    # 5. move each weight a little in that direction
    for p in model.parameters():
        p.data -= 0.05 * p.grad

    if step % 50 == 0:
        print(step, loss.data)

print([round(p.data, 3) for p in preds])