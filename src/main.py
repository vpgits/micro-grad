from micro_grad.layers.linear import LinearLayer
from micro_grad.losses.MSELoss import MSELoss
from micro_grad.optimizers.SGD import SGD
from micro_grad.primitives.Tensor import Tensor
import numpy as np


class SimpleNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.fc1 = LinearLayer(input_size, hidden_size)
        self.fc2 = LinearLayer(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = Tensor(np.maximum(0, x.data), None, x._prev, x.requires_grad)
        x = self.fc2(x)
        return x


def main():

    np.random.seed(0)
    X = Tensor(np.random.randn(100, 3), requires_grad=True)
    Y = Tensor(np.random.randn(100, 1), requires_grad=False)

    model = SimpleNN(input_size=3, hidden_size=2, output_size=1)
    optimizer = SGD(
        tensors=[
            model.fc1.weights,
            model.fc1.biases,
            model.fc2.weights,
            model.fc2.biases,
        ],
        lr=1e-6,
    )
    loss_fn = MSELoss()

    epochs = 100
    for epoch in range(epochs):

        predictions = model.forward(X)
        loss = loss_fn.forward(predictions, Y)

        loss.backward()

        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data.mean()}")


if __name__ == "__main__":
    main()
