import numpy as np
from micro_grad.primitives.Tensor import Tensor


class LinearLayer:
    def __init__(self, in_tensors: int, out_tensors: int) -> None:
        self.weights = Tensor(
            np.random.randn(in_tensors, out_tensors), requires_grad=True
        )
        self.biases = Tensor(
            np.random.randn(
                out_tensors,
            ),
            requires_grad=True,
        )
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors

    def __call__(self, X: Tensor) -> Tensor:
        result = X @ self.weights + self.biases
        return result