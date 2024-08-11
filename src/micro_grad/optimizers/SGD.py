from typing import List
from micro_grad.optimizers.optim import Optimizer
from micro_grad.primitives.Tensor import Tensor


class SGD(Optimizer):
    def __init__(self, lr: float, tensors: List[Tensor]) -> None:
        super().__init__(lr, tensors)

    def step(self) -> None:
        for tensor in self.tensors:
            if isinstance(tensor, Tensor) and tensor.requires_grad:
                if tensor.grad is not None:
                    tensor.data -= self.lr * tensor.grad
                    tensor.grad = None
