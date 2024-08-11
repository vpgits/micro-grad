from typing import List

from micro_grad.primitives.Tensor import Tensor


class Optimizer:
    def __init__(self, lr: float, tensors: List[Tensor]) -> None:
        self.lr = lr
        self.tensors = tensors

    def step(self) -> None:
        raise NotImplementedError
