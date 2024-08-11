from micro_grad.primitives.Tensor import Tensor


class Loss:
    def __init__(self):
        pass

    def forward(self, prediction_targets: Tensor, absolute_targets: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, prediction_targets: Tensor, absolute_targets: Tensor):
        return self.forward
