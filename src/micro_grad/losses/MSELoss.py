from micro_grad.losses.loss import Loss
from micro_grad.primitives.Tensor import Tensor
import numpy as np


class MSELoss(Loss):
    def __init__(self):
        self.loss_tensor = None
        super().__init__()

    def forward(self, prediction_targets, absolute_targets) -> Tensor:
        loss = ((prediction_targets - absolute_targets) ** 2).mean()
        self.loss_tensor = loss

        def _backward() -> None:
            current_tensor = self.loss_tensor
            while len(current_tensor._prev) == 1:
                current_tensor = current_tensor._prev[0]
            prediction_targets: Tensor = current_tensor._prev[0]
            absolute_targets: Tensor = current_tensor._prev[1]
            gradient = (
                2
                * (prediction_targets.data - absolute_targets.data)
                / np.prod(absolute_targets.data.shape)
            )
            if prediction_targets.requires_grad:
                if prediction_targets.grad is None:
                    prediction_targets.grad = np.zeros_like(prediction_targets.data)
                prediction_targets.grad += np.broadcast_to(
                    gradient, prediction_targets.data.shape
                )
            if absolute_targets.requires_grad:
                if absolute_targets.grad is None:
                    absolute_targets.grad = np.zeros_like(absolute_targets.data)
                absolute_targets.grad -= np.broadcast_to(
                    gradient, absolute_targets.data.shape
                )

        self.loss_tensor._backward = _backward

        return self.loss_tensor
