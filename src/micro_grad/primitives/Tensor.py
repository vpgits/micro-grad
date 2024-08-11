from typing import Dict, List, Optional, Union, Iterable
import numpy as np
from micro_grad.primitives.ML_Ops import ML_OPS


class Tensor:

    def __init__(
        self,
        data: np.ndarray,
        op: Optional[ML_OPS] = None,
        children: Optional[Union["Tensor", Iterable["Tensor"]]] = None,
        requires_grad: bool = False,
    ) -> None:
        """
        Initialize a Tensor.

        :param data: The data contained in the tensor.
        :param op: The operation that produced this tensor, if any.
        :param children: Previous tensor(s) in the computation graph.
                         Can be a single Tensor, an iterable of Tensors, or None.
        :param requires_grad: Whether this tensor requires gradients.
        """
        self.data = data
        self.op = op
        self._prev = children if children is not None else []
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None

    @staticmethod
    def build_topological_order(tensor: "Tensor") -> List["Tensor"]:
        """
        Build a topological order of the computation graph.
        """
        topological_order: List[Tensor] = []
        visited: set["Tensor"] = set()

        def _build_topological_order(tensor: "Tensor") -> None:
            if tensor in visited:
                return
            visited.add(tensor)
            for child in tensor._prev:
                _build_topological_order(child)
            topological_order.append(tensor)

        _build_topological_order(tensor)
        return topological_order

    def backward(self) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.ones_like(self.data)
        topo_list = Tensor.build_topological_order(self)
        for tensor in reversed(topo_list):
            if tensor.requires_grad:
                tensor._backward()

    def print_graph(self) -> None:
        """
        Debug method for printing the computation graph.
        """

        def print_tensor(tensor: "Tensor", level: int = 0) -> None:
            indent = "  " * level
            print(f"{indent}Tensor with data shape: {tensor.data.shape}")
            if tensor._prev:
                print(f"{indent}  Operations:")
                for parent in tensor._prev:
                    print_tensor(parent, level + 1)
            else:
                print(f"{indent}  No operations")

        print_tensor(self)

    def __add__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data + other.data,
            ML_OPS.ADD,
            [self, other],
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += np.sum(out.grad, axis=0, keepdims=True).reshape(
                    other.data.shape
                )

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data * other.data,
            ML_OPS.MUL,
            [self, other],
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.broadcast_to(other.data, self.grad.shape) * out.grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += np.broadcast_to(self.data, other.grad.shape) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: int) -> "Tensor":
        out = Tensor(
            self.data**power,
            ML_OPS.POW,
            [self],
            self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if out.grad is None:
                    out.grad = np.ones_like(out.data)
                self.grad += (
                    (self.data ** (power - 1))
                    * power
                    * np.broadcast_to(out.grad, self.grad.shape)
                )

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            self.data - other.data,
            ML_OPS.SUB,
            [self, other],
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.broadcast_to(out.grad, self.grad.shape)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad -= np.broadcast_to(out.grad, other.grad.shape)

        out._backward = _backward
        return out

    def mean(self) -> "Tensor":
        out = Tensor(
            np.mean(self.data),
            ML_OPS.MEAN,
            [self],
            self.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.broadcast_to(out.grad / self.data.size, self.grad.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(
            np.matmul(self.data, other.data),
            ML_OPS.MATMUL,
            [self, other],
            self.requires_grad or other.requires_grad,
        )

        def _backward() -> None:
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if out.grad is None:
                    out.grad = np.ones_like(out.data)
                self.grad += np.matmul(out.grad, other.data.T)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                if out.grad is None:
                    out.grad = np.ones_like(out.data)
                other.grad += np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out
