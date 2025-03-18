"""
For additional transformer related

Sequential
Embedding

"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .module import Module, Parameter
from .nn import one_hot
from .tensor import Tensor
from .tensor_functions import (ones, ones_tensor_from_numpy, rand, tensor,
                               tensor_from_numpy, zeros,
                               zeros_tensor_from_numpy)
from .tensor_ops import TensorBackend


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings  # Vocab size
        self.embedding_dim = embedding_dim  # Embedding Dimension
        ### BEGIN YOUR SOLUTION
        rand_data = np.random.random((num_embeddings, embedding_dim)).astype(np.float32)
        self.weights = Parameter(tensor_from_numpy(rand_data, backend=backend))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN YOUR SOLUTION
        one_hot_embedding = one_hot(x, self.num_embeddings).view(
            1, bs * seq_len, self.num_embeddings
        )
        return (one_hot_embedding @ self.weights.value).view(
            bs, seq_len, self.embedding_dim
        )
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p_dropout: float = 0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor:
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)

        Args:
            x : Tensor of shape (*)

        Returns:
            output : Tensor of shape (*)
        """
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p_dropout == 0:
            return x

        mask = tensor_from_numpy(np.random.binomial(1, 1 - self.p_dropout, x.shape))
        return (x * mask) / (1 - self.p_dropout)
        ### END YOUR SOLUTION


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN YOUR SOLUTION
        self.in_size = in_size
        self.backend = backend
        scale = 1.0 / np.sqrt(in_size)
        self.weights = Parameter(
            (
                rand(
                    (
                        in_size,
                        out_size,
                    ),
                    backend=backend,
                )
                * 2
                * scale
            )
            - scale
        )
        self.bias = Parameter(tensor((out_size,), backend=backend)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.

        Args:
            x : Tensor of shape (n, in_size)

        Returns:
            output : Tensor of shape (n, out_size)
        """
        ### BEGIN YOUR SOLUTION
        if self.bias is not None:
            return x @ self.weights.value + self.bias.value
        else:
            return x @ self.weights.value
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(
        self,
        dim: int,
        eps: float,
        backend: TensorBackend,
        use_fused_kernel: bool = False,
    ):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim              : Expected size of the last dimension to apply layer normalization.
            eps              : A value added for numerical stability.
            backend          : Tensor backend, can use CUDA or simple backend
            use_fused_kernel : If True, use fused attention-softmax and layernorm kernel for speedup
        
        Attributes: 
            weights          : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias             : the learnable bias of the module of shape (self.dim, ) initialized to 0.
            use_fused_kernel : If True, use fused attention-softmax and layernorm kernel for speedup
        """
        self.use_fused_kernel = use_fused_kernel
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))
        ### END YOUR SOLUTION

        if use_fused_kernel:
            self.gamma = Parameter(
                tensor_from_numpy(np.ones((dim,)), backend=backend, requires_grad=True)
            )
            self.beta = Parameter(
                tensor_from_numpy(np.zeros((dim,)), backend=backend, requires_grad=True)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs.
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.

        Input:
            x - Tensor of shape (bs, dim)

        Output:
            output - Tensor of shape (bs, dim)
        """
        ### BEGIN YOUR SOLUTION
        if self.use_fused_kernel:
            return x.layernorm(self.gamma.value, self.beta.value)

        mean = x.mean(1)
        var = x.var(1)
        x_normalized = (x - mean) / ((var + self.eps) ** 0.5)
        return x_normalized * self.weights.value + self.bias.value
        ### END YOUR SOLUTION
