import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        z = np.dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        if activation == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-z))
            return np.round(sig, 5)
        # ReLU: max(0, z)
        elif activation == "relu":
            rel = max(0.0, z)
            return round(rel, 5)