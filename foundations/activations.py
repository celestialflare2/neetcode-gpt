import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        output = []
        for i in z:
            sig = 1 / (1 + np.exp(-i))
            output.append(sig)
        return np.round(output, 5)
        

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        output = []
        for i in z:
            rel = max(0.0, i)
            output.append(rel)
        return np.round(output, 5)