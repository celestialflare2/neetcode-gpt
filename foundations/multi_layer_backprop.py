import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Convert inputs to numpy arrays for easier calculations
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward pass
        z1 = np.dot(W1, x) + b1  # Linear transformation for first layer
        a1 = np.maximum(0, z1)   # ReLU activation
        z2 = np.dot(W2, a1) + b2  # Linear transformation
        loss = np.mean((z2 - y_true) ** 2)  # MSE loss

        # Backward pass
        dz2 = (z2 - y_true) * (2 / len(y_true))
        dW2 = np.outer(dz2, a1)  # Gradient w.r.t. W2
        db2 = dz2  # Gradient w.r.t. b2

        # Propagate gradients back to the first layer
        da1 = np.dot(W2.T, dz2)  # Gradient w.r.t. a1
        dz1 = da1 * (z1 > 0)  # Gradient w.r.t. z1 (ReLU derivative)
        dW1 = np.outer(dz1, x)  # Gradient w.r.t. W1
        db1 = dz1  # Gradient w.r.t. b1

        return {
            'loss': round(float(loss), 4),
            'dW1': [[round(float(val), 4) for val in row] for row in dW1],
            'db1': [round(float(val), 4) for val in db1],
            'dW2': [[round(float(val), 4) for val in row] for row in dW2],
            'db2': [round(float(val), 4) for val in db2]
        }