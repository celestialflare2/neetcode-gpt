import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        n = len(y_true)
        loss = 0
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        for i in range(n):
            loss += (y_true[i] * np.log(y_pred[i])) + ((1 - y_true[i]) * np.log(1 - y_pred[i]))
        return round(-loss/n, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        n = len(y_true)
        C = len(y_true[0])
        loss = 0
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        for i in range(n):
            for c in range(C):
                loss += (y_true[i][c] * np.log(y_pred[i][c]))
        return round(-loss/n, 4)