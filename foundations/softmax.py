import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        exponents = z - max(z)
        exp_values = np.exp(exponents)
        sum_of_elements = 0
        for i in exponents:
            sum_of_elements += np.exp(i)
        return np.round(exp_values/sum_of_elements, 4)