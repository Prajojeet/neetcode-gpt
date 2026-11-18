import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # For a single neuron
        # Pre-activation and output
        z = x.T @ w + b
        y_pred = 1/ (1 + np.exp(-z))

        # Loss
        dL_db = np.round((y_pred - y_true) * y_pred * (1 - y_pred), 5)
        dL_dw = np.round(dL_db * x, 5)
        return (dL_dw, dL_db)
        
