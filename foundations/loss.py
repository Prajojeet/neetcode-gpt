import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return round(-np.mean(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clipping for log stability
        return round((np.sum(-y_true*np.log(y_pred))/len(y_true)), 4)
