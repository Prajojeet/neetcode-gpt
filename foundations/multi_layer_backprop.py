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
        def relu_der(z):
            return np.where(z > 0, 1, 0)

        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0.0, z1)
        z2 = W2 @ a1 + b2
        a2 = np.maximum(0.0, z2)

        MSE_loss = float(np.mean(a2 - y_true)**2) 
       
        # Gradients
        dz2 = (-2/y_true.size) * (y_true - a2) * relu_der(z2)


        dw2 =  dz2.reshape(-1, 1) @  a1.reshape(1, -1)
        db2 =  dz2

        dz1 = (W2.T @ dz2) * relu_der(z1)
        dw1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = dz1

        return {
        'loss':  np.round(MSE_loss, 4),
        'dW1':   np.round(dw1, 4).tolist(),
        'db1':   np.round(db1, 4).tolist(),
        'dW2':   np.round(dw2, 4).tolist(),
        'db2':   np.round(db2, 4).tolist()
        }
