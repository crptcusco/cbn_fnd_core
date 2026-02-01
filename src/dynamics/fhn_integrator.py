import numpy as np
from scipy.integrate import odeint

class FHNIntegrator:
    """
    FitzHugh-Nagumo integrator for a network of coupled neurons.
    """
    def __init__(self, a=0.7, b=0.8, epsilon=0.08):
        self.a = a
        self.b = b
        self.epsilon = epsilon

    def fhn_equations(self, state, t, I, K):
        """
        ODE system for the FitzHugh-Nagumo model.

        Parameters:
        - state: [v1, v2, ..., vn, w1, w2, ..., wn]
        - t: time
        - I: External current for each node (array of length n)
        - K: Coupling matrix (n x n)
        """
        num_nodes = len(I)
        v = state[:num_nodes]
        w = state[num_nodes:]

        # dv/dt = v - v^3/3 - w + I + coupling
        dv = v - (v**3) / 3 - w + I

        if K is not None:
            # Vectorized coupling term: sum_j K_ij * (v_j - v_i)
            # This is equivalent to: (K @ v) - v * sum(K, axis=1)
            dv += K @ v - v * np.sum(K, axis=1)

        # dw/dt = epsilon * (v + a - b * w)
        dw = self.epsilon * (v + self.a - self.b * w)

        return np.concatenate([dv, dw])

    def integrate(self, t, initial_state, I, K=None):
        """
        Integrates the FHN system over the given time points.
        """
        sol = odeint(self.fhn_equations, initial_state, t, args=(I, K))
        return sol
