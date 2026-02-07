import numpy as np

class FHNIntegrator:
    """
    FitzHugh-Nagumo integrator for a network of coupled neurons using RK4.
    """
    def __init__(self, a=0.7, b=0.8, tau=0.08):
        self.a = a
        self.b = b
        self.tau = tau

    def fhn_equations(self, state, I_ext, A):
        """
        ODE system for the FitzHugh-Nagumo model.

        Parameters:
        - state: [v1, v2, ..., vn, w1, w2, ..., wn]
        - I_ext: External current for each node (array of length n)
        - A: Adjacency/Coupling matrix (n x n)
        """
        num_nodes = len(I_ext)
        v = state[:num_nodes]
        w = state[num_nodes:]

        # dv/dt = v - v^3/3 - w + I_ext + sum(A_ab * v_b)
        # Coupling: A @ v
        dv = v - (v**3) / 3 - w + I_ext
        if A is not None:
            dv += A @ v

        # dw/dt = tau * (v + a - b * w)
        dw = self.tau * (v + self.a - self.b * w)

        return np.concatenate([dv, dw])

    def rk4_step(self, state, dt, I_ext, A):
        """
        Performs a single RK4 integration step.
        """
        k1 = self.fhn_equations(state, I_ext, A)
        k2 = self.fhn_equations(state + 0.5 * dt * k1, I_ext, A)
        k3 = self.fhn_equations(state + 0.5 * dt * k2, I_ext, A)
        k4 = self.fhn_equations(state + dt * k3, I_ext, A)

        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate_rk4(self, t, initial_state, I_ext, A=None):
        """
        Integrates the FHN system over time using RK4.
        """
        num_steps = len(t)
        dt = t[1] - t[0]
        state_dim = len(initial_state)
        sol = np.zeros((num_steps, state_dim))
        sol[0] = initial_state

        current_state = initial_state
        for i in range(1, num_steps):
            current_state = self.rk4_step(current_state, dt, I_ext, A)
            sol[i] = current_state

        return sol
