import numpy as np
from .fhn_integrator import FHNIntegrator
from .phi_mapping import PhiMapping
from .cbn_controller import CBNController

class HybridSimulator:
    """
    Framework CBN-FND v2.0 Simulator.
    Connects m FHN continuous systems with a CBN layer using spatial summation.
    """
    def __init__(self, m, n, K_thresholds, adjacency_A, delta_t, theta, refractory_duration=1, fhn_params=None):
        self.m = m  # Number of entities
        self.n = n  # Boolean variables per entity
        self.K = np.array(K_thresholds)  # Activation thresholds Ka
        self.A = np.array(adjacency_A)  # Adjacency matrix
        self.delta_t = delta_t
        self.theta = theta
        self.ref_duration = refractory_duration

        params = fhn_params if fhn_params else {'a': 0.7, 'b': 0.8, 'tau': 0.08}
        self.integrator = FHNIntegrator(**params)
        self.phi = PhiMapping(threshold=theta)
        self.cbn = CBNController(m, n, K_thresholds, adjacency_A)

        # Initial state
        self.v = np.full(m, -1.0)
        self.w = np.full(m, -0.5)
        self.x = np.zeros((m, self.n), dtype=int)
        self.ref_counters = np.zeros(m, dtype=int)

        self.history_v = []
        self.history_x = []
        self.history_t = []
        self.attractor = None

    def run(self, total_time, dt_integration, I_ext):
        """
        Runs the hybrid simulation.
        """
        t = 0
        steps_per_window = int(self.delta_t / dt_integration)
        num_windows = int(total_time / self.delta_t)

        current_fhn_state = np.concatenate([self.v, self.w])

        for k in range(num_windows):
            # 1. Continuous Layer: Integrate FHN for one delta_t window
            window_t = np.linspace(t, t + self.delta_t, steps_per_window + 1)
            sol = self.integrator.integrate_rk4(window_t, current_fhn_state, I_ext, self.A)

            # v_trace for this window
            v_window = sol[:, :self.m]
            current_fhn_state = sol[-1]
            self.v = current_fhn_state[:self.m]
            self.w = current_fhn_state[self.m:]

            # Store history
            self.history_v.append(v_window[:-1])
            self.history_t.append(window_t[:-1])

            # Record state and check for attractors
            if self.attractor is None:
                self.attractor = self.cbn.record_state(self.x)
                if self.attractor:
                    print(f"Attractor detected at window {k}!")

            # Store CURRENT x (x_k) for this window
            window_x = np.tile(self.x, (steps_per_window, 1, 1))
            self.history_x.append(window_x)

            # 2. Projection Operator (Phi_delta_t)
            y = np.array([self.phi.detect_spike(v_window[:, b]) for b in range(self.m)])

            # 3. Discrete Layer (CBN): Spatial Summation (via CBNController)
            new_x = self.cbn.compute_next_state(y, self.ref_counters)

            # Update refractory counters
            for a in range(self.m):
                if new_x[a, 0] == 1:
                    self.ref_counters[a] = self.ref_duration
                elif self.ref_counters[a] > 0:
                    self.ref_counters[a] -= 1

            self.x = new_x
            t += self.delta_t

        return self.get_results()

    def get_results(self):
        full_v = np.concatenate(self.history_v, axis=0)
        full_x = np.concatenate(self.history_x, axis=0)
        full_t = np.concatenate(self.history_t, axis=0)
        return full_t, full_v, full_x
