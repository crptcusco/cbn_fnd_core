import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'core'))
from dynamics.fhn_integrator import FHNIntegrator
from dynamics.simulator import HybridSimulator

class TestV2(unittest.TestCase):
    def test_rk4_solver(self):
        integrator = FHNIntegrator()
        t = np.linspace(0, 1, 101)
        initial_state = np.array([0.0, 0.0])
        I = np.array([0.5])
        sol = integrator.integrate_rk4(t, initial_state, I)
        self.assertEqual(sol.shape, (101, 2))
        # Simple check for evolution
        self.assertNotEqual(sol[-1, 0], 0.0)

    def test_spatial_summation_coincidence(self):
        # 3 inputs to 1 detector
        m = 4
        n = 1
        K = [1, 1, 1, 3] # Ka=3 for index 3
        A = np.zeros((m, m))
        A[3, 0] = 1.0
        A[3, 1] = 1.0
        A[3, 2] = 1.0

        sim = HybridSimulator(m, n, K, A, delta_t=10, theta=1.0)

        # Scenario A: Only 2 inputs spike
        I_ext = np.array([0.5, 0.5, 0.0, 0.0])
        t, v, x = sim.run(total_time=20, dt_integration=0.1, I_ext=I_ext)
        # In window 1, node 3 should NOT fire because spatial_sum=2 < K=3
        self.assertEqual(x[150, 3, 0], 0)

        # Scenario B: 3 inputs spike
        sim = HybridSimulator(m, n, K, A, delta_t=10, theta=1.0) # Reset
        I_ext = np.array([0.5, 0.5, 0.5, 0.0])
        t, v, x = sim.run(total_time=20, dt_integration=0.1, I_ext=I_ext)
        # In window 1, node 3 SHOULD fire because spatial_sum=3 >= K=3
        self.assertEqual(x[150, 3, 0], 1)

    def test_refractory_period(self):
        m = 2
        n = 1
        K = [1, 1]
        A = [[0, 0], [1, 0]] # 0 -> 1

        # Large refractory duration
        sim = HybridSimulator(m, n, K, A, delta_t=10, theta=1.0, refractory_duration=5)

        # Continuous input to node 0
        I_ext = np.array([0.5, 0.0])
        t, v, x = sim.run(total_time=100, dt_integration=0.1, I_ext=I_ext)

        fired_indices = []
        for k in range(10):
            if np.any(x[k*100:(k+1)*100, 1, 0] == 1):
                fired_indices.append(k)

        # Based on FHN dynamics at I=0.5, spikes occur at k=0, 4, 8...
        # k=0 spike -> fires x(1)=1. ref counter set to 5.
        # k=4 spike -> ref counter is 1. No fire. ref counter becomes 0.
        # k=8 spike -> ref counter is 0. Fires x(9)=1.
        self.assertEqual(fired_indices, [1, 9])

if __name__ == '__main__':
    unittest.main()
