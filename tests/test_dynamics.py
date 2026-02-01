import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from dynamics.fhn_integrator import FHNIntegrator
from dynamics.phi_mapping import PhiMapping

class TestDynamics(unittest.TestCase):
    def test_fhn_integrator(self):
        integrator = FHNIntegrator()
        t = np.linspace(0, 10, 100)
        initial_state = np.array([0.0, 0.0])
        I = np.array([0.5])
        sol = integrator.integrate(t, initial_state, I)
        self.assertEqual(sol.shape, (100, 2))
        # Voltage should change from initial 0
        self.assertNotEqual(sol[-1, 0], 0.0)

    def test_coupled_fhn(self):
        integrator = FHNIntegrator()
        t = np.linspace(0, 1, 10)
        # 2 nodes: [v0, v1, w0, w1]
        initial_state = np.array([1.0, 0.0, 0.0, 0.0])
        I = np.array([0.0, 0.0])
        K = np.array([[0.0, 0.0],
                      [0.1, 0.0]]) # Node 1 coupled to Node 0

        # Test vectorized equations directly
        state = initial_state
        res = integrator.fhn_equations(state, 0, I, K)

        # v0 = 1.0, v1 = 0.0
        # dv0 = v0 - v0^3/3 - w0 + I0 + coupling0
        # coupling0 = K[0,0]*(v0-v0) + K[0,1]*(v1-v0) = 0 + 0 = 0
        # dv0 = 1.0 - 1/3 - 0 + 0 + 0 = 0.66666667

        # coupling1 = K[1,0]*(v0-v1) + K[1,1]*(v1-v1) = 0.1*(1.0-0.0) + 0 = 0.1
        # dv1 = v1 - v1^3/3 - w1 + I1 + coupling1 = 0 - 0 - 0 + 0 + 0.1 = 0.1

        self.assertAlmostEqual(res[0], 2/3)
        self.assertAlmostEqual(res[1], 0.1)

    def test_phi_mapping(self):
        phi = PhiMapping(threshold=0.5)
        v_trace = np.array([0.0, 0.6, 0.4, 0.8])
        t = np.array([0.0, 1.0, 2.0, 3.0])

        # Test basic mapping
        bool_trace = phi.map_to_boolean(v_trace)
        np.testing.assert_array_equal(bool_trace, [0, 1, 0, 1])

        # Test apply_phi with windows
        t_long = np.linspace(0, 10, 101)
        v_long = np.zeros(101)
        v_long[50] = 1.0 # spike at t=5.0

        # delta_t = 2.0
        # Windows: [0,2), [2,4), [4,6), [6,8), [8,10)
        # t=5.0 is in [4,6) which is window index 2
        discrete_long = phi.apply_phi(v_long, t_long, delta_t=2.0)

        self.assertEqual(len(discrete_long), 5)
        self.assertEqual(discrete_long[2], 1)
        self.assertEqual(np.sum(discrete_long), 1)

if __name__ == '__main__':
    unittest.main()
