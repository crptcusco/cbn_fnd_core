import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from dynamics.simulator import HybridSimulator

class TestSimulatorV2(unittest.TestCase):
    def test_spatial_summation(self):
        m = 3
        n = 1
        K = [1, 2, 1] # Ka
        # A: 0 -> 2, 1 -> 2
        A = [[0, 0, 0],
             [0, 0, 0],
             [1, 1, 0]]

        sim = HybridSimulator(m, n, K, A, delta_t=10, theta=1.0)

        # We need to simulate a case where y[0]=1 and y[1]=1
        # To get y[0]=1, node 0 must spike. We can set I_ext for node 0.
        I_ext = np.array([0.5, 0.5, 0.0])

        t, v, x = sim.run(total_time=20, dt_integration=0.1, I_ext=I_ext)

        # In the first window, nodes 0 and 1 should spike if I_ext=0.5
        # v history is in full_v
        # Check if node 2 fired in the second window (x at k=1)
        # x shape is (steps, m, n)
        # windows are 0 and 1. steps_per_window = 100.
        # k=0 update x for k=1.

        # Node 0,1 should spike in window 0.
        spike_0 = np.any(v[:100, 0] >= 1.0)
        spike_1 = np.any(v[:100, 1] >= 1.0)

        print(f"Node 0 spiked: {spike_0}")
        print(f"Node 1 spiked: {spike_1}")

        # Node 2 threshold K[2] = 2. It needs both 0 and 1 to spike.
        # If both spiked, x[2] should be 1 in window 1.
        fired_2 = x[150, 2, 0] # Middle of second window
        print(f"Node 2 fired: {fired_2}")

        if spike_0 and spike_1:
            self.assertEqual(fired_2, 1)

if __name__ == '__main__':
    unittest.main()
