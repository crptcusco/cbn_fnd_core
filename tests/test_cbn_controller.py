import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'core'))
from dynamics.cbn_controller import CBNController

class TestCBNController(unittest.TestCase):
    def test_spatial_summation_logic(self):
        m = 3
        n = 1
        K = [1, 1, 2] # Ka=2 for index 2
        adj = [[0, 0, 0],
               [0, 0, 0],
               [1, 1, 0]] # 0->2, 1->2

        controller = CBNController(m, n, K, adj)

        # Scenario 1: Only node 0 active
        y = np.array([1, 0, 0])
        ref = np.array([0, 0, 0])
        next_x = controller.compute_next_state(y, ref)
        self.assertEqual(next_x[2, 0], 0)

        # Scenario 2: Node 0 and 1 active
        y = np.array([1, 1, 0])
        next_x = controller.compute_next_state(y, ref)
        self.assertEqual(next_x[2, 0], 1)

        # Scenario 3: Node 0 and 1 active but Node 2 in refractory
        ref = np.array([0, 0, 1])
        next_x = controller.compute_next_state(y, ref)
        self.assertEqual(next_x[2, 0], 0)

    def test_attractor_detection(self):
        m = 2
        n = 1
        K = [1, 1]
        adj = [[0, 1], [1, 0]]
        controller = CBNController(m, n, K, adj)

        # Sequence of states: S1, S2, S3, S2
        s1 = np.array([[0], [0]])
        s2 = np.array([[1], [0]])
        s3 = np.array([[0], [1]])

        self.assertIsNone(controller.record_state(s1))
        self.assertIsNone(controller.record_state(s2))
        self.assertIsNone(controller.record_state(s3))

        # Repeating s2
        cycle = controller.record_state(s2)
        self.assertIsNotNone(cycle)
        # Cycle should be (S2, S3)
        self.assertEqual(len(cycle), 2)
        self.assertEqual(cycle[0], tuple(s2.flatten()))
        self.assertEqual(cycle[1], tuple(s3.flatten()))

if __name__ == '__main__':
    unittest.main()
