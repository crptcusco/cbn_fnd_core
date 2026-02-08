import numpy as np

class CBNController:
    """
    Handles the discrete logic for Coupled Boolean Networks (CBN).
    Includes spatial summation (K-threshold) and attractor detection.
    """
    def __init__(self, m, n, K_thresholds, adj_matrix):
        self.m = m
        self.n = n
        self.K = np.array(K_thresholds)
        self.adj = np.array(adj_matrix)
        self.history = []
        self.state_map = {} # Maps state tuple to its first occurrence index

    def compute_next_state(self, current_y, current_ref_counters):
        """
        Calculates the next boolean state based on the spatial summation rule.

        Parameters:
        - current_y: Vector of projected signals y_b(k)
        - current_ref_counters: Current refractory counters for each entity
        """
        new_x = np.zeros((self.m, self.n), dtype=int)
        for a in range(self.m):
            # Sum signals from active neighbors (b such that adj[a, b] != 0)
            neighbors = np.where(self.adj[a] != 0)[0]
            spatial_sum = np.sum(current_y[neighbors])

            if spatial_sum >= self.K[a] and current_ref_counters[a] == 0:
                new_x[a, :] = 1
            else:
                new_x[a, :] = 0
        return new_x

    def record_state(self, state):
        """
        Records a discrete state in the history and checks for attractors.

        Parameters:
        - state: The current discrete state vector X(k)

        Returns:
        - attractor: The cycle of states if an attractor is detected, else None.
        """
        state_tuple = tuple(state.flatten())

        if state_tuple in self.state_map:
            start_index = self.state_map[state_tuple]
            cycle = self.history[start_index:]
            # We don't add the repeated state to history to keep it clean,
            # or we could. Let's add it then return.
            self.history.append(state_tuple)
            return cycle

        self.state_map[state_tuple] = len(self.history)
        self.history.append(state_tuple)
        return None

    def reset_history(self):
        self.history = []
        self.state_map = {}
