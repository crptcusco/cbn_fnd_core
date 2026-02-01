import numpy as np

class PhiMapping:
    """
    Operator Phi_delta_t that maps continuous signals to discrete boolean states.
    """
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def map_to_boolean(self, v_trace):
        """
        Maps a continuous voltage trace to a boolean sequence.

        Returns 1 if v > threshold, else 0.
        """
        return (v_trace > self.threshold).astype(int)

    def apply_phi(self, v_trace, t, delta_t):
        """
        Applies the Phi_delta_t operator over time windows.

        Parameters:
        - v_trace: continuous signal
        - t: time points corresponding to v_trace
        - delta_t: time window for discretization

        Returns:
        - discrete_signal: sequence of 0s and 1s representing states in each delta_t window.
        """
        t_max = np.max(t)
        num_windows = int(t_max / delta_t)
        discrete_signal = []

        for k in range(num_windows):
            start_t = k * delta_t
            end_t = (k + 1) * delta_t
            mask = (t >= start_t) & (t < end_t)
            window_v = v_trace[mask]

            if len(window_v) > 0:
                # If the signal exceeds threshold at any point in the window, state is 1
                state = 1 if np.max(window_v) > self.threshold else 0
            else:
                state = 0
            discrete_signal.append(state)

        return np.array(discrete_signal)
