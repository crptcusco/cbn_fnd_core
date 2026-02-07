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

    def detect_spike(self, v_window):
        """
        Returns 1 if any value in v_window exceeds the threshold, else 0.
        """
        if len(v_window) == 0:
            return 0
        return 1 if np.max(v_window) >= self.threshold else 0

    def apply_phi(self, v_trace, t, delta_t):
        """
        Applies the Phi_delta_t operator over time windows.
        """
        t_max = np.max(t)
        num_windows = int(t_max / delta_t)
        discrete_signal = []

        for k in range(num_windows):
            start_t = k * delta_t
            end_t = (k + 1) * delta_t
            mask = (t >= start_t) & (t < end_t)
            window_v = v_trace[mask]
            discrete_signal.append(self.detect_spike(window_v))

        return np.array(discrete_signal)
