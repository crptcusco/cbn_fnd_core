import numpy as np
import matplotlib.pyplot as plt
from dynamics.simulator import HybridSimulator
import os

def run_example():
    # Simulation parameters
    m = 2
    n = 1
    K = [1, 1]

    # Adjacency Matrix: 0 -> 1
    A = np.zeros((m, m))
    A[1, 0] = 0.1

    delta_t = 10.0
    theta = 1.0

    # 1. Initialize Hybrid Simulator v2.0
    # Note: v2.0 uses 'tau' instead of 'epsilon' and 'integrate_rk4' internally.
    sim = HybridSimulator(m, n, K, A, delta_t, theta, refractory_duration=2)

    # 2. Run Simulation
    I_ext = np.array([0.5, 0.0])
    total_time = 200
    dt_integration = 0.1

    t, v, x = sim.run(total_time, dt_integration, I_ext)

    # 3. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, v[:, 0], color='blue', label='v0 (FND)')
    axes[0].plot(t, v[:, 1], color='green', label='v1 (FND)')
    axes[0].axhline(theta, color='red', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Voltage')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Hybrid Dynamics v2.0: FND and CBN coupling')

    axes[1].step(t, x[:, 0, 0], color='blue', label='x0 (CBN)', alpha=0.5)
    axes[1].step(t, x[:, 1, 0], color='green', label='x1 (CBN)')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_ylabel('Boolean State')
    axes[1].set_xlabel('Time (t)')
    axes[1].legend(loc='upper right')

    plt.tight_layout()

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    output_filename = 'hybrid_dynamics_example_v2.png'
    output_path = os.path.join(results_dir, output_filename)
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(base_dir, 'core') not in sys.path:
        sys.path.append(os.path.join(base_dir, 'core'))
    run_example()
