import numpy as np
import matplotlib.pyplot as plt
from dynamics.fhn_integrator import FHNIntegrator
from dynamics.phi_mapping import PhiMapping
import os

def run_example(output_path='hybrid_dynamics_example.png'):
    # Simulation parameters
    t = np.linspace(0, 200, 20000)
    delta_t = 10.0

    # Node parameters: 2 nodes
    # Node 0 has an external current that makes it spike
    # Node 1 is coupled to Node 0
    I = np.array([0.5, 0.0])
    K = np.array([[0.0, 0.0],
                  [0.1, 0.0]]) # Node 1 receives input from Node 0

    initial_state = np.array([-1.0, -1.0, -0.5, -0.5]) # [v0, v1, w0, w1]

    # 1. Continuous Dynamics (FHN)
    integrator = FHNIntegrator(epsilon=0.08, a=0.7, b=0.8)
    solution = integrator.integrate(t, initial_state, I, K)

    v0 = solution[:, 0]
    v1 = solution[:, 1]

    # 2. Discrete Mapping (Phi_delta_t)
    phi = PhiMapping(threshold=1.0)
    discrete_v0 = phi.apply_phi(v0, t, delta_t)
    discrete_v1 = phi.apply_phi(v1, t, delta_t)

    # Time for discrete steps
    t_discrete = np.arange(0, len(discrete_v0)) * delta_t + delta_t/2

    # 3. Visualization
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    # Plot Node 0 Continuous
    axes[0].plot(t, v0, color='blue', label='v0 (Continuous)')
    axes[0].axhline(phi.threshold, color='red', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Voltage v0')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Hybrid Dynamics: FHN to CBN Mapping')

    # Plot Node 0 Discrete
    axes[1].step(t_discrete, discrete_v0, where='mid', color='blue', label='x0 (Discrete)')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_ylabel('State x0')
    axes[1].legend(loc='upper right')

    # Plot Node 1 Continuous
    axes[2].plot(t, v1, color='green', label='v1 (Continuous)')
    axes[2].axhline(phi.threshold, color='red', linestyle='--')
    axes[2].set_ylabel('Voltage v1')
    axes[2].legend(loc='upper right')

    # Plot Node 1 Discrete
    axes[3].step(t_discrete, discrete_v1, where='mid', color='green', label='x1 (Discrete)')
    axes[3].set_ylim([-0.1, 1.1])
    axes[3].set_ylabel('State x1')
    axes[3].set_xlabel('Time (t)')
    axes[3].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

if __name__ == "__main__":
    # Ensure we can import from the src directory
    # If running from the root as 'python src/example_hybrid.py'
    # we might need to add 'src' to path if not already there.
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(base_dir, 'src') not in sys.path:
        sys.path.append(os.path.join(base_dir, 'src'))

    run_example()
