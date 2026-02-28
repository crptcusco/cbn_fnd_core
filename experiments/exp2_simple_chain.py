import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. ROBUST PATH CONFIGURATION
# Identify the project root by going up one level from 'experiments/'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
results_dir = os.path.join(project_root, 'results')

# Add 'src' to the system path to allow importing the 'dynamics' package
if src_path not in sys.path:
    sys.path.append(src_path)

from dynamics.simulator import HybridSimulator

def run_example():
    """
    Basic 2-node simulation to verify FND and CBN coupling.
    Topology: Node 0 -> Node 1.
    """
    # --- Simulation Parameters ---
    m = 2  # Number of nodes
    n = 1  # Dimension per node
    K = [1, 1]  # Thresholds

    # Adjacency Matrix: Node 0 influences Node 1
    A = np.zeros((m, m))
    A[1, 0] = 0.1

    delta_t = 10.0  # CBN window
    theta = 1.0     # FND threshold

    # 1. Initialize Hybrid Simulator v2.0
    # Note: v2.0 uses 'tau' internally and integrate_rk4 for stability.
    sim = HybridSimulator(m, n, K, A, delta_t, theta, refractory_duration=2)

    # 2. Run Simulation
    # External current applied only to the first node
    I_ext = np.array([0.5, 0.0])
    total_time = 200
    dt_integration = 0.1

    t, v, x = sim.run(total_time, dt_integration, I_ext)

    # --- 3. Professional Visualization ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Hybrid Dynamics v2.0: FND and CBN coupling', fontsize=14)

    # Top Plot: Continuous FND Signals
    axes[0].plot(t, v[:, 0], color='blue', label='v0 (FND)')
    axes[0].plot(t, v[:, 1], color='green', label='v1 (FND)')
    axes[0].axhline(theta, color='red', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Voltage (v)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom Plot: Discrete CBN Pulses
    axes[1].step(t, x[:, 0, 0], color='blue', label='x0 (CBN)', alpha=0.5, where='post')
    axes[1].step(t, x[:, 1, 0], color='green', label='x1 (CBN)', where='post')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_ylabel('Boolean State (x)')
    axes[1].set_xlabel('Time (t)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 4. EXPORT TO RESULTS FOLDER ---
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # Using a clean name without versioning for the final output
    output_filename = 'exp2_hybrid_dynamics_example.png'
    output_path = os.path.join(results_dir, output_filename)
    
    # Save with high resolution (300 DPI) and tight margins
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Simulation complete. Plot saved to: {output_path}")

if __name__ == "__main__":
    # The path logic is now handled globally at the top of the script
    run_example()