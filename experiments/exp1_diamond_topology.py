import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. ROBUST PATH CONFIGURATION
# Identify the project root by going up one level from 'experiments/'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
core_path = os.path.join(project_root, 'core')
results_dir = os.path.join(project_root, 'results')

# Add 'core' to the system path to allow importing the 'dynamics' package
if core_path not in sys.path:
    sys.path.append(core_path)

from dynamics.simulator import HybridSimulator

def run_diamond_case():
    """
    Simulates a 4-node diamond topology to test the Coincidence Detector logic (K=2).
    Nodes 1 and 2 receive input from Node 0 and converge onto Node 3.
    """
    # --- Experiment Configuration ---
    m = 4  # Total number of nodes
    n = 1  # State dimensions for each FHN unit
    
    # Thresholds: Node 3 (index 3) is the coincidence detector (K=2)
    K = [1, 1, 1, 2]

    # Adjacency Matrix (Diamond shape)
    # Node 0 triggers 1 and 2; Nodes 1 and 2 both feed into Node 3
    A = np.zeros((m, m))
    A[1, 0] = 1.0 
    A[2, 0] = 1.0 
    A[3, 1] = 1.0 
    A[3, 2] = 1.0 

    delta_t = 10.0  # CBN time window
    theta = 1.0    # FND firing threshold
    fhn_params = {'a': 0.7, 'b': 0.8, 'tau': 0.08}

    # Initialize Simulator with a refractory period of 2 windows
    sim = HybridSimulator(m, n, K, A, delta_t, theta, 
                          refractory_duration=2, 
                          fhn_params=fhn_params)

    # Input: Stimulate only the root node (Node 0) to trigger the cascade
    I_ext = np.array([0.5, 0.0, 0.0, 0.0])
    total_time = 400
    dt_integration = 0.1

    # Run Simulation
    t, v, x = sim.run(total_time, dt_integration, I_ext)

    # --- Professional Visualization ---
    fig, axes = plt.subplots(m, 2, figsize=(12, 10), sharex=True)
    fig.suptitle('Diamond Case (m=4): Coincidence Detector (K4=2)', fontsize=16)

    for i in range(m):
        # Column 0: Continuous Dynamics (FitzHugh-Nagumo)
        axes[i, 0].plot(t, v[:, i], label=f'v{i} (FND)')
        axes[i, 0].axhline(theta, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].set_ylabel(f'Entity {i}')
        if i == 0: 
            axes[i, 0].set_title('Continuous Dynamics (FND)')

        # Column 1: Boolean Dynamics (CBN pulses)
        color = 'g' if i != 3 else 'red' # Highlight the coincidence detector node
        axes[i, 1].step(t, x[:, i, 0], where='post', color=color, linewidth=2)
        axes[i, 1].set_ylim([-0.1, 1.1])
        if i == 0: 
            axes[i, 1].set_title('Boolean Dynamics (CBN)')
        if i == 3: 
            axes[i, 1].set_ylabel('INTEGRATOR', color='red', fontweight='bold')

    axes[m-1, 0].set_xlabel('Time (t)')
    axes[m-1, 1].set_xlabel('Time (t)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- EXPORT TO RESULTS FOLDER ---
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    output_filename = 'exp1_diamond_case.png'
    output_path = os.path.join(results_dir, output_filename)
    
    # Save with high resolution (300 DPI) for the Paper
    plt.savefig(output_path, dpi=300) 
    print(f"Simulation complete. Plot saved to: {output_path}")
    
    # Optional: Display the plot if running interactively
    # plt.show()

if __name__ == "__main__":
    run_diamond_case()
