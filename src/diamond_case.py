import numpy as np
import matplotlib.pyplot as plt
from dynamics.simulator import HybridSimulator
import os

def run_diamond_case():
    m = 4
    n = 1
    # Ka thresholds: Entity 3 (index 3) is the coincidence detector with K=2
    K = [1, 1, 1, 2]

    # Adjacency Matrix (Diamond shape)
    # 0 -> 1, 0 -> 2
    # 1 -> 3, 2 -> 3
    A = np.zeros((m, m))
    A[1, 0] = 1.0
    A[2, 0] = 1.0
    A[3, 1] = 1.0
    A[3, 2] = 1.0

    delta_t = 10.0
    theta = 1.0

    # FHN parameters
    fhn_params = {'a': 0.7, 'b': 0.8, 'tau': 0.08}

    sim = HybridSimulator(m, n, K, A, delta_t, theta, refractory_duration=2, fhn_params=fhn_params)

    # External current: Stimulate only node 0 to start the cascade
    I_ext = np.array([0.5, 0.0, 0.0, 0.0])

    total_time = 400
    dt_integration = 0.1

    t, v, x = sim.run(total_time, dt_integration, I_ext)

    # Visualization
    fig, axes = plt.subplots(m, 2, figsize=(12, 10), sharex=True)
    fig.suptitle('Diamond Case (m=4): Coincidence Detector (K4=2)', fontsize=16)

    for i in range(m):
        # Continuous FHN signal
        axes[i, 0].plot(t, v[:, i], label=f'v{i} (FND)')
        axes[i, 0].axhline(theta, color='r', linestyle='--', alpha=0.5)
        axes[i, 0].set_ylabel(f'Entity {i}')
        if i == 0:
            axes[i, 0].set_title('Continuous Dynamics (FND)')

        # Boolean pulses
        axes[i, 1].step(t, x[:, i, 0], where='post', color='g', label=f'x{i} (CBN)')
        axes[i, 1].set_ylim([-0.1, 1.1])
        if i == 0:
            axes[i, 1].set_title('Boolean Dynamics (CBN)')

    axes[m-1, 0].set_xlabel('Time (t)')
    axes[m-1, 1].set_xlabel('Time (t)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = 'diamond_case_v2.png'
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    run_diamond_case()
