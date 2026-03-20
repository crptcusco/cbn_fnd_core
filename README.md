# CBN-FND Core: Hybrid Systems Dynamics (v2.0)

This repository is a specialized framework for studying hybrid dynamics, bridging continuous biophysical models (FitzHugh-Nagumo) with discrete Coupled Boolean Networks (CBN).

## v2.0 New Features

- **RK4 Integrator:** High-precision Runge-Kutta 4th order solver for the continuous FND layer.
- **Spatial Summation ($K$-threshold):** Boolean state updates based on the sum of incoming signals from neighbors, compared against a specific threshold $K_a$ for each entity.
- **Refractory Period:** Entities enter a refractory state after firing, preventing immediate re-triggering and ensuring biologically plausible dynamics.
- **M-Entity Architecture:** Support for $m$ coupled entities, each with its own FND unit and set of Boolean variables.

## Project Structure

- `src/dynamics/`: Core implementation.
  - `fhn_integrator.py`: RK4 solver for the FitzHugh-Nagumo equations.
  - `simulator.py`: The `HybridSimulator` class that manages the interaction between layers.
- `src/diamond_case.py`: A demonstration of a 4-entity diamond-shaped network where entity 4 acts as a coincidence detector ($K_4=2$).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Diamond Case simulation:
   ```bash
   PYTHONPATH=core python core/diamond_case.py
   ```
3. Check the results in `results/diamond_case_v2.png`.

## Core Logic

### Continuous Layer (FND)
The system solves the following ODEs for each entity $a$:
$$\frac{dv_a}{dt} = v_a - \frac{v_a^3}{3} - w_a + I_{ext} + \sum_{b \in \mathcal{N}_a} A_{ab} v_b(t)$$
$$\frac{dw_a}{dt} = \tau (v_a + a - b w_a)$$

### Projection Operator ($\Phi_{\Delta t}$)
Transforms continuous signals into discrete events. A signal $y^b_a(k) = 1$ if unit $b$ fires ($v_b \geq \theta$) within the time window $\Delta t$.

### Discrete Layer (CBN)
Updates the Boolean state $x_{a,i}$ using the Spatial Summation Rule:
$$x_{a,i}(k+1) = 1 \iff \left( \sum_{b \in \mathcal{N}_a} y^b_a(k) \geq K_a \right) \text{ AND } \text{Ref}_a(k) = 0$$
