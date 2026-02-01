# CBN-FND Core: Hybrid Systems Dynamics

This repository is a specialized fork of `cbnetwork`, evolving from a general Boolean Network framework into a biophysical hybrid system core.

## Overview

The `cbn_fnd_core` project bridges the gap between continuous biophysical models and discrete logic representations. It integrates the **FitzHugh-Nagumo (FHN)** equations with a **Boolean mapping operator ($\Phi_{\Delta t}$)** to create a robust framework for studying hybrid dynamics.

### Key Components

- **Continuous State (FHN):** Uses the FitzHugh-Nagumo model to simulate neuronal or excitable media dynamics.
- **Discrete State (CBN):** Represents the system as Coupled Boolean Networks.
- **The Link ($\Phi_{\Delta t}$):** A mathematical operator that maps continuous signals into discrete pulses, giving biological meaning to boolean transitions.

## Project Structure

- `src/dynamics/`: Core implementation of the hybrid bridge.
  - `fhn_integrator.py`: Solves the continuous ODEs.
  - `phi_mapping.py`: Implements the $\Phi_{\Delta t}$ operator.
- `src/example_hybrid.py`: A minimalist example with 2-3 coupled nodes showing the mapping from continuous to discrete states.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the hybrid dynamics example:
   ```bash
   PYTHONPATH=src python src/example_hybrid.py
   ```
3. View the results in `hybrid_dynamics_example.png`.

## Why this approach?

By using the $\Phi_{\Delta t}$ operator, we provide a "biological engine" to Boolean Networks, allowing for more realistic simulations of complex systems while maintaining the computational efficiency of discrete models.
