Physics-Informed Neural Networks (PINN) - Phase 1Solving Laplace and Poisson Equations📖 OverviewThis repository contains the implementation of Phase 1 of the Neural Networks course project. The primary objective is to solve Partial Differential Equations (PDEs)—specifically the Laplace and Poisson equations—using Physics-Informed Neural Networks (PINNs).The Mesh-Free ApproachIn this project, we aim to shift from traditional mesh-based numerical methods (such as the Finite Element Method (FEM) or Finite Difference Method (FDM)) to a mesh-free Deep Learning approach.Mesh-Free: Unlike classical solvers that require complex mesh generation and discretization, PINNs approximate the solution $u(x,y)$ directly using a neural network.Physics-Driven Loss: The network learns the unknown function by minimizing a composite loss function that enforces both the boundary conditions and the residual of the governing equations at randomly sampled collocation points.Automatic Differentiation: This approach leverages Automatic Differentiation (AD) to compute derivatives exactly, avoiding the truncation errors common in numerical schemes.📂 File StructureThe project is organized to separate configuration, execution, and analysis:.
├── main.py                 # Core execution script (Data generation, Model training, Logic)
├── visualization_utils.py  # Utility module for professional, publication-quality plots
├── Laplace_EXP.csv         # Configuration file specific to the Laplace case
└── Poisson_EXP.csv         # Configuration file specific to the Poisson case
main.py: Handles network initialization, training loops, and calls visualization tools.visualization_utils.py: Computes error metrics and generates side-by-side comparisons of the PINN prediction vs. Ground Truth.🔬 Equations & Methodology1. Laplace EquationThe Laplace equation is a fundamental second-order PDE describing potential fields in regions free of sources (e.g., electrostatics in charge-free regions, steady-state heat conduction). The network minimizes the residual $\mathcal{L}_{PDE} = ||\nabla^2 u||^2$ across the domain defined in Laplace_EXP.csv.$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
$$### 2\. Poisson Equation

The Poisson equation generalizes the Laplace equation to model systems containing a source or sink term. It relates the potential field to a distributed source density, denoted by $f(x, y)$.

$$\nabla^2 u = f(x, y)
$$In this phase, the source function $f(x, y)$ is chosen specifically such that an **analytical exact solution** exists. This allows us to effectively benchmark the neural network's accuracy against a known ground truth.

-----

## 📊 Results & Visualization

Visualizing the raw output of a neural network is insufficient for scientific validation. To rigorously evaluate the model, we calculate quantitative error metrics:

  * **Absolute Error:** Point-wise difference $|u_{exact} - u_{pred}|$.
  * **Relative $L_2$ Error:** Global accuracy metric.

### Visual Outputs

The `visualization_utils.py` script generates a comprehensive composite figure containing three side-by-side plots:

1.  **Exact Solution (Ground Truth):** The mathematically correct solution derived analytically.
2.  **PINN Prediction:** The solution approximated by the trained neural network.
3.  **Absolute Error Map:** A spatial heatmap showing the error distribution. This is critical for identifying regions where the model struggles (e.g., near sharp gradients).

> **Note:** Output plots are saved with the `_comparison_PINN.png` suffix (e.g., `Laplace_comparison_PINN.png`) to distinguish them from legacy outputs.

-----

## 🚀 Usage Guide

### 1\. Installation

Ensure your environment is set up with the necessary scientific computing and deep learning libraries:

```bash
pip install numpy matplotlib torch pandas
```

### 2\. Training & Plotting

Run the main script to initiate the training process. The script will automatically parse the CSV configuration, train the model, and generate results.

```bash
python main.py
```

*Tip: Monitor the console for loss values to ensure the physics loss and boundary loss are converging.*

-----

## ⚙️ Configuration

You can modify experiment parameters in the `.csv` files (`Laplace_EXP.csv` or `Poisson_EXP.csv`) without changing the source code.

| Parameter | Description | Impact |
| :--- | :--- | :--- |
| **`grid_node_num`** | Number of internal collocation points. | Controls where the PDE is enforced. Higher numbers usually improve accuracy but increase training time. |
| **`figure_node_num`** | Resolution of the output images. | Default is usually 200x200. Increase this for higher-quality publication plots. |
| **`hidden_layers_group`** | Network architecture. | Defines the depth and width (neurons per layer) of the neural network. |
| **`domain_limits`** | (Implied in CSV) $x$ and $y$ ranges. | Defines the physical boundaries of the simulation. |

-----

**Developer:** [Faraz-Ardeh-2004](https://www.google.com/search?q=https://github.com/Faraz-Ardeh-2004)$$
