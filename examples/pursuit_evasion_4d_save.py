import imp
import numpy as np
import pickle
import os
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the file that includes dynamic systems
from odp.dynamics import DubinsCapture4D
# Plot options
from odp.Plots import PlotOptions, visualize_plots

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math

# Define grid for 4D: [x, y, theta, v]
g = Grid(
    np.array([-2.0, -2.0, -math.pi, 0.1]),
    np.array([2.0, 2.0, math.pi, 1.0]),
    4,
    np.array([100, 100, 50, 30]),
    [2]
)

# Implicit function for the initial value function
Initial_value_f = CylinderShape(g, [2, 3], np.array([0.0, 0.0, 0.0, 1.0]), 0.5, quadratic=False)

# Look-back length and time step of computation
lookback_length = 2.
t_step = 0.05
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# uMode maximizing means avoiding capture, dMode minimizing means capturing
my_car = DubinsCapture4D(uMode="max", dMode="min")

# Compute Backward Reachable Tube
compMethods = {"TargetSetMode": "minVWithV0"}
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, saveAllTimeSteps=True)

# ============================================================================
# Save grid and results
# ============================================================================
save_dir = "saved_value_functions"
os.makedirs(save_dir, exist_ok=True)

# Save grid (using pickle)
with open(os.path.join(save_dir, "grid_4d.pkl"), "wb") as f:
    pickle.dump(g, f)

# Save results (using numpy)
np.save(os.path.join(save_dir, "value_function_4d_linear.npy"), result)

# Save tau (time array)
np.save(os.path.join(save_dir, "tau_4d.npy"), tau)

# Save dynamics parameters for reference
dynamics_params = {
    "wMax": my_car.wMax,
    "aMax": my_car.aMax,
    "aMin": my_car.aMin,
    "vMax": my_car.vMax,
    "vMin": my_car.vMin,
    "speed": my_car.speed,
    "dMax": my_car.dMax,
    "uMode": my_car.uMode,
    "dMode": my_car.dMode
}
with open(os.path.join(save_dir, "dynamics_params_4d.pkl"), "wb") as f:
    pickle.dump(dynamics_params, f)

print(f"Saved grid, value function, and parameters to {save_dir}/")

# ============================================================================
# Function to get value and derivatives at a specific state and time
# ============================================================================
def get_value_and_derivatives(grid, result, state, time, tau):
    """
    Get value function V and its spatial derivatives at a given state and time.
    
    Args:
        grid: Grid object
        result: Value function array from HJSolver (shape: [grid_points..., time_steps])
        state: State vector [x, y, theta, v] (np.array of shape (4,))
        time: Time value (float)
        tau: Time array used in HJSolver
    
    Returns:
        value: Value function V(state, time) (float)
        derivatives: Spatial derivatives [dV/dx, dV/dy, dV/dtheta, dV/dv] (np.array of shape (4,))
        actual_time: The actual time used (closest to requested time)
    """
    # Find the closest time index
    time_idx = np.argmin(np.abs(tau - time))
    actual_time = tau[time_idx]
    
    # Get value function at this time
    V_at_time = result[..., time_idx]
    
    # Get the value at the given state using grid interpolation
    value = grid.get_values(V_at_time, state.reshape(1, -1))[0]
    
    # Compute spatial derivatives at all grid points for this time
    x_derivative = computeSpatDerivArray(grid, V_at_time, deriv_dim=1, accuracy="medium")
    y_derivative = computeSpatDerivArray(grid, V_at_time, deriv_dim=2, accuracy="medium")
    theta_derivative = computeSpatDerivArray(grid, V_at_time, deriv_dim=3, accuracy="medium")
    v_derivative = computeSpatDerivArray(grid, V_at_time, deriv_dim=4, accuracy="medium")
    
    # Get indices of the state in the grid
    state_indices = grid.get_indices(state.reshape(1, -1))

    # Convert to tuple of integers for indexing
    idx_tuple = tuple(int(arr[0]) if hasattr(arr, '__len__') else int(arr) 
                    for arr in state_indices)

    # Get derivatives at the nearest grid point
    derivatives = np.array([
        float(x_derivative[idx_tuple]),
        float(y_derivative[idx_tuple]),
        float(theta_derivative[idx_tuple]),
        float(v_derivative[idx_tuple])
    ])
    
    return value, derivatives, actual_time

# ============================================================================
# Example usage (optional - can comment out if just saving)
# ============================================================================
if __name__ == "__main__":
    # Define a specific state: [x, y, theta, v]
    query_state = np.array([0.5, 0.25, 0.5*math.pi, 1.0])
    query_time = 2.0

    # Get value and derivatives
    value, derivatives, actual_time = get_value_and_derivatives(g, result, query_state, query_time, tau)

    print("=" * 60)
    print("Value Function and Derivatives at Specific State and Time")
    print("=" * 60)
    print(f"Query state: x={query_state[0]:.3f}, y={query_state[1]:.3f}, "
          f"theta={query_state[2]:.3f}, v={query_state[3]:.3f}")
    print(f"Query time: {query_time:.3f} (actual time used: {actual_time:.3f})")
    print(f"\nValue function V(state, time): {value:.6f}")
    print(f"\nSpatial derivatives:")
    print(f"  dV/dx     = {derivatives[0]:.6f}")
    print(f"  dV/dy     = {derivatives[1]:.6f}")
    print(f"  dV/dtheta = {derivatives[2]:.6f}")
    print(f"  dV/dv     = {derivatives[3]:.6f}")
    print("=" * 60)