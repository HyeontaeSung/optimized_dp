import numpy as np
import pickle
import os
import math

# Plot options
from odp.Plots import PlotOptions, visualize_plots

# ============================================================================
# Load saved grid and value function
# ============================================================================
save_dir = "saved_value_functions"

# Load grid (using pickle)
print(f"Loading grid from {save_dir}/grid_4d.pkl...")
with open(os.path.join(save_dir, "grid_4d.pkl"), "rb") as f:
    grid = pickle.load(f)

# Load value function (using numpy)
print(f"Loading value function from {save_dir}/value_function_4d.npy...")
result = np.load(os.path.join(save_dir, "value_function_4d_linear.npy"))

# Load tau (time array) - optional, for time-specific plotting
print(f"Loading time array from {save_dir}/tau_4d.npy...")
tau = np.load(os.path.join(save_dir, "tau_4d.npy"))

print(f"Grid shape: {grid.pts_each_dim}")
print(f"Value function shape: {result.shape}")
print(f"Time steps: {len(tau)}")
print("=" * 60)

# Create plots directory if it doesn't exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# ============================================================================
# Plotting options
# ============================================================================
from odp.dynamics import DubinsCapture4D
import matplotlib.pyplot as plt

dynamics = DubinsCapture4D()

# Plot value function slices
# Grid dimensions: [x, y, theta, v]
x_min, x_max = grid.min[0], grid.max[0]
y_min, y_max = grid.min[1], grid.max[1]
theta_min, theta_max = grid.min[2], grid.max[2]
v_min, v_max = grid.min[3], grid.max[3]

# Create coordinate arrays
xs = np.linspace(x_min, x_max, grid.pts_each_dim[0])
ys = np.linspace(y_min, y_max, grid.pts_each_dim[1])
thetas = np.linspace(theta_min, theta_max, grid.pts_each_dim[2])
vs = np.linspace(v_min, v_max, grid.pts_each_dim[3])

# Select theta values to plot (5 slices)
theta_slices = np.linspace(theta_min, theta_max, 5)
# Select theta values to plot (5 slices) with explicit labels.
# Note:
# - The grid stores theta for periodic dimension 2 as [-pi, pi) (so +pi is excluded from the grid).
# - Nevertheless, we can request these theta values for plotting; grid.get_values will snap to the
#   nearest stored theta grid point via nearest-neighbor interpolation.
# theta_slices = np.array(
#     [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi],
#     dtype=np.float64,
# )
# theta_display_slices = theta_slices
# Fixed velocity value (1.0)
v_fixed = 1.0

# Time indices to plot
time_indices = [0]  # Can add more: [0, 10, 20, 30, 40]

# Create figure
fig = plt.figure(figsize=(6*len(theta_slices), 5*len(time_indices)), dpi=200)
X, Y = np.meshgrid(xs, ys)

for i, time_idx in enumerate(time_indices):
    for j, theta_val in enumerate(theta_slices):
        # Get value function at this time
        V_at_T = result[..., time_idx]
        
        # Create state array: each row is [x, y, theta, v]
        # Flatten the meshgrid (X, Y are already defined above) and combine with fixed theta and v
        states = np.column_stack([
            X.flatten(),
            Y.flatten(),
            np.full(X.size, theta_val),
            np.full(X.size, v_fixed)
        ])
        
        # Get values for all states and reshape to 2D (matching X, Y shape)
        values = grid.get_values(V_at_T, states)
        BRT_img = values.reshape(X.shape)
        
        # Create subplot
        ax = fig.add_subplot(len(time_indices), len(theta_slices), j + 1 + i*len(theta_slices))
        
        # Plot settings
        max_value = np.amax(BRT_img)
        min_value = np.amin(BRT_img)
        imshow_kwargs = {
            'vmax': max_value,
            'vmin': min_value,
            'cmap': 'coolwarm_r',
            'extent': (x_min, x_max, y_min, y_max),
            'origin': 'lower',
        }
        
        # Plot value function
        s1 = ax.imshow(BRT_img, **imshow_kwargs)
        fig.colorbar(s1, ax=ax)
        
        # Plot zero contour (reachable set boundary)
        zero_contour = ax.contour(X, Y, BRT_img, 
                                  levels=[0.0],  
                                  colors="black",  
                                  linewidths=2,    
                                  linestyles='--')
        
        # Set title
        ax.set_title(f'Time={tau[time_idx]:.2f}, θ={theta_val:.2f}, v={v_fixed:.2f}', 
                     fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pursuit_evasion_4d_slices_quadratic.png'), dpi=200, bbox_inches='tight')
print(f"Plot saved to {plots_dir}/pursuit_evasion_4d_slices.png")
plt.show()  

