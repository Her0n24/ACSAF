import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patheffects as patheffects

# -----------------------------
# 1. Time and geometry setup
# -----------------------------
t = np.arange(0, 1081, 60)  # seconds (0 to 18 min, every 60s)

# For plotting: small angle approximation (used in parabolic ray)
alpha = -5.14e-5 * t  # rad

# Distance grid (km)
x = np.linspace(0, 700, 500)

# Ray parameters
m = 0.0        # reference point
r = 5000.0     # curvature radius (km, adjust if needed)

# Cloud heights (km)
cloud_heights = [1, 4, 9]

# -----------------------------
# 2. Parabolic ray equation
# -----------------------------
def parabolic_ray_equation(x, m, alpha, r, H):
    return (x - m) * np.tan(alpha) + (0.5 * (x - m)**2) / r + H

# -----------------------------
# 3. Reddening calculation based on path below 10 km
# -----------------------------
def compute_reddening_distance_weighted(x, g_x):
    """
    Compute reddening based on path length through atmosphere below 10 km
    with exponential density weighting.
    """
    # Only consider ray segments above ground
    mask = g_x >= 0
    if np.sum(mask) < 2:
        return 0.0

    x_valid = x[mask]
    z_valid = g_x[mask]

    # Path length along the ray
    dx = np.gradient(x_valid)
    dz = np.gradient(z_valid)
    ds = np.sqrt(dx**2 + dz**2)

    # Exponential density weighting (scale height H_scale = 8 km)
    H_scale = 8.0
    k0 = 0.005  # km^-1, reduced for high cloud distances

    # Only count contribution below 10 km
    below_mask = z_valid < 10.0
    if np.sum(below_mask) < 1:
        return 0.0

    k_z = k0 * np.exp(-z_valid / H_scale) * below_mask

    # Weighted optical depth
    tau = np.sum(k_z * ds)

    # Convert to reddening factor (0 to 1)
    reddening = 1 - np.exp(-tau)
    return reddening

# -----------------------------
# 4. Plotting
# -----------------------------
for H in cloud_heights:
    fig, ax = plt.subplots(figsize=(12, 6))

    reddening_values = []

    # First pass: compute reddening per timestep
    for a in alpha:
        g_x = parabolic_ray_equation(x, m, a, r, H)
        reddening = compute_reddening_distance_weighted(x, g_x)
        reddening_values.append(reddening)


    reddening_values = np.array(reddening_values)
    reddening_values = np.array(reddening_values)

    # Select candidate profiles based on cloud height but only among valid (non-submerged) profiles
    start_idx_map = {1: 3, 4: 4, 9: 5}
    start_idx = start_idx_map.get(H, 3)
    # Determine which timesteps produce rays entirely above ground
    valid_indices = []
    for i, a in enumerate(alpha):
        g_x_full = parabolic_ray_equation(x, m, a, r, H)
        if np.all(g_x_full >= 0):
            valid_indices.append(i)

    # Filter candidates to those valid indices starting at start_idx
    candidates = [i for i in valid_indices if i >= start_idx]
    selected_idx = None
    # Select candidate profiles based on cloud height and pick the earliest valid candidate
    # LCC (H=1 km) -> consider 4th profile onward (index 3)
    # MCC (H=4 km) -> consider 5th profile onward (index 4)
    # HCC (H=9 km) -> consider 6th profile onward (index 5)
    start_idx_map = {1: 3, 4: 4, 9: 5}
    start_idx = start_idx_map.get(H, 3)
    selected_idx = None
    for i in range(start_idx, len(alpha)):
        g_x_full = parabolic_ray_equation(x, m, alpha[i], r, H)
        # require entire profile above ground
        if np.all(g_x_full >= 0):
            selected_idx = i
            break

    # Cap colorbar at 95th percentile to avoid yellow/white at the top
    vmax = np.percentile(reddening_values, 95)
    norm = colors.Normalize(vmin=0, vmax=1.0)
    # Truncate the colormap to use only the upper (red-violet) part for high intensity
    from matplotlib.colors import LinearSegmentedColormap
    base_cmap = plt.get_cmap('inferno_r')
    def truncate_colormap(cmap, minval=0.6, maxval=0.9, n=256):
        return LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n)))
    # Use only the upper part (red-violet) for high intensity (not inverted)
    cmap = truncate_colormap(base_cmap, 0.0, 0.6)


    # Second pass: plot and label
    placed_labels = []
    label_min_sep = 0.25  # km minimum vertical separation between labels
    for i, a in enumerate(alpha):
        g_x = parabolic_ray_equation(x, m, a, r, H)
        # Remove profile if it goes under the surface
        if np.any(g_x < 0):
            continue
        # If this is the selected profile, highlight in red
        if selected_idx is not None and i == selected_idx:
            sel_color = 'red'
            ax.plot(x, g_x, color=sel_color, linewidth=2.8, alpha=0.95, zorder=5)
        else:
            color = cmap(norm(reddening_values[i]))
            ax.plot(x, g_x, color=color, alpha=0.9)

        # Add time label on top of the profile for every 2nd profile (even indices)
        if i % 2 == 0:
            valid = np.ones_like(g_x, dtype=bool)
            # anchor label at a point near the profile center (avoid ends)
            idx_anchor = len(x) // 2
            # if that point is invalid (shouldn't be since we removed negative profiles), fall back to last
            if not valid[idx_anchor]:
                valid_idx = np.where(valid)[0]
                if valid_idx.size:
                    idx_anchor = valid_idx[-1]
            x_label = x[idx_anchor]
            y_label = g_x[idx_anchor]
            # Clamp inside axes
            x_label = min(max(x_label, 5.0), 695.0)
            y_label = min(max(y_label, 0.7), 10.3)
            # avoid overlap
            orig_y = y_label
            attempts = 0
            while any(abs(y_label - y0) < label_min_sep for y0 in placed_labels) and attempts < 40:
                y_label += label_min_sep
                attempts += 1
            if y_label > 10.3:
                y_label = orig_y
                attempts = 0
                while any(abs(y_label - y0) < label_min_sep for y0 in placed_labels) and attempts < 40:
                    y_label -= label_min_sep
                    attempts += 1
            y_label = min(max(y_label, 0.7), 10.3)
            placed_labels.append(y_label)
            ax.text(
                x_label, y_label, f"t={t[i]}s",
                fontsize=7, color='white', va='center', ha='center', alpha=0.95, fontweight='bold',
                path_effects=[patheffects.withStroke(linewidth=1.2, foreground='black')],
                clip_on=True
            )

    ax.set_title(f"Sun Ray Profiles (Cloud Height = {H} km)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Height (km)")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 11)

    # Colorbar for reddening
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Reddening intensity [Arbitrary Units] (0=no, 1=full)")

    # Add legend in lower-right: include reddening boundary (red line)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='red', lw=0.8, label='reddening boundary')]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9, prop={'weight':'normal'})

    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"sun_ray_profiles_cloud_{H}km.png")