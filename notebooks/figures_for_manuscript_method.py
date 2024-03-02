# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ast

from functions import q_drydown, exponential_drydown, loss_model, exponential_drydown2

# %%

var_dict = {
    "theta": {
        "column_name": "sm",
        "symbol": r"$\theta$",
        # "label": r"Soil moisture",
        "label": r"Soil moisture, $\theta$",
        "unit": r"(m$^3$ m$^{-3}$)",
        "lim": [0, 0.50],
    },
    "dtheta": {
        "column_name": "",
        "symbol": r"$-\frac{d\theta}{dt}$",
        "label": r"$-\frac{d\theta}{dt}$",
        # "label": r"Change in soil moisture, $-\frac{d\theta}{dt}$",
        "unit": r"(m$^3$ m$^{-3}$ day$^{-1}$)",
        "lim": [-0.10, 0],
    },
    "t": {
        "column_name": "",
        "symbol": r"$t$",
        "label": r"$t$",
        "unit": r"(days)",
    },
}

# var_dict = {
#     "theta": {
#         "column_name": "",
#         "symbol": r"$\theta$",
#         "label": r"Soil moisture $\theta$",
#         "unit": r"$[m^3/m^3]$",
#         "lim": [0, 0.50],
#     },
#     "dtheta": {
#         "column_name": "",
#         "symbol": r"$-d\theta/dt$",
#         "label": r"$-d\theta/dt$",
#         "unit": r"$[m^3/m^3/day]$",
#         "lim": [-0.10, 0],
#     },
#     "theta_norm": {
#         "column_name": "",
#         "symbol": r"$\theta_{norm}$",
#         "label": r"Normalized soil moisture $\theta_{norm}$",
#         "unit": r"$[-]$",
#         "lim": [0, 1.0],
#     },
#     "dtheta_norm": {
#         "column_name": "",
#         "symbol": r"$-d\theta/dt$",
#         "label": r"$-d\theta_{norm}/dt$",
#         "unit": r"$[-/day]$",
#         "lim": [-0.15, 0],
#     },
#     "t": {
#         "column_name": "",
#         "symbol": r"$t$",
#         "label": r"Timestep $t$",
#         "unit": r"$[day]$",
#     },
# }
theta_vardict = var_dict["theta"]
dtheta_vardict = var_dict["dtheta"]


# %%

# Define parameters

# Non-linearity parameters
q0 = 1
q1 = 1.5
q2 = 0.7

# Common parameters
k = 0.3
delta_theta = 0.5
theta_w = 0.02
theta_star = 0.5

# Define variables
theta = np.arange(theta_w, 1, 1e-03)
t = np.arange(0, 10, 1e-03)

# %% Plot
fig = plt.figure(figsize=(8, 4))
plt.rcParams.update({"font.size": 14})
c1 = f"#2c7fb8"
c2 = f"#41b6c4"
c3 = f"#a1dab4"
linewidth = 3

ax1 = fig.add_subplot(1, 2, 1)

# List of (q, color) pairs
q_colors = [(q1, c1), (q0, c2), (q2, c3)]

# Loop through each pair and plot
for q, color in q_colors:
    ax1.plot(
        theta,
        -1 * loss_model(theta=theta, q=q, k=k, theta_wp=theta_w, theta_star=theta_star),
        label=f"q={q}",
        linewidth=linewidth,
        color=color,
    )

ax1.set_ylim([0.0, k])
ax1.set_xlim([0.0, theta_star])
ax1.set(
    xlabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
    ylabel=f'{dtheta_vardict["label"]} {dtheta_vardict["unit"]}',
    title=rf"Normalized loss function $L(\theta)/\Delta z$",
)

# Calculate & plot the drawdown
ax2 = fig.add_subplot(1, 2, 2)

# Calculate & plot d_theta

ax2.plot(
    t,
    q_drydown(
        t=t, q=q1, k=k, delta_theta=delta_theta, theta_star=theta_star, theta_w=theta_w
    ),
    label=f"q={q1}",
    linewidth=linewidth,
    color=c1,
)
ax2.plot(
    t,
    exponential_drydown2(
        t=t, delta_theta=delta_theta, theta_w=theta_w, theta_star=theta_star, k=k
    ),
    label=f"q={q0}",
    linewidth=linewidth,
    color=c2,
)
ax2.plot(
    t,
    q_drydown(
        t=t, q=q2, k=k, delta_theta=delta_theta, theta_star=theta_star, theta_w=theta_w
    ),
    label=f"q={q2}",
    linewidth=linewidth,
    color=c3,
)
ax2.set_ylim([0.0, theta_star])
ax2.set_xlim([0.0, 5])
ax2.set(
    xlabel=f'{var_dict["t"]["label"]} {var_dict["t"]["unit"]}',
    ylabel=f'{theta_vardict["label"]} {theta_vardict["unit"]}',
    title="Soil moisture drydown",
)

ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
fig.tight_layout()


# %% Output

out_path = (
    r"/home/raraki/waves/projects/smap-drydown/output/raraki_2023-11-25_global_95asmax"
)
out_dir = os.path.join(out_path, "figs")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f"Created dir: {out_dir}")


# %%
fig.savefig(os.path.join(out_dir, f"theory.pdf"), dpi=600, bbox_inches="tight")
# %%
