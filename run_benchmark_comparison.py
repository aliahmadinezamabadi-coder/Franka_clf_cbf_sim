import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# IEEE RA-L Publication Quality Configuration
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 5.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "patch.force_edgecolor": True,
})

# Controller configurations matching paper notation & CSV filenames
controllers = [
    {
        "filename": "art_avic_contact_forces.csv",
        "stiffness_filename": "art_avic_stiffness.csv",
        "label": "ART-AVIC",
        "color": "#7E2987",   # Purple
        "ls_ctrl": ":",       # Dotted for stiffness plot
        "ls_raw": ":",
    },
    {
        "filename": "Song_Controller.csv",
        "stiffness_filename": "Song_Controller_stiffness.csv",
        "label": "NAIPC",
        "color": "#D95319",   # Orange
        "ls_ctrl": "--",      # Dashed for stiffness plot
        "ls_raw": "--",
    },
    {
        "filename": "my_controller.csv",
        "stiffness_filename": "my_controller_stiffness.csv",
        "label": "UC-VIAC",
        "color": "#0072BD",   # Blue
        "ls_ctrl": "-",       # Solid for stiffness plot
        "ls_raw": "-.",
    },
]

loaded_force_data = {}
loaded_stiffness_data = {}

# Load available CSV datasets
for ctrl in controllers:
    fname = ctrl["filename"]
    stiff_fname = ctrl["stiffness_filename"]

    if os.path.exists(fname):
        loaded_force_data[fname] = np.genfromtxt(fname, delimiter=",", names=True)
        print(f"Loaded force data: '{fname}'.")
    else:
        print(f"Warning: File '{fname}' not found.")

    if os.path.exists(stiff_fname):
        loaded_stiffness_data[stiff_fname] = np.genfromtxt(stiff_fname, delimiter=",", names=True)
        print(f"Loaded stiffness data: '{stiff_fname}'.")

if not loaded_force_data:
    raise FileNotFoundError("No force CSV datasets found.")


# =============================================================================
# FIGURE 1: Standalone Normal Force Tracking Comparison
# =============================================================================
fig1, ax_f = plt.subplots(figsize=(3.5, 2.5), dpi=300)

target_plotted = False
for ctrl in controllers:
    fname = ctrl["filename"]
    if fname in loaded_force_data:
        data = loaded_force_data[fname]
        colnames = data.dtype.names
        t = data["Time_s"]

        # Target Force Profile F_d(t)
        if "Desired_Force_N" in colnames and not target_plotted:
            ax_f.plot(
                t,
                data["Desired_Force_N"],
                color="#000000",
                linestyle="--",
                linewidth=0.5,
                label=r"$F_d(t)$ (Target)",
                zorder=1,
            )
            target_plotted = True

        # Raw Force F_n^raw
        if "Raw_Force_Z_N" in colnames:
            ax_f.plot(
                t,
                data["Raw_Force_Z_N"],
                color=ctrl["color"],
                linestyle=ctrl["ls_raw"],
                linewidth=0.5,
                alpha=0.35,
                label=rf"$F_n^{{\mathrm{{raw}}}}$ ({ctrl['label']})",
                zorder=2,
            )

        # Filtered Normal Force F_n
        if "Filtered_Normal_Force_N" in colnames:
            ax_f.plot(
                t,
                data["Filtered_Normal_Force_N"],
                color=ctrl["color"],
                linestyle="-",
                linewidth=0.5,
                alpha=0.95,
                label=rf"$F_n$ ({ctrl['label']})",
                zorder=3,
            )

ax_f.set_xlabel(r"Time $t$ [s]")
ax_f.set_ylabel(r"Normal Force $F_n$ [N]")
ax_f.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
ax_f.set_axisbelow(True)

ax_f.legend(
    loc="lower right",
    fontsize=5.8,
    frameon=True,
    framealpha=0.9,
    edgecolor="none",
    ncol=1,
    handlelength=1.4,
    labelspacing=0.25,
)

fig1.tight_layout(pad=0.3)

pdf_path1 = "single_figure_force_comparison.pdf"
png_path1 = "single_figure_force_comparison.png"
fig1.savefig(pdf_path1, format="pdf", bbox_inches="tight")
fig1.savefig(png_path1, format="png", dpi=600, bbox_inches="tight")
print(f"Exported Figure 1: '{pdf_path1}'")
plt.close(fig1)


# =============================================================================
# FIGURE 2: Standalone 6-D Cartesian Stiffness Adaptation (Trans. & Rot.)
# =============================================================================
fig2, (ax_pos, ax_ori) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True, dpi=300)

pos_styles = [
    {"axis": "x", "col_keys": ["Kp_x", "Kp_pos_x"], "color": "#D95319", "symbol": r"$K_{p,x}$"},
    {"axis": "y", "col_keys": ["Kp_y", "Kp_pos_y"], "color": "#2CA02C", "symbol": r"$K_{p,y}$"},
    {"axis": "z", "col_keys": ["Kp_z", "Kp_pos_z"], "color": "#0072BD", "symbol": r"$K_{p,z}$"},
]

ori_styles = [
    {"axis": "roll",  "col_keys": ["Kp_roll", "Kp_ori_x"],  "color": "#7E2987", "symbol": r"$K_{p,\phi}$"},
    {"axis": "pitch", "col_keys": ["Kp_pitch", "Kp_ori_y"], "color": "#77AC30", "symbol": r"$K_{p,\theta}$"},
    {"axis": "yaw",   "col_keys": ["Kp_yaw", "Kp_ori_z"],   "color": "#4DBEEE", "symbol": r"$K_{p,\psi}$"},
]

for ctrl in controllers:
    stiff_fname = ctrl["stiffness_filename"]
    fname = ctrl["filename"]

    stiff_data = None
    if stiff_fname in loaded_stiffness_data:
        stiff_data = loaded_stiffness_data[stiff_fname]
    elif fname in loaded_force_data:
        stiff_data = loaded_force_data[fname]

    if stiff_data is not None:
        colnames = stiff_data.dtype.names
        t = stiff_data["Time_s"]

        # Plot Translational Stiffness (Top Subplot)
        for style in pos_styles:
            col_found = next((key for key in style["col_keys"] if key in colnames), None)
            if col_found:
                ax_pos.plot(
                    t,
                    stiff_data[col_found],
                    color=style["color"],
                    linestyle=ctrl["ls_ctrl"],
                    linewidth=1.1,
                    alpha=0.9,
                    label=f"{style['symbol']} ({ctrl['label']})",
                )

        # Plot Rotational Stiffness (Bottom Subplot)
        for style in ori_styles:
            col_found = next((key for key in style["col_keys"] if key in colnames), None)
            if col_found:
                ax_ori.plot(
                    t,
                    stiff_data[col_found],
                    color=style["color"],
                    linestyle=ctrl["ls_ctrl"],
                    linewidth=1.1,
                    alpha=0.9,
                    label=f"{style['symbol']} ({ctrl['label']})",
                )

# Subplot (a): Translational Stiffness
ax_pos.set_ylabel(r"$\mathbf{k}_p^{\mathrm{pos}}$ [N/m]")
ax_pos.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
ax_pos.set_axisbelow(True)
ax_pos.legend(
    loc="upper right",
    fontsize=5.0,
    frameon=True,
    framealpha=0.85,
    edgecolor="none",
    ncol=3,
    handlelength=1.2,
    columnspacing=0.5,
)

# Subplot (b): Rotational Stiffness
ax_ori.set_xlabel(r"Time $t$ [s]")
ax_ori.set_ylabel(r"$\mathbf{k}_p^{\mathrm{ori}}$ [Nm/rad]")
ax_ori.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
ax_ori.set_axisbelow(True)
ax_ori.legend(
    loc="upper right",
    fontsize=5.0,
    frameon=True,
    framealpha=0.85,
    edgecolor="none",
    ncol=3,
    handlelength=1.2,
    columnspacing=0.5,
)

fig2.tight_layout(pad=0.3)

pdf_path2 = "single_figure_stiffness_comparison.pdf"
png_path2 = "single_figure_stiffness_comparison.png"
fig2.savefig(pdf_path2, format="pdf", bbox_inches="tight")
fig2.savefig(png_path2, format="png", dpi=600, bbox_inches="tight")
print(f"Exported Figure 2: '{pdf_path2}'")
plt.close(fig2)


# =============================================================================
# FIGURE 3: Standalone Dynamic Target Force Tracking (my_controller_forces.csv)
# =============================================================================
dynamic_force_file = "my_controller_forces.csv"

if os.path.exists(dynamic_force_file):
    dyn_data = np.genfromtxt(dynamic_force_file, delimiter=",", names=True)
    t_dyn = dyn_data["Time_s"]

    fig3, ax_dyn = plt.subplots(figsize=(3.5, 2.5), dpi=300)

    # 1. Target Force Profile F_d(t)
    if "Desired_Force_N" in dyn_data.dtype.names:
        ax_dyn.plot(
            t_dyn,
            dyn_data["Desired_Force_N"],
            color="#000000",
            linestyle="--",
            linewidth=1.2,
            label=r"$F_d(t)$ (Target)",
            zorder=1,
        )

    # 2. Raw Force F_n^raw
    if "Raw_Force_Z_N" in dyn_data.dtype.names:
        ax_dyn.plot(
            t_dyn,
            dyn_data["Raw_Force_Z_N"],
            color="#0072BD",
            linestyle="-.",
            linewidth=0.5,
            alpha=0.35,
            label=r"$F_n^{\mathrm{raw}}$ (UC-VIAC)",
            zorder=2,
        )

    # 3. Filtered Normal Force F_n
    if "Filtered_Normal_Force_N" in dyn_data.dtype.names:
        ax_dyn.plot(
            t_dyn,
            dyn_data["Filtered_Normal_Force_N"],
            color="#0072BD",
            linestyle="-",
            linewidth=1.3,
            alpha=0.95,
            label=r"$F_n$ (UC-VIAC)",
            zorder=3,
        )

    ax_dyn.set_xlabel(r"Time $t$ [s]")
    ax_dyn.set_ylabel(r"Normal Force $F_n$ [N]")
    ax_dyn.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.7)
    ax_dyn.set_axisbelow(True)

    ax_dyn.legend(
        loc="lower right",
        fontsize=5.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        ncol=1,
        handlelength=1.4,
        labelspacing=0.25,
    )

    fig3.tight_layout(pad=0.3)

    pdf_path3 = "single_figure_dynamic_force.pdf"
    png_path3 = "single_figure_dynamic_force.png"
    fig3.savefig(pdf_path3, format="pdf", bbox_inches="tight")
    fig3.savefig(png_path3, format="png", dpi=600, bbox_inches="tight")
    print(f"Exported Figure 3: '{pdf_path3}'")
    plt.close(fig3)
else:
    print(f"Warning: File '{dynamic_force_file}' not found. Figure 3 skipped.")