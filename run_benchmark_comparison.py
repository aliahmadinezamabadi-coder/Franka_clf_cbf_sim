import os
import numpy as np
import matplotlib.pyplot as plt

# Controller files and styling configuration
controllers = [
    {
        "filename": "art_avic_contact_forces.csv",
        "label": "ART-AVIC (Wang et al.)",
        "color": "purple",
    },
    {
        "filename": "Song_Controller.csv",
        "label": "Song Controller",
        "color": "forestgreen",
    },
    {
        "filename": "my_controller.csv",
        "label": "Proposed CLF-CBF (Ours)",
        "color": "royalblue",
    },
]

loaded_data = {}

# Load available CSV datasets
for ctrl in controllers:
    fname = ctrl["filename"]
    if os.path.exists(fname):
        data = np.genfromtxt(fname, delimiter=",", names=True)
        loaded_data[fname] = data
        print(f"Successfully loaded '{fname}'.")
    else:
        print(f"Warning: File '{fname}' not found. Skipping...")

if not loaded_data:
    print("Error: No CSV files were loaded. Exiting.")
    exit()

# Single-panel plot creation
fig, ax = plt.subplots(figsize=(12, 6))

# 1. Plot Target Force Profile first (in background)
target_plotted = False
for ctrl in controllers:
    fname = ctrl["filename"]
    if fname in loaded_data:
        data = loaded_data[fname]
        colnames = data.dtype.names
        if "Desired_Force_N" in colnames and not target_plotted:
            ax.plot(
                data["Time_s"],
                data["Desired_Force_N"],
                "r--",
                linewidth=1.8,
                label="Target Force Profile",
                zorder=1,
            )
            target_plotted = True

# 2. Plot Raw and Filtered Forces on the same single axis
for ctrl in controllers:
    fname = ctrl["filename"]
    if fname in loaded_data:
        data = loaded_data[fname]
        colnames = data.dtype.names
        t = data["Time_s"]

        # Raw force signal (light/semi-transparent)
        if "Raw_Force_Z_N" in colnames:
            ax.plot(
                t,
                data["Raw_Force_Z_N"],
                color=ctrl["color"],
                alpha=0.25,
                linewidth=0.8,
                linestyle="-",
                label=f"Raw Fz ({ctrl['label']})",
            )

        # Filtered force signal (solid/bold)
        if "Filtered_Normal_Force_N" in colnames:
            ax.plot(
                t,
                data["Filtered_Normal_Force_N"],
                color=ctrl["color"],
                alpha=0.95,
                linewidth=1.8,
                linestyle="-",
                label=f"Filtered F_normal ({ctrl['label']})",
            )

# Plot formatting
ax.set_title("Raw & Filtered Contact Force Overlay Comparison", fontsize=12, fontweight="bold")
ax.set_xlabel("Time [s]", fontsize=10)
ax.set_ylabel("Normal Force [N]", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left", fontsize=8, ncol=2)

fig.tight_layout()
output_image = "single_figure_force_comparison.png"
fig.savefig(output_image, dpi=300)
print(f"Combined single-figure plot saved to '{output_image}'.")
plt.show()