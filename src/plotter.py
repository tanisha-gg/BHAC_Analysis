import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np
import pandas as pd

class Plotter:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def plot_combined(self, data, b_field, e_field, ratio, L, N_grid):
        """Create a 2-panel plot for E/B ratio and magnetic energy."""
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        plt.subplots_adjust(wspace=0.4)

        # Left panel: E/B ratio
        self._plot_ratio_panel(axs[0], data, ratio, L, N_grid)

        # Right panel: Magnetic energy
        self._plot_magnetic_energy_panel(axs[1], data, b_field, L, N_grid)

        plt.savefig(f"{self.output_folder}/combined_plot.pdf", bbox_inches='tight')
        plt.close()

    def _plot_ratio_panel(self, ax, data, ratio, L, N_grid):
        """Plot E/B ratio panel."""
        p1 = ax.imshow(ratio, cmap="coolwarm", origin="lower", vmin=-1, vmax=1, extent=[-2, 2, -2, 2])
        ax.set_title(r"$\frac{E^2 - B^2}{E^2 + B^2}$", fontsize=16)
        ax.set_xlabel(r"$x$", fontsize=16)
        ax.set_ylabel(r"$y$", fontsize=16)

        for idx, row in data.iterrows():
            if ratio[int(row["x0"] * N_grid / L + N_grid / 2), int(row["y0"] * N_grid / L + N_grid / 2)] > 0.5:
                ellipse = Ellipse(xy=(row["y0"], row["x0"]), width=2.0 * row["a_max"], height=2.0 * row["a_min"], angle=0.0, edgecolor='yellow', facecolor='none')
                ax.add_patch(ellipse)

    def _plot_magnetic_energy_panel(self, ax, data, b_field, L, N_grid):
        """Plot magnetic energy panel."""
        sigma = 10.0
        mag = np.sqrt(b_field[0]**2 + b_field[1]**2)
        p2 = ax.imshow(gaussian(mag, sigma=sigma)**2 / np.mean(mag**2), cmap="cmr.ghostlight", origin="lower", norm=colors.LogNorm(vmin=1e-1, vmax=1e1), extent=[-2, 2, -2, 2])
        ax.streamplot(self.Y, self.X, gaussian(b_field[1], sigma=sigma), gaussian(b_field[0], sigma=sigma), color="w", linewidth=0.05, density=10, arrowsize=0.0, integration_direction="both")
        ax.plot(data["y0"], data["x0"], c='r', marker='o', ls='None', markersize=4)

        for idx, row in data.iterrows():
            ellipse = Ellipse(xy=(row["y0"], row["x0"]), width=2.0 * row["a_max"], height=2.0 * row["a_min"], angle=0.0, edgecolor='yellow', facecolor='none')
            ax.add_patch(ellipse)

        ax.set_title(r"$b^2 / \langle b^2 \rangle$", fontsize=16)
        ax.set_xlabel(r"$x$", fontsize=16)
        ax.set_ylabel(r"$y$", fontsize=16)

        # Add colorbars
        cb1 = fig.colorbar(p1, ax=axs[0], pad=0.01, aspect=20, shrink=0.3)
        cb1.set_label(r"$\frac{E^2 - B^2}{E^2 + B^2}$", fontsize=16)
        cb2 = fig.colorbar(p2, ax=axs[1], pad=0.01, aspect=20, shrink=0.3)
        cb2.set_label(r"$b^2 / \langle b^2 \rangle$", fontsize=16)
