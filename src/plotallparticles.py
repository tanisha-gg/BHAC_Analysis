import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from plot_vtuanalyzenew import *
from matplotlib import rc
import cmasher as cmr

# Set up LaTeX font for plots
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

class ParticleVTUPlotter:
    def __init__(self, folder, i0, i1):
        self.folder = folder
        self.i0 = i0
        self.i1 = i1

    def plot_particles_on_vtu(self):
        """Overlay particles on VTU data."""
        xrange1, xrange2 = 0, 1
        yrange1, yrange2 = -0.5, 0.5

        for vtu_index in range(self.i0, self.i1 + 1, 10):
            # Load VTU data
            mhd_file_path = os.path.join(self.folder, f'data{str(vtu_index).zfill(4)}.vtu')
            print(f"Reading VTU file: {mhd_file_path}")

            try:
                data, names = fast_vtu_reader(mhd_file_path, attr={'lfac', 'curlbz'}, blocks=False)
            except FileNotFoundError:
                print(f"VTU file {mhd_file_path} not found. Skipping.")
                continue

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_raw_data_cells(data, data['lfac'], fig=fig, ax=ax, x_range=(xrange1, xrange2), y_range=(yrange1, yrange2),
                                cmap='cmr.ember', label='$\\gamma_{MHD}$', linewidths=0.2, edgecolors=None, colorbar=False)
            ax.set_aspect('equal')

            # Add colorbar for VTU data
            cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap='cmr.ember', norm=plt.Normalize(vmin=1, vmax=3.5)),
                                 ax=ax, orientation='vertical', pad=0.03, location='right', shrink=0.8, aspect=15)
            cbar1.set_label('$\\gamma_{MHD}$', fontsize=16)
            cbar1.ax.tick_params(labelsize=16)

            # Load corresponding particle data
            particle_file_name = f'data0000_ensemble00{str(vtu_index).zfill(4)}.csv'
            particle_file_path = os.path.join(self.folder, particle_file_name)
            print(f"Reading particle data: {particle_file_path}")

            try:
                df = pd.read_csv(particle_file_path, delimiter=',', on_bad_lines='skip')
                if df.empty:
                    print(f"Particle data file {particle_file_path} is empty. Skipping.")
                    continue

                # Extract particle data
                x1 = df[' x1']
                x2 = df[' x2']
                u1 = df[' u1']
                u2 = df[' u2']
                u3 = df[' u3']
                gamma = np.sqrt(1 + (u1**2 + u2**2 + u3**2))

                # Scatter plot of particles (overlay)
                sc = ax.scatter(x1, x2, s=0.3, c=np.log10(gamma - 1), cmap='cmr.horizon', alpha=0.9)

                # Add colorbar for particle data
                cbar2 = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.14, location='left', shrink=0.8, aspect=15)
                cbar2.set_label('$\\log(\\gamma_P - 1)$', fontsize=16)
                cbar2.ax.tick_params(labelsize=16)
                sc.set_clim(-1, 3)

            except FileNotFoundError:
                print(f"Particle file {particle_file_path} not found. Skipping.")
                continue

            # Set axis limits and labels
            ax.set_xlim([xrange1, xrange2])
            ax.set_ylim([yrange1, yrange2])
            ax.set_xlabel('$x$', fontsize=16)
            ax.set_ylabel('$y$', fontsize=16)
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelsize=16)

            # Set the title
            ax.set_title(f"$t={round(float(vtu_index)/100, 2)}t_c$", fontsize=18)

            # Save the figure
            output_file = f'{self.folder}/zoom_overlay_particles_vtu_{vtu_index}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"Saved plot to {output_file}")
            plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Overlay particles on VTU data")
    parser.add_argument("i0", type=int, help="Initial time index")
    parser.add_argument("i1", type=int, help="Final time index")
    parser.add_argument("folder", type=str, help="Folder containing the data files")
    args = parser.parse_args()

    plotter = ParticleVTUPlotter(args.folder, args.i0, args.i1)
    plotter.plot_particles_on_vtu()

if __name__ == "__main__":
    main()
