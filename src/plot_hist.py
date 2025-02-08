import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set up LaTeX font for plots
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

class ParticleHistogramPlotter:
    def __init__(self, folder, title, time_indices):
        self.folder = folder
        self.title = title
        self.time_indices = time_indices
        self.datasets = []
        self.histograms = []
        self.bin_centers = []
        self.fwhm_values = []

    def load_data(self):
        """Load particle data from CSV files for each time index."""
        for t in self.time_indices:
            data = np.loadtxt(f"{self.folder}/data00{self.time_indices[0]}_ensemble00{t}.csv", delimiter=',', skiprows=1)
            self.datasets.append(data)
            print(f"Loaded data for time index {t} with shape {data.shape}")

    def compute_histograms(self):
        """Compute histograms and FWHM for gamma distributions."""
        for data in self.datasets:
            gg = np.sqrt(data[:, 5]**2 + data[:, 6]**2 + data[:, 7]**2 + 1)  # gamma
            gamma_minus_one = gg - 1

            # Create histogram in linear (gamma - 1) space
            N, edges = np.histogram(gamma_minus_one, bins=100, density=True)
            bin_center = (edges[:-1] + edges[1:]) / 2  # Calculate bin centers
            self.bin_centers.append(bin_center)
            self.histograms.append(N)

            # Find the peak and FWHM
            peak_index = np.argmax(N)
            peak_value = N[peak_index]

            # Find the left and right indices where the histogram is half of the peak value
            left_index = np.where(N[:peak_index] <= peak_value / 2)[0]
            left_index = left_index[-1] if len(left_index) > 0 else 0

            right_index = np.where(N[peak_index:] <= peak_value / 2)[0]
            right_index = right_index[0] + peak_index if len(right_index) > 0 else len(N) - 1

            # Calculate the FWHM in terms of bin centers
            fwhm = bin_center[right_index] - bin_center[left_index]
            self.fwhm_values.append(fwhm)

            print(f"Peak gamma: {bin_center[peak_index]}, FWHM: {fwhm}")

    def plot_histograms(self):
        """Plot histograms in log-log scale."""
        num_colors = len(self.time_indices)
        cmap = plt.cm.get_cmap('turbo', num_colors)

        fig, ax = plt.subplots()
        for i, hist in enumerate(self.histograms):
            ax.loglog(self.bin_centers[i], hist, color=cmap(i), label=f'{int(self.time_indices[0])/10} - FWHM: {self.fwhm_values[i]:.2f}')

        plt.ylabel(r'$\frac{dN}{d (\gamma - 1)}$')
        plt.xlabel(r'$\gamma$ - 1')

        # Add dynamic title based on input
        if self.title == 'MJ1000':
            dist = 'Maxwell-Juttner'
            also = ' $\\theta$ = 0.01 with $\\frac{q}{m}$=1000'
        elif self.title == 'MJ3333':
            dist = 'Maxwell-Juttner'
            also = ' $\\theta$ = 0.01 with $\\frac{q}{m}$=3333'
        elif self.title == 'MJ10000':
            dist = 'Maxwell-Juttner'
            also = ' $\\theta$ = 0.01 with $\\frac{q}{m}$=10000'
        elif self.title == 'MJ100000':
            dist = 'Maxwell-Juttner'
            also = ' $\\theta$ = 0.01 with $\\frac{q}{m}$=100000'
        else:
            also = ''

        # Create discrete colorbar
        norm = BoundaryNorm(np.arange(len(self.time_indices) + 1) - 0.5, len(self.time_indices))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        num_ticks = 11
        if len(self.time_indices) > num_ticks:
            tick_indices = np.linspace(0, len(self.time_indices) - 1, num_ticks, dtype=int)
        else:
            tick_indices = np.arange(len(self.time_indices))

        cbar = plt.colorbar(sm, ticks=tick_indices)
        plt.gca().set_xlim(left=1)
        cbar.ax.set_yticklabels([f'{int(self.time_indices[i])/100}' for i in tick_indices])
        cbar.set_label('Time')

        plt.savefig(f"{self.folder}/particletest_{self.time_indices[0]}_{self.title}.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot particle histograms.")
    parser.add_argument("i", type=str, help="Time of first particle snapshot")
    parser.add_argument("t", type=str, nargs='+', help='Time indices of snapshots')
    parser.add_argument('folder', type=str, help='Folder with ensemble CSV files')
    parser.add_argument('title', type=str, help='Title of the plot')
    args = parser.parse_args()

    plotter = ParticleHistogramPlotter(args.folder, args.title, args.t)
    plotter.load_data()
    plotter.compute_histograms()
    plotter.plot_histograms()

if __name__ == "__main__":
    main()
