import argparse
from src.data_loader import DataLoader
from src.extremum import Extremum
from src.plotter import Plotter
from src.plotallparticles import ParticleVTUPlotter
from src.plot_hist.py import ParticleHistogramPlotter

def main():
    parser = argparse.ArgumentParser(description="Analyze MHD and particle data.")
    parser.add_argument("i", type=int, help="Time index for analysis")
    parser.add_argument("folder", type=str, help="Folder containing data files")
    args = parser.parse_args()

    # Load data
    loader = DataLoader(args.folder)
    sim = loader.load_vtu_data(args.i, "elec")
    e_field = np.array([sim.elecy, sim.elecx])
    loader.save_field_data(e_field, "e_field.npy")

    b_field = np.load(f"{args.folder}/b_field.npy")

    # Analyze extremum
    extremum = Extremum(b_field, output_label=f"./plots/data{str(args.i).zfill(4)}", sigma=10.0, phase_plot=True)

    # Plot results
    plotter = Plotter("./plots")
    data = pd.read_csv(f"./plots/data{str(args.i).zfill(4)}_local_statistics.csv")
    plotter.plot_combined(data, b_field, e_field, ratio, L=4.0, N_grid=2048)
    plotter = ParticleVTUPlotter(args.folder, args.i0, args.i1)
    plotter.plot_particles_on_vtu()
    plotter = ParticleHistogramPlotter(args.folder, args.title, args.t)
    plotter.load_data()
    plotter.compute_histograms()
    plotter.plot_histograms()

if __name__ == "__main__":
    main()
