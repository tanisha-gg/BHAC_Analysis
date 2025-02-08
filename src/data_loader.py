import os
import numpy as np
import pandas as pd
from PLASMAtools.read_funcs.read import Fields

class DataLoader:
    def __init__(self, folder):
        self.folder = folder

    def load_vtu_data(self, vtu_index, attr):
        """Load VTU data for a given index and attributes."""
        vtu_path = os.path.join(self.folder, f'data{str(vtu_index).zfill(4)}.vtu')
        sim = Fields(vtu_path, sim_data_type="bhac")
        sim.read(attr, N_grid_x=2048, N_grid_y=2048)
        return sim

    def load_csv_data(self, csv_file):
        """Load CSV data for particle analysis."""
        csv_path = os.path.join(self.folder, csv_file)
        return pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')

    def save_field_data(self, field, filename):
        """Save field data to a file."""
        np.save(os.path.join(self.folder, filename), field)
