import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from PLASMAtools.aux_funcs import derived_var_funcs as dv
import pandas as pd

# Class below is from James Beattie's PLASMATools

class Extremum:
    def __init__(self, field, output_label, phase_plot=True, L=4.0, debug=False, sigma=0.0, min_distance=1, num_of_peaks=3, rtol=0.05, radius_min=0.0, radius_max=0.7, n_steps_in_integration=500, n_steps_in_limit_cycle=1000):
        self.field = field
        self.output_label = output_label
        self.phase_plot = phase_plot
        self.L = L
        self.debug = debug
        self.sigma = sigma
        self.min_distance = min_distance
        self.num_of_peaks = num_of_peaks
        self.rtol = rtol
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.n_steps_in_integration = n_steps_in_integration
        self.n_steps_in_limit_cycle = n_steps_in_limit_cycle

        self._initialize_grid()
        self._compute_vector_potential()
        self._compute_hessian()
        self._find_extrema()
        self._classify_critical_points()
        self._compute_region_statistics()

    def _initialize_grid(self):
        """Initialize grid for interpolation."""
        self.x = np.linspace(-self.L / 2.0, self.L / 2.0, self.field.shape[1])
        self.y = np.linspace(-self.L / 2.0, self.L / 2.0, self.field.shape[2])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.cs_coords = {"L_x": 0.1 * self.L, "L_y": 0.5 * self.L}

        self.interp_u = RegularGridInterpolator((self.x, self.y), gaussian(self.field[0], sigma=self.sigma))
        self.interp_v = RegularGridInterpolator((self.x, self.y), gaussian(self.field[1], sigma=self.sigma))

    def _compute_vector_potential(self):
        """Compute vector potential and magnitude."""
        self.dvf = dv.DerivedVars(bcs="11", num_of_dims=2)
        self.a_z = self.dvf.vector_potential(self.field)
        self.mag = self.dvf.vector_magnitude(self.field)

    def _compute_hessian(self):
        """Compute Hessian tensor."""
        self.hessian = self.dvf.gradient_tensor(self.dvf.scalar_gradient(self.a_z))

    def _find_extrema(self):
        """Find local extrema in the field."""
        self.minimum_coords = peak_local_max(-gaussian(self.a_z, sigma=self.sigma), min_distance=self.min_distance)
        self.maximum_coords = peak_local_max(gaussian(self.a_z, sigma=self.sigma), min_distance=self.min_distance)

    def _classify_critical_points(self):
        """Classify critical points as O or X points."""
        self.o_points = []
        self.x_points = []

        for coord in np.vstack([self.minimum_coords, self.maximum_coords]):
            eigs = np.linalg.eigvalsh(self.hessian[:, :, coord[0], coord[1]])
            if eigs[0] * eigs[1] > 0:
                self.o_points.append([coord[0], coord[1]])
            else:
                self.x_points.append([coord[0], coord[1]])

        self.o_points = np.array(self.o_points) * self.L / self.field.shape[1] - self.L / 2.0
        self.x_points = np.array(self.x_points) * self.L / self.field.shape[1] - self.L / 2.0

    def _compute_region_statistics(self):
        """Compute statistics for O-point regions."""
        self.o_point_stats = []
        for x0, y0 in self.o_points:
            stats = self._compute_region_for_point(x0, y0)
            if stats:
                self.o_point_stats.append(stats)

        self._save_statistics()

    def _compute_region_for_point(self, x0, y0):
        """Compute region statistics for a single O-point."""
        largest_cycle_radius = 0
        stats = None

        for shift in np.linspace(self.radius_min, self.radius_max, self.n_steps_in_limit_cycle):
            t, x, y = self._solve_trajectory(x0 + shift, y0 + shift)
            is_periodic_x, _ = self._check_for_limit_cycle(x)
            is_periodic_y, _ = self._check_for_limit_cycle(y)

            if is_periodic_x and is_periodic_y:
                a_max = np.max(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))
                a_min = np.min(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))

                if a_max > largest_cycle_radius:
                    largest_cycle_radius = a_max
                    stats = {"x0": x0, "y0": y0, "a_max": a_max, "a_min": a_min}

        return stats

    def _solve_trajectory(self, x0, y0, t_span=[0, 2]):
        """Solve trajectory for a given initial condition."""
        sol = solve_ivp(self._interp_vector_field, t_span, [x0, y0], t_eval=np.linspace(*t_span, self.n_steps_in_integration))
        return sol.t, sol.y[0], sol.y[1]

    def _interp_vector_field(self, t, z):
        """Interpolate vector field for integration."""
        x, y = z
        if (-self.cs_coords["L_x"] <= x <= self.cs_coords["L_x"]) and (-self.cs_coords["L_y"] <= y <= self.cs_coords["L_y"]):
            return [self.interp_u((x, y)), self.interp_v((x, y))]
        return [0.0, 0.0]

    def _check_for_limit_cycle(self, x):
        """Check if a limit cycle is present in the trajectory."""
        peaks, _ = find_peaks(x)
        if len(peaks) < self.num_of_peaks:
            return False, None

        peak_intervals = np.diff(peaks)
        is_periodic = np.allclose(peak_intervals, peak_intervals[0], rtol=self.rtol)
        return is_periodic, peaks

    def _save_statistics(self):
        """Save O-point statistics to a CSV file."""
        df = pd.DataFrame(self.o_point_stats)
        df.to_csv(f"{self.output_label}_local_statistics.csv", index=False)
