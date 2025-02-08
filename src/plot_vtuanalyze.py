# Using fellow graduate student Michael Grehan's BHAC data reader.
"""
BHAC VTU Reader for 2D Data (No General Relativity)

Author: Michael Patrick Grehan 
Date: October 2024
Email: michael.grehan@mail.utoronto.ca

------------------------------------------------------------------------------------
This module provides functionality for reading and processing 2D BHAC VTU data files
that do not include GR effects. The module is designed to read 
VTU files, extract relevant data fields, and return them in a format suitable for 
numerical analysis and visualization.
------------------------------------------------------------------------------------

--------------
Main Features:
--------------
- Efficient reading of VTU files for 2D simulations
- Handles grid and solution data extraction
- Supports loading into NumPy arrays for further manipulation
- Interpolation of field data
- Plotting of vector potential contour lines and arrows
- Plotting of cell boundaries
- Plotting of block boundaries

--------------
Usage Example (interpolation):
--------------
filename = 'data0075.vtu'
data, names = fast_vtu_reader(filename, blocks=True)
Ngrid_x, Ngrid_y = 1024, 2048
grid_x, grid_y, interp_b1 = interpolate_var_to_grid(data, "b1", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y)
_, _, interp_b2 = interpolate_var_to_grid(data, "b2", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y)
_, _, interp_b3 = interpolate_var_to_grid(data, "b3", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y)
_, _, interp_p = interpolate_var_to_grid(data, "p", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y)
B2 = interp_b1**2 + interp_b2**2 + interp_b3**2
_, _, Az_computed = smooth_vect_pot(data, Ngrid_x = Ngrid_x, Ngrid_y = Ngrid_y)

fig, ax = plt.subplots()
p1 = ax.imshow(np.abs(2*interp_p/B2), cmap="hot", origin="lower",
               extent=[data['xpoint'].min(), data['xpoint'].max(), data['ypoint'].min(), data['ypoint'].max()],
               norm=LogNorm(vmax=1e2, vmin=1e-1))
xmin, xmax = -0.1, 0.1
ymin, ymax = -0.2, 0.2
n_levels = 300
contour = ax.contour(grid_x, grid_y, Az_computed, levels=n_levels, colors='w', linewidths=0.5)
for collection in contour.collections:
    for path in collection.get_paths():
        path_data = path.vertices
        x = path_data[:, 0]
        y = path_data[:, 1]
        line, = ax.plot(x, y, color=collection.get_edgecolor(), linewidth=0.5)
        line.set_visible(False)
        add_arrow(line)
ax.set_xlabel('$x/L$')
ax.set_ylabel('$y/L$')
# ax.set_aspect('equal')
cbar = fig.colorbar(p1, ax=ax, pad=0.05,  extend=determine_extend_from_plot(p1), orientation='horizontal',  location='top')
cbar.set_label('$\\beta$')
plot_cells(data, fig=fig, ax=ax, linewidth=0.25, color='w', x_range=(xmin,xmax), y_range=(ymin,ymax))
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin ,ymax)
plt.show()

--------------
Usage Example (raw data plotting):
--------------
filename = 'data0075.vtu'
data, names = fast_vtu_reader(filename, attr={'p', 'b1', 'b2', 'b3'}, blocks=Fale)
fig, ax = plt.subplots()
xmin, xmax = -0.01, -0.007
ymin, ymax = -0.001, 0.001
plot_raw_data_cells(data, 2*data['p']/(data['b1']**2 + data['b2']**2 + data['b3']**2), fig=fig, ax=ax, x_range=(xmin,xmax), y_range=(ymin,ymax), cmap='hot', label='$\\beta$', linewidths=0.1, edgecolors='k', orientation='horizontal',  location='top', use_log_norm=True)
ax.set_xlabel('$x/L$')
ax.set_ylabel('$y/L$')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin ,ymax)
plt.show()

--------------
Libraries Used:
--------------
- struct: For handling binary data.
- numpy: For efficient numerical operations.
- xml.etree.ElementTree: For parsing the VTU XML structure.
- time: For timing the file reading and processing steps.
- base64: For decoding base64-encoded data.
- scipy.integrate, scipy.ndimage, scipy.interpolate: For interpolation and smoothing.
- matplotlib.pyplot: For plotting the data.
"""


import struct
import numpy as np
import xml.etree.ElementTree as ET
import time
import base64
from scipy.integrate import cumtrapz
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection



def fast_vtu_reader(filename, attr='all', blocks=False):
    """
    Reads a VTU file produced by BHAC for 2D simulations.

    Parameters:

    - filename: str, path to the VTU file.
    - attr: list or 'all', attributes to extract (default is 'all').
    - blocks: bool, whether to compute block boundaries for visualization (default is False).

    Returns:
    - data: dict, containing extracted points, attributes, and calculated cell centers.
    - data_keys: list, names of the data arrays present in the file.
    """
    
    print('===============================')
    print(f"Starting to read file: {filename}")
    start_time = time.time()

    with open(filename, 'rb') as f:
        content = f.read()

    appended_data_start = content.find(b'<AppendedData encoding="raw">')
    if appended_data_start == -1:
        raise ValueError("AppendedData section not found")

    data_start = content.find(b'_', appended_data_start) + 1

    xml_content = content[:appended_data_start].decode('utf-8', errors='ignore')
    root = ET.fromstring(xml_content + '</VTKFile>')

    pieces = root.findall('.//Piece')
    num_pieces = len(pieces)
    print(f"Number of Pieces: {num_pieces}")

    cells_per_piece = int(pieces[0].get('NumberOfCells'))
    total_cells = cells_per_piece * num_pieces
    print(f"Cells per piece: {cells_per_piece}")
    print(f"Total number of cells: {total_cells}")

    data = {}
    # Get all unique DataArray names
    data_array_names = set()
    for piece in pieces:
        for data_array in piece.findall('.//DataArray'):
            data_array_names.add(data_array.get('Name'))

    # Read Points (x, y, z coordinates)
    points_data = []
    block_boundaries = []
    for piece in pieces:
        points_data_array = piece.find('.//Points/DataArray')
        if points_data_array is None:
            raise ValueError("Points data not found")

        dtype = points_data_array.get('type')
        format = points_data_array.get('format')

        if format == 'appended':
            offset = int(points_data_array.get('offset', '0'))
            size = struct.unpack('<I', content[data_start+offset:data_start+offset+4])[0]
            raw_data = content[data_start+offset+4:data_start+offset+4+size]
        elif format == 'ascii':
            raw_data = points_data_array.text.strip().split()
        else:  # Assume inline base64
            raw_data = base64.b64decode(points_data_array.text.strip())

        if dtype == 'Float32':
            parsed_data = np.frombuffer(raw_data, dtype=np.float32) if format != 'ascii' else np.array(raw_data, dtype=np.float32)
        elif dtype == 'Float64':
            parsed_data = np.frombuffer(raw_data, dtype=np.float64) if format != 'ascii' else np.array(raw_data, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported data type for Points: {dtype}")
        points_data.append(parsed_data)
        
        if blocks:
            # Reshape the parsed data for this piece
            piece_points = parsed_data.reshape(-1, 3)

            # Vectorized min and max operations for this piece
            x_min, y_min = np.min(piece_points[:, :2], axis=0)
            x_max, y_max = np.max(piece_points[:, :2], axis=0)

            # Define block corners for this piece
            corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min]  # Close the loop
            ])

            # Create block boundaries for this piece
            piece_boundaries = np.array([corners[:-1], corners[1:]]).transpose(1, 0, 2)
            
            block_boundaries.append(piece_boundaries)

    if blocks:
        data['block_coord'] = np.array(block_boundaries)

        

    if points_data:
        points = np.concatenate(points_data).reshape(-1, 3)  # Assuming 3D points (x, y, z)
        data['xpoint'], data['ypoint'], data['zpoint'] = points[:, 0], points[:, 1], points[:, 2]
        print(f"Extracted {len(data['xpoint'])} points")



    # Handle attributes
    if attr == 'all':
        data_array_names.discard(None)
        data_array_names.discard('types')
    else:
        data_array_names = attr
        data_array_names.add('connectivity')
        data_array_names.add('offsets')



    for name in data_array_names:
        combined_data = []

        for piece in pieces:
            piece_data_array = piece.find(f".//DataArray[@Name='{name}']")
            if piece_data_array is None:
                continue

            dtype = piece_data_array.get('type')
            format = piece_data_array.get('format')

            if format == 'appended':
                offset = int(piece_data_array.get('offset', '0'))
                size = struct.unpack('<I', content[data_start+offset:data_start+offset+4])[0]
                raw_data = content[data_start+offset+4:data_start+offset+4+size]
            elif format == 'ascii':
                raw_data = piece_data_array.text.strip().split()
            else:
                raw_data = base64.b64decode(piece_data_array.text.strip())

            if dtype == 'Float32':
                parsed_data = np.frombuffer(raw_data, dtype=np.float32) if format != 'ascii' else np.array(raw_data, dtype=np.float32)
            elif dtype == 'Float64':
                parsed_data = np.frombuffer(raw_data, dtype=np.float64) if format != 'ascii' else np.array(raw_data, dtype=np.float64)
            elif dtype == 'Int32':
                parsed_data = np.frombuffer(raw_data, dtype=np.int32) if format != 'ascii' else np.array(raw_data, dtype=np.int32)
            elif dtype == 'Int64':
                parsed_data = np.frombuffer(raw_data, dtype=np.int64) if format != 'ascii' else np.array(raw_data, dtype=np.int64)
            else:
                raise ValueError(f"Unsupported data type: {dtype}")

            combined_data.append(parsed_data)

        if combined_data:
            data[name] = np.concatenate(combined_data)
            
            
 


    data["ncells"] = total_cells
    data["center_x"], data["center_y"] = calculate_cell_centers(data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished reading file: {filename}")
    print(f"Time taken to read: {elapsed_time:.4f} seconds")
    print('===============================')


    return data, list(data.keys())


def calculate_cell_centers(data):
    """
    Calculate the cell centers by averaging the vertex coordinates.

    Parameters:
    - data: dict, containing vertex coordinates and connectivity information.

    Returns:
    - center_x: ndarray, x-coordinates of cell centers.
    - center_y: ndarray, y-coordinates of cell centers.
    """

    print('===============================')
    print(f"Started finding cell centers")
    start_time = time.time()
    
    x = data['xpoint']
    y = data['ypoint']
    ncells = data['ncells']
    
    offsets = data['offsets']
    connectivity = data['connectivity']
    # Create mod_conn array using broadcasting instead of a for loop
    base_conn = connectivity[:np.max(offsets)]  # Base mod_conn for the first set
    num_iterations = int(4 * ncells / np.max(offsets))  # Number of iterations

    # Use broadcasting to create mod_conn without a loop
    offsets_array = np.arange(num_iterations) * (np.max(base_conn) + 1)  # Calculate all offsets at once
    mod_conn = base_conn + offsets_array[:, None]  # Broadcast and add offsets
    
    # Flatten mod_conn to a 1D array
    mod_conn = mod_conn.ravel()[:ncells * 4]  # Only take enough entries for ncells

    # Reshape mod_conn to group cell vertices (ncells x 4)
    cell_vertices = mod_conn.reshape(ncells, 4)

    # Vectorized calculation of cell centers
    cell_centers_x = np.mean(x[cell_vertices], axis=1)
    cell_centers_y = np.mean(y[cell_vertices], axis=1)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished finding cell centers")
    print(f"Time taken to get centers: {elapsed_time:.4f} seconds")
    print('===============================')

    return cell_centers_x, cell_centers_y


def interpolate_var_to_grid(data, var, Ngrid_x=2048, Ngrid_y=2048, method='nearest', x_range=None, y_range=None):
    """
    Interpolates the specified variable from cell center data onto a uniform 2D grid.

    Parameters:
    - data (dict): A dictionary containing the data, including 'center_x', 'center_y', and the variable to interpolate.
    - var (str): The key in the `data` dictionary corresponding to the variable to be interpolated.
    - Ngrid_x (int): The number of grid points along the x-axis (default is 2048).
    - Ngrid_y (int): The number of grid points along the y-axis (default is 2048).
    - method (str): The interpolation method to use ('nearest', 'linear', or 'cubic'; default is 'nearest').
    - x_range (tuple, optional): A tuple (xmin, xmax) to limit the interpolation to the specified x bounds. If None, no limits are applied.
    - y_range (tuple, optional): A tuple (ymin, ymax) to limit the interpolation to the specified y bounds. If None, no limits are applied.

    Returns:
    - tuple: A tuple containing the grid points in the x-direction, grid points in the y-direction,
              and the interpolated variable on the uniform grid.
    """
    print('===============================')
    print(f"Started interpolating")
    start_time = time.time()

    center_x, center_y = data["center_x"], data["center_y"]

    # Create initial mask for both x and y
    mask = np.ones(center_x.shape, dtype=bool)

    # Apply spatial filtering based on the provided x_range
    if x_range is not None:
        x_mask = (center_x >= x_range[0]) & (center_x <= x_range[1])
        mask &= x_mask  # Combine with the overall mask

    # Apply spatial filtering based on the provided y_range
    if y_range is not None:
        y_mask = (center_y >= y_range[0]) & (center_y <= y_range[1])
        mask &= y_mask  # Combine with the overall mask

    # Filter the center_x, center_y, and variable data based on the combined mask
    filtered_center_x = center_x[mask]
    filtered_center_y = center_y[mask]
    filtered_var_data = data[var][mask]  # Ensure var data is filtered according to the same mask

    # Create a uniform grid based on the range of filtered x and y
    grid_x, grid_y = np.linspace(filtered_center_x.min(), filtered_center_x.max(), Ngrid_x), \
                     np.linspace(filtered_center_y.min(), filtered_center_y.max(), Ngrid_y)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate point data onto the uniform grid
    if method == 'linear':
        # Using LinearNDInterpolator for faster linear interpolation
        interpolator = LinearNDInterpolator((filtered_center_x, filtered_center_y), filtered_var_data)
        interpolated_var = interpolator(grid_x, grid_y)

    else:
        interpolated_var = griddata((filtered_center_x, filtered_center_y), filtered_var_data, (grid_x, grid_y), method=method)

    # Fill NaNs if using linear interpolation
    if method == 'linear':
        interpolated_var = fill_nan(interpolated_var)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished interpolating")
    print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
    print('===============================')

    return grid_x, grid_y, interpolated_var



def interpolate_vect_pot_to_grid(data, Az, Ngrid_x=2048, Ngrid_y=2048, method='nearest', x_range=None, y_range=None):
    """
    Interpolates the specified variable from cell center data onto a uniform 2D grid.

    Parameters:
    - data (dict): A dictionary containing the data, including 'center_x', 'center_y', and the variable to interpolate.
    - var (str): The key in the `data` dictionary corresponding to the variable to be interpolated.
    - Ngrid_x (int): The number of grid points along the x-axis (default is 2048).
    - Ngrid_y (int): The number of grid points along the y-axis (default is 2048).
    - method (str): The interpolation method to use ('nearest', 'linear', or 'cubic'; default is 'nearest').
    - x_range (tuple, optional): A tuple (xmin, xmax) to limit the interpolation to the specified x bounds. If None, no limits are applied.
    - y_range (tuple, optional): A tuple (ymin, ymax) to limit the interpolation to the specified y bounds. If None, no limits are applied.

    Returns:
    - tuple: A tuple containing the grid points in the x-direction, grid points in the y-direction,
              and the interpolated variable on the uniform grid.
    """
    print('===============================')
    print(f"Started interpolating")
    start_time = time.time()

    center_x, center_y = data["center_x"], data["center_y"]

    # Create initial mask for both x and y
    mask = np.ones(center_x.shape, dtype=bool)

    # Apply spatial filtering based on the provided x_range
    if x_range is not None:
        x_mask = (center_x >= x_range[0]) & (center_x <= x_range[1])
        mask &= x_mask  # Combine with the overall mask

    # Apply spatial filtering based on the provided y_range
    if y_range is not None:
        y_mask = (center_y >= y_range[0]) & (center_y <= y_range[1])
        mask &= y_mask  # Combine with the overall mask

    # Filter the center_x, center_y, and variable data based on the combined mask
    filtered_center_x = center_x[mask]
    filtered_center_y = center_y[mask]
    filtered_Az = Az[mask]  # Ensure var data is filtered according to the same mask

    # Create a uniform grid based on the range of filtered x and y
    grid_x, grid_y = np.linspace(filtered_center_x.min(), filtered_center_x.max(), Ngrid_x), \
                     np.linspace(filtered_center_y.min(), filtered_center_y.max(), Ngrid_y)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate point data onto the uniform grid
    if method == 'linear':
        # Using LinearNDInterpolator for faster linear interpolation
        interpolator = LinearNDInterpolator((filtered_center_x, filtered_center_y), filtered_Az)
        interpolated_var = interpolator(grid_x, grid_y)

    else:
        interpolated_var = griddata((filtered_center_x, filtered_center_y), filtered_Az, (grid_x, grid_y), method=method)

    # Fill NaNs if using linear interpolation
    if method == 'linear':
        interpolated_var = fill_nan(interpolated_var)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished interpolating")
    print(f"Time taken to interpolate: {elapsed_time:.4f} seconds")
    print('===============================')

    return grid_x, grid_y, interpolated_var

def fill_nan(grid):
    """
    Fills NaN values in a 2D array using nearest neighbor interpolation.

    Parameters:
    - grid (ndarray): A 2D array with NaN values that need to be filled.

    Returns:
    - ndarray: The input array with NaN values filled.
    """
    nan_mask = np.isnan(grid)
    if np.any(nan_mask):
        # Find the indices of non-NaN values
        x_non_nan, y_non_nan = np.where(~nan_mask)
        non_nan_values = grid[~nan_mask]
        
        # Fill NaN values with nearest neighbor interpolation
        grid[nan_mask] = griddata(
            (x_non_nan, y_non_nan),
            non_nan_values,
            (np.where(nan_mask)[0], np.where(nan_mask)[1]),
            method='nearest'
        )
    return grid

def smooth_vect_pot(data, Ngrid_x = 2048, Ngrid_y = 2048, sigma=5, method='nearest',  x_range=None, y_range=None):
    """
    Interpolates the magnetic fields Bx and By, integrates them to obtain the vector potential Az, 
    and applies Gaussian smoothing to the result.

    Parameters:
    - data (dict): A dictionary containing the magnetic field data with keys 'b1' and 'b2'.
    - Ngrid_x (int): The number of grid points along the x-axis (default is 2048).
    - Ngrid_y (int): The number of grid points along the y-axis (default is 2048).
    - sigma (float): The standard deviation for Gaussian smoothing (default is 5).
    - method (str): The interpolation method to use ('nearest', 'linear', or 'cubic'; default is 'nearest').

    Returns:
    - ndarray: The smoothed vector potential Az.
    """

    grid_x, grid_y, Bx_interp = interpolate_var_to_grid(data, "b1", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y, method=method, x_range=x_range, y_range=y_range)
    _, _, By_interp = interpolate_var_to_grid(data, "b2", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y, method=method, x_range=x_range, y_range=y_range)

    F = cumtrapz(Bx_interp, grid_y, axis=0, initial=0)        
    G = cumtrapz(-By_interp, grid_x, axis=1, initial=0) - F        
    Az_computed = F + G
    # Enforce periodic boundary conditions for Az_computed
    # Az_computed[:, -1] = Az_computed[:, 0]
    # Az_computed[-1, :] = Az_computed[0, :]


    # Apply Gaussian smoothing with a standard deviation (sigma) of your choice
    sigma = sigma  # Adjust this value as needed
    Az_smooth = gaussian_filter(Az_computed, sigma=sigma)

    Az_computed = Az_smooth
    
    
    return grid_x, grid_y, Az_computed
                
def unsmooth_vect_pot(data, Ngrid_x = 2048, Ngrid_y = 2048, method='nearest',  x_range=None, y_range=None):
    """
    Interpolates the magnetic fields Bx and By, integrates them to obtain the vector potential Az 
    without applying any smoothing.

    Parameters:
    - data (dict): A dictionary containing the magnetic field data with keys 'b1' and 'b2'.
    - Ngrid_x (int): The number of grid points along the x-axis (default is 2048).
    - Ngrid_y (int): The number of grid points along the y-axis (default is 2048).
    - method (str): The interpolation method to use ('nearest', 'linear', or 'cubic'; default is 'nearest').

    Returns:
    - ndarray: The unsmoothed vector potential Az.
    """

    grid_x, grid_y, Bx_interp = interpolate_var_to_grid(data, "b1", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y, method=method, x_range=x_range, y_range=y_range)
    _, _, By_interp = interpolate_var_to_grid(data, "b2", Ngrid_x=Ngrid_x, Ngrid_y=Ngrid_y, method=method, x_range=x_range, y_range=y_range)

    F = cumtrapz(Bx_interp, grid_y, axis=0, initial=0)        
    G = cumtrapz(-By_interp, grid_x, axis=1, initial=0) - F        
    Az_computed = F + G
    # Enforce periodic boundary conditions for Az_computed
    # Az_computed[:, -1] = Az_computed[:, 0]
    # Az_computed[-1, :] = Az_computed[0, :]
    
    return grid_x, grid_y, Az_computed

    
def add_arrow(line, position=None, direction='right', size=7, color=None):
    """
    Adds an arrow to a contour plot to indicate direction.

    Parameters:
    - line (Line2D): The line object to which the arrow will be added.
    - position (int, optional): The index of the position on the line to place the arrow (default is the middle).
    - direction (str): The direction of the arrow ('right' for forward, 'left' for backward; default is 'right').
    - size (int): The size of the arrow (default is 7).
    - color (str, optional): The color of the arrow (default is the color of the line).

    Returns:
    - None
    """

    if color is None:
        color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if position is None:
        position = xdata.size // 2
    if direction == 'right':
        dx = xdata[position + 1] - xdata[position]
        dy = ydata[position + 1] - ydata[position]
    else:
        dx = xdata[position - 1] - xdata[position]
        dy = ydata[position - 1] - ydata[position]
    line.axes.annotate('',
                       xytext=(xdata[position], ydata[position]),
                       xy=(xdata[position] + dx, ydata[position] + dy),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size)
    
    
def determine_extend_from_plot(plot_obj):
    """
    Determines the 'extend' parameter for the colorbar based on the data of the plot object.

    Parameters:
    - plot_obj (ScalarMappable): The ScalarMappable object from which to determine the color range.

    Returns:
    - str: A string indicating which end(s) of the colorbar should be extended ('neither', 'both', 'min', 'max').
    """

    norm = plot_obj.norm
    vmin = norm.vmin
    vmax = norm.vmax
    data = plot_obj.get_array()
    
    if np.any(data < vmin) and np.any(data > vmax):
        return 'both'
    elif np.any(data < vmin):
        return 'min'
    elif np.any(data > vmax):
        return 'max'
    else:
        return 'neither'
    
    

    

def format_sci_notation(value):
    """
    Formats a given numerical value into scientific notation for LaTeX.

    Parameters:
    - value (float): The numerical value to format.

    Returns:
    - str: A string representing the formatted value in scientific notation (e.g., "1.0 \times 10^{3}").
    """

    
    coeff, exp = f"{value:.1E}".split("E")
    exp = int(exp)
    if coeff == '1.0':
        return f"10^{{{exp}}}"
    else:
        return f"{coeff} \\times 10^{{{exp}}}"
    
    
def plot_blocks(data, fig=None, ax=None, linewidth=0.1, color='k', 
                          x_range=None, y_range=None):
    """
    Plot block boundaries efficiently using LineCollection with optional spatial range filtering.
    
    Parameters:
    - data: dict containing 'block_coord' numpy array of shape (N, 4, 2, 2)
    - fig: matplotlib figure object (optional)
    - ax: matplotlib axis object (optional)
    - linewidth: width of the boundary lines
    - color: color of the boundary lines
    - x_range: tuple (xmin, xmax) to limit plotted blocks within x bounds (optional)
    - y_range: tuple (ymin, ymax) to limit plotted blocks within y bounds (optional)
    """
    print('===============================')
    print("Started plotting block boundaries")
    start_time = time.time()

    block_coords = data['block_coord']
    N = block_coords.shape[0]  # Number of blocks

    # Prepare segments for LineCollection
    segments = []

    for i in range(N):
        block = block_coords[i]
        
        # Extract the x and y coordinates of the block
        x_block = block[:, 0, 0]  # Shape (4,)
        y_block = block[:, 0, 1]  # Shape (4,)
        
        # Check if the block falls within the provided spatial range (x_range, y_range)
        if x_range is not None:
            if np.all(x_block < x_range[0]) or np.all(x_block > x_range[1]):
                continue  # Skip this block if it lies entirely outside the x_range

        if y_range is not None:
            if np.all(y_block < y_range[0]) or np.all(y_block > y_range[1]):
                continue  # Skip this block if it lies entirely outside the y_range

        # Create segments for each side of the block
        segments.extend([
            [block[0, 0], block[1, 0]],  # Bottom edge
            [block[1, 0], block[2, 0]],  # Right edge
            [block[2, 0], block[3, 0]],  # Top edge
            [block[3, 0], block[0, 0]]   # Left edge
        ])

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('$x/L$')
        ax.set_ylabel('$y/L$')

    # Create and add LineCollection
    lc = LineCollection(segments, linewidths=linewidth, colors=color)
    ax.add_collection(lc)

    # Set plot limits based on provided ranges or data bounds
    if x_range is None:
        ax.set_xlim(block_coords[:, :, 0, 0].min(), block_coords[:, :, 0, 0].max())
    else:
        ax.set_xlim(x_range)
    
    if y_range is None:
        ax.set_ylim(block_coords[:, :, 0, 1].min(), block_coords[:, :, 0, 1].max())
    else:
        ax.set_ylim(y_range)


    


    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Finished plotting block boundaries")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print('===============================')

    return fig, ax


def plot_cells(data, fig=None, ax=None, linewidth=0.1, color='k', 
                             x_range=None, y_range=None):
    """
    Optimized plotting of grid cells using LineCollection with optional spatial range filtering.
    
    Parameters:
    - data: dictionary containing 'xpoint', 'ypoint', 'ncells', 'offsets', 'connectivity'
    - fig: matplotlib figure object (optional)
    - ax: matplotlib axis object (optional)
    - linewidth: width of the boundary lines
    - color: color of the boundary lines
    - x_range: tuple (xmin, xmax) to limit plotted cells within x bounds (optional)
    - y_range: tuple (ymin, ymax) to limit plotted cells within y bounds (optional)
    """
    print('===============================')
    print("Started plotting grid cells")
    start_time = time.time()

    x = data['xpoint']
    y = data['ypoint']
    ncells = data['ncells']
    offsets = data['offsets']
    connectivity = data['connectivity']

    # Create mod_conn array using broadcasting
    base_conn = connectivity[:np.max(offsets)]
    num_iterations = int(4 * ncells / np.max(offsets))
    offsets_array = np.arange(num_iterations) * (np.max(base_conn) + 1)
    mod_conn = (base_conn + offsets_array[:, None]).ravel()[:ncells * 4]
    cell_vertices = mod_conn.reshape(ncells, 4)

    # Extract x and y coordinates for all cells at once
    x_vals = x[cell_vertices]
    y_vals = y[cell_vertices]

    # Apply spatial filtering based on the provided x_range and y_range
    if x_range is not None:
        x_mask = (x_vals.min(axis=1) >= x_range[0]) & (x_vals.max(axis=1) <= x_range[1])
    else:
        x_mask = np.ones(ncells, dtype=bool)

    if y_range is not None:
        y_mask = (y_vals.min(axis=1) >= y_range[0]) & (y_vals.max(axis=1) <= y_range[1])
    else:
        y_mask = np.ones(ncells, dtype=bool)

    # Combine masks to filter cells
    valid_cells = x_mask & y_mask
    x_vals = x_vals[valid_cells]
    y_vals = y_vals[valid_cells]
    filtered_ncells = len(x_vals)

    # Create segments for LineCollection
    segments = []
    for i in range(filtered_ncells):
        # Define the four corners of each cell
        corners = [
            (x_vals[i, 0], y_vals[i, 0]),
            (x_vals[i, 1], y_vals[i, 0]),
            (x_vals[i, 1], y_vals[i, 2]),
            (x_vals[i, 0], y_vals[i, 2]),
            (x_vals[i, 0], y_vals[i, 0])  # Close the loop
        ]
        # Add segments for each side of the cell
        segments.extend([
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]]
        ])

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('$x/L$')
        ax.set_ylabel('$y/L$')

    # Create and add LineCollection
    lc = LineCollection(segments, linewidths=linewidth, colors=color)
    ax.add_collection(lc)

    # Set plot limits
    ax.set_xlim(x_range if x_range is not None else (x.min(), x.max()))
    ax.set_ylim(y_range if y_range is not None else (y.min(), y.max()))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Finished plotting grid cells")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print('===============================')

    return fig, ax



def plot_raw_data_cells(data, field_data, fig=None, ax=None, x_range=None, y_range=None, 
                        vmin=None, vmax=None, cmap='viridis', label=None, 
                        edgecolors=None, linewidths=0.1, orientation='vertical',  
                        location='right', use_log_norm=False, pad=0.1, colorbar=True):
    """
    Plots raw simulation data by filling each cell based on the field value associated with that cell.

    Parameters:
    - data (dict): A dictionary containing the following keys:
        - 'xpoint' (ndarray): 1D array of x-coordinates for cell vertices.
        - 'ypoint' (ndarray): 1D array of y-coordinates for cell vertices.
        - 'ncells' (int): The total number of cells to be plotted.
        - 'offsets' (ndarray): 1D array specifying the starting index of each cell's vertices in the connectivity array.
        - 'connectivity' (ndarray): 1D array that defines the connections between cell vertices.
    - field_data (ndarray): 1D array of field values corresponding to each cell. These values determine the color of each cell.
    - fig (matplotlib.figure.Figure, optional): A Matplotlib figure object. If not provided, a new figure is created.
    - ax (matplotlib.axes.Axes, optional): A Matplotlib axis object. If not provided, a new axis is created.
    - x_range (tuple, optional): A tuple (xmin, xmax) to limit plotted cells within specified x bounds. If None, no limits are applied.
    - y_range (tuple, optional): A tuple (ymin, ymax) to limit plotted cells within specified y bounds. If None, no limits are applied.
    - vmin (float, optional): Minimum value for color mapping. If None, the minimum value from filtered_field_data is used.
    - vmax (float, optional): Maximum value for color mapping. If None, the maximum value from filtered_field_data is used.
    - cmap (str, optional): Colormap used to color the cells. Default is 'viridis'.
    - label (str, optional): Label for the colorbar.
    - edgecolors (str, optional): Color of the edges of the cells. Default is None, which uses the default edge color.
    - linewidths (float, optional): Width of the cell boundaries. Default is 0.1.
    - orientation (str, optional): Orientation of the colorbar. Default is 'vertical'.
    - location (str, optional): Location of the colorbar. Default is 'right'.
    - use_log_norm (bool, optional): If True, applies logarithmic normalization to the field data for color mapping. Default is False.
    - pad: The colorbar padding. Default is 0.1.
    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plot.
    - ax (matplotlib.axes.Axes): The axis object with the plotted data.
    """
    print('===============================')
    print("Started plotting raw data cells")
    start_time = time.time()

    x = data['xpoint']
    y = data['ypoint']
    ncells = data['ncells']
    offsets = data['offsets']
    connectivity = data['connectivity']

    # Create mod_conn array using broadcasting
    base_conn = connectivity[:np.max(offsets)]
    num_iterations = int(4 * ncells / np.max(offsets))
    offsets_array = np.arange(num_iterations) * (np.max(base_conn) + 1)
    mod_conn = (base_conn + offsets_array[:, None]).ravel()[:ncells * 4]
    cell_vertices = mod_conn.reshape(ncells, 4)

    # Extract x and y coordinates for all cells at once
    x_vals = x[cell_vertices]
    y_vals = y[cell_vertices]

    # Apply spatial filtering based on the provided x_range and y_range
    if x_range is not None:
        x_mask = (x_vals.min(axis=1) >= x_range[0]) & (x_vals.max(axis=1) <= x_range[1])
    else:
        x_mask = np.ones(ncells, dtype=bool)

    if y_range is not None:
        y_mask = (y_vals.min(axis=1) >= y_range[0]) & (y_vals.max(axis=1) <= y_range[1])
    else:
        y_mask = np.ones(ncells, dtype=bool)


    # Combine masks to filter cells
    valid_cells = x_mask & y_mask
    x_vals = x_vals[valid_cells]
    y_vals = y_vals[valid_cells]
    filtered_field_data = field_data[valid_cells]
    filtered_ncells = len(x_vals)

    # Create polygons for PolyCollection (one polygon per cell)
    polygons = []
    for i in range(filtered_ncells):
        polygon = [
            (x_vals[i, 0], y_vals[i, 0]),
            (x_vals[i, 1], y_vals[i, 0]),
            (x_vals[i, 1], y_vals[i, 2]),
            (x_vals[i, 0], y_vals[i, 2])
        ]
        polygons.append(polygon)

    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('$x/L$')
        ax.set_ylabel('$y/L$')

    if vmin == None:
        vmin = np.min(filtered_field_data)
        
    if vmax == None:
        vmax = np.max(filtered_field_data)
        
    if use_log_norm:
        # Create a log normalization instance
        norm = LogNorm(vmin=vmin, 
                   vmax=vmax)
        poly_collection = PolyCollection(polygons, norm=norm, array=filtered_field_data, 
                                         cmap=cmap, edgecolors=edgecolors, 
                                         clim=(vmin,vmax), linewidths=linewidths)

    else:
        # Create a PolyCollection with the polygons and color them by field data
        poly_collection = PolyCollection(polygons, array=filtered_field_data, 
                                         cmap=cmap, edgecolors=edgecolors, 
                                         clim=(vmin,vmax), linewidths=linewidths)
    ax.add_collection(poly_collection)
    


    # Add colorbar with 'extend' parameter determined from the data
    if colorbar:
        # Determine extend based on comparisons
        extend_type = 'neither'  # Default
        if np.any(filtered_field_data < vmin):
            extend_type = 'min'
        if np.any(filtered_field_data > vmax):
            extend_type = 'max'
        if np.any(filtered_field_data < vmin) and np.any(filtered_field_data > vmax):
            extend_type = 'both'
        cbar = plt.colorbar(poly_collection, ax=ax, extend=extend_type, 
                            label=label, orientation=orientation, location=location, 
                            pad=pad)

    # Set plot limits
    # ax.set_xlim(x_range if x_range is not None else (x.min(), x.max()))
    # ax.set_ylim(y_range if y_range is not None else (y.min(), y.max()))
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())


    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Finished plotting raw data cells")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print('===============================')

    return fig, ax



def extract_1d_slice_2d(axis, slice_value, cell_centers_x, cell_centers_y, *variables):
    """
    Extract a 1D slice of the data from a 2D grid, along a specified axis.

    Args:
        axis: the axis to slice along (0 for x, 1 for y).
        slice_value: the value of the axis where the slice should be taken.
        cell_centers_x: the x-coordinates of the cell centers (1D array).
        cell_centers_y: the y-coordinates of the cell centers (1D array).
        variables: any number of cell-centered data arrays to slice (e.g., e1, e2, b1, etc.).
    
    Returns:
        The coordinates along the slice and the sliced variables.
    """
    # Choose the axis for slicing (0 = x-axis, 1 = y-axis)
    if axis == 0:  # Slice along x (i.e., fixed x = slice_value)
        slice_indices = np.isclose(cell_centers_x, slice_value, atol=1e-2)
        slice_coord = cell_centers_y[slice_indices]
    elif axis == 1:  # Slice along y (i.e., fixed y = slice_value)
        print(cell_centers_y)
        slice_indices = np.isclose(cell_centers_y, slice_value, atol=1e-2)
        slice_coord = cell_centers_x[slice_indices]
    else:
        raise ValueError("Invalid axis. Use 0 for x and 1 for y.")
    
    if not np.any(slice_indices):
        raise ValueError(f"No data found for slice at {slice_value} along axis {axis}.")
    
    # Extract the slice from each variable
    sliced_vars = [var[slice_indices] for var in variables]

    return slice_coord, sliced_vars

