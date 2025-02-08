
# MHD and Particle Data Analysis

This repository contains tools for analyzing MHD (Magnetohydrodynamics) and particle data, including visualization, extremum detection, and distribution analysis. The code is modular and organized into classes for easy reuse and extension.

## **Requirements**

- Python 3.8+
- Required Python packages:
  ```
  numpy
  pandas
  matplotlib
  seaborn
  scipy
  skimage
  PLASMAtools
  cmasher  
  
To run, 

``` python main.py --i 100 --folder data/  ```

## Outputs
Plots:
- Combined E/B ratio and magnetic energy plots.
- -Particle overlays on VTU data.
- Particle distribution histograms.

Statistics:
-CSV files with O-point and X-point statistics.
