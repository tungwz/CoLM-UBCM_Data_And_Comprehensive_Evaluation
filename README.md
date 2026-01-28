# A New 3D Urban Building Community Model for Earth System Modeling: Data and Comprehensive Evaluation


## Overview
This repository contains the plot code of the evaluation CoLM-UBCM, supporting the manuscript:
> "A New 3D Urban Building Community Model for Earth System Modeling: Data and Comprehensive Evaluation" submitted to _Journal of Advances in Modeling Earth Systems_. A related dataset is open access at: .

## Code Organization

The code is organized into three main functional directories:

1. **data_analysis** (`data_analysis/`)
   Verification and comparison of high-resolution urban canopy data.

2. **site_analysis** (`site_analysis/`)
   Verification of CoLM-UBCM in flux towers simulation.

3. **global_analysis** (`global_analysis/`)
   Global and regional validation of CoLM-UBCM.


## Directory Structure
```bash
├── data_analysis
│   ├── alb_roof
│   │   ├── Fit_vs_LUT.py                # LCZ roof albedo vs Fit roof albedo
│   │   ├── Global_FIT_ALB_ROOF_0.5.nc   # Global 0p5 deg fit roof albedo
│   │   ├── Global_NCAR_ALB_ROOF_0.5.nc  # Global 0p5 deg NCAR roof albedo
|   |   ├── LCZ_vs_NCAR.py               # LCZ roof albedo vs urban density roof albedo
│   │   └── NCAR_urban_properties.nc     # Urban density prescribed properties
│   ├── GFCC_ETH
│   │   ├── HTOP.csv                     # Tree height and percentage of Urban-Plumber2 sites form global data
│   │   ├── Scatter_ETH.py               # Global tree height and percentage vs site obs
│   │   └── SiteInfo.xlsx                # Site information
│   └── LUT_vs_Grid
│       ├── LCZ.txt                      # Urban morphology from LCZ LUT
│       ├── NCAR.txt                     # Urban morphology from urban density LUT
│       ├── OBS.txt                      # Urban morphology from site obs
│       └── UCPs_plot.py             
├── global_analysis
│   ├── AHF
│   │   └── plot_AHE.py                  
│   └── OpenBench
│       └── albeo_le_plot.py
└── site_analysis
    ├── 21_sites
    │   └── 21_sites_box_plot.py
    └── taylor_plot
        └── taylor.py
```
<br>





