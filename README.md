# The New Urban Building Community Model for Earth System Modeling: Data and Comprehensive Evaluation

Data, model output, and analysis notebooks for the comprehensive evaluation of the Common Land Model with Urban Building Community Model (CoLM-UBCM). This repository supports the manuscript:

> "The New Urban Building Community Model for Earth System Modeling: Data and Comprehensive Evaluation", submitted to _Journal of Advances in Modeling Earth Systems_.

The repository is data-heavy because it includes NetCDF model output, site observations, global gridded data products, and precomputed evaluation metrics.

## What Is Included

- CoLM-UBCM point-site simulation output for 21 urban flux-tower sites.
- Four site experiment groups: `slab`, `urb`, `veg`, and `ucps`.
- CLM5 comparison data in site-level CSV files.
- Cleaned Urban-PLUMBER2-style observation files and auxiliary global datasets.
- Notebooks for high-resolution urban canopy data evaluation, site-scale flux evaluation, Taylor diagrams, and global/regional comparisons.
- Precomputed metrics used by the plotting notebooks.

## Repository Structure

```text
.
├── data_analysis/
│   ├── alb_roof/
│   │   ├── Global_FIT_ALB_ROOF_0.5.nc
│   │   ├── Global_NCAR_ALB_ROOF_0.5.nc
│   │   ├── Global_USrf_ALB_ROOF_0.5.nc
│   │   ├── LCZ_vs_NCAR.ipynb
│   │   ├── NCAR_urban_properties.nc
│   │   └── Plot_alb_roof.ipynb
│   ├── GFCC_ETH/
│   │   ├── HTOP.csv
│   │   ├── Scatter_GFCC_ETH.ipynb
│   │   └── exact_eth_gfcc.ipynb
│   ├── HL/
│   │   ├── Plot_HL.ipynb
│   │   ├── data_for_HL_plot.csv
│   │   └── site_hl_csv.ipynb
│   ├── LUT_vs_Grid/
│   │   ├── UCPs_data.csv
│   │   ├── UCPs_extract.ipynb
│   │   └── UCPs_plot.ipynb
│   └── SiteInfo.csv
├── global_analysis/
│   ├── AHF/
│   │   └── plot_AHE.ipynb
│   └── OpenBench/
│       └── openbench_plot.ipynb
├── model_output/
│   ├── clm5/
│   ├── slab/
│   ├── urb/
│   ├── veg/
│   └── ucps/
├── obs/
│   ├── *_clean_observations_v1.nc
│   ├── AH4GUC_2010_monmean.nc
│   └── Flanner_2005.nc
├── site_analysis/
│   ├── 21_sites/
│   │   ├── 21_sites_box_plot.ipynb
│   │   ├── 21_sites_calculate_metrics.ipynb
│   │   └── 21_sites_metrics.csv
│   └── taylor_plot/
│       ├── taylor_metrics.csv
│       ├── taylor_metrics.ipynb
│       └── taylor_plot.ipynb
└── README.md
```

Each `model_output/{experiment}/{site}/` directory generally contains:

- `history/`: monthly CoLM history output files named `{site}_hist_{YYYY-MM}.nc`
- `restart/`: restart files used by the corresponding run
- `landdata/`: site surface data such as `srfdata.nc`

The `model_output/{experiment}/nml/` directories contain the namelists used for the point-site simulations.

## Sites

The site simulations cover 21 urban flux-tower sites:

| Region | Sites |
| --- | --- |
| Australia | `AU-Preston`, `AU-SurreyHills` |
| Canada | `CA-Sunset` |
| Finland | `FI-Kumpula`, `FI-Torni` |
| France | `FR-Capitole` |
| Greece | `GR-HECKOR` |
| Japan | `JP-Yoyogi` |
| Korea | `KR-Jungnang`, `KR-Ochang` |
| Mexico | `MX-Escandon` |
| Netherlands | `NL-Amsterdam` |
| Poland | `PL-Lipowa`, `PL-Narutowicza` |
| Singapore | `SG-TelokKurau06` |
| United Kingdom | `UK-KingsCollege`, `UK-Swindon` |
| United States | `US-Baltimore`, `US-Minneapolis1`, `US-Minneapolis2`, `US-WestPhoenix` |

Site metadata are stored in `data_analysis/SiteInfo.csv`. Cleaned observation files are stored in `obs/`.

## Model Configurations

| Directory | Short label | Description |
| --- | --- | --- |
| `model_output/slab` | `Slab` | Traditional slab urban parameterization |
| `model_output/urb` | `Urb` | CoLM urban configuration without the urban tree component |
| `model_output/veg` | `Urb_Veg` | CoLM-UBCM with urban vegetation and related urban energy/water processes |
| `model_output/ucps` | `UCPs` | CoLM-UBCM using urban canopy parameters derived from the UCP workflow |
| `model_output/clm5` | `CLM5` | CLM5 comparison data stored as site-level CSV files |

The exact run settings are recorded in the corresponding `Site_*.nml` files.

## Evaluated Variables

The site notebooks compare radiation and turbulent/storage heat fluxes against observations.

| Diagnostic | Typical CoLM variable | Unit |
| --- | --- | --- |
| `SWup` | `f_sr` | W m-2 |
| `LWup` | `f_olrg` | W m-2 |
| `Rnet` | `f_rnet` | W m-2 |
| `Qh` | `f_fsena` | W m-2 |
| `Qle` | `f_lfevpa` | W m-2 |
| `Qg` | `f_fgrnd` | W m-2 |

Precomputed site metrics include correlation (`R`), root-mean-square error (`RMSE`), mean absolute error (`MAE`), mean bias error (`MBE`), and standard-deviation ratios where applicable.

## Analysis Notebooks

### `data_analysis/alb_roof/`

Evaluates global roof albedo datasets. `Plot_alb_roof.ipynb` compares fitted, NCAR, and USRF roof albedo products, while `LCZ_vs_NCAR.ipynb` compares LCZ-based and NCAR urban-density-based estimates.

### `data_analysis/GFCC_ETH/`

Extracts and evaluates global tree height and tree cover information against site-scale information. `HTOP.csv` stores the processed tree height and percentage values used by the scatter plots.

### `data_analysis/HL/`

Processes and plots urban building height and characteristic length-scale diagnostics. `data_for_HL_plot.csv` stores the prepared plotting table.

### `data_analysis/LUT_vs_Grid/`

Compares urban canopy parameters derived from lookup-table assumptions with gridded/site values. `UCPs_extract.ipynb` prepares `UCPs_data.csv`, and `UCPs_plot.ipynb` generates the comparison figures.

### `site_analysis/21_sites/`

Calculates and visualizes model performance across all 21 sites. `21_sites_calculate_metrics.ipynb` writes `21_sites_metrics.csv`, and `21_sites_box_plot.ipynb` creates summary box plots for the main model configurations.

### `site_analysis/taylor_plot/`

Computes and plots Taylor-diagram metrics for the site simulations. `taylor_metrics.ipynb` prepares `taylor_metrics.csv`, and `taylor_plot.ipynb` generates the Taylor diagrams.

### `global_analysis/AHF/`

Plots global or regional anthropogenic heat flux diagnostics using the auxiliary observation/global datasets in `obs/`.

### `global_analysis/OpenBench/`

Creates OpenBench-style global comparison plots for CoLM-UBCM evaluation.

## Environment

Recommended Python packages:

```text
python >= 3.8
jupyter
numpy
pandas
xarray
netCDF4
matplotlib
seaborn
scipy
```

Optional but useful for larger NetCDF workloads:

```text
dask
h5netcdf
cartopy
```

## Usage

Clone the repository:

```bash
git clone git@github.com:tungwz/CoLM-UBCM_Data_And_Comprehensive_Evaluation.git
cd CoLM-UBCM_Data_And_Comprehensive_Evaluation
```

Create an environment and install dependencies, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas xarray netCDF4 matplotlib seaborn scipy
```

Run notebooks from the repository root or from their own directories, depending on the relative paths used in each notebook. For example:

```bash
jupyter notebook data_analysis/alb_roof/Plot_alb_roof.ipynb
jupyter notebook site_analysis/21_sites/21_sites_calculate_metrics.ipynb
jupyter notebook site_analysis/taylor_plot/taylor_plot.ipynb
jupyter notebook global_analysis/OpenBench/openbench_plot.ipynb
```

If a notebook uses local absolute paths from the original analysis environment, update those path variables before running it.

## Notes For Reuse

- NetCDF model output is organized by experiment and site; use the checked-in namelists to identify the corresponding model settings.
- The notebooks assume the variable names used by the stored CoLM history files and cleaned observation files. If using newer CoLM output, update the variable mapping blocks before recalculating metrics.
- Precomputed CSV files are included so plotting notebooks can be rerun without recomputing every intermediate metric.
- Generated figures are analysis products and may not all be tracked in this repository.

## References

- Urban-PLUMBER2: https://urban-plumber.github.io/
- CoLM: https://github.com/CoLM-SYSU/CoLM202X
