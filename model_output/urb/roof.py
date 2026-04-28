import xarray as xr
import numpy as np
import os
import glob
from netCDF4 import Dataset

site = ['AU-Preston', 'AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole', \
         'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam', \
         'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon', 'US-Baltimore', \
         'US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

for i in range(21):
    print(f'Processing {site[i]}')
    base_dir = f'./{site[i]}/restart/const/'
    file_list = glob.glob(os.path.join(base_dir, f"{site[i]}_restart_urb_const_lc2005_*.nc"))

    ds       = xr.open_dataset(file_list[0])
    alb_roof_= ds['ALB_ROOF'][0,0,0].values
    print(alb_roof_)
    df_mod   = xr.open_dataset(f"./{site[i]}/history/{site[i]}.nc",engine='netcdf4')
    df_obs   = xr.open_dataset(f"/stu01/dongwz/data/inputdata/single_point/urban_flux/v1/{site[i]}_metforcing_v1.nc",engine='netcdf4')

    solarln= df_mod['f_solvdln'] + df_mod['f_solviln'] + df_mod['f_solndln'] + df_mod['f_solniln']
    snowf  = df_mod['f_fsno']
    wt_roof= df_obs['roof_area_fraction'][0,0].values
    alb_obs= df_obs['average_albedo_at_midday'][0,0].values

    urb_srs= df_mod['f_srvdln'] + df_mod['f_srviln'] - (alb_roof_)*(df_mod['f_solvdln'] + df_mod['f_solviln'])*wt_roof
    urb_srd= df_mod['f_srndln'] + df_mod['f_srniln'] - (alb_roof_)*(df_mod['f_solndln'] + df_mod['f_solniln'])*wt_roof

    urb_sr = urb_srs + urb_srd

    urb_alb_ = xr.where(snowf==0, urb_sr/solarln, np.nan)

    # print(f'monthly mean canyon albedo {urb_alb_.values}')
    urb_alb = urb_alb_.mean('time')
    # print(f'{site[i]} mean canyon albedo {urb_alb.values}')

    fit_roof= (alb_obs-urb_alb[0].values)/wt_roof

    if fit_roof>=1:
        fit_roof=0.9

    if fit_roof<=0:
        print('ALB_ROOF<0!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(f'{site[i]} fit roof albedo {fit_roof}')
    df_mod.close()
    df_obs.close
    ds.close()

    with Dataset(file_list[0], mode='r+') as nc:
        nc.variables['ALB_ROOF'][0,:,:] = fit_roof

