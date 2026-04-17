import xarray as xr
import numpy as np
import pandas as pd
import sys
import os
import math

info = pd.read_excel("./SiteInfo_HL.xlsx")

site_name = []

hl_ghsl = []
wt_ghsl = []
ht_ghsl = []

hl_glob3d = []
wt_glob3d = []
ht_glob3d = []

hl_obs = []
ht_obs = []
wt_obs = []

GloBFP_path= '/tera12/yuanhua/dongwz/urban_data/raw_urban/HL/GloBFP/global_nc/500m/n_bld_5x5'
GHSL_path  = '/tera12/yuanhua/dongwz/urban_data/raw_urban/GHSL/roof_height_fraction_GHSL'
Glob3D_path= '/tera12/yuanhua/data/CoLMrawdata/urban_morphology/roof_height_fraction_Li'
site_path  = '/stu01/dongwz/data/inputdata/single_point/urban_flux/v1'

for i in range(len(info['site'])):
    target_lat = info['Lat'][i]
    target_lon = info['Lon'][i]

    site_name.append(info['site'][i])
    if target_lat>=0:
        reg_slat = int(target_lat/5)*5
        reg_elat = int(target_lat/5)*5+5
    else:
        reg_slat = int(target_lat/5)*5-5
        reg_elat = int(target_lat/5)*5

    if target_lon>=0:
        reg_slon = int(target_lon/5)*5
        reg_elon = int(target_lon/5)*5+5
    else:
        reg_slon = int(target_lon/5)*5-5
        reg_elon = int(target_lon/5)*5

    ii     = 0

    reg_mod   = 'RG_'+str(reg_elat)+'_'+str(reg_slon)+'_'+str(reg_slat)+'_'+str(reg_elon)+'.nBLD.nc'

    print(reg_mod)
    if os.path.exists(f'{GloBFP_path}/{str(reg_mod)}'):
        obs_file = xr.open_dataset(f"{site_path}/{info['site'][i]}_metforcing_v1.nc")

        reg_mod     = f'RG_{str(reg_elat)}_{str(reg_slon)}_{str(reg_slat)}_{str(reg_elon)}'
        globfp_file = xr.open_dataset(f'{GloBFP_path}/{str(reg_mod)}.nBLD.nc')
        ghsl_file   = xr.open_dataset(f'{GHSL_path}/2020/{str(reg_mod)}.ROOF500m.GHSL.2020.nc')
        glob3d_file = xr.open_dataset(f'{Glob3D_path}/{str(reg_mod)}.ROOF1km.Li.nc')

        obs_wt = obs_file['roof_area_fraction'][:,:].values
        obs_ht = obs_file['building_mean_height'][:,:].values
        obs_aw = obs_file['wall_to_plan_area_ratio'][:,:].values
        obs_hw = obs_file['canyon_height_width_ratio'][:,:].values

        print(f'lat {target_lat}, lon {target_lon}')
        lat_idx = abs(ghsl_file.lat - target_lat).argmin().item()
        lon_idx = abs(ghsl_file.lon - target_lon).argmin().item()

        lat = ghsl_file['lat'][:].isel(lat=lat_idx).values
        lon = ghsl_file['lon'][:].isel(lon=lon_idx).values

        ghsl_ht = ghsl_file['HT_ROOF'][:,:].isel(lat=lat_idx, lon=lon_idx).values
        ghsl_wt = ghsl_file['PCT_ROOF'][:,:].isel(lat=lat_idx, lon=lon_idx).values

        glob3d_ht = glob3d_file['HT_ROOF'][:,:].isel(lat=lat_idx, lon=lon_idx).values
        glob3d_wt = glob3d_file['PCT_ROOF'][:,:].isel(lat=lat_idx, lon=lon_idx).values

        bld_num = globfp_file['Num_BLD'][:,:].isel(lat=lat_idx, lon=lon_idx).values

        pi = 4.0 * math.atan(1.0)
        deg2rad = pi / 180.0
        re = 6.37122e6 * 0.001

        dx = 1/240*deg2rad
        latn = lat + 1/480
        lats = lat - 1/480
        dy = math.sin(latn * deg2rad) - math.sin(lats * deg2rad)
        area = dx*dy*re*re*1e6

        obs_hl   = obs_ht/np.sqrt(area*obs_wt/bld_num)
        ghsl_hl  = ghsl_ht/np.sqrt(area*ghsl_wt/bld_num)
        glob3d_hl= glob3d_ht/np.sqrt(area*glob3d_wt/bld_num)

        hl_obs.append(obs_hl[0,0])
        ht_obs.append(obs_ht)
        wt_obs.append(obs_wt)

        hl_ghsl.append(ghsl_hl)
        ht_ghsl.append(ghsl_ht)
        wt_ghsl.append(ghsl_wt)

        hl_glob3d.append(glob3d_hl)
        ht_glob3d.append(glob3d_ht)
        wt_glob3d.append(glob3d_wt)


# Create a DataFrame
df = pd.DataFrame({'Site': site_name, 'HL_GHSL': hl_ghsl, 'HL_Glob3D':hl_glob3d, 'HL_obs': hl_obs})

# Save to a CSV file
df.to_csv('HL_sites.csv', index=False)
ii = 0



