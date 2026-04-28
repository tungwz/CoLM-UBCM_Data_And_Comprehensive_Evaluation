import sys
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def calculate_metrics(observed, predicted, sw):
    # Exclude NaN values
    valid_indices = ~np.isnan(observed) & ~np.isnan(predicted) & ~np.isnan(sw)
    observed = observed[valid_indices]
    predicted = predicted[valid_indices]

    # Calculate correlation coefficient (R)
    correlation_coefficient, _ = pearsonr(observed, predicted)

    # Calculate root mean square error (RMSE)
    rmse = np.sqrt(mean_squared_error(observed, predicted))

    mod_sd = np.std(predicted)
    obs_sd = np.std(observed)
    # Calculate mean bias error (MBE)
    mbe = np.mean(predicted - observed)

    return correlation_coefficient, rmse, mbe, mod_sd, obs_sd

def main():
    # Check if a file path is provided as a command line argument
    # if len(sys.argv) != 2:
        # print("Usage: python script.py <netcdf_file_path>")
        # sys.exit(1)

    # Extract the file path from the command line argument
    # file_path = sys.argv[1]
    # site=['AU-Preston','FR-Capitole','AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','JP-Yoyogi','NL-Amsterdam','PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon']
    site = ['AU-Preston', 'AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole', \
           'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam', \
           'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon', 'US-Baltimore', \
           'US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']
    # site = ['AU-Preston']
    # site = ['FR-Capitole']

    # site=['AU-Preston', 'AU-Preston_solar']
    for i in range(len(site)):
        try:
            # Define the time range for selection
            start_time = "0993-11-28"# 14:00"
            end_time = "3005-11-28"# 13:30"

            # Read the NetCDF file with the specified time range
            # dataset = xr.open_dataset('/stu01/dongwz/point_case/urban_point/LCZ/irr/'+str(file_path)+'_ctl/history/'+str(file_path)+'.nc').sel(time=slice(start_time, end_time))
            dataset = xr.open_dataset(f'/tera12/yuanhua/dongwz/point_new/0110/urb/{site[i]}/history/{site[i]}.nc')#.sel(time=slice(start_time, end_time))
            obs_data = xr.open_dataset(f'/stu01/dongwz/data/inputdata/single_point/obs/v1/{site[i]}_clean_observations_v1.nc')#.sel(time=slice(start_time, end_time))

            print(f'#################{site[i]}##################')
            # SW UP
            #print(obs_data['time'][0].values)
            predicted = dataset['f_sr'][:, 0].values
            observed = obs_data['SWup'][:-1].values
            sw       = obs_data['SWdown'][:-1].values
            sw_      = dataset ['f_xy_solarin'][:,0].values

            sw = np.where(sw_==0, 0, sw)
            #print(len(predicted))
            # print(predicted[predicted==dataset['f_xy_solarin'][:,0].values])
            # valid_indices = (observed > 0) & (predicted > 0) & (predicted < dataset['f_xy_solarin'][:,0].values)
            # observed = observed[valid_indices]
            # predicted = predicted[valid_indices]

            correlation_coefficient, rmse, mbe, mod_sd, obs_sd = calculate_metrics(observed, predicted, sw)
            print(f"----------SW UP------------")
            print(f"Correlation Coefficient (R): {correlation_coefficient:.4f}")
            print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
            print(f"Mean Bias Error (MBE): {mbe:.4f}")
            print(f"Model Stand Error (mod_std): {mod_sd:.4f}")
            print(f"OBS Stand Error (obs_std): {obs_sd:.4f}")
            print(f"---------------------------")

            # LW UP
            #predicted = dataset['f_olrg'][:, 0].values
            #observed = obs_data['LWup'][:-1].values
            #correlation_coefficient, rmse, mbe, mod_sd, obs_sd = calculate_metrics(observed, predicted, sw)
            #print(f"----------LW UP------------")
            #print(f"Correlation Coefficient (R): {correlation_coefficient:.4f}")
            #print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
            #print(f"Mean Bias Error (MBE): {mbe:.4f}")
            #print(f"Model Stand Error (mod_std): {mod_sd:.4f}")
            #print(f"OBS Stand Error (obs_std): {obs_sd:.4f}")
            #print(f"---------------------------")

            # Qh
            #predicted = dataset['f_fsena'][:, 0].values
            #observed = obs_data['Qh'][:-1].values
            #correlation_coefficient, rmse, mbe, mod_sd, obs_sd = calculate_metrics(observed, predicted, sw)
            #print(f"------------Qh-------------")
            #print(f"Correlation Coefficient (R): {correlation_coefficient:.4f}")
            #print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
            #print(f"Mean Bias Error (MBE): {mbe:.4f}")
            #print(f"Model Stand Error (mod_std): {mod_sd:.4f}")
            #print(f"OBS Stand Error (obs_std): {obs_sd:.4f}")
            #print(f"---------------------------")

            # Qle
            #predicted = dataset['f_lfevpa'][:, 0].values
            #observed = obs_data['Qle'][:-1].values
            #correlation_coefficient, rmse, mbe, mod_sd, obs_sd = calculate_metrics(observed, predicted, sw)
            #print(f"------------Qle------------")
            #print(f"Correlation Coefficient (R): {correlation_coefficient:.4f}")
            #print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
            #print(f"Mean Bias Error (MBE): {mbe:.4f}")
            #print(f"Model Stand Error (mod_std): {mod_sd:.4f}")
            #print(f"OBS Stand Error (obs_std): {obs_sd:.4f}")
            #print(f"---------------------------")
            print(f'#########################################')
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

