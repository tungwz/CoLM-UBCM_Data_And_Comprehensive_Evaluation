import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scienceplots
import xarray as xr
from matplotlib import cm
import sys
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import pandas as pd

simhei = fm.FontProperties(fname="/stu01/dongwz/miniconda3/envs/pyplot/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf", size=45, weight='bold')
simhei1= fm.FontProperties(fname="/stu01/dongwz/miniconda3/envs/pyplot/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
# mpl.rcParams['font.weight'] = 'bold'
plt.style.use(['science','no-latex', 'retro'])
plt.rc('font', family='Arial')

plot_font = 50

if len(sys.argv) != 2:
    print("Usage: python script.py var_type")
    sys.exit(1)

if sys.argv[1] == 'turbulent':
    var_mod   = ['f_fsena', 'f_lfevpa', 'f_fgrnd']
    var_obs   = ['Qh', 'Qle', 'Qg']
else:
    var_mod   = ['f_sr', 'f_olrg', 'f_rnet']
    var_obs   = ['SWup', 'LWup', 'Rnet']

mod_label = ['Slab', 'Urb', 'Urb_Veg', 'CLM5U']
mod_color = ['peru', '#3B5387', '#D94738']
mod_marker= ['o', 'o', 'o', 'o']

def calculate_metrics(observed, predicted):
    valid_indices = ~np.isnan(observed) & ~np.isnan(predicted)
    observed = observed[valid_indices]
    predicted = predicted[valid_indices]

    correlation_coefficient, _ = np.corrcoef(observed, predicted)
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    mbe = np.mean(predicted - observed)

    # Calculate normalized standard deviation
    std_ratio = np.std(predicted) / np.std(observed)

    return correlation_coefficient, rmse, mbe, std_ratio

def plot_taylor_diagram_cesm(axe, ref_data, model_data, label, marker, color, num_i, var_obs):
    axe.set_thetalim(thetamin=0, thetamax=90)
    r_small, r_big, r_interval = 0, 1.5, 0.25
    axe.set_rlim(r_small, r_big)

    rad_ticks = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    axe.set_yticks(rad_ticks)
    axe.set_yticklabels(['0', '0.25', '0.5', '0.75', '1', '1.25', '1.5'], fontsize=plot_font)  # Format labels as needed
    axe.tick_params(axis='y', labelsize=plot_font, pad=20)

    rad_list = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1]

    angle_list = np.rad2deg(np.arccos(rad_list))
    angle_list_rad = np.arccos(rad_list)

    axe.set_thetagrids(angle_list, [], fontsize=plot_font)

    # for label in axe.get_xticklabels():
    #     label.set_rotation(90)

    for angle, rad in zip(angle_list, rad_list):
        if rad == 0:
            rotation = 0
            label_text = '0'
        elif rad == 1:
            rotation = -90
            label_text = '1'
        else:
            rotation = angle - 90
            label_text = f'{rad:.2f}'

        axe.text(np.deg2rad(angle), axe.get_rmax() + 0.11, label_text,
                 rotation=angle,
                 ha='center', va='center',
                 fontsize=plot_font)

    angle_linewidth, angle_length, angle_minor_length = 2.5, 0.02, 0.01
    tick = [axe.get_rmax(), axe.get_rmax() * (1 - angle_length)]
    tick_minor = [axe.get_rmax(), axe.get_rmax() * (1 - angle_minor_length)]

    for t in angle_list_rad:
        axe.plot([t, t], tick, lw=angle_linewidth, color="k")

    for t in angle_list_rad:
        axe.plot([t, t], tick_minor, lw=angle_linewidth, color="k")

    circle = plt.Circle((1, 0), 0.25, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                        linestyle='--', linewidth=2.5)
    axe.add_artist(circle)

    x = 0.095*np.pi
    axe.text(x, 1, s=f'0.25', fontsize=plot_font, ha='center', va='center', color='grey')

    circle1 = plt.Circle((1, 0), 0.5, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                         linestyle='--', linewidth=2.5)
    axe.add_artist(circle1)
    # axe.text(0, 0.5, s=f'0.5', fontsize=plot_font, ha='center', va='top')
    x = 0.175*np.pi
    axe.text(x, 1, s=f'0.50', fontsize=plot_font, ha='center', va='center', color='grey')

    circle2 = plt.Circle((1, 0), 0.75, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                         linestyle='--', linewidth=2.5)
    axe.add_artist(circle2)
    x = 0.269*np.pi
    axe.text(x, 1, s=f'0.75', fontsize=plot_font, ha='center', va='center', color='grey')

    circle3 = plt.Circle((1, 0), 1, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                         linestyle='--', linewidth=2.5)
    axe.add_artist(circle3)
    x = 0.36*np.pi
    axe.text(x, 1, s=f'1.0', fontsize=plot_font, ha='center', va='center', color='grey')

    circle_std = plt.Circle((0, 0), 1, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='k',linestyle='-', linewidth=1)
    axe.add_artist(circle_std)

    axe.text(0, 1, '\u2605', fontfamily='DejaVu Sans',
         fontsize=plot_font+10, ha='center', va='center')
    axe.set_xlabel('Normalized Standard Deviation', labelpad=75, fontsize=plot_font)#, weight='bold')
    axe.text(np.deg2rad(45), 1.65, s='Correlation', fontsize=plot_font, ha='center', va='bottom', rotation=-45)

    sim_start = max(model_data['time'].min(), ref_data['time'].min())
    sim_end = min(model_data['time'].max(), ref_data['time'].max())

    obs_filtered  = ref_data[(ref_data['time'] >= sim_start) & (ref_data['time'] <= sim_end)]
    sim1_filtered = model_data[(model_data['time'] >= sim_start) & (model_data['time'] <= sim_end)]

    if var_obs != 'Rnet' and var_obs != 'Qg':
        obs_valid = obs_filtered.dropna(subset=[f'{var_obs}_obs'])
    elif var_obs == 'Rnet':
        obs_valid = obs_filtered.dropna(subset=['Rn_obs'])
    else:
        obs_valid = obs_filtered.dropna(subset=['Qg_obs'])

    sim1_valid = sim1_filtered[sim1_filtered['time'].isin(obs_valid['time'])]

    if var_obs != 'Rnet' and var_obs != 'Qg':
        ref1_data = obs_valid[f'{var_obs}_obs'].values
        model1_data = sim1_valid[f'{var_obs}_cesmlcz'].values
    elif var_obs == 'Rnet':
        ref1_data = obs_valid[f'Rn_obs'].values
        model1_data = sim1_valid[f'Rn_cesmlcz'].values
    else:
        ref1_data = obs_valid[f'Qg_obs'].values
        model1_data = sim1_valid[f'Qg_cesmlcz'].values

    # model1_data= sim1_valid.values

    if (num_i==11) & (var_obs=='LWup' or var_obs=='Qg' or var_obs=='Rnet'):
        correlation_coefficient, rmse, mbe, std_ratio = np.nan, np.nan, np.nan, np.nan
    else:
        correlation_coefficient, rmse, mbe, std_ratio = calculate_metrics(ref1_data, model1_data)
        print(correlation_coefficient[1], np.arccos(correlation_coefficient[1]), rmse)

    if (num_i==11) & (var_obs=='LWup' or var_obs=='Qg' or var_obs=='Rnet'):
        axe.plot(np.arccos(correlation_coefficient), std_ratio, marker, color=color, markersize=25, label=label, mec='black', mew=2.5)
        # axe.text(np.arccos(correlation_coefficient), std_ratio+0.02, s=str(num_i), fontsize=plot_font)
    else:
        axe.plot(np.arccos(correlation_coefficient[1]), std_ratio, marker, color=color, markersize=25, label=label, mec='black', mew=2.5)
        # axe.text(np.arccos(correlation_coefficient[1]), std_ratio+0.02, s=str(num_i), fontsize=plot_font)

def plot_taylor_diagram(axe, ref_data, model_data, label, marker, color, num_i, var_obs):
    axe.set_thetalim(thetamin=0, thetamax=90)
    r_small, r_big, r_interval = 0, 1.5, 0.25
    axe.set_rlim(r_small, r_big)

    rad_ticks = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    axe.set_yticks(rad_ticks)
    axe.set_yticklabels(['0', '0.25', '0.5', '0.75', '1', '1.25', '1.5'], fontsize=plot_font)  # Format labels as needed
    axe.tick_params(axis='y', labelsize=plot_font, pad=20)

    rad_list = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1]

    angle_list = np.rad2deg(np.arccos(rad_list))
    angle_list_rad = np.arccos(rad_list)

    axe.set_thetagrids(angle_list, [], fontsize=plot_font)

    # for label in axe.get_xticklabels():
    #     label.set_rotation(90)

    for angle, rad in zip(angle_list, rad_list):
        if rad == 0:
            rotation = 0
            label_text = '0'
        elif rad == 1:
            rotation = -90
            label_text = '1'
        else:
            rotation = angle - 90
            label_text = f'{rad:.2f}'

        axe.text(np.deg2rad(angle), axe.get_rmax() + 0.11, label_text,
                 rotation=angle,
                 ha='center', va='center',
                 fontsize=plot_font)

    angle_linewidth, angle_length, angle_minor_length = 2.5, 0.02, 0.01
    tick = [axe.get_rmax(), axe.get_rmax() * (1 - angle_length)]
    tick_minor = [axe.get_rmax(), axe.get_rmax() * (1 - angle_minor_length)]

    for t in angle_list_rad:
        axe.plot([t, t], tick, lw=angle_linewidth, color="k")

    for t in angle_list_rad:
        axe.plot([t, t], tick_minor, lw=angle_linewidth, color="k")

    circle = plt.Circle((1, 0), 0.25, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                        linestyle='--', linewidth=2.5)
    axe.add_artist(circle)

    x = 0.095*np.pi
    axe.text(x, 1, s=f'0.25', fontsize=plot_font, ha='center', va='center', color='grey')

    circle1 = plt.Circle((1, 0), 0.5, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                         linestyle='--', linewidth=2.5)
    axe.add_artist(circle1)
    # axe.text(0, 0.5, s=f'0.5', fontsize=plot_font, ha='center', va='top')
    x = 0.175*np.pi
    axe.text(x, 1, s=f'0.50', fontsize=plot_font, ha='center', va='center', color='grey')

    circle2 = plt.Circle((1, 0), 0.75, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                         linestyle='--', linewidth=2.5)
    axe.add_artist(circle2)
    x = 0.269*np.pi
    axe.text(x, 1, s=f'0.75', fontsize=plot_font, ha='center', va='center', color='grey')

    circle3 = plt.Circle((1, 0), 1, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',
                         linestyle='--', linewidth=2.5)
    axe.add_artist(circle3)
    x = 0.36*np.pi
    axe.text(x, 1, s=f'1.0', fontsize=plot_font, ha='center', va='center', color='grey')

    circle_std = plt.Circle((0, 0), 1, transform=axe.transData._b, facecolor=(0, 0, 0, 0), edgecolor='k',linestyle='-', linewidth=1)
    axe.add_artist(circle_std)

    # axe.text(0, 1,  r'$\star$', color='k', fontsize=plot_font+10,ha='center', va='center')
    axe.text(0, 1, '\u2605', fontfamily='DejaVu Sans',
         fontsize=plot_font+10, ha='center', va='center')
    axe.set_xlabel('Normalized Standard Deviation', labelpad=75, fontsize=plot_font)#, weight='bold')
    axe.text(np.deg2rad(45), 1.65, s='Correlation', fontsize=plot_font, ha='center', va='bottom', rotation=-45)

    sim_start = max(model_data['time'].min(), ref_data['time'].min())
    sim_end = min(model_data['time'].max(), ref_data['time'].max())

    obs_filtered = ref_data.sel(time=slice(sim_start, sim_end))
    sim1_filtered = model_data.sel(time=slice(sim_start, sim_end))

    obs_valid = obs_filtered.dropna(dim='time')

    sim1_valid = sim1_filtered.sel(time=obs_valid['time'])

    if num_i <= 21:
        ref1_data = obs_valid.values
    else:
        ref1_data = obs_valid[:,0,0].values
    model1_data= sim1_valid.values

    if (num_i==11) & (var_obs=='LWup' or var_obs=='Rnet' or var_obs=='Qg'):
        correlation_coefficient, rmse, mbe, std_ratio = np.nan, np.nan, np.nan, np.nan
    else:
        correlation_coefficient, rmse, mbe, std_ratio = calculate_metrics(ref1_data, model1_data)
        print(correlation_coefficient[1], np.arccos(correlation_coefficient[1]), rmse)

    if (num_i==11) & (var_obs=='LWup' or var_obs=='Rnet' or var_obs=='Qg'):
        axe.plot(np.arccos(correlation_coefficient), std_ratio, marker, color=color, markersize=25, label=label, mec='black', mew=2.5)
        # axe.text(np.arccos(correlation_coefficient), std_ratio+0.02, s=str(num_i), fontsize=plot_font)
    else:
        axe.plot(np.arccos(correlation_coefficient[1]), std_ratio, marker, color=color, markersize=25, label=label, mec='black', mew=2.5)
        # axe.text(np.arccos(correlation_coefficient[1]), std_ratio+0.02, s=str(num_i), fontsize=plot_font)

def main():

    label  = ['AU-Preston', 'AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole', \
         'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam', \
         'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon', 'US-Baltimore', \
         'US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

    label_i = 0
    fig = plt.figure(figsize=(58, 58), dpi=300)
    for path in label:
        print('Processing Site '+str(path))

        for imod in range(4):
            marker  = mod_marker[imod]
            mod_name= mod_label [imod]


            if imod == 0:
                file_p = '/tera12/yuanhua/dongwz/point_new/slab/'
            elif imod == 1:
                file_p = '/tera12/yuanhua/dongwz/point_new/0110/urb/'
            elif imod == 2:
                file_p = '/tera12/yuanhua/dongwz/point_new/0110/veg/'
            else:
                file_p = '/tera12/yuanhua/dongwz/github/thsis/section_4/4_2/clm5/output'

            if imod == 3:
                pred_df = pd.read_csv(f'{file_p}/{path}.csv')
            else:
                pred_dataset= xr.open_dataset(file_p+str(path)+'/history/'+str(path)+'.nc')
                ref_dataset = xr.open_dataset('/stu01/dongwz/data/inputdata/single_point/obs/v1/'+str(path)+'_clean_observations_v1.nc')

            for ivar in range(3):
                axe     = plt.subplot(3, 4, imod+ivar*4+1, projection='polar')
                axe.spines[:].set_linewidth(5)
                obs_var = var_obs[ivar]
                mod_var = var_mod[ivar]

                if obs_var=='SWup' or obs_var=='Qh':
                    if imod==0:
                        axe.set_title(r'$\mathregular{(a)~Slab}$', fontsize=60, y=1.15, weight='bold')
                    if imod==1:
                        axe.set_title(r'$\mathregular{(b)~Urb}$', fontsize=60, y=1.15, weight='bold')
                    if imod==2:
                        axe.set_title(r'$\mathregular{(c)~Urb_{Veg}}$', fontsize=60, y=1.15, weight='bold')
                    if imod==3:
                        axe.set_title(r'$\mathregular{(d)~CLM5U}$', fontsize=60, y=1.15, weight='bold')

                if obs_var=='LWup' or obs_var=='Qle':
                    if imod==0:
                        axe.set_title(r'$\mathregular{(e)~Slab}$', fontsize=60, y=1.15, weight='bold')
                    if imod==1:
                        axe.set_title(r'$\mathregular{(f)~Urb}$', fontsize=60, y=1.15, weight='bold')
                    if imod==2:
                        axe.set_title(r'$\mathregular{(g)~Urb_{Veg}}$', fontsize=60, y=1.15, weight='bold')
                    if imod==3:
                        axe.set_title(r'$\mathregular{(h)~CLM5U}$', fontsize=60, y=1.15, weight='bold')

                if obs_var=='Rnet' or obs_var=='Qg':
                    if imod==0:
                        axe.set_title(r'$\mathregular{(i)~Slab}$', fontsize=60, y=1.15, weight='bold')
                    if imod==1:
                        axe.set_title(r'$\mathregular{(j)~Urb}$', fontsize=60, y=1.15, weight='bold')
                    if imod==2:
                        axe.set_title(r'$\mathregular{(k)~Urb_{Veg}}$', fontsize=60, y=1.15, weight='bold')
                    if imod==3:
                        axe.set_title(r'$\mathregular{(l)~CLM5U}$', fontsize=60, y=1.15, weight='bold')

                if imod == 3:
                    if obs_var != 'Rnet' and obs_var != 'Qg':
                        model_data = pred_df[f'{obs_var}_cesmlcz']
                        reference_data = pred_df[f'{obs_var}_obs']
                    else:
                        if obs_var == 'Qg':
                            model_data_ = pd.DataFrame()
                            pred_df['Qg_cesmlcz'] = pred_df[f'Rn_cesmlcz'] - pred_df[f'Qh_cesmlcz'] - pred_df[f'Qle_cesmlcz']
                            model_data_['Qg_cesmlcz'] = pred_df[f'Rn_cesmlcz'] - pred_df[f'Qh_cesmlcz'] - pred_df[f'Qle_cesmlcz']
                            model_data_['time'] = pred_df['time']
                            model_data = model_data_

                            reference_data_ = pd.DataFrame()
                            pred_df['Qg_obs'] = pred_df[f'Rn_obs'] - pred_df[f'Qh_obs'] - pred_df[f'Qle_obs']
                            reference_data_['Qg_obs'] = pred_df[f'Rn_obs'] - pred_df[f'Qh_cesmlcz'] - pred_df[f'Qle_cesmlcz']
                            reference_data_['time'] = pred_df['time']
                            reference_data = reference_data_
                        else:
                            model_data = pred_df[f'Rn_cesmlcz']
                            reference_data = pred_df[f'Rn_obs']
                else:
                    if obs_var!= 'Rnet' and obs_var != 'Qg':
                        reference_data = ref_dataset [obs_var][:]
                        model_data     = pred_dataset[mod_var][:,0]
                    else:
                        if obs_var == 'Qg':
                            model_data = pred_dataset['f_xy_solarin']+pred_dataset['f_xy_frl']-pred_dataset['f_sr']-pred_dataset['f_olrg']-pred_dataset['f_fsena']-pred_dataset['f_lfevpa']

                            SWdown   = xr.where((pred_dataset['f_xy_solarin']==0), 0, ref_dataset['SWdown'][:-1])
                            SWup     = xr.where((pred_dataset['f_xy_solarin']==0), 0, ref_dataset['SWup'][:-1])

                            LWdown   = ref_dataset['LWdown'][:-1]
                            LWup     = ref_dataset['LWup'][:-1]
                            reference_data  = SWdown + LWdown - SWup - LWup - ref_dataset['Qh'][:-1] - ref_dataset['Qle'][:-1]
                        else:
                            model_data = pred_dataset['f_xy_solarin']+pred_dataset['f_xy_frl']-pred_dataset['f_sr']-pred_dataset['f_olrg']

                            SWdown   = xr.where((pred_dataset['f_xy_solarin']==0), 0, ref_dataset['SWdown'][:-1])
                            SWup     = xr.where((pred_dataset['f_xy_solarin']==0), 0, ref_dataset['SWup'][:-1])

                            LWdown   = ref_dataset['LWdown'][:-1]
                            LWup     = ref_dataset['LWup'][:-1]
                            reference_data  = SWdown + LWdown - SWup - LWup

                label_mod = label[label_i]
                if label_i<20:
                    color = cm.tab20(np.linspace(0, 1, 21))[label_i]
                else:
                    color = 'peru'

                if imod != 3:
                    plot_taylor_diagram(axe, reference_data, model_data, label_mod, marker, color, label_i+1, obs_var)
                else:
                    plot_taylor_diagram_cesm(axe, pred_df, pred_df, label_mod, marker, color, label_i+1, obs_var)

        label_i += 1

    handles, labels = axe.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))

    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', fontsize=45, ncol=7, frameon=False, bbox_to_anchor=(0.51, 0.025))

    if sys.argv[1] == 'turbulent':
        fig.suptitle('Sensible Heat',
                y=0.92, fontsize=75, weight='bold')

        fig.text(0.5, 0.64, "Latent Heat",
            ha='center', va='center', fontsize=75, weight='bold')

        fig.text(0.5, 0.36, "Storage Heat",
            ha='center', va='center', fontsize=75, weight='bold')

        fig.subplots_adjust(hspace=0.3)
    else:
        fig.suptitle('Upward Shortwave',
                y=0.92, fontsize=75, weight='bold')

        fig.text(0.5, 0.64, "Upward Longwave",
            ha='center', va='center', fontsize=75, weight='bold')

        fig.text(0.5, 0.36, "Net Radiation",
            ha='center', va='center', fontsize=75, weight='bold')

        fig.subplots_adjust(hspace=0.3)

    if sys.argv[1] == 'turbulent':
        plt.savefig(f'./taylor_diagram_turbulent.jpg', dpi=300)
    else:
        plt.savefig(f'./taylor_diagram_radiation.jpg', dpi=300)

if __name__ == "__main__":
    main()
