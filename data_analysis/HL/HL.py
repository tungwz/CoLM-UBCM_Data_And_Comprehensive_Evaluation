# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scienceplots
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
import xarray as xr
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    # —— 字体族和大小
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Calibri'],  
    'font.size':          55,    # 全局文字（小四）
    'axes.titlesize':     55,    # 子图标题
    'axes.labelsize':     45,    # 坐标轴标签
    'xtick.labelsize':    45,    # 刻度文字
    'ytick.labelsize':    45,
    'legend.fontsize':    32,    # 图例文字

    # —— 坐标轴线和刻度线
    'axes.linewidth':     3,   # 边框线宽 1 pt
    'xtick.major.size':   16,     # 主刻度长度 6 pt
    'xtick.major.width':  3,   # 主刻度线宽 1 pt
    'ytick.major.size':   16,
    'ytick.major.width':  3,
    'xtick.minor.size':   10,     # 次刻度长度 3 pt
    'xtick.minor.width':  2.,   # 次刻度线宽 0.8 pt
    'ytick.minor.size':   10,
    'ytick.minor.width':  2.,
    'xtick.minor.visible': True, # 显示次刻度
    'ytick.minor.visible': True,
    
    # —— 刻度线方向：画到轴内
    'xtick.direction':    'in',
    'ytick.direction':    'in',

    # —— 刻度标签与轴线距离（pt）
    'xtick.major.pad':    15,     # x 轴主刻度标签距轴线距离
    'ytick.major.pad':    15,     # y 轴主刻度标签距轴线距离
    'xtick.minor.pad':    15,     # x 轴次刻度标签距轴线距离
    'ytick.minor.pad':    15,     # y 轴次刻度标签距轴线距离

    # —— 标签和标题的 padding（pt）
    'axes.labelpad':      15,     # 坐标轴标签距轴线距离
    'axes.titlepad':      15,    # 子图标题距顶端边框距离
})

def calculate_metrics(hwr, war, fb, H):

    W = H/hwr
    L1= W*np.sqrt(fb)/(1-np.sqrt(fb))
    L2= 4*H*fb/war
    return L1, L2

info = pd.read_excel("./SiteInfo_HL.xlsx")

# Step 1: Read the NCL RGB file and extract RGB color values
rgb_file = "/tera10/yuanhua/dongwz/github/RF_LAI/plot_Fig/Fig5/colormap/blue_red.rgb"  # Replace with the path to your NCL RGB file
with open(rgb_file, "r") as file:
    lines = file.readlines()

rgb_data = []
for line in lines:
    if line.strip() and not line.startswith("#"):
        rgb_values = line.split()
        rgb_data.append([float(rgb_values[0]), float(rgb_values[1]), float(rgb_values[2])])

# Step 2: Normalize the color values to the range [0, 1]
norm_rgb_data = np.array(rgb_data) / 255.0

# Step 3: Create a custom colormap using the normalized color values
cmap = LinearSegmentedColormap.from_list('custom_colormap', norm_rgb_data, N=len(norm_rgb_data))


fig = plt.figure(figsize=(35, 13),dpi=300)

for ii in range(2):
    ax1 = plt.subplot(1,2,ii+1)
    # 用于累积所有站点的数据
    all_obs = []
    all_ncar = []

    ncar_df = pd.read_csv(f'gfHWR_{ii+1}.csv')
    for i in range(len(info['site'])):
        site_name = info['site'][i]
        print(f'Processing {site_name}')

        # Read the NetCDF file with the specified time range
        site_data = xr.open_dataset('/stu01/dongwz/data/inputdata/single_point/urban_flux/v1/'+str(site_name)+'_metforcing_v1.nc')

        hwr = site_data['canyon_height_width_ratio'].values
        war = site_data['wall_to_plan_area_ratio'].values
        fb  = site_data['roof_area_fraction'].values
        H   = site_data['building_mean_height'].values

        L1, L2 = calculate_metrics(hwr, war, fb, H)

        # ncar_hwr = H//ncar_df['HL'][i]
        ncar_hwr = ncar_df['HL'][i]#.values
        obs_hwr  = ncar_df['OBS_HL'][i]
        # print(ncar_df.columns)
        # print(obs_hwr)
        # all_obs.extend(H/L2)
        if i==5:
            ncar_hwr = np.nan
        all_obs.append(obs_hwr)
        all_ncar.append(ncar_hwr)


        s_size = 400
        color = cm.tab20(np.linspace(0, 1, 21))[i]

        ax1.scatter(obs_hwr, ncar_hwr, s=s_size, c=color, linewidths=0.5, label=f'{site_name}', alpha=0.8)
    
    all_obs = np.array(all_obs).flatten()
    all_ncar = np.array(all_ncar).flatten()
    r1 = np.corrcoef(all_ncar[~np.isnan(all_ncar)], all_obs[~np.isnan(all_ncar)])[0, 1]
    rmse1 = np.sqrt(np.mean((all_ncar[~np.isnan(all_ncar)] - all_obs[~np.isnan(all_ncar)])**2))
    mae1 = np.mean(all_ncar[~np.isnan(all_ncar)] - all_obs[~np.isnan(all_ncar)])

    ax1.text(0.04, 0.97, f'R = {r1:.2f}\nRMSE = {rmse1:.2f}\nMBE = {mae1:.2f}',
            ha='left', va='top', transform=ax1.transAxes)

    if ii==0:
        ax1.set_ylabel("GHSL HL (-)")
        ax1.set_title('(A) GHSL HL vs Site HL', loc='left')
    else:
        ax1.set_ylabel("Global-3D HL (-)")
        ax1.set_title('(B) Global-3D HL vs Site HL', loc='left')
    ax1.set_xlabel("Site HL (-)")

    ax1.set_xlim(-0.09, 1.09)
    ax1.set_ylim(-0.09, 1.09)
    ax1.plot((-0.09, 1.09), (-0.09, 1.09), ls='--', c='black')

    ax1.tick_params(axis='x', which='major', top=False)
    ax1.tick_params(axis='x', which='minor', top=False)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.xaxis.set_major_locator(MultipleLocator(0.2))

    ax1.tick_params(axis='y', which='major', right=False)
    ax1.tick_params(axis='y', which='minor', right=False)
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))

    for location in ['left', 'right', 'top', 'bottom']:
        ax1.spines[location].set_linewidth(3.5)

    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.2))

fig.subplots_adjust(wspace=0.35)
plt.savefig('./compare_HL.pdf', format='pdf', bbox_inches='tight', dpi=300)
