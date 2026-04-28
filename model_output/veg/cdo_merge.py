import os

label = ['AU-Preston', 'AU-SurreyHills','CA-Sunset','FI-Kumpula','FI-Torni','FR-Capitole', \
         'GR-HECKOR','JP-Yoyogi','KR-Jungnang','KR-Ochang','MX-Escandon','NL-Amsterdam', \
         'PL-Lipowa','PL-Narutowicza','SG-TelokKurau06','UK-KingsCollege','UK-Swindon', 'US-Baltimore', \
         'US-Minneapolis1','US-Minneapolis2','US-WestPhoenix']

file_p = '/tera12/yuanhua/dongwz/point_new/0110/veg/'
for path in label:
    print('Processing Site '+str(path))

    # os.system('mv '+file_p+str(path)+f'/history/{path}.nc '+file_p+str(path)+f'/history/{path}_forc.nc')
    # print(f'mv {file_p}{path}/history/{path}.nc {file_p}{path}/history/{path}_bak.nc')
    os.system(f'rm -f ./{path}/history/{path}.nc')
    #print('cdo mergetime '+file_p+str(path)+'/history/*_hist_*.nc '+file_p+str(path)+'/history/'+str(path)+'.nc')
    #os.system('cdo mergetime '+file_p+str(path)+'/history/*_hist_*.nc '+file_p+str(path)+'/history/'+str(path)+'.nc &')

