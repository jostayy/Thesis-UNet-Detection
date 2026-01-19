"""
Kod bazowy udostępniony przez promotora.
Wykorzystany do wstępnego przetwarzania sygnałów wibrotermograficznych.
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:03:13 2022

@author: PADuser
"""

#%%
# from Signal import Signal as Sig
# from Matrix_Calculating import Matrix_Calculating as MC
# from Matrix_Filter import Matrix_Filter as MFil
# from Matrix_Ploter import Matrix_Ploter as Mplt
# from  Matrix_Analizer import Matrix_Analizer as MA
# import matplotlib.pyplot as plt
# 


# import json

import numpy as np
from Matrix_Calculating import Matrix_Calculating as MC
from  Matrix_Analizer import Matrix_Analizer as MA
# from Matrix_Filter import Matrix_Filter as MFil
# from Matrix_Ploter import Matrix_Ploter as Mplt
# import matplotlib.pyplot as plt
# import numpy as np
# import json

# The following code loads time data, calculates damage indices and then stores resulting data in a form of another set of matrices.
# Finally, it saves them so we would not need to calculate them again
# STARA WERSJA:
# def LoadAndProcess(Baseline,Measurement):
# NOWA WERSJA:
def LoadAndProcess(Baseline, Measurement, t0=1000, t1=5500):
    file_dirs = [Baseline,Measurement]
    time_matrices = []  
    for file in file_dirs:
        mx_obj = MC(file)
        #mx_obj = MC(file,x_dim=122,y_dim=115) #PW szklo 1
        time_matrices.append(mx_obj.get_mx_time())
        
    list_mx_dict = []
    num = 2    
    for mx in time_matrices[1:]:    
        # list_mx_dict.append(['1_' + str(num),MA.mx_dict_corr(time_matrices[0],mx,t0=1000,t1=5500)])
        # STARA WERSJA:
# list_mx_dict.append(['1_' + str(num),MA.mx_dict_corr(time_matrices[0],mx,t0=1000,t1=5500)])
# NOWA WERSJA:
        list_mx_dict.append(['1_' + str(num), MA.mx_dict_corr(time_matrices[0], mx, t0=t0, t1=t1)])
        #jak jest odwrocone wymuszenie to -
        #list_mx_dict.append(['1_' + str(num),MA.mx_dict_corr(-time_matrixes[0],mx,t0=1000,t1=2000)])
        num += 1    
    return list_mx_dict

# Baseline = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11\1_PZT_11_freq_300_kHz_av_60_Vpp_178.svd'
# Measurement = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11\5_PZT_11_freq_300_kHz_av_60_Vpp_178.svd'
# Result = LoadAndProcess(Baseline,Measurement)
# np.save("D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11_1to5_300kHz.npy",Result, allow_pickle = True)

# Baseline = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11\1_PZT_11_freq_200_kHz_av_60_Vpp_180.svd'
# Measurement = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11\5_PZT_11_freq_200_kHz_av_60_Vpp_178.svd'
# Result = LoadAndProcess(Baseline,Measurement)
# np.save("D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11_1to5_200kHz.npy",Result, allow_pickle = True)

# Baseline = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11\1_PZT_11_chirp_100_300_kHz_av_60_Vpp_180.svd'
# Measurement = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11\5_PZT_11_chirp_100_300_kHz_av_60_Vpp_180.svd'
# Result = LoadAndProcess(Baseline,Measurement)
# np.save("D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT11_1to5_200kHz_Chirp.npy",Result, allow_pickle = True) 

# Baseline = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1\7_PZT_1_freq_300_kHz_av_60_Vpp_178.svd'
# Measurement = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1\11_PZT_1_freq_300_kHz_av_60_Vpp_178.svd'
# Result = LoadAndProcess(Baseline,Measurement)
# np.save("D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1_7to11_300kHz.npy",Result, allow_pickle = True)

# Baseline = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1\7_PZT_1_freq_200_kHz_av_60_Vpp_178.svd'
# Measurement = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1\11_PZT_1_freq_200_kHz_av_60_Vpp_178.svd'
# Result = LoadAndProcess(Baseline,Measurement)
# np.save("D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1_7to11_200kHz.npy",Result, allow_pickle = True)

# Baseline = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1\7_PZT_1_chirp_100_300_kHz_av_60_Vpp_180.svd'
# Measurement = r'D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1\11_PZT_1_chirp_100_300_kHz_av_60_Vpp_180.svd'
# Result = LoadAndProcess(Baseline,Measurement)
# np.save("D:\AGH_Archive\__DATASETS\DatasetsTAPSES\VibrometrProbka1\PZT1_7to11_200kHz_Chirp.npy",Result, allow_pickle = True) 










     
 
#%% # The following fragment apparently calculates correlation matrices, it is not necessary right now.



#%%
#time_matrix[i,j,k]
#i,j indeksy przestrzenne
#k indeks czasowy

#snapshot pola propagacji fali dla chwili czasowej o indeksie 2000


## Show of all calculated DIs:
# for tup in list_mx_dict:
#     i = 0 
#     for key in tup[1]:
#         plt.figure()
#         plt.imshow(tup[1][key],cmap='jet',interpolation = 'gaussian')
#         plt.title(str(tup[0])+'_'+str(i)+'_'+str(key)+' mean:' + str(np.round(np.mean(tup[1][key]),4)))
#         ###                               --------------------------- Dlaczego wygląda na to, że wskaźniki nie idą po kolei?

# # Now lets see selected DI maps and their combination:
    
# plt.figure()
# plt.imshow(list_mx_dict[0][1]['diffAmp'])
# plt.savefig('diffAmp.png')    
# plt.imshow(list_mx_dict[0][1]['diffRMS']) 
# plt.savefig('diffRMS.png')
# list_mx_dict[0][1]['summary'] = list_mx_dict[0][1]['divAmp'] +  list_mx_dict[0][1]['diffAmp']  
# plt.imshow(list_mx_dict[0][1]['summary'])
# plt.savefig('summary.png')

# #wykres przebiegu czasowego sygnalu dla wspolrzednnych przestrzennych o indeksach 30, 30
# plt.plot(time_matrixes[0][30,30,:])

# # Mapa wartosci wskaźników uszkodzeń:

    

#%%
#wykres macierzy wskaznikow uszkodzen, np. korelacji
# for tup in list_mx_dict:
#     i = 0
#     for key in tup[1]:
             
#         plt.figure()
#         plt.imshow(tup[1][key], cmap='jet', interpolation='gaussian')
            
#         plt.title(str(tup[0]) +'__' + str(i) +'__' + str(key) + '  max:' +str(np.round(np.max(tup[1][key]),4)) + '  mean' + str(np.round(np.mean(tup[1][key]),4)))
#         i += 1




