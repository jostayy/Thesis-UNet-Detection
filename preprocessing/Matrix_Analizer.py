"""
Kod bazowy udostępniony przez promotora.
Wykorzystany do wstępnego przetwarzania sygnałów wibrotermograficznych.
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:31:15 2022

@author: PADuser
"""
import numpy as np
from scipy.signal import *

class Matrix_Analizer():
    
    def matrix_dispersion(mx_fft,delta_k,delta_om):
        num1 = round(mx_fft[:,1,1].size/2)+1-2*delta_k
        num2 = round(mx_fft[1,1,:].size/2)+1-2*delta_om
        mx_dis = np.zeros([num1,num2 ]) 
        mx_fft_shift = np.fft.fftshift(mx_fft)
        a =  round(mx_fft[:,1,1].size/2) + 1 
        b=   round(mx_fft[1,:,1].size/2) + 1              
        for r in range(delta_k,round(mx_fft[:,1,1].size/2)+1-delta_k):
            r1 = r + delta_k
            r2 = r - delta_k
            r_signals = []
            for x in range(a -r1,a - r2):
                for y in range(b -r1,b + r2):
                 
                    if (np.power(x-a,2)+np.power(y-b,2) < r1**2) and (np.power(x-a,2)+np.power(y-b,2) > r2**2):
                        r_signals.append(np.fft.fftshift(mx_fft_shift[x,y,:]))
            for x in range(a -r2,a+r2):
                for y in range(b-r1,b-r2):
                 
                    if (np.power(x-a,2)+np.power(y-b,2) < r1**2) and (np.power(x-a,2)+np.power(y-b,2) > r2**2):
                        r_signals.append(np.fft.fftshift(mx_fft_shift[x,y,:]))
                        
            for x in range(a -r2,a+r2):
                for y in range(b+r1,b+r2):
                 
                    if (np.power(x-a,2)+np.power(y-b,2) < r1**2) and (np.power(x-a,2)+np.power(y-b,2) > r2**2):
                        r_signals.append(np.fft.fftshift(mx_fft_shift[x,y,:]))
                        
            for x in range(a +r2,a +r1):
                for y in range(b -r1,b + r2):
                 
                    if (np.power(x-a,2)+np.power(y-b,2) < r1**2) and (np.power(x-a,2)+np.power(y-b,2) > r2**2):
                        r_signals.append(np.fft.fftshift(mx_fft_shift[x,y,:]))
                        
            for om in range(delta_om,round(mx_fft[1,1,:].size/2)-delta_om ):
                value = 0
                for sig in r_signals:
                    value += np.sum(np.abs(sig[om-delta_om:om+delta_om]))
                value = value/len(r_signals)
                mx_dis[r - delta_k,om - delta_om] = value        
                        
        return mx_dis     

            
    
         
    def define_local_wavenumber_map_bank(mx,k_list,sigma,start_time_index = 0):
          
          map_bank = {'kn': [], 'map': []}
          
          for _kn in k_list:
              
              print(_kn) 
              print('calculating filter matrix')
              filter_matrix = np.zeros([mx[:,1,1].size,mx[1,:,1].size,mx[1,1,:].size]) 
          
              slice_matrix = np.zeros([mx[:,1,1].size,mx[1,:,1].size])
              x_center = mx[:,1,1].size/2
              y_center = mx[1,:,1].size/2
              
              for x in range(0,mx[:,1,1].size):
                  for y in range(0,mx[1,:,1].size):
                      xn = x-x_center
                      yn = y-y_center
                      slice_matrix[x,y] = np.exp(-(((np.sqrt(xn**2+yn**2)-_kn)**2)/(2*sigma)))
                      
              slice_matrix = np.fft.fftshift(slice_matrix)
          
              for k in range(0,mx[1,1,:].size):
                  filter_matrix[:,:,k] = slice_matrix
              
              print('calculating inverse')
              filter_matrix = mx*filter_matrix/np.sum(filter_matrix)
              inv_mx_time = np.fft.ifftn(filter_matrix)
              
              'calculating RMS map'
              RMS_map = np.zeros([mx[:,1,1].size,mx[1,:,1].size])
              
              for x in range(0,mx[:,1,1].size):
                  for y in range(0,mx[1,:,1].size):
                      RMS_map[x,y] = np.sum(np.abs(hilbert(np.real(inv_mx_time)[x,y,:])[start_time_index:]))
              
              map_bank['kn'].append(_kn)
              map_bank['map'].append(RMS_map)
                 
              
          return map_bank
      
        
      
            
    def kc_finder(mx_fft,x_dim,f_dim,f_step,f_limits):
        
        kc = np.ones(mx_fft[1,1,:].size)  
        
        idx = []
        for f in range(f_dim[0],f_dim[1],f_step):
            a = np.abs(mx_fft[x_dim[0]:x_dim[1],0,f:(f+f_step)])       
            idx.append(np.unravel_index(a.argmax(), a.shape))
            idx[-1] = (idx[-1][0] + x_dim[0], idx[-1][1] + f)
            
        y_coords, x_coords = zip(*idx)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords)[0]
        
        kc[:f_limits[0]] = mx_fft[:,1,1].size + 100
        kc[f_limits[0]:f_limits[1]] = np.array(range(f_limits[0],f_limits[1]))*m + c
        kc[f_limits[1]:] = mx_fft[:,1,1].size + 100
        
        return kc
    
    
    
    
    
    
    
    def reverse_matix_list(matrix_list):
        for mx in matrix_list:
            mx =  [np.abs(x) for x in np.fft.ifftn(mx)]
        return matrix_list
    
    
    
    
    def get_wavenumber_map(mx_bank):   
        wavenumber_map = np.zeros([mx_bank['map'][0][:,1].size,mx_bank['map'][0][1,:].size])
        for x in range(0,mx_bank['map'][0][:,1].size):
            for y in range(0,mx_bank['map'][0][1,:].size):
                list_kn = [val for val in mx_bank['kn']]
                list_values = [value[x,y] for value in mx_bank['map']]
                wavenumber_map[x,y]= list_kn[np.argmax(list_values)]
        return wavenumber_map
    
    
    
    
    

    def mx_dict_corr(mx_time1,mx_time2,t0=0,t1=None):
        if not t1:
            t1 = len(mx_time1[1,1,:])
        mx_dict_corr = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size]) 
        mx_dict_envCorr = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size]) 
        mx_dict_sigDiffRMS = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size]) 
        mx_dict_diffRMS = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size]) 
        mx_dict_diffAmp= np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size]) 
        mx_dict_divAmp = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size])
        mx_raw_difference_sum = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size])
        mx_sig_diff_squaredSum = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size])
        mx_ZD_DifferenceSquaredSum = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size])
        mx_ZD_MaxAmp = np.zeros([mx_time1[:,1,1].size,mx_time1[1,:,1].size])
        for x in range(0,mx_time1[:,1,1].size):
            for y in range(0,mx_time2[1,:,1].size):
                    
                
                signal = mx_time1[x,y,t0:t1]
                baseline = mx_time2[x,y,t0:t1]
                
                hilb_signal = hilbert(signal)
                hilb_baseline = hilbert(baseline)
                env_signal = np.absolute(hilb_signal)
                env_baseline = np.absolute(hilb_baseline)
            
                corr = np.corrcoef(signal, baseline)[0, 1]
                mx_dict_corr[x,y]= 1 - corr
                 
                #a jakby tak z hilberta zrobic instantaneous phase i instantaneous amplitude i zrobic podobna chmure jak jest robiona dla
                #Transfer Impedance???? - byloby troche szumow, ale jakby je odfiltrowac to powinny byc fajne chmurki
            
                #print(np.corrcoef(env_signal, env_baseline))    #zwraca macierz, mozna wrzucic do corrcoef wiecej i wtedy wziac np max z kazdego wiersza
                corr = np.corrcoef(env_signal, env_baseline)[0, 1]
                mx_dict_envCorr[x,y] = 1 - corr
                dt = 2e-7
                sig_diff = signal - baseline
                
                # ZD: Raw, not-normalized difference:
                mx_raw_difference_sum[x,y] = np.sum(sig_diff)
                
                mx_ZD_DifferenceSquaredSum[x,y] = np.sum(np.multiply(sig_diff,sig_diff))

                              
                sig_diff_squared = np.multiply(sig_diff, sig_diff)
                sig_diff_squared_integral = np.sum(sig_diff_squared) * dt
                baseline_squared = np.multiply(baseline, baseline)
                baseline_squared_integral = np.sum(baseline_squared) * dt
                mx_dict_sigDiffRMS[x,y] = np.sqrt(sig_diff_squared_integral / baseline_squared_integral)
                
                
                sig_squared = np.multiply(signal, signal)
                sig_squared_integral = np.sum(sig_squared) * dt
                mx_dict_diffRMS[x,y] = np.sqrt(np.absolute((sig_squared_integral - baseline_squared_integral) / baseline_squared_integral))
            
                #Hilbert transform zachowuje norme L2 - trzeba byloby wziac inna norme
                # env_squared = np.multiply(env_signal, env_signal)
                # env_squared_integral = np.sum(env_squared) * dt
                # _measurement.DIs['diffEnvRMS'].append(np.sqrt(np.absolute((env_squared_integral - env_baseline_squared_integral) / env_baseline_squared_integral)))
            
                max_baseline = np.max(baseline)
                max_signal = np.max(signal)
                mx_ZD_MaxAmp[x,y] = max_signal
                mx_dict_diffAmp[x,y] = np.absolute((max_baseline-max_signal)/max_baseline)
                mx_dict_divAmp[x,y] = np.absolute(np.log10(max_baseline/max_signal))
                
                # print(str([x,y]))
        mx_dict = {'corr': mx_dict_corr, 'envCorr': mx_dict_envCorr, 'sigDiffRMS':   mx_dict_sigDiffRMS ,'diffRMS':  mx_dict_diffRMS , 
                   'diffAmp':  mx_dict_diffAmp, 'divAmp': mx_dict_divAmp, 'sigRawDiff': mx_raw_difference_sum, 
                   'ZD_DifferenceSquaredSum': mx_ZD_DifferenceSquaredSum, 'mx_ZD_MaxAmp':mx_ZD_MaxAmp}        
        #mx_dict = [ mx_dict_corr,mx_dict_diffAmp, mx_dict_divAmp]
        return mx_dict


            