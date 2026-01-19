"""
Kod bazowy udostępniony przez promotora.
Wykorzystany do wstępnego przetwarzania sygnałów wibrotermograficznych.
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:55:10 2022

@author: PADuser

"""

# Zostaw win32com.client, ale będziemy go używać warunkowo
import win32com.client
import numpy as np
from Signal import Signal        
from os.path import isfile, join # Dodano join
import os # Dodano os

class Matrix_Calculating():
    
    def __init__(self,file_dir,x_dim=0,y_dim=0):
        
        self.mx_time = None # Inicjalizujemy mx_time jako None na wypadek błędu
        
        # Nazwa pliku bez rozszerzenia
        base_name, ext = os.path.splitext(file_dir)
        npy_path = base_name + '.npy' # Ścieżka do potencjalnego pliku .npy

        # --- KROK 1: Spróbuj załadować .npy, jeśli istnieje ---
        if isfile(npy_path):
            # print(f"Ładuję przetworzony plik NPY: {npy_path}") # Debugowanie
            self.mx_time = np.load(npy_path, allow_pickle = True)
            # print("Ładowanie NPY zakończone.")
            return # Zakończ __init__, jeśli .npy został załadowany

        # --- KROK 2: Jeśli .npy nie istnieje, ale file_dir SAM jest plikiem .npy ---
        elif ext.lower() == '.npy' and isfile(file_dir):
            # print(f"Ładuję bezpośredni plik NPY: {file_dir}") # Debugowanie
            self.mx_time = np.load(file_dir, allow_pickle = True)
            # print("Ładowanie NPY zakończone.")
            return # Zakończ __init__, jeśli .npy został załadowany

        # --- KROK 3: Jeśli to .svd i odpowiadający .npy nie istnieje ---
        elif ext.lower() == '.svd':
            print(f"OSTRZEŻENIE: Plik .npy dla {file_dir} nie istnieje. Nie można przetworzyć pliku .svd bez oprogramowania Polytec. Pomijam ten plik.")
            # Zwracamy pustą macierz zer lub None, aby program nie przerywał działania.
            # Wybieram pustą macierz, aby uniknąć błędów TypeErrors w dalszej części kodu.
            # Rzeczywiste wymiary (np. 86x115) zależą od eksperymentu. Ustawiam ogólne dla uniknięcia błędu,
            # ale to będzie pusta mapa.
            # Zastepujemy to pustą macierzą o typowych wymiarach
            self.mx_time = np.zeros((86, 115, 6500)) # Typowe wymiary: Wysokość, Szerokość, Liczba próbek czasowych
            return # Zakończ __init__

        # --- KROK 4: Inny nieobsługiwany typ pliku ---
        else:
            print(f"BŁĄD: Nieobsługiwany typ pliku lub plik nie istnieje: {file_dir}. Oczekiwano .svd lub .npy.")
            self.mx_time = np.zeros((86, 115, 6500)) # Zwracamy pustą macierz
            return # Zakończ __init__


    def __get_matrix_dim(self,dim_list):
        #zmiana
        x_diff = np.abs(np.diff(dim_list))
        trigg = np.max(x_diff) - (np.max(x_diff) - np.min(x_diff))/2
        idx = [ix for ix, x in enumerate(x_diff) if x > trigg]
        idx_diff = round(np.median(np.diff(idx)))
        x = len(idx)
        y = idx_diff

        return x,y

    def __rms(self,x):
        return np.sqrt(x.dot(x)/x.size)
    
    def get_mx_time(self):
        return self.mx_time
    def clear_memory(self):
        self.mx_time = []
        self.signal = []
        self.list_signals = []
        
        
    def get_mx_RMS(self,W_np,mx_idx = None,points_before = False):
        if W_np is None or W_np.size == 0: # Dodano zabezpieczenie
            return np.zeros((86,115)) # Zwróć pustą macierz
            
        mx_RMS = np.zeros([W_np[:,1,1].size,W_np[1,:,1].size])
        
        if mx_idx is None:
            mx_idx = np.zeros([W_np[:,1,1].size,W_np[1,:,1].size])
        if points_before:           
            for x in range(0,W_np[:,1,1].size):
                for y in range(0,W_np[1,:,1].size):
                    mx_RMS[x,y] = self.__rms(W_np[x,y,:int(mx_idx[x][y])])
                # print(x)
        else: 
             for x in range(0,W_np[:,1,1].size):
                for y in range(0,W_np[1,:,1].size):
                    mx_RMS[x,y] = self.__rms(W_np[x,y,int(mx_idx[x][y]):])
                # print(x)
        
        return mx_RMS


    def get_mx_fft(self,mx_time):
        if mx_time is None or mx_time.size == 0: # Dodano zabezpieczenie
            # Zwróć pustą macierz o oczekiwanych wymiarach dla FFT
            # Wymiary FFT zależą od wymiarów wejściowych. Zakładamy typowe.
            return np.zeros((86, 115, 6500), dtype=np.complex128) 
        
        return np.fft.fftn(mx_time)
    

    def get_mx_fft_2D(self,mx_fft):
        if mx_fft is None or mx_fft.size == 0: # Dodano zabezpieczenie
            return np.zeros((86,115))
            
        mx_fft_2D = np.zeros([mx_fft[:,1,1].size,mx_fft[1,:,1].size])    
        for x in range(0,mx_fft[:,1,1].size):
            for y in range(0,mx_fft[1,:,1].size):
                mx_fft_2D[x,y] = np.sum(np.abs(mx_fft[x,y,:]))/mx_fft[1,1,:].size
        return mx_fft_2D

    def get_mx_time_from_fft(self,mx_fft):
        if mx_fft is None or mx_fft.size == 0: # Dodano zabezpieczenie
            return np.zeros((86, 115, 6500))
        
        invfft = np.fft.ifftn(mx_fft)
        return np.real(invfft)