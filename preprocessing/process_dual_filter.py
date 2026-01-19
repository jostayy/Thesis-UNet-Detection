"""
Skrypt autorski (na bazie bibliotek promotora) do przygotowania zbioru danych dla sieci neuronowej.
Autor: Joanna Szczypkowska

Opis:
Skrypt przetwarza pary sygnałów (referencja-pomiar), oblicza mapy cech,
a następnie aplikuje filtry odszumiające (Medianowy oraz Gaussa) w celu
generacji dwóch wariantów zbiorów danych.
"""

import os
import numpy as np
import cv2
import config
from ZD_AutomaticProcesser_v01 import LoadAndProcess

# ==========================================
# KONFIGURACJA EKSPERYMENTU
# ==========================================
EXPERIMENT = 2
WINDOW = [1300, 1801]
EXCITATION_FREQ = '300'
FEATURE_VECTOR = ['envCorr', 'diffAmp', 'sigDiffRMS']

# Parametry filtracji
FILTER_SIZE_MEDIAN = 3
FILTER_SIZE_GAUSS = (3, 3)

# Ścieżki wyjściowe
BASE_OUTPUT_DIR = os.path.dirname(config.PROCESSED_DATA_PATH_EXP2)
OUTPUT_PATH_MEDIAN = os.path.join(BASE_OUTPUT_DIR, f"ProcessedExp2_Filtered_Median_W{WINDOW[0]}_{WINDOW[1]}")
OUTPUT_PATH_GAUSS = os.path.join(BASE_OUTPUT_DIR, f"ProcessedExp2_Filtered_Gauss_W{WINDOW[0]}_{WINDOW[1]}")

os.makedirs(OUTPUT_PATH_MEDIAN, exist_ok=True)
os.makedirs(OUTPUT_PATH_GAUSS, exist_ok=True)

# Definicja par pomiarowych (Baseline vs Measurement)
BASELINE_MAP = {
    'state_15_vs_14': {'base': 14, 'meas': 15, 'dmg_id': 3},  'state_12_vs_11': {'base': 11, 'meas': 12, 'dmg_id': 3},
    'state_16_vs_14': {'base': 14, 'meas': 16, 'dmg_id': 3},  'state_16_vs_15': {'base': 15, 'meas': 16, 'dmg_id': 3},
    'state_4_vs_3':   {'base': 3,  'meas': 4,  'dmg_id': 5},  'state_5_vs_4':   {'base': 4,  'meas': 5,  'dmg_id': 5},
    'state_5_vs_3':   {'base': 3,  'meas': 5,  'dmg_id': 5},
    'state_6_vs_5':   {'base': 5,  'meas': 6,  'dmg_id': 7},  'state_7_vs_6':   {'base': 6,  'meas': 7,  'dmg_id': 7},
    'state_7_vs_5':   {'base': 5,  'meas': 7,  'dmg_id': 7},
    'state_10_vs_9':  {'base': 9,  'meas': 10, 'dmg_id': 9},  'state_11_vs_10': {'base': 10, 'meas': 11, 'dmg_id': 9},
    'state_11_vs_9':  {'base': 9,  'meas': 11, 'dmg_id': 9},
    'state_13_vs_12': {'base': 12, 'meas': 13, 'dmg_id': 11}, 'state_14_vs_12': {'base': 12, 'meas': 14, 'dmg_id': 11},
    'state_14_vs_13': {'base': 13, 'meas': 14, 'dmg_id': 11},
    'state_2_vs_1':   {'base': 1,  'meas': 2,  'dmg_id': 13}, 'state_3_vs_2':   {'base': 2,  'meas': 3,  'dmg_id': 13},
    'state_3_vs_1':   {'base': 1,  'meas': 3,  'dmg_id': 13},
    'state_8_vs_7':   {'base': 7,  'meas': 8,  'dmg_id': 15}, 'state_9_vs_8':   {'base': 8,  'meas': 9,  'dmg_id': 15},
    'state_9_vs_7':   {'base': 7,  'meas': 9,  'dmg_id': 15},
}

def find_source_file(sensor_folder_path, state_num, sensor_id, freq):
    """Znajduje odpowiedni plik pomiarowy .svd lub .npy w katalogu sensora."""
    if not os.path.exists(sensor_folder_path):
        return None
    mask_str = f"state_{state_num}_PZT_{sensor_id}_freq_{freq}_kHz"
    for filename in os.listdir(sensor_folder_path):
        if filename.startswith(mask_str) and (filename.endswith('.svd') or filename.endswith('.npy')):
            return filename
    return None

# ==========================================
# GŁÓWNA PĘTLA PRZETWARZANIA
# ==========================================
if __name__ == "__main__":
    print("--- Rozpoczynam przetwarzanie danych (Filtracja: Median & Gauss) ---")
    total_processed = 0

    for state_key, info in BASELINE_MAP.items():
        base_num, meas_num = info['base'], info['meas']
        
        for sensor_id in range(1, 13):
            try:
                # Ścieżki do plików źródłowych
                path_to_sensor_folder = os.path.join(config.RAW_DATA_PATH_EXP2, f'PZT{sensor_id}')
                base_filename = find_source_file(path_to_sensor_folder, base_num, sensor_id, EXCITATION_FREQ)
                meas_filename = find_source_file(path_to_sensor_folder, meas_num, sensor_id, EXCITATION_FREQ)

                if not base_filename or not meas_filename:
                    continue
                
                baseline_path = os.path.join(path_to_sensor_folder, base_filename)
                measurement_path = os.path.join(path_to_sensor_folder, meas_filename)
                output_filename = f'PZT{sensor_id}_{base_num}to{meas_num}_{EXCITATION_FREQ}kHz_W{WINDOW[0]}_{WINDOW[1]}.npy'
                
                # --- Wariant 1: Filtr Medianowy ---
                dest_path_median = os.path.join(OUTPUT_PATH_MEDIAN, output_filename)
                if not os.path.exists(dest_path_median):
                    print(f"Generowanie (Median): {output_filename}")
                    
                    # Obliczenia
                    result_median = LoadAndProcess(baseline_path, measurement_path, t0=WINDOW[0], t1=WINDOW[1])
                    di_dict = result_median[0][1]
                    
                    # Aplikacja filtra
                    for feature in FEATURE_VECTOR:
                        if feature in di_dict:
                            raw_map = np.nan_to_num(di_dict[feature].astype(np.float32))
                            di_dict[feature] = np.nan_to_num(cv2.medianBlur(raw_map, FILTER_SIZE_MEDIAN))
                    
                    np.save(dest_path_median, result_median, allow_pickle=True)
                    total_processed += 1
                
                # --- Wariant 2: Filtr Gaussa ---
                dest_path_gauss = os.path.join(OUTPUT_PATH_GAUSS, output_filename)
                if not os.path.exists(dest_path_gauss):
                    print(f"Generowanie (Gauss): {output_filename}")
                    
                    # Ponowne obliczenia dla czystej instancji danych
                    result_gauss = LoadAndProcess(baseline_path, measurement_path, t0=WINDOW[0], t1=WINDOW[1])
                    di_dict_gauss = result_gauss[0][1]
                    
                    # Aplikacja filtra
                    for feature in FEATURE_VECTOR:
                        if feature in di_dict_gauss:
                            raw_map = np.nan_to_num(di_dict_gauss[feature].astype(np.float32))
                            di_dict_gauss[feature] = np.nan_to_num(cv2.GaussianBlur(raw_map, FILTER_SIZE_GAUSS, 0))
                            
                    np.save(dest_path_gauss, result_gauss, allow_pickle=True)
                    total_processed += 1

            except Exception as e:
                print(f"Błąd przetwarzania PZT{sensor_id} ({base_num}->{meas_num}): {e}")

    print(f"\nZakończono. Przetworzono łącznie {total_processed} plików.")