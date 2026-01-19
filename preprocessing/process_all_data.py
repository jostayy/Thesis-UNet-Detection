"""
Skrypt autorski (na bazie bibliotek promotora) do przygotowania zbioru danych dla sieci neuronowej.
Autor: Joanna Szczypkowska

Opis:
Skrypt masowo przetwarza dane surowe z Eksperymentu 2, obsługując różne częstotliwości wzbudzenia
(200kHz, 300kHz, Chirp). Automatycznie paruje stany referencyjne z uszkodzonymi
i zapisuje przetworzone macierze do formatu .npy.
"""

import os
import numpy as np
import config
from ZD_AutomaticProcesser_v01 import LoadAndProcess

# ==========================================
# KONFIGURACJA
# ==========================================
FREQUENCIES_TO_PROCESS = ['200', '300', 'chirp']
OUTPUT_DIR = config.PROCESSED_DATA_PATH_EXP2v4

# Mapa parowania stanów (Referencja -> Pomiar -> ID Uszkodzenia)
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

# Mapa sensorów (Filtrowanie: które sensory widzą dane uszkodzenie)
SENSORS_MAP = {
    3: [4, 8, 12], 
    5: [2, 3], 
    7: [1, 2, 6, 9, 10],
    9: [1, 2, 5, 9, 11], 
    11: [4, 8, 12], 
    13: [1, 2, 3, 7, 11], 
    15: [1, 2, 3, 6, 9, 10]
}

def find_source_file(sensor_folder_path, state_num, sensor_id, freq):
    """
    Wyszukuje plik źródłowy (.svd lub .npy) dla danego stanu i częstotliwości.
    Obsługuje specyficzne nazewnictwo dla sygnałów 'chirp'.
    """
    if not os.path.exists(sensor_folder_path):
        return None
    
    if freq == 'chirp':
        mask_prefix = f"state_{state_num}_PZT_{sensor_id}"
        mask_infix = "_chirp_"
        for filename in os.listdir(sensor_folder_path):
            if filename.startswith(mask_prefix) and mask_infix in filename and (filename.endswith('.svd') or filename.endswith('.npy')):
                return filename
    else:
        # Dla stałych częstotliwości (np. 200, 300 kHz)
        mask_str = f"state_{state_num}_PZT_{sensor_id}_freq_{freq}_kHz"
        for filename in os.listdir(sensor_folder_path):
            if filename.startswith(mask_str) and (filename.endswith('.svd') or filename.endswith('.npy')):
                return filename
    return None

# ==========================================
# GŁÓWNA PĘTLA PRZETWARZANIA
# ==========================================
if __name__ == "__main__":
    print(f"--- Rozpoczynam przetwarzanie masowe danych (Exp 2) ---")
    print(f"Folder docelowy: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_processed = 0

    for state_key, info in BASELINE_MAP.items():
        base_num, meas_num, dmg_id = info['base'], info['meas'], info['dmg_id']
        
        # Przetwarzamy tylko sensory zdefiniowane w mapie dla danego uszkodzenia
        if dmg_id not in SENSORS_MAP:
            continue
            
        target_sensors = SENSORS_MAP[dmg_id]
        
        for sensor_id in target_sensors:
            for freq in FREQUENCIES_TO_PROCESS:
                try:
                    path_to_sensor_folder = os.path.join(config.RAW_DATA_PATH_EXP2, f'PZT{sensor_id}')

                    base_filename = find_source_file(path_to_sensor_folder, base_num, sensor_id, freq)
                    meas_filename = find_source_file(path_to_sensor_folder, meas_num, sensor_id, freq)

                    if not base_filename or not meas_filename:
                        # Brak kompletnej pary plików
                        continue
                    
                    baseline_path = os.path.join(path_to_sensor_folder, base_filename)
                    measurement_path = os.path.join(path_to_sensor_folder, meas_filename)

                    # Ustalenie nazwy pliku wyjściowego
                    if freq == 'chirp':
                        output_filename = f'PZT{sensor_id}_{base_num}to{meas_num}_chirp.npy'
                    else:
                        output_filename = f'PZT{sensor_id}_{base_num}to{meas_num}_{freq}kHz.npy'
                    
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    # Pomijamy, jeśli plik już istnieje (wznawianie pracy)
                    if os.path.exists(output_path):
                        continue

                    print(f"Przetwarzam: PZT{sensor_id}, para {base_num}->{meas_num}, freq: {freq}...")
                    
                    # --- GŁÓWNE PRZETWARZANIE ---
                    result = LoadAndProcess(baseline_path, measurement_path)
                    
                    np.save(output_path, result, allow_pickle=True)
                    print(f"--> Zapisano: {output_filename}")
                    total_processed += 1

                except Exception as e:
                    print(f"!!! Błąd przetwarzania: PZT{sensor_id}, para {base_num}->{meas_num}, freq {freq}: {e}")

    print(f"\n--- Zakończono ---")
    print(f"Przetworzono łącznie {total_processed} nowych plików.")