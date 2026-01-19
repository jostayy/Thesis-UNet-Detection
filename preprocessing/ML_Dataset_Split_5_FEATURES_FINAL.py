"""
Skrypt autorski (na bazie bibliotek promotora) do przygotowania zbioru danych dla sieci neuronowej.
Autor: Joanna Szczypkowska

Opis:
Skrypt generuje finalny zbiór danych (Tensor 5-kanałowy + Maska binarna).
Dokonuje podziału na zbiory Train/Val/Test na podstawie ID uszkodzenia,
skaluje obrazy i normalizuje mapy cech.
"""

import os
import numpy as np
import cv2
import config
from FusionToolbox import FusionToolbox as FT

# ==========================================
# KONFIGURACJA
# ==========================================
EXPERIMENT = 2
IMG_SHAPE = (80, 112)
NEIGHBORHOOD_SIZE = 3
EXCITATION_FREQ = '300'
WINDOW = [1300, 1801]

# Wybór cech (5 kanałów)
FEATURES_TO_USE = ['corr', 'diffRMS', 'envCorr', 'diffAmp', 'sigDiffRMS']
NUM_CHANNELS = len(FEATURES_TO_USE)

# Ścieżki
INPUT_DIR = os.path.join(os.path.dirname(config.PROCESSED_DATA_PATH_EXP2), 
                         f"ProcessedExp2_Filtered_Gauss_W{WINDOW[0]}_{WINDOW[1]}FINAL")
OUTPUT_DIR = os.path.join(config.FIGURE_DROP_PATH, "ML_Dataset_Split_5_FEATURES_FINAL_test1")

# Strategia podziału danych (Damage ID)
TRAIN_DMG_IDS = [11, 5, 15]
VAL_DMG_IDS = [7]
TEST_DMG_IDS = [9, 13]

# Średnice uszkodzeń [mm]
DAMAGE_DIAMETERS_MM = {3: 1, 5: 1, 7: 1, 9: 1, 11: 1, 13: 1, 15: 1}

# Mapa par pomiarowych
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

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def create_ground_truth_mask(image_shape, damage_center_xy, damage_diameter_mm):
    """Generuje binarną maskę uszkodzenia (Ground Truth)."""
    mask = np.zeros(image_shape, dtype=np.float32)
    # Przeliczenie mm na piksele (zgodnie z fizycznymi wymiarami próbki)
    radius_pixels = int((damage_diameter_mm / 1.0) / 2) + 3
    scaled_center_x = int(damage_center_xy[0] * (image_shape[1] / 115.0))
    scaled_center_y = int(damage_center_xy[1] * (image_shape[0] / 86.0))
    
    cv2.circle(mask, (scaled_center_x, scaled_center_y), radius_pixels, 1.0, -1)
    return mask

def get_dmg_id_from_pair(base, meas):
    """Zwraca ID uszkodzenia dla danej pary pomiarowej."""
    for key, info in BASELINE_MAP.items():
        if info['base'] == base and info['meas'] == meas:
            return info['dmg_id']
    return None

def validate_input_data(path):
    """Sprawdza, czy pliki w folderze zawierają wymagane kanały."""
    if not os.path.exists(path):
        print(f"BŁĄD: Folder {path} nie istnieje!")
        exit()
        
    for fname in os.listdir(path):
        if fname.endswith('.npy'):
            try:
                sample_data = np.load(os.path.join(path, fname), allow_pickle=True)
                sample_dict = sample_data[0][1]
                missing = [f for f in FEATURES_TO_USE if f not in sample_dict]
                
                if missing:
                    print(f"BŁĄD KRYTYCZNY: Brak wskaźników w plikach: {missing}")
                    exit()
                else:
                    print("Weryfikacja danych pomyślna (znaleziono 5 kanałów).")
                    return
            except Exception as e:
                continue
    print("OSTRZEŻENIE: Nie znaleziono plików .npy do weryfikacji.")

# ==========================================
# GŁÓWNA PĘTLA PRZETWARZANIA
# ==========================================
if __name__ == "__main__":
    print("--- Rozpoczynam generowanie zbioru danych ML ---")
    validate_input_data(INPUT_DIR)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(f'_{EXCITATION_FREQ}kHz_W{WINDOW[0]}_{WINDOW[1]}.npy')])
    
    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        
        try:
            # Parsowanie nazwy pliku
            parts = filename.split('_')
            pair = parts[1].split('to')
            base_num, meas_num = int(pair[0]), int(pair[1])
            
            damage_id = get_dmg_id_from_pair(base_num, meas_num)
            if damage_id is None:
                continue

            # Przypisanie do zbioru
            if damage_id in TRAIN_DMG_IDS:
                subset = "train"
            elif damage_id in VAL_DMG_IDS:
                subset = "val"
            elif damage_id in TEST_DMG_IDS:
                subset = "test"
            else:
                continue # Ignorujemy uszkodzenia spoza listy

            # Wczytanie danych
            data = np.load(filepath, allow_pickle=True)
            di_dict = data[0][1]

            if not all(k in di_dict for k in FEATURES_TO_USE):
                continue

            # Przetwarzanie kanałów (Skalowanie i Normalizacja)
            feature_maps = []
            for feature in FEATURES_TO_USE:
                raw_map = np.nan_to_num(di_dict[feature].astype('float32'))
                
                # Resize do docelowego kształtu sieci
                resized_map = cv2.resize(raw_map, (IMG_SHAPE[1], IMG_SHAPE[0]))
                
                # Min-Max Normalization
                min_val, max_val = np.min(resized_map), np.max(resized_map)
                if max_val > min_val:
                    norm_map = (resized_map - min_val) / (max_val - min_val)
                else:
                    norm_map = resized_map
                
                # Wygładzanie (usuwanie artefaktów pikselowych)
                smooth_map = cv2.GaussianBlur(norm_map, (NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE), 0)
                feature_maps.append(smooth_map)

            # Tworzenie tensora X (H, W, Channels)
            x_tensor = np.stack(feature_maps, axis=-1)

            # Tworzenie maski Y (Ground Truth)
            dmg_pos_xy = FT.PosDmg(EXPERIMENT, damage_id)
            dmg_diameter = DAMAGE_DIAMETERS_MM.get(damage_id)
            
            if dmg_pos_xy is None or dmg_diameter is None:
                continue
                
            y_mask = create_ground_truth_mask(IMG_SHAPE, dmg_pos_xy, dmg_diameter)

            # Zapis wyników
            dest_x = os.path.join(OUTPUT_DIR, subset, "images_X")
            dest_y = os.path.join(OUTPUT_DIR, subset, "masks_Y")
            os.makedirs(dest_x, exist_ok=True)
            os.makedirs(dest_y, exist_ok=True)

            out_name = f"{os.path.splitext(filename)[0]}_feature_channels_5.npy"
            np.save(os.path.join(dest_x, out_name), x_tensor)
            np.save(os.path.join(dest_y, out_name), y_mask)

            counts[subset] += 1

        except Exception as e:
            print(f"Błąd przetwarzania {filename}: {e}")

    print("\n--- Zakończono ---")
    print(f"Zapisano w: {OUTPUT_DIR}")
    print(f"Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")