import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


IMG_SHAPE = (80, 112)
NUM_CHANNELS = 5

DATASET_DIR_NAME = "ML_Dataset_Split_5_FEATURES_FINAL_test"

MODEL_DIR = "Eksperyment_UNet_Final"
MODEL_FILENAME = os.path.join(MODEL_DIR, "unet_best_model.h5")

RESULTS_DIR = "model_result"
os.makedirs(RESULTS_DIR, exist_ok=True)

def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.flatten(y_true) * K.flatten(y_pred))
    score = (2. * intersection + smooth) / (K.sum(K.flatten(y_true)) + K.sum(K.flatten(y_pred)) + smooth)
    return 1 - score

def weighted_dice_bce_loss(y_true, y_pred):

    dice_weight = 0.7
    bce_weight = 0.3
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return (bce_weight * bce) + (dice_weight * dice)

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(K.flatten(y_true) * K.flatten(y_pred))
    return (2. * intersection + smooth) / (K.sum(K.flatten(y_true)) + K.sum(K.flatten(y_pred)) + smooth)

def load_data_from_folder(path_x, path_y, img_shape, num_channels):
    images, masks = [], []
    if not os.path.exists(path_x):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono folderu: {path_x}")
        return np.array([]), np.array([])
    
    files = sorted([f for f in os.listdir(path_x) if f.endswith('.npy')])
    
    for filename in files:
        img = np.load(os.path.join(path_x, filename))
        mask = np.load(os.path.join(path_y, filename))
        if img.shape == (img_shape[0], img_shape[1], num_channels):
            images.append(img.astype('float32'))
            masks.append(np.expand_dims(mask, axis=-1).astype('float32'))
            
    return np.array(images), np.array(masks)

path_X_test = os.path.join(DATASET_DIR_NAME, "test", "images_X")
path_Y_test = os.path.join(DATASET_DIR_NAME, "test", "masks_Y")

print(f">>> Wczytuję dane testowe z: {DATASET_DIR_NAME}...")
X_test, Y_test = load_data_from_folder(path_X_test, path_Y_test, IMG_SHAPE, NUM_CHANNELS)

if len(X_test) == 0:
    print("BŁĄD: Brak danych testowych! Sprawdź, czy folder z danymi jest rozpakowany obok skryptu.")
    exit()

print(f"    Załadowano {len(X_test)} próbek testowych.")

print(f"\n>>> Wczytuję wytrenowany model: {MODEL_FILENAME}")

if not os.path.exists(MODEL_FILENAME):
    print(f"BŁĄD: Nie znaleziono pliku {MODEL_FILENAME}. Upewnij się, że jest w tym samym folderze co skrypt.")
    exit()

try:
    model = load_model(MODEL_FILENAME, custom_objects={
        'weighted_dice_bce_loss': weighted_dice_bce_loss,
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss
    })
    print("    Model wczytany poprawnie.")
except Exception as e:
    print(f"    Błąd wczytywania modelu: {e}")
    print("    Spróbuję wczytać bez kompilacji (tylko do predykcji)...")
    model = load_model(MODEL_FILENAME, compile=False)


preds_test = model.predict(X_test, batch_size=4, verbose=1)

dice_scores = []
for i in range(len(preds_test)):
    y_true_t = tf.convert_to_tensor(Y_test[i], dtype=tf.float32)
    y_pred_t = tf.convert_to_tensor(preds_test[i], dtype=tf.float32)
    score = dice_coefficient(y_true_t, y_pred_t).numpy()
    dice_scores.append(score)

mean_dice = np.mean(dice_scores)
std_dice = np.std(dice_scores)
min_dice = np.min(dice_scores)
max_dice = np.max(dice_scores)

print("\n" + "#"*60)
print(f"RAPORT WERYFIKACJI MODELU")
print("#"*60)
print(f"Analizowany zbiór:       Testowy ({len(preds_test)} próbek)")
print(f"Plik modelu:             {MODEL_FILENAME}")
print("-" * 30)
print(f"Średni Dice Score:       {mean_dice:.4f}")
print(f"Odchylenie (Std Dev):    {std_dice:.4f}")
print(f"Wynik Min:               {min_dice:.4f}")
print(f"Wynik Max:               {max_dice:.4f}")
print("#"*60 + "\n")

print(f">>> Zapisuję wizualizacje do folderu: {RESULTS_DIR}")

for i in range(len(preds_test)):
    score = dice_scores[i]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Wejście (kanał 2 - envCorr)
    ax[0].imshow(X_test[i][:, :, 2], cmap='viridis')
    ax[0].set_title('Dane Wejściowe (envCorr)')
    ax[0].axis('off')
    
    # Ground Truth
    ax[1].imshow(Y_test[i].squeeze(), cmap='gray')
    ax[1].set_title('Maska Referencyjna')
    ax[1].axis('off')
    
    # Predykcja
    ax[2].imshow(preds_test[i].squeeze(), cmap='gray')
    ax[2].set_title(f'Predykcja Modelu (Dice: {score:.3f})')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"weryfikacja_probka_{i}.png"))
    plt.close()

print(f"Gotowe! Wszystkie wyniki zapisano w: {RESULTS_DIR}")