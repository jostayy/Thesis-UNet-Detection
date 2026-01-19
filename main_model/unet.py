import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, RandomFlip, RandomRotation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

IMG_SHAPE = (80, 112)
NUM_CHANNELS = 5
BATCH_SIZE = 4
EPOCHS = 200

DATASET_BASE_PATH = "ML_Dataset_Split_5_FEATURES_FINAL_test"
path_X_train = os.path.join(DATASET_BASE_PATH, "train", "images_X")
path_Y_train = os.path.join(DATASET_BASE_PATH, "train", "masks_Y")
path_X_test = os.path.join(DATASET_BASE_PATH, "test", "images_X")
path_Y_test = os.path.join(DATASET_BASE_PATH, "test", "masks_Y")
path_X_val = os.path.join(DATASET_BASE_PATH, "val", "images_X")
path_Y_val = os.path.join(DATASET_BASE_PATH, "val", "masks_Y")

SAVE_DIR = "new_model"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "unet_best_model.h5")

def load_data_from_folder(path_x, path_y, img_shape, num_channels):
    images, masks = [], []
    if not os.path.exists(path_x):
        print(f"BŁĄD: Ścieżka nie istnieje: {path_x}")
        return np.array([]), np.array([])
    for filename in os.listdir(path_x):
        if filename.endswith('.npy'):
            img = np.load(os.path.join(path_x, filename))
            mask = np.load(os.path.join(path_y, filename))
            if img.shape == (img_shape[0], img_shape[1], num_channels):
                images.append(img.astype('float32'))
                masks.append(np.expand_dims(mask, axis=-1).astype('float32'))
    return np.array(images), np.array(masks)

print("Wczytuję dane (Zbiór Pełny)...")
X_train, Y_train = load_data_from_folder(path_X_train, path_Y_train, IMG_SHAPE, NUM_CHANNELS)
X_val, Y_val = load_data_from_folder(path_X_val, path_Y_val, IMG_SHAPE, NUM_CHANNELS)
X_test, Y_test = load_data_from_folder(path_X_test, path_Y_test, IMG_SHAPE, NUM_CHANNELS)

if len(X_train) == 0: 
    print("Brak danych!")
    exit()


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


def build_unet(input_shape):
    inputs = Input(input_shape)
    
    x = RandomFlip("horizontal_and_vertical")(inputs)
    x = RandomRotation(0.1)(x)

    c1 = Conv2D(32, (3, 3), padding='same')(x); c1 = BatchNormalization()(c1); c1 = tf.keras.layers.Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), padding='same')(p1); c2 = BatchNormalization()(c2); c2 = tf.keras.layers.Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c5 = Conv2D(128, (3, 3), padding='same')(p2); c5 = BatchNormalization()(c5); c5 = tf.keras.layers.Activation('relu')(c5)


 
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u7 = concatenate([u7, c2])
    c7 = Conv2D(64, (3, 3), padding='same')(u7); c7 = BatchNormalization()(c7); c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = Dropout(0.2)(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c1]) 
    c8 = Conv2D(32, (3, 3), padding='same')(u8); c8 = BatchNormalization()(c8); c8 = tf.keras.layers.Activation('relu')(c8)
    c8 = Dropout(0.2)(c8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    
    return Model(inputs=[inputs], outputs=[outputs], name="Classic_UNet_3Level")

model = build_unet((IMG_SHAPE[0], IMG_SHAPE[1], NUM_CHANNELS))

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=weighted_dice_bce_loss,
              metrics=[dice_coefficient])
model.summary()

checkpoint = ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_dice_coefficient', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_dice_coefficient', patience=30, mode='max', verbose=1, restore_best_weights=True)

print("\nRozpoczynam trening (U-Net)...")
history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, Y_val),
                    callbacks=[checkpoint, early_stop])

print("\nGenerowanie wyników...")

# 1. Wykres historii treningu
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Strata trenowania')
plt.plot(history.history['val_loss'], label='Strata walidacji')
plt.plot(history.history['dice_coefficient'], label='Dice (train)', linestyle='--')
plt.plot(history.history['val_dice_coefficient'], label='Dice (val)', linestyle='--')
plt.title('Historia treningu U-Net (Rozwiązanie Optymalne)')
plt.xlabel('Epoka')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "historia_treningu_v5.png"))
plt.close()

preds_test = model.predict(X_test)
all_dice_scores = [] 

for i in range(len(preds_test)):

    y_true_tensor = tf.convert_to_tensor(Y_test[i], dtype=tf.float32)
    y_pred_tensor = tf.convert_to_tensor(preds_test[i], dtype=tf.float32)
    

    test_dice = dice_coefficient(y_true_tensor, y_pred_tensor).numpy()
    all_dice_scores.append(test_dice) 

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(X_test[i][:, :, 2], cmap='viridis'); ax[0].set_title('Dane Wejściowe (envCorr)')
    ax[1].imshow(Y_test[i].squeeze(), cmap='gray'); ax[1].set_title('Ground Truth')
    ax[2].imshow(preds_test[i].squeeze(), cmap='gray'); ax[2].set_title(f'U-Net (Dice: {test_dice:.3f})')

    plt.savefig(os.path.join(SAVE_DIR, f"weryfikacja_probka{i}.png")) 
    plt.close()

mean_dice = np.mean(all_dice_scores)
std_dice = np.std(all_dice_scores)
min_dice = np.min(all_dice_scores)
max_dice = np.max(all_dice_scores)

print("\n" + "="*40)
print(f"PODSUMOWANIE WYNIKÓW (Zbiór Testowy - {len(preds_test)} próbek)")
print("="*40)
print(f"Średni Dice Score:       {mean_dice:.4f}")
print(f"Odchylenie standardowe:  {std_dice:.4f}")
print(f"Najgorszy wynik (Min):   {min_dice:.4f}")
print(f"Najlepszy wynik (Max):   {max_dice:.4f}")
print("="*40 + "\n")

print(f"Gotowe! Wyniki zapisano w: {SAVE_DIR}")