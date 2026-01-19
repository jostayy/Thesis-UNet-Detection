import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, 
                                     concatenate, BatchNormalization, RandomFlip, RandomRotation, 
                                     Activation, add, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import config

IMG_SHAPE = (80, 112)
NUM_CHANNELS = 5
BATCH_SIZE = 4
EPOCHS = 200

# Paths
DATASET_BASE_PATH = os.path.join(config.FIGURE_DROP_PATH, "ML_Dataset_Split_5_FEATURES_FINAL")
DIRS = {
    'train': (os.path.join(DATASET_BASE_PATH, "train", "images_X"), os.path.join(DATASET_BASE_PATH, "train", "masks_Y")),
    'val': (os.path.join(DATASET_BASE_PATH, "val", "images_X"), os.path.join(DATASET_BASE_PATH, "val", "masks_Y")),
    'test': (os.path.join(DATASET_BASE_PATH, "test", "images_X"), os.path.join(DATASET_BASE_PATH, "test", "masks_Y"))
}

SAVE_DIR = os.path.join(config.FIGURE_DROP_PATH, "Eksperyment_AttentionUNet")
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "attention_unet_best.h5")

def load_dataset(path_x, path_y, img_shape, num_channels):
    """Loads and returns data as numpy arrays."""
    if not os.path.exists(path_x):
        return np.array([]), np.array([])
    
    images, masks = [], []
    files = sorted([f for f in os.listdir(path_x) if f.endswith('.npy')])
    
    for filename in files:
        img = np.load(os.path.join(path_x, filename))
        mask = np.load(os.path.join(path_y, filename))
        
        if img.shape == (img_shape[0], img_shape[1], num_channels):
            images.append(img.astype('float32'))
            masks.append(np.expand_dims(mask, axis=-1).astype('float32'))
            
    return np.array(images), np.array(masks)

def plot_history(history, save_path):
    """Plots training and validation metrics."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['dice_coefficient'], label='Dice (Train)', linestyle='--')
    plt.plot(history.history['val_dice_coefficient'], label='Dice (Val)', linestyle='--')
    plt.title('Training History - Attention U-Net')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.flatten(y_true) * K.flatten(y_pred))
    return (2. * intersection + smooth) / (K.sum(K.flatten(y_true)) + K.sum(K.flatten(y_pred)) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def weighted_dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return (0.3 * bce) + (0.7 * dice)

def attention_gate(x, g, inter_channels):
    """
    Attention Gate mechanism.
    x: Input feature map from encoder (skip connection)
    g: Gating signal from decoder
    inter_channels: Number of channels in intermediate layers
    """
    # 1. Gating signal processing
    phi_g = Conv2D(inter_channels, (1, 1), padding='same')(g)
    
    # 2. Encoder feature processing
    theta_x = Conv2D(inter_channels, (1, 1), padding='same')(x)
    
    # 3. Add and Activation
    f = add([phi_g, theta_x])
    f = Activation('relu')(f)
    
    # 4. Attention coefficients
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    rate = Activation('sigmoid')(psi_f)
    
    # 5. Feature scaling
    att_x = multiply([x, rate])
    return att_x

def build_attention_unet(input_shape):
    inputs = Input(input_shape)
    
    # Augmentation
    x = RandomFlip("horizontal_and_vertical")(inputs)
    x = RandomRotation(0.1)(x)

    # --- ENCODER ---
    # Block 1
    c1 = Conv2D(32, (3, 3), padding='same')(x); c1 = BatchNormalization()(c1); c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Block 2
    c2 = Conv2D(64, (3, 3), padding='same')(p1); c2 = BatchNormalization()(c2); c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Block 3 (Bottleneck)
    c3 = Conv2D(128, (3, 3), padding='same')(p2); c3 = BatchNormalization()(c3); c3 = Activation('relu')(c3)

    # --- DECODER WITH ATTENTION ---
    
    # Up 2
    u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    att2 = attention_gate(x=c2, g=u2, inter_channels=32) # Attention Gate
    u2 = concatenate([u2, att2])
    
    c4 = Conv2D(64, (3, 3), padding='same')(u2); c4 = BatchNormalization()(c4); c4 = Activation('relu')(c4)
    c4 = Dropout(0.2)(c4)

    # Up 1
    u1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4)
    att1 = attention_gate(x=c1, g=u1, inter_channels=16) # Attention Gate
    u1 = concatenate([u1, att1])
    
    c5 = Conv2D(32, (3, 3), padding='same')(u1); c5 = BatchNormalization()(c5); c5 = Activation('relu')(c5)
    c5 = Dropout(0.2)(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    return Model(inputs=[inputs], outputs=[outputs], name="Attention_UNet")

if __name__ == "__main__":
    print("Loading datasets...")
    X_train, Y_train = load_dataset(*DIRS['train'], IMG_SHAPE, NUM_CHANNELS)
    X_val, Y_val = load_dataset(*DIRS['val'], IMG_SHAPE, NUM_CHANNELS)
    X_test, Y_test = load_dataset(*DIRS['test'], IMG_SHAPE, NUM_CHANNELS)

    if len(X_train) == 0:
        raise RuntimeError("Training set is empty.")

    # Build and Compile
    model = build_attention_unet((IMG_SHAPE[0], IMG_SHAPE[1], NUM_CHANNELS))
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss=weighted_dice_bce_loss,
                  metrics=[dice_coefficient])
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='val_dice_coefficient', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_dice_coefficient', patience=30, mode='max', verbose=1, restore_best_weights=True)
    ]

    # Training
    print("\nStarting training...")
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    # Evaluation
    print("\nSaving results...")
    plot_history(history, os.path.join(SAVE_DIR, "training_history.png"))

    preds = model.predict(X_test, verbose=0)
    for i in range(len(preds)):
        dice = dice_coefficient(tf.convert_to_tensor(Y_test[i], dtype=tf.float32), 
                                tf.convert_to_tensor(preds[i], dtype=tf.float32)).numpy()
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(X_test[i][:, :, 2], cmap='viridis')
        ax[0].set_title('Input (envCorr)')
        ax[0].axis('off')
        
        ax[1].imshow(Y_test[i].squeeze(), cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')
        
        ax[2].imshow(preds[i].squeeze(), cmap='gray')
        ax[2].set_title(f'Attention U-Net (Dice: {dice:.3f})')
        ax[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"result_{i}.png"))
        plt.close()

    print(f"Done. Results saved in: {SAVE_DIR}")