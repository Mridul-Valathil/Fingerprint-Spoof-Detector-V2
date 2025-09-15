import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, 
    BatchNormalization, Activation, Dropout, GRU, Reshape,
    GlobalAveragePooling2D, Dense, Add, Multiply
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- ADVANCED MODEL ARCHITECTURE: GRU-Attention-UNet ---

def conv_block(inputs, num_filters):
    """A standard convolutional block with two conv layers."""
    x = Conv2D(num_filters, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def attention_gate(encoder_output, decoder_output, num_filters):
    """Attention Gate to focus on relevant features."""
    g = Conv2D(num_filters, (1, 1), padding="same")(decoder_output)
    g = BatchNormalization()(g)
    
    x = Conv2D(num_filters, (1, 1), padding="same")(encoder_output)
    x = BatchNormalization()(x)
    
    psi = Add()([g, x])
    psi = Activation("relu")(psi)
    
    psi = Conv2D(1, (1, 1), padding="same")(psi)
    psi = Activation("sigmoid")(psi)
    
    return Multiply()([encoder_output, psi])

def build_material_model(input_shape, num_classes):
    """
    Builds and compiles a GRU-Attention-UNet model for material classification.
    """
    inputs = Input(shape=input_shape)

    # --- Encoder Path (Down-sampling) ---
    enc1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D((2, 2))(enc1)
    enc2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D((2, 2))(enc2)
    enc3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D((2, 2))(enc3)
    
    # --- Bottleneck with GRU ---
    bottleneck = conv_block(pool3, 512)
    # Reshape for GRU: treat spatial dims as a sequence
    shape = tf.keras.backend.int_shape(bottleneck)
    # The sequence length is height * width
    reshaped_bottleneck = Reshape((shape[1] * shape[2], shape[3]))(bottleneck)
    gru = GRU(256, return_sequences=False)(reshaped_bottleneck)
    
    # --- Classification Head ---
    # We use the output of the GRU, which has captured spatial context,
    # as the input for our final classification decision.
    outputs = Dense(num_classes, activation='softmax')(gru)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- The rest of the script remains the same ---

def train():
    """Loads material data and trains the multi-class classification model."""
    print("--- Training PHASE 2: Material Classification Model (GRU-AUnet) ---")
    
    print("Loading preprocessed material data...")
    try:
        with np.load('material_data.npz') as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data['X_val']
            y_val = data['y_val']
    except FileNotFoundError:
        print("Error: 'material_data.npz' not found. Please run 'data_utils_v2.py' first.")
        return

    print(f"Data loaded. Training on {len(X_train)} samples.")
    
    input_shape = X_train.shape[1:]
    
    try:
        with open('material_label_map.json', 'r') as f:
            num_classes = len(json.load(f))
    except FileNotFoundError:
        print("Error: 'material_label_map.json' not found. Please run 'data_utils_v2.py'.")
        return
        
    print(f"Number of material classes: {num_classes}")

    model = build_material_model(input_shape, num_classes)
    model.summary()

    # Callbacks for smart training
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, verbose=1) # Increased patience for complex model
    model_checkpoint = ModelCheckpoint('material_model_v2_advanced.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    print("\nStarting model training...")
    model.fit(
        X_train, y_train,
        epochs=40, # Increased epochs for complex model
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    print("\nMaterial model training complete! 🎉")
    print("The best version of the model has been saved to 'material_model_v2_advanced.keras'")

if __name__ == '__main__':
    train()

