import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_liveness_model(input_shape):
    """
    Builds and compiles a CNN model specifically for binary (Live vs. Fake) classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # A single output neuron with a sigmoid activation is best for binary classification.
        # It outputs a value between 0 (for 'fake') and 1 (for 'live').
        Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # The standard loss function for binary problems
                  metrics=['accuracy'])
    return model

def train():
    """Loads liveness data and trains the binary classification model."""
    print("--- Training PHASE 1: Liveness Detection Model ---")
    
    print("Loading preprocessed liveness data...")
    try:
        with np.load('liveness_data.npz') as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data['X_val']
            y_val = data['y_val']
    except FileNotFoundError:
        print("Error: 'liveness_data.npz' not found. Please run 'data_utils_v2.py' first.")
        return

    print(f"Data loaded. Training on {len(X_train)} samples.")
    
    input_shape = X_train.shape[1:]
    
    model = build_liveness_model(input_shape)
    model.summary()

    # Callbacks for smart training
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('liveness_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    print("\nStarting model training...")
    model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    print("\nLiveness model training complete! 🎉")
    print("The best version of the model has been saved to 'liveness_model.keras'")

if __name__ == '__main__':
    train()
