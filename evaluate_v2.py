import numpy as np
import tensorflow as tf
import json
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
LIVENESS_MODEL_FILE = 'liveness_model.keras'
MATERIAL_MODEL_FILE = 'material_model_v2.keras'
MATERIAL_LABEL_MAP_FILE = 'material_label_map.json'
TEST_DATA_DIR = 'data/LivDet2015/Testing'
IMG_SIZE = 96

def preprocess_image_for_prediction(image):
    """Prepares a raw image for model prediction."""
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    resized_image = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE))
    normalized_image = resized_image.astype(np.float32) / 255.0
    return np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=-1)

def load_test_data():
    """Loads all test images and their ground truth labels."""
    images = []
    true_labels = []
    
    print("Loading test data...")
    for sensor in os.listdir(TEST_DATA_DIR):
        sensor_path = os.path.join(TEST_DATA_DIR, sensor)
        if not os.path.isdir(sensor_path): continue

        # Load live images
        live_path = os.path.join(sensor_path, 'Live')
        if os.path.exists(live_path):
            for filename in os.listdir(live_path):
                img = cv2.imread(os.path.join(live_path, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    true_labels.append('live')

        # Load fake images
        fake_path = os.path.join(sensor_path, 'Fake')
        if os.path.exists(fake_path):
            for material in os.listdir(fake_path):
                material_path = os.path.join(fake_path, material)
                if os.path.isdir(material_path):
                    for filename in os.listdir(material_path):
                        img = cv2.imread(os.path.join(material_path, filename), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            images.append(img)
                            true_labels.append(f'fake-{material}')
    return images, true_labels

def evaluate_v2_system():
    # 1. Load models and label map
    print("Loading models...")
    liveness_model = tf.keras.models.load_model(LIVENESS_MODEL_FILE)
    material_model = tf.keras.models.load_model(MATERIAL_MODEL_FILE)
    with open(MATERIAL_LABEL_MAP_FILE, 'r') as f:
        label_map = json.load(f)
        material_class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    # 2. Load test data
    test_images, y_true = load_test_data()
    y_pred = []

    print(f"\nRunning predictions on {len(test_images)} test images...")
    # 3. Run two-phase prediction on all test images
    for image in tqdm(test_images, desc="Evaluating"):
        input_tensor = preprocess_image_for_prediction(image)
        
        # Phase 1
        liveness_pred = liveness_model.predict(input_tensor, verbose=0)[0][0]
        is_live = liveness_pred > 0.5

        if is_live:
            y_pred.append('live')
        else:
            # Phase 2
            material_preds = material_model.predict(input_tensor, verbose=0)[0]
            material_index = np.argmax(material_preds)
            material_name = material_class_names[material_index]
            y_pred.append(f'fake-{material_name}')
            
    # 4. Generate report and confusion matrix
    print("\n--- V2 System Evaluation Complete ---")
    
    # Get all unique class names from both true and predicted lists
    all_class_names = sorted(list(set(y_true + y_pred)))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=all_class_names, zero_division=0))
    
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=all_class_names)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_class_names, yticklabels=all_class_names)
    plt.title('Confusion Matrix for Two-Phase System (V2)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_v2.png')
    print("Confusion matrix saved to 'confusion_matrix_v2.png'")
    # plt.show() # Use this if running locally outside of a script

if __name__ == '__main__':
    evaluate_v2_system()
