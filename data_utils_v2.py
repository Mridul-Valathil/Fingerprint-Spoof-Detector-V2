import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import Counter

# (Re-usable helper function from the first project)
def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
                path = os.path.join(dirpath, filename)
                parent_folder = os.path.basename(os.path.dirname(path))
                grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
                label = None
                if parent_folder.lower() == 'live':
                    label = 'live'
                elif grandparent_folder.lower() == 'fake':
                    label = f'fake-{parent_folder}'
                if label:
                    image_paths.append(path)
                    labels.append(label)
    return image_paths, labels

def process_for_liveness_model():
    """
    Processes data for the binary (Live vs. Fake) model.
    All spoof types are mapped to a single 'fake' class.
    """
    print("--- Processing data for PHASE 1: Liveness Model ---")
    all_paths, all_labels = get_image_paths_and_labels('data/LivDet2015/Training')
    
    # Simplify all 'fake-*' labels to just 'fake'
    binary_labels = ['fake' if 'fake' in label else 'live' for label in all_labels]
    
    unique_labels = sorted(list(set(binary_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    with open('liveness_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Liveness label map created: {label_map}")

    # Process images...
    images, labels = [], []
    for i, path in enumerate(all_paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            resized = cv2.resize(image, (96, 96)).astype(np.float32) / 255.0
            images.append(resized)
            labels.append(label_map[binary_labels[i]])

    X = np.expand_dims(np.array(images), axis=-1)
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.savez_compressed('liveness_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("Saved 'liveness_data.npz'. Ready for liveness model training.")

def process_for_material_model():
    """
    Processes data for the multi-class (Material ID) model.
    It ONLY uses the 'fake' images.
    """
    print("\n--- Processing data for PHASE 2: Material Model ---")
    all_paths, all_labels = get_image_paths_and_labels('data/LivDet2015/Training')
    
    # Filter to only include fake images
    fake_paths = [p for i, p in enumerate(all_paths) if 'fake' in all_labels[i]]
    fake_labels = [l for l in all_labels if 'fake' in l]
    
    unique_labels = sorted(list(set(fake_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    with open('material_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Material label map created with {len(unique_labels)} classes.")

    # Process images...
    images, labels = [], []
    for i, path in enumerate(fake_paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            resized = cv2.resize(image, (96, 96)).astype(np.float32) / 255.0
            images.append(resized)
            labels.append(label_map[fake_labels[i]])

    X = np.expand_dims(np.array(images), axis=-1)
    y = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.savez_compressed('material_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print("Saved 'material_data.npz'. Ready for material model training.")


if __name__ == '__main__':
    process_for_liveness_model()
    process_for_material_model()
    print("\nAll data processing for V2 is complete!")

