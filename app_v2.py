from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import random
from datetime import datetime
import base64

app = Flask(__name__)

# --- Configuration for V2 ---
LIVENESS_MODEL_FILE = 'liveness_model.keras'
MATERIAL_MODEL_FILE = 'material_model_v2.keras'
MATERIAL_LABEL_MAP_FILE = 'material_label_map.json'
LOG_FILE = 'scan_log_v2.json'
IMG_SIZE = 96

# --- Lazy Load Globals for V2 ---
liveness_model = None
material_model = None
material_class_names = None

def load_models_once():
    """Loads both specialized models if they haven't been loaded yet."""
    global liveness_model, material_model, material_class_names
    if liveness_model is None:
        try:
            print("--- Loading V2 models for the first time... ---")
            liveness_model = tf.keras.models.load_model(LIVENESS_MODEL_FILE)
            material_model = tf.keras.models.load_model(MATERIAL_MODEL_FILE)
            with open(MATERIAL_LABEL_MAP_FILE, 'r') as f:
                label_map = json.load(f)
                # Ensure the class names are sorted by index
                material_class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
            print("--- V2 models and label map loaded successfully. ---")
        except Exception as e:
            print(f"FATAL ERROR: Could not load V2 models or label map. {e}")

def preprocess_image_for_prediction(image):
    """Prepares a raw image for model prediction."""
    # Ensure image is grayscale
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    resized_image = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE))
    normalized_image = resized_image.astype(np.float32) / 255.0
    # Add batch and channel dimensions: (1, 96, 96, 1)
    return np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=-1)

def log_event(log_data):
    """Appends a new log entry to the JSON log file."""
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    logs.insert(0, log_data)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

def analyze_image_v2(image, source, filename="N/A"):
    """
    Implements the two-phase prediction pipeline.
    """
    load_models_once() # Ensure models are loaded
    if not all([liveness_model, material_model, material_class_names]):
        return {"error": "Models are not loaded, cannot perform analysis."}

    input_tensor = preprocess_image_for_prediction(image)
    
    # --- PHASE 1: Liveness Detection ---
    liveness_pred = liveness_model.predict(input_tensor)[0][0]
    is_live = liveness_pred > 0.5 # Sigmoid outputs a probability
    liveness_confidence = liveness_pred if is_live else 1 - liveness_pred

    # Prepare initial result from phase 1
    result = {
        "is_spoof": not is_live,
        "confidence": float(liveness_confidence),
        "predicted_class": "live" if is_live else "fake"
    }

    # --- PHASE 2: Material Classification (only if fake) ---
    if not is_live:
        material_preds = material_model.predict(input_tensor)[0]
        material_index = np.argmax(material_preds)
        material_confidence = np.max(material_preds)
        
        material_name = material_class_names[material_index]
        result["suspected_material"] = material_name
        result["confidence"] = float(material_confidence) # Update confidence to material model's
        result["predicted_class"] = f"fake-{material_name}"

    # --- Logging ---
    _, buffer = cv2.imencode('.png', cv2.resize(image, (100, 100)))
    img_preview_b64 = base64.b64encode(buffer).decode('utf-8')

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_id": str(random.randint(100000, 999999)),
        "input_source": source,
        "image_file": filename,
        "predicted_class": result["predicted_class"],
        "confidence": f"{result['confidence']:.2%}",
        "is_spoof": result["is_spoof"],
        "image_preview": f"data:image/png;base64,{img_preview_b64}"
    }
    if "suspected_material" in result:
        log_entry["suspected_material"] = result["suspected_material"]

    log_event(log_entry)
    
    # Return the result object for the frontend
    return result

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def history():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    return render_template('history.html', logs=logs)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('fingerprint_image')
    if not file or file.filename == '':
        return jsonify({"error": "No image file provided."}), 400

    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        return jsonify({"error": "Could not decode image."}), 400

    result = analyze_image_v2(image, "image_upload", file.filename)
    return jsonify(result)

if __name__ == '__main__':
    # Required for macOS compatibility with TensorFlow
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    app.run(debug=True, use_reloader=False)

