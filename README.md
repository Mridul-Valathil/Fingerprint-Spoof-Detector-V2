Fingerprint Spoof Detector V2
This project is an advanced, two-phase deep learning system for fingerprint liveness detection and spoof material classification. It uses a dual-model architecture to first determine if a fingerprint is genuine or a spoof, and then, if it is a spoof, to classify the material used to create it.

Project Overview
The system is built as a Flask web application with a modern, intuitive user interface. It provides real-time analysis of uploaded fingerprint images and maintains a detailed log of all scans.

Core Features
Two-Phase AI Architecture: Utilizes two specialized models for higher accuracy and modularity.

Liveness Detection: A binary classification model to distinguish between live and fake fingerprints.

Material Classification: A multi-class classification model to identify the specific material of a spoofed print (e.g., Gelatine, WoodGlue, Latex).

Web Interface: A user-friendly front-end for uploading images and viewing results.

Scan History: A dedicated page to review the logs of all past analyses.

The Two-Phase Architecture
This project employs a "Gatekeeper and Specialist" model pipeline:

Phase 1 (The Gatekeeper): A highly accurate CNN (liveness_model.keras) is used to perform a binary classification. It achieved ~94% validation accuracy in determining if a print is live or fake.

Phase 2 (The Specialist): If the Gatekeeper model flags an image as fake, a second, specialized CNN (material_model_v2.keras) is activated. This model was trained exclusively on spoofed images and is responsible for classifying the material. It achieved ~79% validation accuracy.

How to Run This Project
1. Setup the Environment
First, clone the repository and set up the Python environment.

# Clone this repository
git clone [https://github.com/Mridul-Valathil/Fingerprint-Spoof-Detector-V2.git](https://github.com/Mridul-Valathil/Fingerprint-Spoof-Detector-V2.git)
cd Fingerprint-Spoof-Detector-V2

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all required dependencies
pip install -r requirements.txt

2. Add the Dataset (Not Included)
This repository does not include the dataset due to its large size. You will need to download the LivDet2015 dataset and place it in the following structure:

./data/LivDet2015/
├── Training/
└── Testing/

3. Preprocess and Train Models (One-Time Step)
Before running the app, you must preprocess the data and train the two models.

# Preprocess the data into two separate datasets
python data_utils_v2.py

# Train the liveness model
python train_liveness_model.py

# Train the material model
python train_material_model.py

4. Run the Application
Once the models are trained, you can run the web application.

# On macOS, you may need to set this environment variable first
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Run the Flask server
python app_v2.py

Now, open your web browser and navigate to http://127.0.0.1:5000.

Technologies Used
Backend: Python, Flask, TensorFlow, Keras, OpenCV, scikit-learn

Frontend: HTML, Tailwind CSS, JavaScript

Dataset: LivDet2015