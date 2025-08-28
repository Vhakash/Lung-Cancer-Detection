#importing the necessary libraries
import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
import keras
import json


#configuration
IMG_SIZE = 224
MODEL_FILE = 'lung_cancer_detector_balanced.h5'
CLASS_NAMES_FILE = 'class_names.json'


#INITIALIZE THE FLASK APP
app = Flask(__name__)


# --- Load Model and Class Names ---
# Load the trained model
# We do this once when the app starts to avoid loading it on every request
print("Loading model...")
model = keras.models.load_model(MODEL_FILE)
print("Model loaded.")


#Load the class names
with open(CLASS_NAMES_FILE, 'r') as f:
    class_names_dict = json.load(f)

# Create a reverse mapping: index -> class name
class_names = {v: k for k, v in class_names_dict.items()}
print(f"Class names loaded: {class_names}")
print(f"Index to class mapping: {class_names}")


# --- Helper Function for Preprocessing ---
def preprocess_image(image_path):
    """
    Reads an image file, decodes it, resizes, and normalizes it.
    """
    # Read the file stream and convert it to a NumPy array
    filestr = image_path.read()
    npimg = np.frombuffer(filestr, np.uint8)

    # Decode the image from the NumPy array into OpenCv image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Check if image was decoded successfully
    if image is None:
        return {"error": "Failed to decode image. Please upload a valid image file."}

    # Resize the image to the target size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize the image
    image = image / 255.0

    # Expand dimensions to create a batch of 1
    # The model expects input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image


@app.route('/', methods = ['GET'])
def index():
    """
    Renders the main HTML page.
    """
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    """
    Handles image upload and returns prediction results.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        processed_image = preprocess_image(file)
        
        # Make a prediction
        prediction = model.predict(processed_image)  # type: ignore
        
        # Create a dictionary of class probabilities
        # The output of the model is a list of probabilities
        results = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        
        # Return the results as JSON
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # The 'debug=True' is helpful for development
    # The server will automatically reload if you make changes
    app.run(debug=True)
