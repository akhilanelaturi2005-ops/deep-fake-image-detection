from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile

app = Flask(__name__)

# Load the trained deepfake detection model
MODEL_PATH = 'my_modell1.h5'  # Updated to relative path
model = load_model(MODEL_PATH)

def preprocess_image(img):
    """Preprocess an image for model prediction."""
    # Convert BGR to RGB (OpenCV reads as BGR, but model expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Resize to match training input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] as per training
    return img_array

@app.route('/')
def index():
    """Render the main page with the upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and predict if it's real or fake."""
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        img_file.save(temp_img_path)

        # Read and preprocess image
        img = cv2.imread(temp_img_path)
        if img is None:
            os.unlink(temp_img_path)
            return jsonify({'error': 'Invalid image file'}), 400

        img_array = preprocess_image(img)
        print("Preprocessed shape:", img_array.shape)  # Debug print
        print("Prediction score:", model.predict(img_array)[0][0])  # Debug print
        prediction = model.predict(img_array)[0][0]

        # Determine result and probability
        result = "Real" if prediction > 0.55 else "Fake"
        probability = prediction if result == "Real" else 1 - prediction

        # Convert image to Base64 for display
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Clean up temporary file
        os.unlink(temp_img_path)

        return jsonify({
            "result": result,
            "probability": f"{probability:.2f}",
            "image": img_base64
        })

    except Exception as e:
        if os.path.exists(temp_img_path):
            os.unlink(temp_img_path)  # Clean up in case of error
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)