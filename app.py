from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
class_names = ['cellulitis', 'nail-fungus', 'ringworm', 'chickenpox', 'shingles']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use safe absolute path for the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    predicted_label = class_names[class_index]

    print("Prediction index:", class_index)
    print("Prediction label:", predicted_label)

    return render_template('index.html', prediction=predicted_label, image_path=file_path)
