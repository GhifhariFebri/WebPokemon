from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.compat.v1.keras.models.load_model('pokemon_classifier.h5')
image_base_dir = r"C:\Users\AGIF\Downloads\archive(1)\archive\extracted_images"

# List available directories in the base path
available_directories = os.listdir(image_base_dir)
class_names = available_directories

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))  # Adjust size according to your model's input size
    img = img.convert('RGB')  # Convert to RGB mode if necessary
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize image data

    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_prob = np.max(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    return jsonify({
        'prediction': predicted_class_name
    })


if __name__ == '__main__':
    app.run(debug=True)
