import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
CLASS_NAMES = ["Clean Water", "Polluted Water"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Save temporarily
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))   # make sure this matches your training size
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Handle binary (1 value) vs softmax (2 values)
        if prediction.shape[-1] == 1:
            prob = float(prediction[0][0])
            predicted_class = "Polluted Water" if prob >= 0.5 else "Clean Water"
            confidence = prob if predicted_class == "Polluted Water" else 1 - prob
        else:
            probs = prediction[0]
            class_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[class_idx]
            confidence = float(probs[class_idx])

        # Clean up
        os.remove(filepath)

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

