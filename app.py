from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------
# Load TFLite Model
# -----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names
CLASS_NAMES = ["Clean Water", "Polluted Water"]

# -----------------------
# Helper: preprocess image
# -----------------------
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))   # match your training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_array = prepare_image(file.read())

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        result = {
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        print("ERROR during prediction:", str(e))
        result = {"error": str(e)}

    return jsonify(result)

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


