from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from PIL import Image
import tensorflow as tf

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

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names (order may need adjustment)
CLASS_NAMES = ["Clean Water", "Polluted Water"]


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

    # Save uploaded file temporarily
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", file.filename)
    file.save(temp_path)

    try:
        # Preprocess image
        img = Image.open(temp_path).convert("RGB")
        img = img.resize((224, 224))  # adjust if your model expects other size
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize if model expects float32 input
        if input_details[0]['dtype'] == np.float32:
            img_array = img_array / 255.0

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        # Debug: raw model output
        print("DEBUG OUTPUT:", preds, "Shape:", preds.shape)

        # Handle sigmoid (binary) vs softmax (multi-class)
        if preds.shape[-1] == 1:
            prob = float(preds[0][0])
            class_idx = 1 if prob > 0.5 else 0
            confidence = prob if class_idx == 1 else 1 - prob
        else:
            class_idx = int(np.argmax(preds))
            confidence = float(preds[0][class_idx])

        return jsonify({
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)


