from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from PIL import Image
import tflite_runtime.interpreter as tflite   # ✅ use lightweight TFLite runtime

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------
# Load TFLite Model
# -----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tflite")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names
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

    # Save uploaded file
    temp_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(temp_path)

    # Preprocess image (⚠️ same input size you used in training, here assumed 224x224)
    img = Image.open(temp_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Debug log
    print("Raw predictions:", preds)

    # Delete file after prediction
    os.remove(temp_path)

    return jsonify({
        "predicted_class": CLASS_NAMES[class_idx],
        "confidence": round(confidence, 4)
    })

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)


