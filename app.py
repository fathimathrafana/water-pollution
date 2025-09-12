from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------
# Load Model
# -----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
model = load_model(MODEL_PATH)

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

    # Preprocess image (⚠️ do NOT divide by 255 because model already has Rescaling layer)
    img = image.load_img(temp_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions
    preds = model.predict(img_array)
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