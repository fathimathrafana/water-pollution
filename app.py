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

    # Save uploaded file temporarily
    temp_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(temp_path)

    try:
        # Preprocess image
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Debug logs
        print("DEBUG: Image shape:", img_array.shape)
        print("DEBUG: Min pixel:", np.min(img_array), "Max pixel:", np.max(img_array))

        # Predict
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Debug logs
        print("DEBUG: Raw predictions:", preds)
        print("DEBUG: Predicted class index:", class_idx)
        print("DEBUG: Confidence:", confidence)

        result = {
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        print("ERROR during prediction:", str(e))
        result = {"error": str(e)}

    finally:
        # Always remove the file to avoid filling storage
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    # Set host to 0.0.0.0 so Render can access it
    app.run(host="0.0.0.0", port=5000, debug=True)
