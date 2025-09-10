import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
import cv2

app = Flask(__name__)

MODEL_PATH = "models/brain_tumor_model.h5"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

CATEGORIES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = 128

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No file selected")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        if img is None:
            return render_template("index.html", message="Invalid image format")

        prediction = model.predict(img)
        predicted_class = CATEGORIES[np.argmax(prediction)]

        return render_template("result.html", filename=file.filename, prediction=predicted_class)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
