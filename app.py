from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained CNN model
model = load_model("models/detection_male_female_model_.keras")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Class labels
labels = ["Male", "Female"]

@app.get("/")
def home():
    return {"message": "Welcome to Face Detection & Gender Classification API 🚀"}

@app.post("/predict/")
async def predict_gender(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Convert image to numpy array
    image_np = np.array(image.convert("RGB"))

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return {"error": "No face detected"}

    predictions = []
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = image_np[y:y+h, x:x+w]  # Crop face
        face_roi = cv2.resize(face_roi, (96, 96))  # Resize to match model input
        face_roi = img_to_array(face_roi) / 255.0  # Normalize
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict gender
        pred = model.predict(face_roi)
        label = labels[int(pred[0] > 0.5)]  # Assuming binary classification (Male=0, Female=1)
        predictions.append({"face_coordinates": (x, y, w, h), "gender": label})

    return {"predictions": predictions}
