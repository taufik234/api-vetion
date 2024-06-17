from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from typing import Union
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the trained model (replace 'my_model.h5' with your actual path)
model = load_model('MyModel.h5')

# Define labels (replace with your actual class names)
labels = {
    0: 'Brokoli Hijau Bagus',
    1: 'Brokoli Hijau Jelek',
    2: 'Daun Pepaya Bagus',
    3: 'Daun Pepaya Jelek',
    4: 'Daun Singkong Bagus',
    5: 'Daun Singkong Jelek',
    6: 'Daun Kelor Bagus',
    7: 'Daun Kelor Jelek',
    8: 'Kembang Kol Bagus',
    9: 'Kembang Kol Jelek',
    10: 'Kubis Hijau Bagus',
    11: 'Kubis Hijau Jelek',
    12: 'Paprika Merah Bagus',
    13: 'Paprika Merah Jelek',
    14: 'Sawi sendok atau Pakcoy Bagus',
    15: 'Sawi sendok atau Pakcoy Jelek',
    16: 'Tomat Merah Bagus',
    17: 'Tomat Merah Jelek',
    18: 'Wortel Nantes Bagus',
    19: 'Wortel Nantes Jelek',
}

# Define function to preprocess an image


def preprocess_image(image_path: Path) -> np.ndarray:
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define function to predict class for an image


def predict_class(image_path: Path) -> Union[str, None]:
    try:
        preprocessed_image = preprocess_image(image_path)
        prediction = model.predict(preprocessed_image)[0]
        predicted_class_index = np.argmax(prediction)
        return labels[predicted_class_index]
    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        return None


# Import FastAPI and dependencies

app = FastAPI()

# API endpoint to classify an uploaded image


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read image data
        image_bytes = await image.read()
        image_array = cv2.imdecode(np.fromstring(
            image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_path = Path("temp_image.jpg")  # Temporary file for processing
        cv2.imwrite(str(image_path), image_array)

        # Predict class
        predicted_class = predict_class(image_path)

        # Remove temporary image
        image_path.unlink()

        if predicted_class:
            return JSONResponse({"class": predicted_class})
        else:
            return JSONResponse({"error": "Error occurred during prediction"}, status_code=400)
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
