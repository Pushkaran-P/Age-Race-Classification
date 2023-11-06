from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from PIL import Image
import os
import requests
import cv2
import numpy as np
import tensorflow as tf


# import dlib # comment when using mtcnn
from mtcnn import MTCNN
def hogDetectFaces(image, image_path):
    # Initialize HOG face detector only once
    # face_detector = dlib.get_frontal_face_detector()
    face_detector = MTCNN()

    def detect_face(image):
        height, width, _ = image.shape

        # OpenCV reads images in BGR format by default
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = imgRGB.copy()
        
        # results = face_detector(imgRGB, 0) # in case of dlib
        results = face_detector.detect_faces(imgRGB)

        # Initialize an empty list to store cropped face images
        cropped_images = []

        # Loop over each face detected
        for bbox in results:
            # Some images have bounding box in their borders
            

            # x1, y1 = max(0, bbox.left()), max(0, bbox.top()) # in case of dlib
            # x2, y2 = max(0, bbox.right()), max(0, bbox.bottom()) # in case of dlib

            x1, y1 = max(0, bbox['box'][0]), max(0, bbox['box'][1])
            x2, y2 = x1 + bbox['box'][2], y1 + bbox['box'][3]

            cropped_image_height, cropped_image_width = y2 - y1, x2 - x1

            if 71 < cropped_image_height < 643 and 68 < cropped_image_width < 642:
                # Append the cropped face image to the list
                cropped_images.append(output_image[y1:y2, x1:x2])

        # Return the list of cropped face images
        return cropped_images

    return detect_face(image)

import tensorflow as tf
def macro_f1(y_true, y_pred):
    # Convert predicted probabilities to class labels
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    # Calculate the number of true positive, false positive, and false negative predictions for each class
    true_positives = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=0), tf.float32)
    false_positives = tf.cast(tf.math.count_nonzero((1 - y_true) * y_pred, axis=0), tf.float32)
    false_negatives = tf.cast(tf.math.count_nonzero(y_true * (1 - y_pred), axis=0), tf.float32)

    # Calculate precision and recall for each class
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    # Calculate the F1 score for each class
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # Calculate the macro-averaged F1 score by taking the mean of the F1 scores for each class
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def get_predictions(cropped_images):
    # Load the model
    custom_objects = {'macro_f1': macro_f1}
    model = tf.keras.models.load_model('model', custom_objects=custom_objects)

    predictions = []
    for result in cropped_images:
        # Create a black image of size (642,641,3)
        padded_image = np.zeros((642, 641, 3), dtype=np.uint8)
        # Compute the center offset to place the cropped image on the black image
        x_offset = (padded_image.shape[1] - result.shape[1]) // 2
        y_offset = (padded_image.shape[0] - result.shape[0]) // 2
        # Place the cropped image onto the black image
        padded_image[y_offset:y_offset+result.shape[0], x_offset:x_offset+result.shape[1]] = result
        result = padded_image

        result = result.astype(float)
        result /= 255.
        real_image = result[np.newaxis, ...]

        # Make the prediction
        prediction = model.predict(real_image)

        predicted_class = np.argmax(prediction[0])

        if predicted_class == 0:
            output_range = '1-6'
        elif predicted_class == 1:
            output_range = '7-12'
        elif predicted_class == 2:
            output_range = '13-19'
        elif predicted_class == 3:
            output_range = '20-26'
        elif predicted_class == 4:
            output_range = '26-30'
        else:
            output_range = 'Invalid class'

        predictions.append(output_range)

    return predictions
    
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(request: Request, image: UploadFile = File(...)):
    file_path = os.path.join(os.getcwd(),'static','images',"uploaded.jpg")
    with open(file_path, "wb") as f:
        f.write(image.file.read())
    return await result(request)

@app.post("/upload-url")
async def upload_image_url(request: Request, image_url: str):
    # Download the image from the provided URL
    response = requests.get(image_url, stream=True)
    response.raise_for_status()

    file_path = os.path.join(os.getcwd(),"static","images" , "uploaded.jpg")
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return await result(request)

from PIL import Image
@app.get("/result")
async def result(request: Request):
    image_path = os.path.join(os.getcwd(),'static','images', "uploaded.jpg")
    image = cv2.imread(image_path)
    cropped_images = hogDetectFaces(image, image_path)
    predictions = get_predictions(cropped_images)

    # Save each cropped image as a separate file and collect the filenames
    filenames = []
    for i, img in enumerate(cropped_images):
        filename = f"cropped_{i}.jpg"
        Image.fromarray(img).save(os.path.join('static','images', filename))
        filenames.append(filename)

    results = list(zip(filenames, predictions))
    return templates.TemplateResponse("result.html", {"request": request, "results": results})
