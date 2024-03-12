import cv2
import numpy as np
import json

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image from {image_path}")

    # Convert image to RGB (if it's in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to a standard size (e.g., 224x224 for deep learning models)
    image = cv2.resize(image, (224, 224))

    # Perform any other preprocessing steps as needed

    return image

def detect_face(image):
    # Use a face detection algorithm to locate the face in the input image
    # For demonstration purposes, we'll use the Haar cascade classifier
    # You can replace this with a more advanced face detection model if needed
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        raise RuntimeError("No faces detected in the image")

    # Assume the first face detected as the target face
    face_bbox = faces[0]

    return face_bbox

def extract_features(image, face_bbox):
    # Extract relevant features from the detected face
    # For demonstration purposes, we'll simply return the bounding box coordinates
    return face_bbox

def classify_skin_type(features):
    # Placeholder function to classify skin type based on extracted features
    # For demonstration purposes, we'll simply return a fixed skin type
    skin_type = "Spring"  # Placeholder, replace with actual classification
    return skin_type

def process_image(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Detect face in the preprocessed image
    face_bbox = detect_face(preprocessed_image)

    # Extract features from the detected face
    features = extract_features(preprocessed_image, face_bbox)

    # Classify skin type based on extracted features
    skin_type = classify_skin_type(features)

    return skin_type

if __name__ == "__main__":
    import sys

    # Get the image path from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python skin.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Process the image and generate output
        skin_type = process_image(image_path)

        # Create JSON string with output
        output = {"skin_type": skin_type}

        # Print JSON string
        print(json.dumps(output))

    except Exception as e:
        print("Error:", e)