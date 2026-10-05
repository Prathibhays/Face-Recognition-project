import cv2
import numpy as np

IMAGE_SIZE = (64, 64)


def decode_image(image_bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid image.")
    return image


def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return gray, False

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return gray[y:y+h, x:x+w], True


def preprocess_image(image):
    face, detected = detect_and_crop_face(image)
    resized = cv2.resize(face, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float64) / 255.0
    return normalized.flatten(), detected


def preprocess_uploaded_image(image_bytes):
    return preprocess_image(decode_image(image_bytes))
