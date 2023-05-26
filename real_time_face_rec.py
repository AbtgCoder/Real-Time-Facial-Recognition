import os
import numpy as np
import cv2
import tensorflow as tf
from model_training import L1Dist

# Constants
DETECTION_THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5
VERIFICATION_IMAGES_DIR = os.path.join("application_data", "verification_images")
INPUT_IMAGE_DIR = os.path.join("application_data", "input_image")
INPUT_IMAGE_PATH = os.path.join(INPUT_IMAGE_DIR, "input_image.jpg")

# Load the trained model
model = tf.keras.models.load_model("siamese_model.h5",
                                   custom_objects={
                                       "L1Dist": L1Dist,
                                       "BinaryCrossentropy": tf.losses.BinaryCrossentropy
                                   })

# Preprocess input image
def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# Verification function
def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(VERIFICATION_IMAGES_DIR):
        validation_img = preprocess_image(os.path.join(VERIFICATION_IMAGES_DIR, image))
        result = model.predict(list(np.expand_dims([validation_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(VERIFICATION_IMAGES_DIR))
    verified = verification > verification_threshold

    return results, verified

# Capture video and perform verification
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow("Verification", frame)

    frame = frame[120:120+2250, 200:200+250, :]
    if cv2.waitKey(10) & 0xFF == ord("v"):
        cv2.imwrite(INPUT_IMAGE_PATH, frame)

        # Run verification
        results, verified = verify(model, DETECTION_THRESHOLD, VERIFICATION_THRESHOLD)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
