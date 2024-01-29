import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from PIL import Image

# Load the object detection model
model_url = "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1"
detector = hub.load(model_url)

# Function to perform object detection on the uploaded image
def detect_objects(image):
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Perform object detection
    detector_output = detector(tf.convert_to_tensor([image_np]), as_dict=True)
    class_names = detector_output["detection_class_names"]

    # Draw bounding boxes on the image
    image_with_boxes = image_np.copy()
    for i in range(len(class_names)):
        bbox = detector_output["detection_boxes"][i].numpy()
        ymin, xmin, ymax, xmax = bbox
        xmin = int(xmin * image_np.shape[1])
        xmax = int(xmax * image_np.shape[1])
        ymin = int(ymin * image_np.shape[0])
        ymax = int(ymax * image_np.shape[0])

        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return image_with_boxes

# Streamlit GUI
st.title("Object Detection with TensorFlow and Streamlit")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the image
    image = Image.open(uploaded_image)

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    result_image = detect_objects(image)

    # Display the image with detected objects
    st.image(result_image, caption="Image with Detected Objects", use_column_width=True)

    # Display detected object class names
    st.write("Detected Classes:")
    st.write(detector(tf.convert_to_tensor([np.array(image)]), as_dict=True)["detection_class_names"])

# Streamlit command to run the app: streamlit run your_app_file.py
