import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import pickle
import tensorflow
from tensorflow.keras.models import load_model
import io
import os

st.set_page_config(
    page_title="3K2X",
    page_icon=":tooth:"
)
col1, col2 = st.columns([2, 4])
with col1:
    col1,col2=st.columns([2,2])
    with col1:
        st.title("3K2X")
        st.write("An AI app")
    with col2:
        st.image("tooth.png",width=30)
with col2:
    st.write("")
# with col3:
#     st.header("")
st.divider()
col1, col2 = st.columns([4, 2])
with col1:
    st.subheader("Check Your X-ray")
with col2:
    st.subheader("Articles")
col1, col2,col3 = st.columns([4,1, 2])
pred = 0

with col1:
    # Define a function to convert an image to grayscale
    def convert_to_grayscale(image):
        if image is None:
            return None
        try:
            image = Image.open(image)
        except Exception as e:
            st.warning("Error: Unable to open the uploaded image. Please upload a valid image.")
            return None
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            return image_np
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    st.write("Check you have wisdom teeth on both sides or not")
    uploaded_image = st.file_uploader("Choose panoramic photo in jpg, png, and jpeg extension", type=["jpg", "png", "jpeg"])
    pred = None

    if uploaded_image is not None:
        grayscale_image = convert_to_grayscale(uploaded_image)
        if grayscale_image is not None:
            st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

        # Resize and preprocess the image (if valid)
        if grayscale_image is not None:
            image = cv2.resize(grayscale_image, (224, 224))
            image = np.expand_dims(image, axis=-1)
            model = load_model('my_model1.h5')
            pred = model.predict(np.expand_dims(image, axis=0))
            if pred[0][0] >= 0.5:
                st.write("You have wisdom teeth on both sides on the lower jaw.")
                
            else:
                st.write("You don't have wisdom teeth on both sides of the lower jaw.")

        else:
            pred = np.array([[0]])
    st.divider()
    st.write("Detect your tooth location")
    uploaded_file2 = st.file_uploader("Choose panoramic photo in jpg, png, and jpeg extension")
    st.divider()
    model = load_model('final_tooth_mask_generation.h5')
    def segment_tooth(model, image):
        image = image.reshape(1, 224, 224, 1)
        mask = model.predict(image)
        threshold = 0.5
        segmented_mask = (mask > threshold).astype(np.uint8)
        return segmented_mask

    def convert_to_grayscale(image):
        if image is None:
            return None
        try:
            image = Image.open(image)
        except Exception as e:
            st.warning("Error: Unable to open the uploaded image. Please upload a valid image.")
            return None
        image_np = np.array(image)
        if len (image_np.shape) == 2:
            return image_np
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    if uploaded_file2 is not None:
        grayscale_image = convert_to_grayscale(uploaded_file2)  # Load image in grayscale
        if grayscale_image is not None:
            st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
            resized_image = cv2.resize(grayscale_image, (224, 224))  # Resize to model input size
            preprocessed_image = resized_image / 255.0  # Normalize pixel values
            preprocessed_image = preprocessed_image.reshape(1, 224, 224, 1)  # Reshape for model input
            segmented_tooth_mask = segment_tooth(model, preprocessed_image)
            segmented_tooth_mask = segmented_tooth_mask.squeeze()
            kernel1 = np.ones((2, 2), dtype=np.float32)
            image = cv2.erode(segmented_tooth_mask, kernel1, iterations=1)
            output_image = resized_image.copy()
        # Rest of your code...


            for x1 in range(image.shape[1]):
                for y1 in range(image.shape[0] - 1, image.shape[0] // 2, -1):
                    region = image[y1 - 2:y1 + 2, x1 - 2:x1 + 2]
                    if np.all(region == 1) and x1 > 30:
                        print(x1, y1)
                        print(f"First white pixel found at coordinates (x1={x1}, y1={y1})")
                        break
                region = image[y1 - 2:y1 + 2, x1 - 2:x1 + 2]
                if np.all(region == 1) and x1 > 30:
                    break

            for x2 in range(image.shape[1] - 1, -1, -1):
                for y2 in range(image.shape[0] - 1, image.shape[0] // 2, -1):
                    region = image[y2 - 2:y2 + 2, x2 - 2:x2 + 2]
                    if np.all(region == 1) and x2 < image.shape[1] - 30:
                        print(x2, y2)
                        print(f"First white pixel found at coordinates (x2={x2}, y2={y2})")
                        break
                region = image[y2 - 2:y2 + 2, x2 - 2:x2 + 2]
                if np.all(region == 1) and x2 < image.shape[1] - 30:
                    break

            result = cv2.rectangle(output_image, (x1 - 17, y1 - 25), (x1 + 10, y1 + 10), (0, 255, 0), 2)
            final_result = cv2.rectangle(result, (x2 - 15, y2 - 27), (x2 + 10, y2 + 10), (0, 255, 0), 2)
            st.image(final_result, caption='Sunrise by the mountains')
            st.write("You can check whether wisdom tooth exists or not in the black boxes and whether your tooth is in the jawbone")
with col2:
    st.write("")
with col3:
        st.image("1.webp",width=200)
        st.write("Why tooth Decay")
        st.caption("Dr. Arthur M.")
        st.divider()
        st.image("2.jpeg",width=200)
        st.write("Why should we do teeth implant")
        st.caption("Dr. Catherine K.")
        st.divider()
        st.image("3.jpeg",width=200)
        st.write("Tips about braces and cribs")
        st.caption("Dr. Matthew")
        st.divider()
