
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/image_classifier.keras")

model = load_model()

st.title("🖼️ Image Classification")
st.write("Upload an image and let the CNN predict what it contains.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    resized_image = image.resize((32, 32))
    image_array = np.array(resized_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Classify Image"):
        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index] * 100

        st.success(f"Prediction: {class_names[predicted_index]}")
        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.subheader("Prediction Scores")

        for i, name in enumerate(class_names):
            st.write(f"{name}: {predictions[i] * 100:.2f}%")
            st.progress(float(predictions[i]))

