import streamlit as st
import tensorflow as tf


@st.cache
def load_model():
    model = tf.keras.models.load_model("./convolutional.h5")
    return model


st.title("Handwriting Recognition")
image = st.file_uploader(
    label="Upload an image",
    type=["jpg", "png"],
    accept_multiple_files=False,
    help="Upload an image to predict",
)

if image is not None and st.button("Predict"):
    st.image(image, use_column_width=True)
    model = load_model()
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    prediction = model.predict(image)
    st.write(prediction)
