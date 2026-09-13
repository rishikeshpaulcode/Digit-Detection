import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models


# page configuration
st.set_page_config(
    page_title="Digit Detection App",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# custom styling
st.markdown(
    """
    <style>
    div.stProgress > div > div > div > div {
        background-color: #28a745; /* Green */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# session state variable
if "display_predictions" not in st.session_state:
    st.session_state["display_predictions"] = False


# load model
model = models.load_model("model/digit_detection_model.keras")


# app title and subtitle
st.title(":blue[Digit Detection App]")
st.write(":grey[This is a Neural Network based basic Digit Detection and Classification app built using TensorFlow, Keras, OpenCV, NumPy, and Streamlit.]")


# main content layout (columns)
col1, col2 = st.columns([1, 1])

with col1:
    # input through drawable canvas
    st.subheader("Canvas Input:")

    canvas_result = st_canvas(
        stroke_width=28,
        stroke_color="#055dff",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict Digit", type="primary"):
        st.session_state["display_predictions"] = True

with col2:
    # display model predictions
    st.subheader("Model Prediction:")

    if st.session_state["display_predictions"]:
        raw_image = canvas_result.image_data

        if np.any(raw_image[:, :, :3]):
            gray_image = cv2.cvtColor(raw_image.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
            normalized_image = resized_image.astype("float64") / 255.0

            model_predicted_array = model.predict(normalized_image[np.newaxis, :])
            model_verdict = np.argmax(model_predicted_array)
            model_confidence = model_predicted_array[0][model_verdict]            

            st.subheader(f"Final Verdict: :green[{model_verdict}]")
            st.progress(
                float(model_confidence),
                text=f":grey[Confidence:] :green[{(model_confidence * 100):.2f}%]"
            )
            st.write(":grey[Detailed Class Probabilities:]")
            st.bar_chart(
                model_predicted_array[0],
                color="#28a745",
                height=270
            )
        else:
            st.warning("Empty canvas cannot be processed.", icon="⚠️")

        st.session_state["display_predictions"] = False
    else:
        st.write(":grey[👈 Draw in Canvas and click 'Predict Digit' button to view model verdict.]")

