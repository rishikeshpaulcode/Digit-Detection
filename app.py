import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# page configuration
st.set_page_config(
    page_title="Digit Detection App",
    page_icon="👁️‍🗨️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# app title and subtitle
st.title(":green[Digit Detection App]")
st.write(":grey[This is a Neural Network based basic Digit Detection and Classification app built using TensorFlow, Keras, NumPy, Streamlit.]")


# main content layout (columns)
col1, col2 = st.columns([1, 1])

with col1:
    # input through drawable canvas
    st.subheader("Canvas Input:")

    if st.get_option("theme.base") == "dark":
        canvas_bg = "#02112e"
    else:
        canvas_bg = "#989799"

    canvas_result = st_canvas(
        stroke_width=28,
        stroke_color="#27d604",
        background_color=canvas_bg,
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.button("Predict Digit", type="primary")

with col2:
    # display model predictions
    st.subheader("Model Prediction:")
    st.write(":grey[👈 Draw in Canvas and click 'Predict Digit' button to view model verdict.]")

